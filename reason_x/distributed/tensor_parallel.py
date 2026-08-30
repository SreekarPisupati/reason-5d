"""
Reason-X: Module 2.1 - Megatron-Style Tensor Parallelism (TP) & Sequence Parallelism (SP)
========================================================================================
Architectural Specification:
- ColumnParallelLinear: Splits weight matrix W along output features (columns).
  * Forward: X @ W_col -> Output partition.
  * Backward: Custom autograd function performs NCCL/Gloo all_reduce over input gradients dX.
- RowParallelLinear: Splits weight matrix W along input features (rows).
  * Forward: X_partition @ W_row -> all_reduce across TP group.
  * Backward: Custom autograd computes local dW and dX without extra communication.
- Sequence Parallelism (SP): Slices LayerNorm/RMSNorm & Dropout activations along the sequence
  dimension S / TP_SIZE, eliminating redundant activation memory.
  * Reduces activation memory by TP_SIZE fold.
  * Transitions between SP and TP domains via reduce_scatter and all_gather.
- Gloo & CPU fallback support for local unit tests and NCCL for multi-GPU training.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# -----------------------------------------------------------------------------
# Autograd Comm Functions for Tensor Parallelism
# -----------------------------------------------------------------------------

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Passes input forward unchanged; all-reduces gradients on backward."""

    @staticmethod
    def forward(ctx, input_, process_group=None):
        ctx.process_group = process_group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.process_group is not None and dist.is_initialized() and dist.get_world_size(ctx.process_group) > 1:
            dist.all_reduce(grad_output, group=ctx.process_group)
        return grad_output, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduces input forward; passes gradients backward unchanged."""

    @staticmethod
    def forward(ctx, input_, process_group=None):
        ctx.process_group = process_group
        if ctx.process_group is not None and dist.is_initialized() and dist.get_world_size(ctx.process_group) > 1:
            output = input_.clone()
            dist.all_reduce(output, group=ctx.process_group)
            return output
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """
    Forward: Scatter tensor along sequence dimension (dim 1) from [B, S, H] -> [B, S/TP, H].
    Backward: All-gather gradients along sequence dimension.
    """

    @staticmethod
    def forward(ctx, input_, process_group=None, dim=1):
        ctx.process_group = process_group
        ctx.dim = dim
        if ctx.process_group is None or not dist.is_initialized() or dist.get_world_size(ctx.process_group) <= 1:
            return input_

        world_size = dist.get_world_size(ctx.process_group)
        rank = dist.get_rank(ctx.process_group)
        seq_len = input_.size(dim)
        assert seq_len % world_size == 0, f"Sequence length {seq_len} must be divisible by TP size {world_size}"
        
        chunks = torch.chunk(input_, world_size, dim=dim)
        return chunks[rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.process_group is None or not dist.is_initialized() or dist.get_world_size(ctx.process_group) <= 1:
            return grad_output, None, None

        world_size = dist.get_world_size(ctx.process_group)
        tensor_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(tensor_list, grad_output.contiguous(), group=ctx.process_group)
        return torch.cat(tensor_list, dim=ctx.dim).contiguous(), None, None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """
    Forward: All-gather tensor along sequence dimension (dim 1) from [B, S/TP, H] -> [B, S, H].
    Backward: Reduce-scatter gradients along sequence dimension.
    """

    @staticmethod
    def forward(ctx, input_, process_group=None, dim=1):
        ctx.process_group = process_group
        ctx.dim = dim
        if ctx.process_group is None or not dist.is_initialized() or dist.get_world_size(ctx.process_group) <= 1:
            return input_

        world_size = dist.get_world_size(ctx.process_group)
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_.contiguous(), group=ctx.process_group)
        return torch.cat(tensor_list, dim=dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.process_group is None or not dist.is_initialized() or dist.get_world_size(ctx.process_group) <= 1:
            return grad_output, None, None

        world_size = dist.get_world_size(ctx.process_group)
        rank = dist.get_rank(ctx.process_group)
        chunks = torch.chunk(grad_output, world_size, dim=ctx.dim)
        return chunks[rank].contiguous(), None, None


# -----------------------------------------------------------------------------
# Parallel Linear Layers
# -----------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """
    Linear layer partitioned across columns (out_features / TP_SIZE).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        sequence_parallel: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

        self.tp_world_size = dist.get_world_size(process_group) if (process_group and dist.is_initialized()) else 1
        self.tp_rank = dist.get_rank(process_group) if (process_group and dist.is_initialized()) else 0

        assert out_features % self.tp_world_size == 0, "out_features must be divisible by TP world size"
        self.split_out_features = out_features // self.tp_world_size

        self.weight = nn.Parameter(torch.empty(self.split_out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.split_out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Master initialization equivalent to standard Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sequence_parallel:
            # Sliced along sequence length -> all-gather along sequence dimension
            input_parallel = _GatherFromSequenceParallelRegion.apply(x, self.process_group)
        else:
            input_parallel = _CopyToModelParallelRegion.apply(x, self.process_group)

        output_parallel = F.linear(input_parallel, self.weight, self.bias)

        if self.gather_output and self.tp_world_size > 1:
            # All-gather along features dimension if explicitly requested
            tensor_list = [torch.empty_like(output_parallel) for _ in range(self.tp_world_size)]
            dist.all_gather(tensor_list, output_parallel.contiguous(), group=self.process_group)
            return torch.cat(tensor_list, dim=-1).contiguous()

        return output_parallel


class RowParallelLinear(nn.Module):
    """
    Linear layer partitioned across rows (in_features / TP_SIZE).
    Performs all-reduce on output or scatter to sequence parallel domain.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
        sequence_parallel: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

        self.tp_world_size = dist.get_world_size(process_group) if (process_group and dist.is_initialized()) else 1
        self.tp_rank = dist.get_rank(process_group) if (process_group and dist.is_initialized()) else 0

        assert in_features % self.tp_world_size == 0, "in_features must be divisible by TP world size"
        self.split_in_features = in_features // self.tp_world_size

        self.weight = nn.Parameter(torch.empty(out_features, self.split_in_features))
        if bias:
            # Bias is added once on rank 0 or added after all_reduce
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_parallel = x
        if not self.input_is_parallel and self.tp_world_size > 1:
            chunks = torch.chunk(x, self.tp_world_size, dim=-1)
            input_parallel = chunks[self.tp_rank].contiguous()

        output_parallel = F.linear(input_parallel, self.weight)

        if self.sequence_parallel:
            # Reduce-scatter directly into sequence parallel domain [B, S/TP, H]
            output_ = _ScatterToSequenceParallelRegion.apply(output_parallel, self.process_group)
        else:
            output_ = _ReduceFromModelParallelRegion.apply(output_parallel, self.process_group)

        if self.bias is not None:
            output_ = output_ + self.bias

        return output_


class SequenceParallelRMSNorm(nn.Module):
    """
    RMSNorm operating directly on sequence-parallel shards [B, S/TP, H].
    Since normalization is computed along the hidden dimension H, it executes
    with ZERO inter-GPU communication across sequence length shards.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * normed
