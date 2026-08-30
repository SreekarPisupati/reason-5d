"""
Reason-X: Module 2.6 - 1F1B (One-Forward-One-Backward) Pipeline Parallelism Scheduler
====================================================================================
Architectural Specification:
- Implements Megatron-LM style 1F1B micro-batch scheduling.
- Bounds peak stored activation memory to at most p stages (where p is pipeline depth),
  eliminating the O(m) activation explosion of GPipe.
- Manages non-blocking P2P activation transfers (forward) and gradient transfers (backward)
  between adjacent pipeline ranks (stage i <-> stage i+1).
- Theoretical Bubble Fraction: F_bubble = (p - 1) / (m + p - 1).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist


class PipelineStage(nn.Module):
    """
    Represents a partitioned subset of model layers running on a specific pipeline rank.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        stage_id: int,
        num_stages: int,
        process_group: Optional[dist.ProcessGroup] = None
    ):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.process_group = process_group
        self.is_first_stage = (stage_id == 0)
        self.is_last_stage = (stage_id == num_stages - 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, **kwargs)
        return h


class OneForwardOneBackwardScheduler:
    """
    1F1B Pipeline Schedule Coordinator.
    """

    def __init__(
        self,
        stage: PipelineStage,
        num_microbatches: int,
        forward_step_fn: Callable[[Any, Any], Tuple[torch.Tensor, torch.Tensor]],
        backward_step_fn: Callable[[torch.Tensor, torch.Tensor], None]
    ):
        self.stage = stage
        self.num_microbatches = num_microbatches
        self.forward_step_fn = forward_step_fn
        self.backward_step_fn = backward_step_fn

        self.p = stage.num_stages
        self.m = num_microbatches
        self.stage_id = stage.stage_id

    @property
    def theoretical_bubble_fraction(self) -> float:
        """Computes theoretical idle bubble ratio: (p - 1) / (m + p - 1)."""
        return (self.p - 1) / (self.m + self.p - 1)

    def run_1f1b_schedule(
        self,
        microbatches: List[Any],
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, float]:
        """
        Executes the 1F1B schedule across warmup, steady-state, and cooldown phases.
        """
        num_warmup_microbatches = min(self.p - self.stage_id - 1, self.m)
        num_1f1b_microbatches = self.m - num_warmup_microbatches

        saved_inputs: List[torch.Tensor] = []
        saved_outputs: List[torch.Tensor] = []
        total_loss = 0.0

        # -------------------------------------------------------------
        # 1. Warmup Phase: Forward only to populate pipeline
        # -------------------------------------------------------------
        for mb_idx in range(num_warmup_microbatches):
            input_tensor, output_tensor, loss = self.forward_step_fn(microbatches[mb_idx], mb_idx)
            saved_inputs.append(input_tensor)
            saved_outputs.append(output_tensor)
            if loss is not None:
                total_loss += loss.item()

        # -------------------------------------------------------------
        # 2. 1F1B Steady-State: 1 Forward followed by 1 Backward
        # -------------------------------------------------------------
        for mb_idx in range(num_1f1b_microbatches):
            # Forward step
            actual_mb = mb_idx + num_warmup_microbatches
            input_tensor, output_tensor, loss = self.forward_step_fn(microbatches[actual_mb], actual_mb)
            saved_inputs.append(input_tensor)
            saved_outputs.append(output_tensor)
            if loss is not None:
                total_loss += loss.item()

            # Backward step on oldest stored microbatch
            bwd_input = saved_inputs.pop(0)
            bwd_output = saved_outputs.pop(0)
            self.backward_step_fn(bwd_input, bwd_output)

        # -------------------------------------------------------------
        # 3. Cooldown Phase: Backward on remaining microbatches
        # -------------------------------------------------------------
        while saved_inputs:
            bwd_input = saved_inputs.pop(0)
            bwd_output = saved_outputs.pop(0)
            self.backward_step_fn(bwd_input, bwd_output)

        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad()

        return {
            "loss": total_loss / max(1, self.m),
            "bubble_fraction": self.theoretical_bubble_fraction
        }
