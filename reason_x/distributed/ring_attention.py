"""
Reason-X: Module 2.2 - Context Parallelism (CP) via Ring-Attention
===================================================================
Architectural Specification:
- Implements Liu et al. (UC Berkeley, 2023) RingAttention with Blockwise Transformers.
- Slices context length across W context-parallel GPUs (S_local = S / W).
- Ring Communication: Cyclic non-blocking P2P Key/Value block passing (dist.P2POp / isend / irecv)
  overlapped with attention computation.
- Numerically stable Online Softmax: Progressively accumulates attention statistics
  (running max m and running denominator l) across ring steps without materializing the full S x S matrix.
- Parity SLA: Output matches global single-GPU scaled dot-product attention to < 1e-5 tolerance.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def online_softmax_step(
    q: torch.Tensor,               # [B, H, S_q, D]
    k: torch.Tensor,               # [B, H, S_kv, D]
    v: torch.Tensor,               # [B, H, S_kv, D]
    out_accum: torch.Tensor,       # [B, H, S_q, D]
    max_accum: torch.Tensor,       # [B, H, S_q, 1]
    lse_accum: torch.Tensor,       # [B, H, S_q, 1]
    scale: float,
    causal: bool = False,
    step_idx: int = 0,
    rank: int = 0,
    world_size: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Executes one blockwise online softmax attention step and updates running stats.
    """
    # Dot product attention scores: [B, H, S_q, S_kv]
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    # Causal masking logic across ring steps
    # Global query range: [rank * S_q, (rank + 1) * S_q)
    # Key range at step_idx: held KV originated from rank_kv = (rank - step_idx) % world_size
    if causal:
        kv_rank = (rank - step_idx) % world_size
        if kv_rank > rank:
            # Entire KV block is in the future -> fully masked
            return out_accum, max_accum, lse_accum
        elif kv_rank == rank:
            # Same rank -> apply causal lower-triangular mask
            S_q = q.shape[-2]
            S_kv = k.shape[-2]
            causal_mask = torch.triu(torch.ones(S_q, S_kv, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
        else:
            # kv_rank < rank -> entire KV block is in the past -> fully visible
            pass

    # Block max: [B, H, S_q, 1]
    block_max = torch.max(scores, dim=-1, keepdim=True)[0]
    # Replace -inf with large negative finite number to avoid NaN in exp
    block_max = torch.where(torch.isneginf(block_max), torch.full_like(block_max, -1e4), block_max)

    # Compute new running max
    new_max = torch.maximum(max_accum, block_max)

    # Exponentials
    exp_scores = torch.exp(scores - new_max)
    block_sum = torch.sum(exp_scores, dim=-1, keepdim=True)

    # Scale previous sum and accumulator
    alpha = torch.exp(max_accum - new_max)
    new_lse = lse_accum * alpha + block_sum

    # Block attention output
    block_out = torch.matmul(exp_scores, v)

    # Update output accumulator
    new_out = (out_accum * (lse_accum * alpha) + block_out) / torch.clamp(new_lse, min=1e-8)

    return new_out, new_max, new_lse


def ring_attention_forward(
    q: torch.Tensor,                          # Local query: [B, H, S_local, D]
    k: torch.Tensor,                          # Local key:   [B, H, S_local, D]
    v: torch.Tensor,                          # Local value: [B, H, S_local, D]
    process_group: Optional[dist.ProcessGroup] = None,
    causal: bool = True,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    RingAttention forward pass with cyclic KV non-blocking communication across CP ranks.
    """
    B, H, S_q, D = q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(D)

    if process_group is None or not dist.is_initialized() or dist.get_world_size(process_group) <= 1:
        # Fallback to local scaled dot product attention
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    # Running accumulators
    out_accum = torch.zeros_like(q)
    max_accum = torch.full((B, H, S_q, 1), fill_value=-1e4, device=q.device, dtype=q.dtype)
    lse_accum = torch.zeros((B, H, S_q, 1), device=q.device, dtype=q.dtype)

    curr_k = k.contiguous()
    curr_v = v.contiguous()

    # Pre-allocate receive buffers
    next_k = torch.empty_like(k)
    next_v = torch.empty_like(v)

    for step in range(world_size):
        # 1. Asynchronously send curr_k, curr_v to next_rank and receive from prev_rank (for step < world_size - 1)
        reqs = []
        if step < world_size - 1:
            send_k_op = dist.P2POp(dist.isend, curr_k, next_rank, group=process_group)
            send_v_op = dist.P2POp(dist.isend, curr_v, next_rank, group=process_group)
            recv_k_op = dist.P2POp(dist.irecv, next_k, prev_rank, group=process_group)
            recv_v_op = dist.P2POp(dist.irecv, next_v, prev_rank, group=process_group)
            reqs = dist.batch_isend_irecv([send_k_op, send_v_op, recv_k_op, recv_v_op])

        # 2. Overlap computation: Compute online attention with currently held KV block
        out_accum, max_accum, lse_accum = online_softmax_step(
            q=q,
            k=curr_k,
            v=curr_v,
            out_accum=out_accum,
            max_accum=max_accum,
            lse_accum=lse_accum,
            scale=scale,
            causal=causal,
            step_idx=step,
            rank=rank,
            world_size=world_size
        )

        # 3. Wait for communication to finish
        if reqs:
            for req in reqs:
                req.wait()
            curr_k, next_k = next_k, curr_k
            curr_v, next_v = next_v, curr_v

    return out_accum
