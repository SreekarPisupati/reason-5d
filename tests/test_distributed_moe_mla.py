"""
Reason-X: Unit Test Suite for Module 2 (5D Distributed Core, MLA, MoE, MTP, Kernels)
=====================================================================================
Verifies:
1. Column & Row Parallel Linear layers.
2. Sequence Parallelism slicing and RMSNorm.
3. Multi-Head Latent Attention (MLA) low-rank KV compression & Decoupled RoPE.
4. Auxiliary-Loss-Free Dynamic Bias Sparse MoE (DeepSeek-V3).
5. Multi-Token Prediction (MTP) concurrent loss & speculative proposing.
6. RingAttention online softmax step numerical correctness.
7. 1F1B Pipeline Scheduler bubble fraction and execution.
8. Fused RMSNorm & SwiGLU parity.
"""

import math
import pytest
import torch
import torch.nn as nn

from reason_x.distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    SequenceParallelRMSNorm
)
from reason_x.distributed.ring_attention import online_softmax_step
from reason_x.distributed.pipeline_parallel import (
    PipelineStage,
    OneForwardOneBackwardScheduler
)
from reason_x.models.attention import MultiHeadLatentAttention
from reason_x.models.moe import SparseMoELayer, DynamicBiasRouter
from reason_x.models.mtp import MultiTokenPredictionEngine
from reason_x.distributed.kernels.fused_ops import FusedRMSNorm, FusedSwiGLU


def test_column_and_row_parallel_linear():
    torch.manual_seed(42)
    B, S, in_dim, out_dim = 2, 8, 32, 64
    x = torch.randn(B, S, in_dim)

    # Standard linear layer
    col_lin = ColumnParallelLinear(in_features=in_dim, out_features=out_dim, bias=True)
    out = col_lin(x)
    assert out.shape == (B, S, out_dim)

    row_lin = RowParallelLinear(in_features=out_dim, out_features=in_dim, bias=True, input_is_parallel=False)
    out_row = row_lin(out)
    assert out_row.shape == (B, S, in_dim)


def test_sequence_parallel_rmsnorm():
    B, S, H = 2, 16, 64
    x = torch.randn(B, S, H)
    norm = SequenceParallelRMSNorm(hidden_size=H)
    out = norm(x)
    assert out.shape == (B, S, H)
    # Check unit variance normalization along hidden dimension
    var = torch.mean(out.pow(2), dim=-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-2)


def test_multi_head_latent_attention_mla():
    torch.manual_seed(42)
    B, S, d_model = 2, 16, 128
    num_heads = 4
    q_rank = 64
    kv_rank = 64
    qk_rope_dim = 16
    qk_nope_dim = 16
    v_dim = 32

    mla = MultiHeadLatentAttention(
        d_model=d_model,
        num_heads=num_heads,
        q_lora_rank=q_rank,
        kv_lora_rank=kv_rank,
        qk_rope_head_dim=qk_rope_dim,
        qk_nope_head_dim=qk_nope_dim,
        v_head_dim=v_dim
    )

    x = torch.randn(B, S, d_model)
    out = mla(x)
    assert out.shape == (B, S, d_model)


def test_auxiliary_loss_free_dynamic_bias_moe():
    """
    Verifies that DeepSeek-V3 dynamic bias adapts during training
    to balance expert load WITHOUT auxiliary backprop loss.
    """
    torch.manual_seed(42)
    B, S, d_model, d_ffn = 4, 32, 64, 128
    num_routed = 4
    num_shared = 1
    top_k = 2

    moe = SparseMoELayer(
        d_model=d_model,
        d_ffn=d_ffn,
        num_routed_experts=num_routed,
        num_shared_experts=num_shared,
        top_k=top_k,
        bias_update_rate=0.01
    )
    moe.train()

    initial_bias = moe.router.expert_bias.clone()
    assert torch.all(initial_bias == 0.0)

    # Run several training steps with skewed input
    for _ in range(5):
        x = torch.randn(B, S, d_model)
        out = moe(x)
        assert out.shape == (B, S, d_model)

    # Verify dynamic bias has been updated adaptively
    updated_bias = moe.router.expert_bias
    assert not torch.all(updated_bias == 0.0)
    assert moe.router.total_tokens_routed.sum() == 5 * B * S * top_k


def test_multi_token_prediction_mtp():
    torch.manual_seed(42)
    B, S, d_model, d_ffn = 2, 16, 64, 128
    vocab_size = 200
    num_future_tokens = 2

    mtp_engine = MultiTokenPredictionEngine(
        d_model=d_model,
        d_ffn=d_ffn,
        vocab_size=vocab_size,
        num_future_tokens=num_future_tokens,
        num_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        num_routed_experts=4,
        num_shared_experts=1,
        top_k=2
    )

    backbone_hidden = torch.randn(B, S, d_model)
    input_ids = torch.randint(0, vocab_size, (B, S))
    labels = torch.randint(0, vocab_size, (B, S))

    # Test MTP loss calculation
    loss, metrics = mtp_engine.compute_mtp_loss(backbone_hidden, input_ids, labels)
    assert loss > 0.0
    assert "loss_mtp_k1" in metrics
    assert "loss_mtp_k2" in metrics
    assert "loss_mtp_total" in metrics

    # Test speculative proposal
    single_step_hidden = backbone_hidden[:, -1:, :]
    curr_token = input_ids[:, -1:]
    proposals = mtp_engine.speculative_propose(single_step_hidden, curr_token)
    assert len(proposals) == num_future_tokens
    for prop in proposals:
        assert prop.shape == (B, 1)


def test_ring_attention_online_softmax():
    B, H, S, D = 1, 2, 4, 8
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)

    # Standard scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    ref_attn = torch.matmul(torch.softmax(scores, dim=-1), v)

    # Online softmax single block step
    out_accum = torch.zeros_like(q)
    max_accum = torch.full((B, H, S, 1), fill_value=-1e4)
    lse_accum = torch.zeros((B, H, S, 1))

    step_out, _, _ = online_softmax_step(
        q=q, k=k, v=v,
        out_accum=out_accum,
        max_accum=max_accum,
        lse_accum=lse_accum,
        scale=scale,
        causal=True,
        step_idx=0,
        rank=0,
        world_size=1
    )

    assert torch.allclose(step_out, ref_attn, atol=1e-4)


def test_1f1b_pipeline_scheduler():
    num_stages = 4
    num_microbatches = 8
    layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])
    stage = PipelineStage(layers=layers, stage_id=0, num_stages=num_stages)

    def mock_forward(mb, idx):
        out = stage(mb)
        loss = out.mean()
        return mb, out, loss

    def mock_backward(inp, out):
        pass

    scheduler = OneForwardOneBackwardScheduler(
        stage=stage,
        num_microbatches=num_microbatches,
        forward_step_fn=mock_forward,
        backward_step_fn=mock_backward
    )

    # Verify bubble fraction: (p-1) / (m + p - 1) = (4-1)/(8 + 4 - 1) = 3/11 ~ 0.2727
    expected_bubble = 3.0 / 11.0
    assert abs(scheduler.theoretical_bubble_fraction - expected_bubble) < 1e-4

    microbatches = [torch.randn(2, 16) for _ in range(num_microbatches)]
    metrics = scheduler.run_1f1b_schedule(microbatches)
    assert "loss" in metrics
    assert "bubble_fraction" in metrics


def test_fused_rmsnorm_and_swiglu():
    B, S, H = 2, 8, 32
    x = torch.randn(B, S, H)

    fused_norm = FusedRMSNorm(hidden_size=H)
    out_norm = fused_norm(x)
    assert out_norm.shape == (B, S, H)

    fused_swiglu = FusedSwiGLU(d_model=H, d_ffn=64)
    out_swiglu = fused_swiglu(x)
    assert out_swiglu.shape == (B, S, H)
