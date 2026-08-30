"""
Reason-5D: Daily Drill A1 - Multi-Head Attention (MHA) with KV-Cache
====================================================================
Architecture:
- Projects input X into Q, K, V heads.
- Prefill vs Decode:
  * Prefill (step 0): Computes attention over all S prompt tokens, populates KV cache.
  * Decode (step > 0): Receives 1 new token [B, 1, D], appends new K_t and V_t to KV cache,
    and attends across the entire cached history [B, H, S_total, head_dim].
- Output projection: Multi-head output merged and projected back to d_model.

Your Goal:
Implement `MultiHeadAttentionWithKVCache` below and pass the verification test.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionWithKVCache(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        # Projections
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,                                      # [B, S, d_model]
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # (K_cached, V_cached)
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes MHA and returns (output, (new_k_cache, new_v_cache)).
        """
        # TODO: Implement MHA with KV cache update
        raise NotImplementedError("Implement MultiHeadAttentionWithKVCache.forward")


def verify_mha_kv_cache():
    torch.manual_seed(42)
    B, d_model, num_heads = 1, 32, 4
    mha = MultiHeadAttentionWithKVCache(d_model=d_model, num_heads=num_heads)
    mha.eval()

    # Step 1: Prefill prompt of length 4
    prompt = torch.randn(B, 4, d_model)
    out_prefill, kv_cache = mha(prompt, kv_cache=None)
    assert out_prefill.shape == (B, 4, d_model)
    assert kv_cache[0].shape == (B, num_heads, 4, 8)

    # Step 2: Autoregressive decoding step 1 (1 token)
    next_token = torch.randn(B, 1, d_model)
    out_decode1, kv_cache = mha(next_token, kv_cache=kv_cache)
    assert out_decode1.shape == (B, 1, d_model)
    assert kv_cache[0].shape == (B, num_heads, 5, 8)

    # Step 3: Verify equivalence with full unrolled sequence (prompt + next_token)
    full_seq = torch.cat([prompt, next_token], dim=1)
    out_full, _ = mha(full_seq, kv_cache=None)

    diff = (out_decode1[:, 0, :] - out_full[:, -1, :]).abs().max().item()
    print(f"Max absolute difference between cached decode and full unroll: {diff:.6e}")
    assert diff < 1e-5, f"KV cache decode output does not match full unroll (diff={diff})"
    print(" PASSED: MHA with KV-Cache matches exact autoregressive generation!")


if __name__ == "__main__":
    verify_mha_kv_cache()
