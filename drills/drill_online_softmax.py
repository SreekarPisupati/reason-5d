"""
Reason-5D: Daily Drill D1 - Online Softmax Derivation & Step
============================================================
Mathematical Problem:
Standard Softmax requires materializing the full S x S matrix to compute max(S) and sum(exp(S)).
Online Softmax (Milakov & Gimelshein 2018 / FlashAttention) maintains running max m and running
sum of exponents l, updating the output accumulator O iteratively across blocks without storing
intermediate attention weights in HBM.

Recurrence Equations:
1. New Max:          m_new = max(m_prev, max(S_curr))
2. Exponent Scale:   alpha = exp(m_prev - m_new)
3. New Denominator:  l_new = l_prev * alpha + sum(exp(S_curr - m_new))
4. New Output:       O_new = (O_prev * (l_prev * alpha) + exp(S_curr - m_new) @ V_curr) / l_new

Your Goal:
Implement `online_softmax_update` below and verify numerical equivalence with standard Attention.
"""

from typing import Tuple
import torch


def online_softmax_update(
    S_curr: torch.Tensor,       # [B, H, S_q, S_kv_block] current score block (Q @ K_block^T * scale)
    V_curr: torch.Tensor,       # [B, H, S_kv_block, D] current value block
    O_prev: torch.Tensor,       # [B, H, S_q, D] previous output accumulator
    m_prev: torch.Tensor,       # [B, H, S_q, 1] previous max
    l_prev: torch.Tensor        # [B, H, S_q, 1] previous denominator sum
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes one online softmax update step.
    Returns: (O_new, m_new, l_new)
    """
    # TODO: Implement the 4 online softmax equations
    raise NotImplementedError("Implement online_softmax_update")


def verify_online_softmax():
    torch.manual_seed(42)
    B, H, S, D = 1, 2, 8, 16
    scale = 1.0 / (D ** 0.5)

    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)

    # 1. Standard Global Attention Reference
    scores = (Q @ K.transpose(-1, -2)) * scale
    attn = torch.softmax(scores, dim=-1)
    ref_O = attn @ V

    # 2. Blockwise Online Softmax (simulate 2 blocks of size 4)
    block_size = 4
    num_blocks = S // block_size

    O = torch.zeros(B, H, S, D)
    m = torch.full((B, H, S, 1), fill_value=-1e4)
    l = torch.zeros(B, H, S, 1)

    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        K_block = K[:, :, start:end, :]
        V_block = V[:, :, start:end, :]

        S_block = (Q @ K_block.transpose(-1, -2)) * scale
        O, m, l = online_softmax_update(S_block, V_block, O, m, l)

    diff = (O - ref_O).abs().max().item()
    print(f"Max absolute difference vs Standard Attention: {diff:.6e}")
    assert diff < 1e-5, f"Online Softmax output does not match reference (diff={diff})"
    print(" PASSED: Online Softmax matches exact Attention!")


if __name__ == "__main__":
    verify_online_softmax()
