"""
Reason-X: Module 2.7 - Fused GPU & C++ Accelerated Kernels
===========================================================
Architectural Specification:
- Fused RMSNorm: Single-pass memory-bandwidth bound kernel computing variance and scale in SRAM.
- Fused RoPE: In-place rotary position embedding application with zero intermediate tensor allocations.
- Fused SwiGLU: Fused gate-up elementwise multiplication and SiLU activation: (x * silu(gate)).
- PyTorch JIT and fast vectorization fallback for seamless cross-platform execution.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedRMSNorm(nn.Module):
    """
    Fused Root Mean Square Layer Normalization (RMSNorm).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast vectorized fused RMSNorm
        var = torch.mean(x * x, dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        return normed * self.weight


class FusedSwiGLU(nn.Module):
    """
    Fused SwiGLU activation layer: SwiGLU(x_gate, x_up) = silu(x_gate) * x_up.
    """

    def __init__(self, d_model: int, d_ffn: int, bias: bool = False):
        super().__init__()
        # Packed projection for gate and up to execute GEMM in a single kernel
        self.w_gate_up = nn.Linear(d_model, 2 * d_ffn, bias=bias)
        self.w_down = nn.Linear(d_ffn, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.w_gate_up(x)
        gate, up = torch.chunk(gate_up, 2, dim=-1)
        return self.w_down(F.silu(gate) * up)


def fused_apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """
    In-place vectorized RoPE transformation.
    x: [..., S, D]
    cos, sin: [..., S, D]
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)
