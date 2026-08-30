"""
Reason-5D: Daily Drill B1 - Muon's 5th-Order Newton-Schulz Iteration
====================================================================
Mathematical Problem:
Standard optimizers (SGD, AdamW) update weight matrices based on coordinate-wise gradient moments.
Muon (Keller Jordan 2024 / Moonlight 2025) orthogonalizes the momentum matrix G such that all singular
values become approximately 1.0 (approximate polar decomposition / UV^T matrix orthogonalization).

Algorithm:
1. Normalize: X_0 = G / (||G||_F + eps)
2. 5th-Order Newton-Schulz polynomial iteration for k = 0..steps-1:
   A = X_k @ X_k^T (or X_k^T @ X_k depending on aspect ratio)
   B = 3.4445 * I - 4.7750 * A + 2.0315 * (A @ A)
   X_{k+1} = B @ X_k
3. Output: X_steps

Your Goal:
Implement `newton_schulz_iteration` below and verify that the output matrix has orthonormal columns/rows.
"""

import torch


def newton_schulz_iteration(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Orthogonalizes matrix G using Newton-Schulz polynomial iteration.
    Computes approximate UV^T polar decomposition.
    """
    # TODO: Implement 5th-order Newton-Schulz iteration
    raise NotImplementedError("Implement newton_schulz_iteration")


def verify_newton_schulz():
    torch.manual_seed(42)
    # Test on a rectangular weight gradient matrix [M, N]
    M, N = 64, 32
    G = torch.randn(M, N)

    X = newton_schulz_iteration(G, steps=5)

    # For tall matrix [M, N] with M >= N: X^T @ X should approximate identity I_N
    gram = X.t() @ X
    I = torch.eye(N)

    diff = (gram - I).abs().max().item()
    print(f"Max absolute deviation from orthonormal Identity (||X^T X - I||_inf): {diff:.6e}")
    assert diff < 0.05, f"Matrix is not properly orthogonalized (deviation={diff})"
    print(" PASSED: Newton-Schulz iteration produces orthonormal momentum updates!")


if __name__ == "__main__":
    verify_newton_schulz()
