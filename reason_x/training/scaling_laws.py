"""
Reason-X: Module 3.1 - Empirical Scaling Laws & Chinchilla Compute-Optimal Frontier
==================================================================================
Architectural Specification:
- Parametric Power-Law Fitting (Hoffmann et al., DeepMind Chinchilla 2022):
  L(N, D) = E + A / N^alpha + B / D^beta
- Micro-architecture sweep across parameter scales (10M, 25M, 50M, 100M).
- Computes compute-optimal parameter-to-data allocation for any target compute budget C = 6ND.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import curve_fit


def power_law_loss(
    grid: Tuple[np.ndarray, np.ndarray],
    E: float,
    A: float,
    B: float,
    alpha: float,
    beta: float
) -> np.ndarray:
    """Parametric power-law loss function: L(N, D) = E + A * N^(-alpha) + B * D^(-beta)."""
    N, D = grid
    return E + A / (N ** alpha) + B / (D ** beta)


class ScalingLawFitter:
    """
    Fits empirical scaling laws from measured pretraining evaluation loss sweeps.
    """

    def __init__(self):
        self.params: Dict[str, float] = {}

    def fit(
        self,
        n_params: List[float],
        n_tokens: List[float],
        val_losses: List[float]
    ) -> Dict[str, float]:
        """
        Fits (E, A, B, alpha, beta) parameters on empirical sweep data.
        """
        N = np.array(n_params, dtype=np.float64)
        D = np.array(n_tokens, dtype=np.float64)
        L = np.array(val_losses, dtype=np.float64)

        # Initial parameter estimates
        initial_guess = [1.5, 400.0, 400.0, 0.35, 0.35]
        bounds = ([0.1, 1.0, 1.0, 0.05, 0.05], [5.0, 1e5, 1e5, 1.0, 1.0])

        popt, _ = curve_fit(
            power_law_loss,
            (N, D),
            L,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000
        )

        self.params = {
            "E": float(popt[0]),
            "A": float(popt[1]),
            "B": float(popt[2]),
            "alpha": float(popt[3]),
            "beta": float(popt[4])
        }
        return self.params

    def predict_loss(self, N: float, D: float) -> float:
        """Predicts validation loss for an arbitrary parameter count N and token count D."""
        if not self.params:
            raise ValueError("Model must be fitted before predicting loss.")
        E = self.params["E"]
        A = self.params["A"]
        B = self.params["B"]
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        return float(E + A / (N ** alpha) + B / (D ** beta))

    def compute_optimal_allocation(self, compute_flops_budget: float) -> Tuple[float, float]:
        """
        Given a FLOPs budget C = 6ND, computes compute-optimal N_opt and D_opt.
        """
        if not self.params:
            raise ValueError("Model must be fitted first.")
        
        A = self.params["A"]
        B = self.params["B"]
        alpha = self.params["alpha"]
        beta = self.params["beta"]

        # Ratio of exponents: a / b
        # N_opt ~ C^(beta / (alpha + beta)), D_opt ~ C^(alpha / (alpha + beta))
        exponent_n = beta / (alpha + beta)
        exponent_d = alpha / (alpha + beta)

        coeff_ratio = (alpha * A) / (beta * B)
        c_base = compute_flops_budget / 6.0

        n_opt = (c_base * (coeff_ratio ** (1.0 / beta))) ** (beta / (alpha + beta))
        d_opt = compute_flops_budget / (6.0 * max(1.0, n_opt))

        return float(n_opt), float(d_opt)
