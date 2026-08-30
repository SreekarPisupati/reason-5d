"""
Reason-X: Module 3.3 - Numerical Stability Suite & Automated Loss Spike Recovery
================================================================================
Architectural Specification:
- Logit Z-Loss: L_z = eps_z * (log sum exp(logits))^2 to penalize unbounded logits.
- Gradient Spike Detector: Monitors ||g||_2 across steps. If ||g|| > tau, triggers automated
  checkpoint rollback, skips corrupted batch, and adjusts gradient clipping threshold.
- QK-Norm layer: Normalizes Query and Key head vectors before dot product to prevent softmax saturation.
"""

import copy
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_z_loss(logits: torch.Tensor, eps_z: float = 1e-4) -> torch.Tensor:
    """
    Computes Logit Z-Loss: eps_z * log(sum(exp(logits)))^2.
    Prevents logit drift and numerical overflow during long pretraining runs.
    """
    logsumexp = torch.logsumexp(logits, dim=-1)
    z_loss = eps_z * torch.mean(logsumexp.pow(2))
    return z_loss


class QKNorm(nn.Module):
    """Per-head RMS normalization for Query and Key tensors."""

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, S, D]
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.scale


class GradientSpikeAutoRecovery:
    """
    Automated loss spike detector and recovery controller.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        spike_threshold: float = 10.0,
        max_recovery_attempts: int = 3
    ):
        self.model = model
        self.optimizer = optimizer
        self.spike_threshold = spike_threshold
        self.max_recovery_attempts = max_recovery_attempts
        
        self.last_clean_model_state: Optional[Dict[str, Any]] = None
        self.last_clean_opt_state: Optional[Dict[str, Any]] = None
        self.recovery_count = 0

    def save_clean_checkpoint(self) -> None:
        """Saves CPU shadow copy of clean weights and optimizer states."""
        self.last_clean_model_state = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }
        self.last_clean_opt_state = copy.deepcopy(self.optimizer.state_dict())

    def check_and_recover(self) -> Tuple[bool, float]:
        """
        Calculates global gradient norm. If ||g|| > spike_threshold or NaN/Inf,
        rolls back model and optimizer state to the last clean checkpoint.
        
        Returns:
            (is_recovered, grad_norm)
        """
        # Compute global gradient norm
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm_sq += param_norm ** 2

        grad_norm = total_norm_sq ** 0.5
        is_spike = (grad_norm > self.spike_threshold) or torch.isnan(torch.tensor(grad_norm)).item() or torch.isinf(torch.tensor(grad_norm)).item()

        if is_spike:
            if self.last_clean_model_state is not None:
                # Rollback weights
                self.model.load_state_dict(self.last_clean_model_state)
                self.optimizer.load_state_dict(self.last_clean_opt_state)
                self.optimizer.zero_grad()
                self.recovery_count += 1
                return True, grad_norm
            else:
                self.optimizer.zero_grad()
                return True, grad_norm

        return False, grad_norm
