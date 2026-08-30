"""
Reason-X: Module 4.2 - DeepSeek-R1 Group Relative Policy Optimization (GRPO)
=============================================================================
Architectural Specification:
- DeepSeek-R1 Pure Reinforcement Learning Algorithm (DeepSeek-AI, 2025).
- Eliminates the Critic / Value network entirely (saving 50% VRAM and removing value drift).
- Group Relative Advantage Normalization across G sampled rollouts:
  A_i = (r_i - mean({r})) / (std({r}) + eps)
- Clipped surrogate policy gradient objective with per-token reference model KL regularization:
  L_GRPO = - (1 / G) sum_i [ min(ratio_i * A_i, clip(ratio_i, 1-eps, 1+eps) * A_i) - beta * KL(pi_theta || pi_ref) ]
"""

from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRPOTrainer:
    """
    Group Relative Policy Optimization (GRPO) Trainer.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: Optional[nn.Module],
        optimizer: torch.optim.Optimizer,
        group_size: int = 4,
        clip_eps: float = 0.2,
        kl_beta: float = 0.04,
        max_grad_norm: float = 1.0
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        self.max_grad_norm = max_grad_norm

    @staticmethod
    def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Computes group-normalized advantages: A_i = (r_i - mu) / (sigma + eps).
        rewards: [G] or [B, G]
        """
        mean = torch.mean(rewards, dim=-1, keepdim=True)
        std = torch.std(rewards, dim=-1, keepdim=True)
        advantages = (rewards - mean) / (std + eps)
        return advantages

    def compute_grpo_loss(
        self,
        prompt_ids: torch.Tensor,              # [B, S_prompt]
        generated_ids: torch.Tensor,           # [B, G, S_gen]
        old_log_probs: torch.Tensor,           # [B, G, S_gen]
        rewards: torch.Tensor,                 # [B, G]
        gen_mask: Optional[torch.Tensor] = None# [B, G, S_gen]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the GRPO surrogate loss with per-token reference model KL penalty.
        """
        B, G, S_gen = generated_ids.shape
        S_prompt = prompt_ids.shape[-1]

        # 1. Compute Group Advantages: [B, G]
        advantages = self.compute_group_advantages(rewards)  # [B, G]
        adv_expanded = advantages.unsqueeze(-1)              # [B, G, 1]

        # 2. Flatten batch and group for parallel model forward pass
        # Full sequence: [prompt, generated] -> [B * G, S_prompt + S_gen]
        prompt_expanded = prompt_ids.unsqueeze(1).expand(-1, G, -1).contiguous().view(B * G, S_prompt)
        gen_flat = generated_ids.view(B * G, S_gen)
        full_ids = torch.cat([prompt_expanded, gen_flat], dim=-1)

        # 3. Compute current policy log probs
        logits = self.policy_model(full_ids)  # [B * G, S_full, Vocab]
        gen_logits = logits[:, S_prompt - 1 : -1, :]  # Logits predicting generated tokens
        curr_log_probs = -F.cross_entropy(
            gen_logits.contiguous().view(-1, gen_logits.shape[-1]),
            gen_flat.contiguous().view(-1),
            reduction="none"
        ).view(B, G, S_gen)

        # 4. Compute reference policy log probs (if ref_model is provided)
        if self.ref_model is not None:
            with torch.no_grad():
                ref_logits = self.ref_model(full_ids)
                ref_gen_logits = ref_logits[:, S_prompt - 1 : -1, :]
                ref_log_probs = -F.cross_entropy(
                    ref_gen_logits.contiguous().view(-1, ref_gen_logits.shape[-1]),
                    gen_flat.contiguous().view(-1),
                    reduction="none"
                ).view(B, G, S_gen)
                
                # Per-token KL divergence: exp(log_ref - log_curr) - (log_ref - log_curr) - 1 (Schulman approximation)
                log_ratio = ref_log_probs - curr_log_probs
                kl_penalty = torch.exp(log_ratio) - log_ratio - 1.0
        else:
            kl_penalty = torch.zeros_like(curr_log_probs)

        # 5. Clipped Policy Ratio
        ratio = torch.exp(curr_log_probs - old_log_probs)
        surr1 = ratio * adv_expanded
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_expanded
        surrogate_loss = torch.min(surr1, surr2)

        # 6. Combined objective with KL regularization
        per_token_obj = surrogate_loss - self.kl_beta * kl_penalty

        if gen_mask is not None:
            per_token_obj = per_token_obj * gen_mask
            loss = -torch.sum(per_token_obj) / torch.clamp(gen_mask.sum(), min=1.0)
        else:
            loss = -torch.mean(per_token_obj)

        metrics = {
            "loss_grpo": loss.item(),
            "mean_reward": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "approx_kl": kl_penalty.mean().item()
        }

        return loss, metrics

    def step(
        self,
        prompt_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        gen_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Performs one optimizer step on GRPO objective."""
        self.optimizer.zero_grad()
        loss, metrics = self.compute_grpo_loss(prompt_ids, generated_ids, old_log_probs, rewards, gen_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return metrics
