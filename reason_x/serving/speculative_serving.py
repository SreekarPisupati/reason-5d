"""
Reason-X: Module 5.2 - Self-Speculative Serving Engine with MTP Heads
=====================================================================
Architectural Specification:
- High-Throughput Speculative Serving utilizing native DeepSeek-V3 MTP heads.
- Draft Phase: MTP modules propose K candidate tokens concurrently in a single forward step.
- Target Verification Phase: Main model validates all K candidates in parallel in 1 forward GEMM.
- SLA: >= 2.0x wall-clock speedup and >= 70% acceptance rate on reasoning benchmarks.
"""

from typing import Callable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from reason_x.models.mtp import MultiTokenPredictionEngine


class MTPSpeculativeEngine:
    """
    Self-Speculative Serving Coordinator using trained Multi-Token Prediction heads.
    """

    def __init__(
        self,
        main_model: nn.Module,
        mtp_engine: MultiTokenPredictionEngine,
        k_speculative_tokens: int = 2
    ):
        self.main_model = main_model
        self.mtp_engine = mtp_engine
        self.k = k_speculative_tokens

        # Verification metrics
        self.total_drafted = 0
        self.total_accepted = 0

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / max(1, self.total_drafted)

    @torch.no_grad()
    def speculative_step(
        self,
        context_ids: torch.Tensor,          # [1, S]
        backbone_hidden_last: torch.Tensor # [1, 1, d_model]
    ) -> Tuple[torch.Tensor, int]:
        """
        Executes one speculative generation & verification iteration.
        Returns:
            (newly_accepted_tokens, num_accepted)
        """
        curr_token = context_ids[:, -1:]

        # 1. MTP Draft Proposal: Propose K future tokens
        draft_tokens_list = self.mtp_engine.speculative_propose(backbone_hidden_last, curr_token)
        if not draft_tokens_list:
            return curr_token, 0

        draft_tokens = torch.cat(draft_tokens_list, dim=-1)  # [1, K]
        self.total_drafted += self.k

        # 2. Parallel Target Verification
        # Candidate sequence to evaluate: [context_ids, draft_tokens]
        candidate_seq = torch.cat([context_ids, draft_tokens], dim=-1)  # [1, S + K]
        target_logits = self.main_model(candidate_seq)                  # [1, S + K, Vocab]

        # 3. Check acceptance sequentially along the draft chain
        accepted_tokens = []
        S = context_ids.shape[-1]

        for i in range(self.k):
            # Target prediction for position S + i
            pred_logits = target_logits[:, S + i - 1, :]
            target_token = torch.argmax(pred_logits, dim=-1, keepdim=True)
            draft_token = draft_tokens[:, i : i + 1]

            if target_token.item() == draft_token.item():
                accepted_tokens.append(draft_token)
                self.total_accepted += 1
            else:
                # Misprediction -> take target model's corrected token and stop speculative chain
                accepted_tokens.append(target_token)
                break
        else:
            # All K tokens accepted -> append 1 bonus target token predicted at the end
            bonus_logits = target_logits[:, S + self.k - 1, :]
            bonus_token = torch.argmax(bonus_logits, dim=-1, keepdim=True)
            accepted_tokens.append(bonus_token)

        result = torch.cat(accepted_tokens, dim=-1)
        return result, len(accepted_tokens)
