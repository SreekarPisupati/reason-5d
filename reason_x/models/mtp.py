"""
Reason-X: Module 2.5 - DeepSeek-V3 Multi-Token Prediction (MTP)
===============================================================
Architectural Specification:
- Multi-Token Prediction (DeepSeek-V3 Technical Report, 2024).
- Predicts k future tokens concurrently (t+1, t+2, ..., t+K) during training.
- Boosts pretraining sample efficiency and natively enables self-speculative decoding at serving time
  without requiring a separate draft model.
- Shared embedding projection and output vocabulary head weights across MTP modules.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from reason_x.models.attention import MultiHeadLatentAttention
from reason_x.models.moe import SparseMoELayer


class MTPModule(nn.Module):
    """
    Single Multi-Token Prediction (MTP) depth-k prediction block.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        num_heads: int = 16,
        q_lora_rank: int = 512,
        kv_lora_rank: int = 512,
        num_routed_experts: int = 8,
        num_shared_experts: int = 1,
        top_k: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1. Representation projection combining previous hidden state + current token embedding
        self.norm_h = nn.RMSNorm(d_model)
        self.norm_emb = nn.RMSNorm(d_model)
        self.fuse_proj = nn.Linear(2 * d_model, d_model, bias=False)

        # 2. Transformer Block (MLA + MoE)
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = MultiHeadLatentAttention(
            d_model=d_model,
            num_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank
        )
        
        self.ffn_norm = nn.RMSNorm(d_model)
        self.moe = SparseMoELayer(
            d_model=d_model,
            d_ffn=d_ffn,
            num_routed_experts=num_routed_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k
        )

    def forward(
        self,
        h_prev: torch.Tensor,       # [B, S, d_model] hidden representation from stage k-1
        tok_emb: torch.Tensor,      # [B, S, d_model] embedding of target token at step t+k-1
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Fuse normalized representation and token embedding
        h_norm = self.norm_h(h_prev)
        emb_norm = self.norm_emb(tok_emb)
        fused = self.fuse_proj(torch.cat([h_norm, emb_norm], dim=-1))

        # Residual Attention Block
        res = fused
        h = res + self.attn(self.attn_norm(fused), attention_mask=attention_mask)

        # Residual MoE Block
        h = h + self.moe(self.ffn_norm(h))
        return h


class MultiTokenPredictionEngine(nn.Module):
    """
    Full MTP engine managing K parallel prediction heads and self-speculative decoding.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        vocab_size: int,
        num_future_tokens: int = 2,
        num_heads: int = 16,
        q_lora_rank: int = 512,
        kv_lora_rank: int = 512,
        num_routed_experts: int = 8,
        num_shared_experts: int = 1,
        top_k: int = 2,
        mtp_loss_weight: float = 0.3
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_future_tokens = num_future_tokens
        self.mtp_loss_weight = mtp_loss_weight

        # Shared Token Embedding
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        
        # Output Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.final_norm = nn.RMSNorm(d_model)

        # K Sequential MTP Modules
        self.mtp_modules = nn.ModuleList([
            MTPModule(
                d_model=d_model,
                d_ffn=d_ffn,
                num_heads=num_heads,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                num_routed_experts=num_routed_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k
            )
            for _ in range(num_future_tokens)
        ])

    def compute_mtp_loss(
        self,
        backbone_hidden_states: torch.Tensor,  # [B, S, d_model]
        input_ids: torch.Tensor,               # [B, S]
        labels: torch.Tensor,                  # [B, S]
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes concurrent MTP prediction losses for k=1..K future tokens.
        """
        B, S, _ = backbone_hidden_states.shape
        total_mtp_loss = torch.tensor(0.0, device=backbone_hidden_states.device)
        metrics: Dict[str, float] = {}

        curr_h = backbone_hidden_states

        for k, module in enumerate(self.mtp_modules):
            future_step = k + 1  # Predicting token at index t + future_step
            if S <= future_step:
                break

            # Slices for prediction: tokens at [0 : S - future_step] predict labels at [future_step : S]
            h_slice = curr_h[:, : S - future_step, :]
            target_ids = input_ids[:, : S - future_step]
            target_embs = self.embed_tokens(target_ids)
            target_labels = labels[:, future_step : S]

            # Forward through MTP module
            h_k = module(h_slice, target_embs, attention_mask=None)
            logits_k = self.lm_head(self.final_norm(h_k))  # [B, S - future_step, vocab_size]

            # Cross entropy loss
            loss_k = F.cross_entropy(
                logits_k.view(-1, self.vocab_size),
                target_labels.contiguous().view(-1)
            )

            total_mtp_loss = total_mtp_loss + loss_k
            metrics[f"loss_mtp_k{future_step}"] = loss_k.item()
            curr_h = h_k  # Cascaded representation

        weighted_mtp_loss = self.mtp_loss_weight * total_mtp_loss
        metrics["loss_mtp_total"] = weighted_mtp_loss.item()
        return weighted_mtp_loss, metrics

    @torch.no_grad()
    def speculative_propose(
        self,
        backbone_hidden_last: torch.Tensor,  # [B, 1, d_model]
        current_token_id: torch.Tensor       # [B, 1]
    ) -> List[torch.Tensor]:
        """
        Proposes K future speculative tokens in a single fast forward pass.
        Returns list of predicted token IDs [token_t+1, token_t+2, ...].
        """
        proposals: List[torch.Tensor] = []
        curr_h = backbone_hidden_last
        curr_token = current_token_id

        for module in self.mtp_modules:
            curr_emb = self.embed_tokens(curr_token)
            curr_h = module(curr_h, curr_emb)
            logits = self.lm_head(self.final_norm(curr_h))
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            proposals.append(next_token)
            curr_token = next_token

        return proposals
