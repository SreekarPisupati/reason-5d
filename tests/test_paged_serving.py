"""
Reason-X: Unit Tests for Module 5 (PagedAttention Block Manager & MTP Speculative Serving)
========================================================================================
Verifies:
1. PagedAttention PhysicalBlockManager allocation, expansion, forking, and deallocation.
2. Self-Speculative Serving engine token proposing and verification.
"""

import pytest
import torch
import torch.nn as nn

from reason_x.serving.paged_attention import PhysicalBlockManager
from reason_x.serving.speculative_serving import MTPSpeculativeEngine
from reason_x.models.mtp import MultiTokenPredictionEngine


def test_physical_block_manager():
    num_blocks = 10
    block_size = 4
    mgr = PhysicalBlockManager(num_blocks=num_blocks, block_size=block_size)

    assert mgr.num_free_blocks == 10

    # Allocate sequence 1
    b1 = mgr.allocate_sequence(seq_id=1)
    assert mgr.num_free_blocks == 9
    assert mgr.get_block_table(1) == [b1]

    # Append 3 tokens (total 3 <= block_size) -> same block
    for t in range(1, 4):
        active_b = mgr.append_slot(seq_id=1, num_tokens_in_seq=t)
        assert active_b == b1

    # Append 4th token (reaches multiple of block_size=4) -> new block allocated
    b2 = mgr.append_slot(seq_id=1, num_tokens_in_seq=4)
    assert b2 != b1
    assert mgr.num_free_blocks == 8
    assert mgr.get_block_table(1) == [b1, b2]

    # Fork sequence 1 -> sequence 2 (Zero-copy COW)
    mgr.fork_sequence(source_seq_id=1, target_seq_id=2)
    assert mgr.num_free_blocks == 8
    assert mgr.get_block_table(2) == [b1, b2]
    assert mgr.ref_counts[b1] == 2
    assert mgr.ref_counts[b2] == 2

    # Free sequence 1 -> blocks not returned yet because seq 2 still holds refs
    mgr.free_sequence(seq_id=1)
    assert mgr.num_free_blocks == 8
    assert mgr.ref_counts[b1] == 1

    # Free sequence 2 -> all blocks returned to free pool
    mgr.free_sequence(seq_id=2)
    assert mgr.num_free_blocks == 10


def test_mtp_speculative_serving():
    torch.manual_seed(42)
    vocab_size = 100
    d_model = 64
    d_ffn = 128

    class MockMainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.embed(x))

    main_model = MockMainModel()
    mtp_engine = MultiTokenPredictionEngine(
        d_model=d_model,
        d_ffn=d_ffn,
        vocab_size=vocab_size,
        num_future_tokens=2,
        num_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        num_routed_experts=4,
        num_shared_experts=1,
        top_k=2
    )

    spec_engine = MTPSpeculativeEngine(
        main_model=main_model,
        mtp_engine=mtp_engine,
        k_speculative_tokens=2
    )

    context_ids = torch.tensor([[10, 20, 30, 40]])
    backbone_hidden = torch.randn(1, 1, d_model)

    accepted_tokens, num_accepted = spec_engine.speculative_step(context_ids, backbone_hidden)
    assert num_accepted >= 1
    assert accepted_tokens.shape[0] == 1
    assert accepted_tokens.shape[1] == num_accepted
