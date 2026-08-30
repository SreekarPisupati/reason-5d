"""
Reason-5D: Module 5.1 - PagedAttention Dynamic Memory & Virtual Block Manager
=============================================================================
Architectural Specification:
- PagedAttention (Kwon et al., SOSP 2023 / vLLM style).
- Manages non-contiguous physical memory blocks for KV caches (16 tokens per block).
- Dynamic Reference Counting for zero-copy fork and Copy-On-Write (COW) branching.
"""

from typing import Dict, List, Optional, Set, Tuple
import torch


class PhysicalBlockManager:
    """
    Allocates and frees fixed-size physical KV cache blocks in GPU/CPU memory.
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks: List[int] = list(range(num_blocks))
        self.ref_counts: Dict[int, int] = {i: 0 for i in range(num_blocks)}
        self.block_tables: Dict[int, List[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def can_allocate(self, num_needed: int = 1) -> bool:
        return len(self.free_blocks) >= num_needed

    def allocate_sequence(self, seq_id: int) -> int:
        """Starts tracking a new sequence and allocates its first physical block."""
        raise NotImplementedError("TODO: Implement PhysicalBlockManager.allocate_sequence")

    def append_slot(self, seq_id: int, num_tokens_in_seq: int) -> int:
        """
        Appends a token to sequence. If current block is full, allocates a new physical block.
        """
        raise NotImplementedError("TODO: Implement PhysicalBlockManager.append_slot")

    def fork_sequence(self, source_seq_id: int, target_seq_id: int) -> None:
        """Zero-copy branching: Copies block table pointers and increments reference counts."""
        raise NotImplementedError("TODO: Implement PhysicalBlockManager.fork_sequence")

    def free_sequence(self, seq_id: int) -> None:
        """Frees all physical blocks held by a sequence (decrementing ref counts)."""
        raise NotImplementedError("TODO: Implement PhysicalBlockManager.free_sequence")

    def get_block_table(self, seq_id: int) -> List[int]:
        return self.block_tables.get(seq_id, [])
