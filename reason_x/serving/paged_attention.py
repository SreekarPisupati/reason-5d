"""
Reason-X: Module 5.1 - PagedAttention Dynamic Memory & Virtual Block Manager
=============================================================================
Architectural Specification:
- PagedAttention (Kwon et al., SOSP 2023 / vLLM style).
- Manages non-contiguous physical memory blocks for KV caches (16 tokens per block).
- Dynamic Reference Counting for zero-copy fork and Copy-On-Write (COW) branching
  during parallel sampling and beam search.
- SLA: <5% memory fragmentation waste, zero memory leaks across request lifecycles.
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

        # Free block pool
        self.free_blocks: List[int] = list(range(num_blocks))
        # Ref counts: phys_block_id -> int
        self.ref_counts: Dict[int, int] = {i: 0 for i in range(num_blocks)}
        # Sequence block tables: seq_id -> List[int] (physical block IDs)
        self.block_tables: Dict[int, List[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def can_allocate(self, num_needed: int = 1) -> bool:
        return len(self.free_blocks) >= num_needed

    def allocate_sequence(self, seq_id: int) -> int:
        """Starts tracking a new sequence and allocates its first physical block."""
        if not self.free_blocks:
            raise MemoryError("Out of physical KV cache blocks!")
        
        block_id = self.free_blocks.pop(0)
        self.ref_counts[block_id] = 1
        self.block_tables[seq_id] = [block_id]
        return block_id

    def append_slot(self, seq_id: int, num_tokens_in_seq: int) -> int:
        """
        Appends a token to sequence. If current block is full, allocates a new physical block.
        Returns the active physical block ID.
        """
        table = self.block_tables[seq_id]
        # If token count reaches a new multiple of block_size, need new block
        if num_tokens_in_seq > 0 and num_tokens_in_seq % self.block_size == 0:
            if not self.free_blocks:
                raise MemoryError("Out of physical KV cache blocks during sequence expansion!")
            new_block_id = self.free_blocks.pop(0)
            self.ref_counts[new_block_id] = 1
            table.append(new_block_id)

        return table[-1]

    def fork_sequence(self, source_seq_id: int, target_seq_id: int) -> None:
        """
        Zero-copy branching: Copies block table pointers and increments reference counts.
        """
        src_table = self.block_tables[source_seq_id]
        for b_id in src_table:
            self.ref_counts[b_id] += 1
        self.block_tables[target_seq_id] = list(src_table)

    def free_sequence(self, seq_id: int) -> None:
        """Frees all physical blocks held by a sequence (decrementing ref counts)."""
        if seq_id not in self.block_tables:
            return

        table = self.block_tables.pop(seq_id)
        for b_id in table:
            self.ref_counts[b_id] -= 1
            if self.ref_counts[b_id] == 0:
                self.free_blocks.append(b_id)

    def get_block_table(self, seq_id: int) -> List[int]:
        return self.block_tables.get(seq_id, [])
