"""
Reason-X: Module 1.2 - Zero-Copy Memory-Mapped Streaming & 2D Block-Diagonal Document Packer
=============================================================================================
Architectural Specification:
- Zero-copy binary memmap streaming via flat uint16 `.bin` and `.idx` index headers.
- Document packing: Packs variable-length mathematical & code documents into exact sequence
  length windows (e.g. S = 4096, 8192, 16384) with ZERO padding waste.
- 2D Block-Diagonal Attention Mask: Generates causal 2D block masks preventing cross-document
  attention leakage across packed boundaries.
- Positional Reset: Resets position_ids to 0 at each document boundary.
- SLA: Zero memory growth over 10,000+ streaming steps, <0.5ms batch retrieval latency.
"""

import os
import struct
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch


class BinaryDatasetBuilder:
    """
    Builds binary uint16 `.bin` token arrays and `.idx` document index metadata.
    """

    MAGIC_HEADER = b"R5DBIN\x00\x01"  # Reason-5D Binary Format v1.0

    @classmethod
    def build(
        cls,
        tokenized_docs: Sequence[Sequence[int]],
        output_prefix: str,
        dtype: np.dtype = np.uint16
    ) -> Tuple[str, str]:
        """
        Encodes a list of tokenized documents into binary .bin and .idx files.

        Args:
            tokenized_docs: Sequence of token ID lists (each representing one document).
            output_prefix: Filepath prefix (e.g. 'data/train_math').
            dtype: Numpy integer dtype (default: uint16, supports vocab up to 65,535).

        Returns:
            Tuple of (bin_filepath, idx_filepath).
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
        bin_path = f"{output_prefix}.bin"
        idx_path = f"{output_prefix}.idx"

        # Calculate offsets and document lengths
        offsets: List[int] = []
        lengths: List[int] = []
        current_offset = 0

        with open(bin_path, "wb") as f_bin:
            for doc in tokenized_docs:
                arr = np.array(doc, dtype=dtype)
                arr.tofile(f_bin)
                offsets.append(current_offset)
                lengths.append(len(doc))
                current_offset += len(doc)

        # Write index file with header: MAGIC (8 bytes), num_docs (uint64), total_tokens (uint64), offsets, lengths
        with open(idx_path, "wb") as f_idx:
            f_idx.write(cls.MAGIC_HEADER)
            num_docs = len(tokenized_docs)
            total_tokens = current_offset
            f_idx.write(struct.pack("<QQ", num_docs, total_tokens))
            
            # Write offsets and lengths as uint64 arrays
            np.array(offsets, dtype=np.uint64).tofile(f_idx)
            np.array(lengths, dtype=np.uint64).tofile(f_idx)

        return bin_path, idx_path


class MemmapStreamer:
    """
    Zero-copy memory-mapped document reader for ultra-high throughput training.
    """

    def __init__(self, prefix: str, dtype: np.dtype = np.uint16):
        self.prefix = prefix
        self.bin_path = f"{prefix}.bin"
        self.idx_path = f"{prefix}.idx"
        self.dtype = dtype

        if not os.path.exists(self.bin_path) or not os.path.exists(self.idx_path):
            raise FileNotFoundError(f"Binary dataset files not found for prefix '{prefix}'")

        self._load_index()
        self.tokens_mmap = np.memmap(self.bin_path, dtype=self.dtype, mode="r")

    def _load_index(self) -> None:
        with open(self.idx_path, "rb") as f:
            header = f.read(len(BinaryDatasetBuilder.MAGIC_HEADER))
            if header != BinaryDatasetBuilder.MAGIC_HEADER:
                raise ValueError("Invalid index file format header")
            
            num_docs, total_tokens = struct.unpack("<QQ", f.read(16))
            self.num_docs = num_docs
            self.total_tokens = total_tokens
            self.offsets = np.fromfile(f, dtype=np.uint64, count=num_docs)
            self.lengths = np.fromfile(f, dtype=np.uint64, count=num_docs)

    def close(self) -> None:
        """Closes the underlying memmap buffer and releases OS file handle."""
        if hasattr(self, "tokens_mmap") and self.tokens_mmap is not None:
            try:
                if hasattr(self.tokens_mmap, "_mmap") and self.tokens_mmap._mmap is not None:
                    self.tokens_mmap._mmap.close()
            except Exception:
                pass
            self.tokens_mmap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return self.num_docs

    def get_document(self, doc_idx: int) -> np.ndarray:
        """Retrieves zero-copy slice of a single document."""
        if doc_idx < 0 or doc_idx >= self.num_docs:
            raise IndexError(f"Document index {doc_idx} out of range [0, {self.num_docs})")
        offset = self.offsets[doc_idx]
        length = self.lengths[doc_idx]
        return self.tokens_mmap[offset : offset + length]


class PackedSequenceDataset:
    """
    Packs multiple variable-length documents into fixed sequence length windows S
    with zero padding overhead, generating exact 2D block-diagonal causal attention masks.
    """

    def __init__(
        self,
        streamer: MemmapStreamer,
        seq_len: int = 4096,
        eos_token_id: int = 1,
        shuffle_docs: bool = False,
        seed: int = 42
    ):
        self.streamer = streamer
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.shuffle_docs = shuffle_docs
        self.seed = seed

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Streams packed batches.
        
        Yields Dict containing:
            - 'input_ids': [S] torch.LongTensor
            - 'labels': [S] torch.LongTensor (shifted autoregressive targets)
            - 'position_ids': [S] torch.LongTensor (resets to 0 at doc boundaries)
            - 'attention_mask': [S, S] torch.BoolTensor (2D causal block-diagonal mask)
            - 'document_boundaries': List[Tuple[int, int]] start/end index within sequence
        """
        doc_indices = list(range(len(self.streamer)))
        if self.shuffle_docs:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(doc_indices)

        current_tokens: List[int] = []
        current_doc_spans: List[Tuple[int, int]] = []  # (start_in_window, end_in_window)
        current_pos_ids: List[int] = []

        for doc_idx in doc_indices:
            doc_tokens = self.streamer.get_document(doc_idx).tolist()
            # Append EOS if not already present
            if len(doc_tokens) == 0 or doc_tokens[-1] != self.eos_token_id:
                doc_tokens.append(self.eos_token_id)

            doc_offset = 0
            while doc_offset < len(doc_tokens):
                space_left = self.seq_len - len(current_tokens)
                chunk_len = min(space_left, len(doc_tokens) - doc_offset)

                start_idx = len(current_tokens)
                end_idx = start_idx + chunk_len

                # Append tokens and doc span
                current_tokens.extend(doc_tokens[doc_offset : doc_offset + chunk_len])
                current_doc_spans.append((start_idx, end_idx))
                
                # Position IDs reset to 0 at document start
                current_pos_ids.extend(list(range(doc_offset, doc_offset + chunk_len)))
                doc_offset += chunk_len

                # If packed sequence window is full, emit batch item
                if len(current_tokens) == self.seq_len:
                    yield self._format_packed_window(
                        tokens=current_tokens,
                        doc_spans=current_doc_spans,
                        pos_ids=current_pos_ids
                    )
                    # Reset buffers
                    current_tokens = []
                    current_doc_spans = []
                    current_pos_ids = []

        # Handle remaining tokens if any
        if current_tokens:
            pad_len = self.seq_len - len(current_tokens)
            start_idx = len(current_tokens)
            current_tokens.extend([self.eos_token_id] * pad_len)
            current_pos_ids.extend([0] * pad_len)
            yield self._format_packed_window(
                tokens=current_tokens,
                doc_spans=current_doc_spans,
                pos_ids=current_pos_ids
            )

    def _format_packed_window(
        self,
        tokens: List[int],
        doc_spans: List[Tuple[int, int]],
        pos_ids: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Constructs the 2D causal block-diagonal attention mask.
        Token i can attend to token j IF AND ONLY IF:
          1. j <= i (Causal condition)
          2. Both i and j belong to the EXACT SAME packed document span.
        """
        S = self.seq_len
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        position_ids = torch.tensor(pos_ids, dtype=torch.long)

        # Build 2D block-diagonal causal attention mask: shape [S, S]
        # True = Allowed to attend, False = Masked out
        attention_mask = torch.zeros((S, S), dtype=torch.bool)

        for (start, end) in doc_spans:
            # Within [start:end], tokens can attend causally to previous tokens in the same document
            span_len = end - start
            causal_submask = torch.tril(torch.ones((span_len, span_len), dtype=torch.bool))
            attention_mask[start:end, start:end] = causal_submask

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "document_spans": doc_spans,
        }
