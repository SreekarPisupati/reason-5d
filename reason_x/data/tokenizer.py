"""
Reason-X: Module 1.1 - Arithmetic-Preserving & Code-Aware BPE Tokenizer
======================================================================
Architectural Specification:
- Custom pre-tokenization regex forcing discrete single-digit splits (\\d).
  * Why: Prevents inconsistent multi-digit merges (e.g. "12" + "34" -> "1234"),
    preserving digit-wise positional alignment essential for transformer math reasoning.
- Discrete whitespace & code indentation retention (2-space and 4-space blocks).
- Byte-level fallback (256 base bytes) guaranteeing 0% out-of-vocabulary (<unk>) occurrences.
- Target vocabulary size: 32,000 tokens (or configurable).
- Special tokens: <|begin_of_text|>, <|end_of_text|>, <|pad|>, <|think|>, <|end_of_think|>,
                 <|user|>, <|assistant|>, <|system|>
"""

import json
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import regex as re


class ArithmeticTokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer engineered for mathematical reasoning and code.
    
    Attributes:
        vocab_size (int): Total vocabulary size (default: 32,000).
        vocab (Dict[int, bytes]): Mapping from token ID to raw byte sequence.
        inverse_vocab (Dict[bytes, int]): Mapping from raw byte sequence to token ID.
        merges (Dict[Tuple[int, int], int]): BPE merge rules mapping (token_id_1, token_id_2) -> new_token_id.
        merge_ranks (Dict[Tuple[int, int], int]): Rank priority for merges.
        special_tokens (Dict[str, int]): Registered special token strings to IDs.
    """

    # GPT-4 / LLaMA style regex modified with discrete single-digit splitting: (?:\\d)
    # This guarantees that numbers are never grouped into multi-digit tokens during pre-tokenization.
    SPLIT_REGEX = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\d| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+|\S"""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.special_tokens: Dict[str, int] = {}
        self.inverse_special_tokens: Dict[int, str] = {}
        self.vocab: Dict[int, bytes] = {}
        self.inverse_vocab: Dict[bytes, int] = {}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.merge_ranks: Dict[Tuple[int, int], int] = {}
        self._compiled_regex = re.compile(self.SPLIT_REGEX)
        
        self._init_special_tokens()
        self._init_byte_vocab()

    def _init_special_tokens(self) -> None:
        """Register essential special tokens for pretraining, reasoning traces, and padding."""
        specials = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|pad|>",
            "<|think|>",
            "<|end_of_think|>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>"
        ]
        for idx, token_str in enumerate(specials):
            self.special_tokens[token_str] = idx
            self.inverse_special_tokens[idx] = token_str

    def _init_byte_vocab(self) -> None:
        """
        Initialize the base vocabulary with 256 individual byte values (0..255).
        Offset by the number of special tokens so that IDs do not collide.
        """
        offset = len(self.special_tokens)
        for b in range(256):
            token_id = offset + b
            byte_val = bytes([b])
            self.vocab[token_id] = byte_val
            self.inverse_vocab[byte_val] = token_id

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["<|pad|>"]

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens["<|end_of_text|>"]

    @property
    def bos_token_id(self) -> int:
        return self.special_tokens["<|begin_of_text|>"]

    def pre_tokenize(self, text: str) -> List[str]:
        """
        Pre-tokenizes raw string into regex-split lexical chunks according to SPLIT_REGEX.
        Enforces discrete digit splits and preserves whitespace/indentation.
        """
        return self._compiled_regex.findall(text)

    def train(self, texts: Sequence[str], min_frequency: int = 2) -> None:
        """
        Learns BPE merge rules from an input corpus until `self.vocab_size` is reached.
        """
        # Step 1: Pre-tokenize all texts and convert to initial byte-token lists
        corpus_chunks: List[List[int]] = []
        for text in texts:
            chunks = self.pre_tokenize(text)
            for chunk in chunks:
                chunk_bytes = chunk.encode("utf-8")
                # Map each byte directly to base token ID
                token_seq = [self.inverse_vocab[bytes([b])] for b in chunk_bytes]
                if len(token_seq) > 0:
                    corpus_chunks.append(token_seq)

        current_vocab_size = len(self.special_tokens) + len(self.vocab)
        num_merges_target = self.vocab_size - current_vocab_size
        if num_merges_target <= 0:
            return

        merge_idx = 0
        while len(self.vocab) + len(self.special_tokens) < self.vocab_size:
            # Step 2: Count adjacent pair frequencies across all chunk sequences
            pair_counts: Dict[Tuple[int, int], int] = {}
            for chunk in corpus_chunks:
                for i in range(len(chunk) - 1):
                    pair = (chunk[i], chunk[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            # Find pair with maximum frequency
            best_pair = max(pair_counts.items(), key=lambda x: x[1])
            pair, count = best_pair
            if count < min_frequency:
                break

            # Assign new token ID
            new_token_id = len(self.special_tokens) + len(self.vocab)
            new_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            self.merges[pair] = new_token_id
            self.merge_ranks[pair] = merge_idx
            self.vocab[new_token_id] = new_bytes
            self.inverse_vocab[new_bytes] = new_token_id
            merge_idx += 1

            # Replace occurrences in corpus_chunks
            new_corpus_chunks: List[List[int]] = []
            for chunk in corpus_chunks:
                i = 0
                new_chunk: List[int] = []
                while i < len(chunk):
                    if i < len(chunk) - 1 and (chunk[i], chunk[i + 1]) == pair:
                        new_chunk.append(new_token_id)
                        i += 2
                    else:
                        new_chunk.append(chunk[i])
                        i += 1
                new_corpus_chunks.append(new_chunk)
            corpus_chunks = new_corpus_chunks

    def encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        """
        Encodes a single pre-tokenized byte chunk using learned BPE merge rules.
        """
        if not chunk_bytes:
            return []
        
        # Start with individual byte token IDs
        tokens = [self.inverse_vocab[bytes([b])] for b in chunk_bytes]
        if len(tokens) <= 1 or not self.merge_ranks:
            return tokens

        while len(tokens) >= 2:
            # Find candidate pairs and pick the one with lowest merge rank
            best_pair: Optional[Tuple[int, int]] = None
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            # Merge the best pair
            new_token_id = self.merges[best_pair]
            new_tokens: List[int] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str, allowed_special: Union[Set[str], str] = "all") -> List[int]:
        """
        Tokenizes arbitrary text into a list of token IDs with 0% UNK guarantee.
        Handles special tokens (<|think|>, <|end_of_text|>, etc.) without splitting them.
        """
        if not text:
            return []

        active_specials: Set[str] = set()
        if allowed_special == "all":
            active_specials = set(self.special_tokens.keys())
        elif isinstance(allowed_special, (set, list, tuple)):
            active_specials = set(s for s in allowed_special if s in self.special_tokens)

        if not active_specials:
            chunks = self.pre_tokenize(text)
            tokens: List[int] = []
            for chunk in chunks:
                tokens.extend(self.encode_chunk(chunk.encode("utf-8")))
            return tokens

        # Compile pattern for special tokens
        special_pattern = "(" + "|".join(re.escape(s) for s in sorted(active_specials, key=len, reverse=True)) + ")"
        parts = re.split(special_pattern, text)

        tokens: List[int] = []
        for part in parts:
            if not part:
                continue
            if part in active_specials:
                tokens.append(self.special_tokens[part])
            else:
                chunks = self.pre_tokenize(part)
                for chunk in chunks:
                    tokens.extend(self.encode_chunk(chunk.encode("utf-8")))

        return tokens

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """
        Decodes a sequence of token IDs back into a Unicode string.
        """
        byte_chunks: List[bytes] = []
        for tid in token_ids:
            if tid in self.inverse_special_tokens:
                if not skip_special_tokens:
                    special_str = self.inverse_special_tokens[tid]
                    byte_chunks.append(special_str.encode("utf-8"))
            elif tid in self.vocab:
                byte_chunks.append(self.vocab[tid])
            else:
                # Handle unexpected token ID gracefully
                pass

        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, filepath_prefix: str) -> None:
        """
        Saves tokenizer state (merges, vocabulary, special tokens) to JSON files.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath_prefix)), exist_ok=True)
        
        # Serialize merges as list of [t1, t2, new_id]
        merges_serialized = [
            [pair[0], pair[1], new_id, self.merge_ranks[pair]]
            for pair, new_id in self.merges.items()
        ]
        
        # Serialize vocab as list of [id, hex_string]
        vocab_serialized = {
            str(tid): b.hex() for tid, b in self.vocab.items()
        }
        
        metadata = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "merges": merges_serialized,
            "vocab": vocab_serialized
        }
        
        with open(f"{filepath_prefix}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def load(self, filepath_prefix: str) -> None:
        """
        Loads tokenizer state from serialized JSON file.
        """
        json_path = filepath_prefix if filepath_prefix.endswith(".json") else f"{filepath_prefix}.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.vocab_size = data["vocab_size"]
        self.special_tokens = data["special_tokens"]
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        self.vocab = {}
        self.inverse_vocab = {}
        for tid_str, hex_str in data["vocab"].items():
            tid = int(tid_str)
            raw_bytes = bytes.fromhex(hex_str)
            self.vocab[tid] = raw_bytes
            self.inverse_vocab[raw_bytes] = tid
            
        self.merges = {}
        self.merge_ranks = {}
        for item in data["merges"]:
            t1, t2, new_id, rank = item
            pair = (t1, t2)
            self.merges[pair] = new_id
            self.merge_ranks[pair] = rank
