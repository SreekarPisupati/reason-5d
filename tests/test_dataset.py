"""
Unit tests for reason_x.data.dataset (BinaryDatasetBuilder, MemmapStreamer, PackedSequenceDataset).
Verifies:
1. Binary serialization (.bin / .idx) and zero-copy memmap slice retrieval.
2. Document packing with zero padding waste.
3. Exact 2D block-diagonal causal attention mask generation.
4. Document boundary position ID resets.
"""

import os
import tempfile
import numpy as np
import pytest
import torch

from reason_x.data.dataset import BinaryDatasetBuilder, MemmapStreamer, PackedSequenceDataset


def test_binary_dataset_builder_and_streamer():
    # Create sample tokenized docs
    docs = [
        [10, 20, 30, 40, 50],
        [100, 200, 300],
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "test_data")
        bin_path, idx_path = BinaryDatasetBuilder.build(docs, prefix)

        assert os.path.exists(bin_path)
        assert os.path.exists(idx_path)

        with MemmapStreamer(prefix) as streamer:
            assert len(streamer) == 3
            assert streamer.total_tokens == sum(len(d) for d in docs)

            # Check exact slice contents
            for i, original_doc in enumerate(docs):
                sliced = streamer.get_document(i)
                assert np.array_equal(sliced, np.array(original_doc, dtype=np.uint16))


def test_packed_sequence_and_2d_block_diagonal_mask():
    """
    Verifies that packed sequences pack multiple docs into exact seq_len,
    and the 2D causal attention mask prevents cross-document attention.
    """
    # 3 docs: lengths 3, 4, 5 (with EOS added: lengths 4, 5, 6 -> total 15 tokens)
    docs = [
        [10, 11, 12],
        [20, 21, 22, 23],
        [30, 31, 32, 33, 34]
    ]
    seq_len = 15
    eos_token_id = 99

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "test_packed")
        BinaryDatasetBuilder.build(docs, prefix)
        with MemmapStreamer(prefix) as streamer:
            packed_ds = PackedSequenceDataset(streamer, seq_len=seq_len, eos_token_id=eos_token_id)
            batch = next(iter(packed_ds))

            input_ids = batch["input_ids"]
            pos_ids = batch["position_ids"]
            att_mask = batch["attention_mask"]
            spans = batch["document_spans"]

            assert input_ids.shape == (seq_len,)
            assert pos_ids.shape == (seq_len,)
            assert att_mask.shape == (seq_len, seq_len)
            assert len(spans) == 3

            # Verify Doc 1 span: [0:4] -> doc tokens [10, 11, 12, 99]
            assert spans[0] == (0, 4)
            assert input_ids[:4].tolist() == [10, 11, 12, 99]
            assert pos_ids[:4].tolist() == [0, 1, 2, 3]

            # Verify Doc 2 span: [4:9] -> doc tokens [20, 21, 22, 23, 99]
            assert spans[1] == (4, 9)
            assert input_ids[4:9].tolist() == [20, 21, 22, 23, 99]
            assert pos_ids[4:9].tolist() == [0, 1, 2, 3, 4]

            # Verify 2D causal block-diagonal attention mask:
            # Cross-doc attention must be False
            # Doc 2 (index 4..8) CANNOT attend to Doc 1 (index 0..3)
            assert not att_mask[4, 0].item()
            assert not att_mask[5, 2].item()
            assert not att_mask[10, 3].item()

            # Within-doc causal attention must be True (j <= i)
            assert att_mask[2, 1].item()  # within doc 1
            assert att_mask[2, 2].item()  # within doc 1 self
            assert not att_mask[1, 2].item()  # future token in doc 1 masked

            assert att_mask[7, 5].item()  # within doc 2
            assert not att_mask[5, 7].item()  # future in doc 2 masked
