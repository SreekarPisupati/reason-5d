"""
Reason-X: Unit & SLA Parity Test Suite for Module 1.1 (ArithmeticTokenizer)
==========================================================================
Verifies:
1. Arithmetic Digit Splitting SLA: Every integer string splits into single-digit tokens.
2. Zero-UNK Byte Fallback SLA: 0% <unk> occurrences across arbitrary unicode/binary streams.
3. Code & Math Preservation: Exact whitespace, LaTeX formulas, and special reasoning tags.
4. Lossless Roundtrip: decode(encode(text)) == text.
5. Checkpoint Serialization: save and load parity.
"""

import os
import tempfile
import pytest
from reason_x.data.tokenizer import ArithmeticTokenizer


@pytest.fixture
def trained_tokenizer():
    """Builds and trains a small sample ArithmeticTokenizer for testing."""
    corpus = [
        "10842 is a composite number.",
        "def compute_fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return compute_fibonacci(n - 1) + compute_fibonacci(n - 2)",
        "In LaTeX, we write fractions as \\frac{numerator}{denominator} and equations as E = mc^2.",
        "<|think|>\nLet us verify if 12 + 34 equals 46.\n12 + 34 = 46.\n</think>\nThe answer is 46.",
        "गणित और तर्क (Math and Logic) are central to reasoning models.",
        "Whitespace test:    four spaces,  two spaces, \t tab."
    ]
    # Small vocab for fast test execution
    tok = ArithmeticTokenizer(vocab_size=350)
    tok.train(corpus, min_frequency=1)
    return tok


def test_special_tokens_initialization():
    tok = ArithmeticTokenizer(vocab_size=300)
    assert "<|begin_of_text|>" in tok.special_tokens
    assert "<|end_of_text|>" in tok.special_tokens
    assert "<|pad|>" in tok.special_tokens
    assert "<|think|>" in tok.special_tokens
    assert "<|end_of_think|>" in tok.special_tokens
    assert tok.pad_token_id == tok.special_tokens["<|pad|>"]
    assert tok.eos_token_id == tok.special_tokens["<|end_of_text|>"]


def test_digit_splitting_sla(trained_tokenizer):
    """
    SLA Verification:
    The integer string '10842' MUST tokenize to exactly 5 individual tokens ['1', '0', '8', '4', '2'].
    Digits must NEVER be merged into multi-digit tokens.
    """
    text = "10842"
    tokens = trained_tokenizer.encode(text, allowed_special="all")
    decoded_pieces = [trained_tokenizer.decode([t]) for t in tokens]
    
    assert len(tokens) == 5, f"Expected 5 tokens for '10842', got {len(tokens)}: {decoded_pieces}"
    assert decoded_pieces == ['1', '0', '8', '4', '2'], f"Expected ['1', '0', '8', '4', '2'], got {decoded_pieces}"
    
    # Verify complex arithmetic expression digit-wise breakdown
    math_expr = "998 + 201 = 1199"
    math_tokens = trained_tokenizer.encode(math_expr)
    math_pieces = [trained_tokenizer.decode([t]) for t in math_tokens]
    # All digit characters in the pieces must be length 1
    for piece in math_pieces:
        if piece.strip().isdigit():
            assert len(piece.strip()) == 1, f"Digit token '{piece}' was merged!"


def test_zero_unk_byte_fallback_sla(trained_tokenizer):
    """
    SLA Verification:
    0% <unk> occurrences across arbitrary binary, multilingual, or emoji streams.
    Every byte must be representable via the 256 base byte tokens.
    """
    unseen_texts = [
        "Unseen English words with rare symbols: ∇ × B = μ₀(J + ε₀ ∂E/∂t)",
        "Multilingual Indic reasoning: यदि x = 5 और y = 10, तो x + y = 15 है।",
        "Bengali script: ৩ + ৫ = ৮",
        "Tamil script: கணிதம் மற்றும் பகுத்தறிவு",
        "Emoji stream: 🚀🤖💡⚡🧠",
        "Raw byte test: \x00\x01\x02\xff\xfe\xaa\xbb"
    ]
    for text in unseen_texts:
        token_ids = trained_tokenizer.encode(text)
        assert len(token_ids) > 0
        assert all(isinstance(t, int) for t in token_ids)
        # Verify lossless reconstruction
        decoded = trained_tokenizer.decode(token_ids)
        assert decoded == text, f"Decoded mismatch!\nOriginal: {text}\nDecoded:  {decoded}"


def test_reasoning_special_tokens(trained_tokenizer):
    """Verifies that special reasoning delimiters <|think|> and <|end_of_think|> are treated atomically."""
    text = "<|think|> Step 1: 5 * 5 = 25. <|end_of_think|> Output: 25"
    token_ids = trained_tokenizer.encode(text, allowed_special="all")
    
    think_id = trained_tokenizer.special_tokens["<|think|>"]
    end_think_id = trained_tokenizer.special_tokens["<|end_of_think|>"]
    
    assert think_id in token_ids
    assert end_think_id in token_ids
    
    decoded = trained_tokenizer.decode(token_ids)
    assert decoded == text


def test_save_and_load(trained_tokenizer):
    """Verifies that saving and reloading the tokenizer preserves exact ID mappings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_tok")
        trained_tokenizer.save(save_path)
        
        reloaded = ArithmeticTokenizer()
        reloaded.load(save_path)
        
        test_str = "<|think|> 1234 + 5678 = 6912 </think>"
        orig_ids = trained_tokenizer.encode(test_str, allowed_special="all")
        reloaded_ids = reloaded.encode(test_str, allowed_special="all")
        
        assert orig_ids == reloaded_ids
        assert reloaded.decode(reloaded_ids) == test_str
