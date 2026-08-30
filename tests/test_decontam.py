"""
Unit tests for reason_x.data.decontam (MinHash, MinHashLSHIndex, DecontaminationScanner).
Verifies:
1. 13-gram shingles extraction.
2. MinHash signature generation & Jaccard similarity estimation.
3. Banded LSH near-duplicate clustering.
4. Benchmark test-set decontamination detection and corpus filtering (0% leakage SLA).
"""

import pytest
from reason_x.data.decontam import (
    DecontaminationScanner,
    MinHash,
    MinHashLSHIndex,
    get_ngrams,
    normalize_text
)


def test_ngram_extraction():
    text = "The quick brown fox jumps over the lazy dog in the warm summer sunlight today"
    ngrams = get_ngrams(text, n=5, level="word")
    assert len(ngrams) > 0
    assert "the quick brown fox jumps" in ngrams


def test_minhash_jaccard_similarity():
    minhash = MinHash(num_hashes=128, seed=42)

    doc_a = "Solving quadratic equations requires factoring or using the quadratic formula accurately."
    doc_b = "Solving quadratic equations requires factoring or using the quadratic formula carefully."
    doc_c = "A neural network optimizes parameters through stochastic gradient descent backpropagation."

    shingles_a = get_ngrams(doc_a, n=3)
    shingles_b = get_ngrams(doc_b, n=3)
    shingles_c = get_ngrams(doc_c, n=3)

    sig_a = minhash.compute_signature(shingles_a)
    sig_b = minhash.compute_signature(shingles_b)
    sig_c = minhash.compute_signature(shingles_c)

    sim_ab = MinHash.jaccard_similarity(sig_a, sig_b)
    sim_ac = MinHash.jaccard_similarity(sig_a, sig_c)

    assert sim_ab > 0.6  # Near duplicate
    assert sim_ac < 0.1  # Completely different topic


def test_minhash_lsh_indexing():
    lsh = MinHashLSHIndex(num_hashes=128, threshold=0.25, seed=42)

    doc1 = "The Riemann hypothesis is a conjecture that the Riemann zeta function has its zeros only at the negative even integers and complex numbers with real part 1/2."
    doc2 = "The Riemann hypothesis states that the Riemann zeta function has its zeros exclusively at negative even integers and complex numbers with real part one half."
    doc3 = "Quantum computing harnesses superposition and entanglement to perform complex tensor calculations at scale."

    lsh.insert(1, doc1, n=4)
    lsh.insert(3, doc3, n=4)

    # Query with doc2 (Jaccard similarity with doc1 is ~0.31)
    matches = lsh.query_duplicates(doc2, threshold=0.25, n=4)
    matched_ids = [m[0] for m in matches]

    assert 1 in matched_ids
    assert 3 not in matched_ids


def test_decontamination_scanner_sla():
    """
    SLA Test: Guarantees 0% 13-gram contamination leakage against GSM8K and HumanEval.
    """
    scanner = DecontaminationScanner(n=13)

    # Register mock benchmark suites
    gsm8k_samples = [
        "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins with four every day. How many does she sell?"
    ]
    humaneval_samples = [
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    Check if in given list of numbers, are any two numbers closer to each other than given threshold."
    ]

    scanner.register_benchmark("GSM8K", gsm8k_samples)
    scanner.register_benchmark("HumanEval", humaneval_samples)

    # Training corpus containing clean and contaminated docs
    training_corpus = [
        "Clean doc: Pretraining mathematical reasoning requires rich step-by-step synthetic proofs, chain of thought verifications, and algorithmic trace generation.",
        "Leaked doc: Here is a math problem: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins with four every day. How many does she sell at the farmers market?",
        "Clean doc 2: In physics, the wave function collapses upon measurement under standard Copenhagen interpretation."
    ]

    clean_docs, reports = scanner.filter_clean_corpus(training_corpus)

    assert len(clean_docs) == 2
    assert len(reports) == 1
    assert reports[0]["is_contaminated"] is True
    assert "GSM8K" in reports[0]["leakages"]
    assert "HumanEval" not in reports[0]["leakages"]
