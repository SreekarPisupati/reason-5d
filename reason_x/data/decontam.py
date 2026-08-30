"""
Reason-X: Module 1.3 - 13-Gram MinHash LSH Deduplication & Test Set Decontamination Scanner
=============================================================================================
Architectural Specification:
- 13-gram shingles generation over normalized text.
- MinHash LSH: 128 hash permutations with linear modulo hash family: h_i(x) = (a_i * x + b_i) mod p.
- Optimal Banded LSH index: Dynamically finds optimal (b, r) such that (1/b)^(1/r) ~ threshold.
- Pre-tokenization Test Set Decontamination Scanner: Checks training corpora against GSM8K, MATH,
  and HumanEval benchmarks for 13-gram containment, enforcing a strict 0% test contamination SLA.
"""

import hashlib
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np


def normalize_text(text: str) -> str:
    """Normalizes text by lowercasing and collapsing whitespace/punctuation."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_ngrams(text: str, n: int = 13, level: str = "word") -> Set[str]:
    """
    Extracts n-grams from text at word level or character level.
    """
    norm = normalize_text(text)
    if level == "word":
        tokens = norm.split(" ")
    else:
        tokens = list(norm)

    if len(tokens) < n:
        return set([" ".join(tokens)] if level == "word" else ["".join(tokens)])

    ngrams = set()
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i : i + n]) if level == "word" else "".join(tokens[i : i + n])
        ngrams.add(ngram)
    return ngrams


class MinHash:
    """
    Computes a K-dimensional MinHash signature for a set of shingles.
    """

    PRIME = (1 << 31) - 1  # Mersenne prime 2^31 - 1

    def __init__(self, num_hashes: int = 128, seed: int = 42):
        self.num_hashes = num_hashes
        self.seed = seed
        rng = np.random.default_rng(seed)
        
        # Linear hash parameters: a_i in [1, PRIME-1], b_i in [0, PRIME-1]
        self.a = rng.integers(1, self.PRIME, size=num_hashes, dtype=np.int64)
        self.b = rng.integers(0, self.PRIME, size=num_hashes, dtype=np.int64)

    def _hash_shingle(self, shingle: str) -> int:
        """32-bit hash of a string shingle."""
        return int(hashlib.md5(shingle.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF

    def compute_signature(self, shingles: Set[str]) -> np.ndarray:
        """
        Computes the MinHash signature array of length `num_hashes`.
        """
        if not shingles:
            return np.zeros(self.num_hashes, dtype=np.int64)

        # Hash all shingles to 32-bit integers
        shingle_hashes = np.array([self._hash_shingle(s) for s in shingles], dtype=np.int64)

        # Vectorized signature computation: min_x ((a * x + b) % PRIME)
        # Shape: [num_hashes, len(shingles)]
        h_matrix = (np.outer(self.a, shingle_hashes) + self.b[:, None]) % self.PRIME
        signature = np.min(h_matrix, axis=1)
        return signature

    @staticmethod
    def jaccard_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimates Jaccard similarity between two MinHash signatures."""
        return float(np.mean(sig_a == sig_b))


class MinHashLSHIndex:
    """
    Locality-Sensitive Hashing (LSH) index using banded MinHash signatures.
    """

    @staticmethod
    def optimal_bands_and_rows(num_hashes: int, threshold: float) -> Tuple[int, int]:
        """Calculates optimal (b, r) partitions minimizing |(1/b)^(1/r) - threshold|."""
        best_diff = float("inf")
        best_b = 1
        best_r = num_hashes
        for r in range(1, num_hashes + 1):
            if num_hashes % r == 0:
                b = num_hashes // r
                curr_t = (1.0 / b) ** (1.0 / r)
                diff = abs(curr_t - threshold)
                if diff < best_diff:
                    best_diff = diff
                    best_b = b
                    best_r = r
        return best_b, best_r

    def __init__(
        self,
        num_hashes: int = 128,
        num_bands: Optional[int] = None,
        threshold: float = 0.8,
        seed: int = 42
    ):
        self.num_hashes = num_hashes
        if num_bands is not None:
            self.num_bands = num_bands
            assert num_hashes % num_bands == 0, "num_hashes must be divisible by num_bands"
            self.rows_per_band = num_hashes // num_bands
        else:
            self.num_bands, self.rows_per_band = self.optimal_bands_and_rows(num_hashes, threshold)

        self.minhash = MinHash(num_hashes=num_hashes, seed=seed)
        
        # List of hash tables (one per band)
        self.band_tables: List[Dict[Tuple[int, ...], List[int]]] = [
            {} for _ in range(self.num_bands)
        ]
        self.doc_signatures: Dict[int, np.ndarray] = {}

    def insert(self, doc_id: int, text: str, n: int = 13) -> None:
        """Inserts a document into the LSH index."""
        shingles = get_ngrams(text, n=n, level="word")
        sig = self.minhash.compute_signature(shingles)
        self.doc_signatures[doc_id] = sig

        # Insert into each band bucket
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_tuple = tuple(sig[start:end])
            bucket = self.band_tables[band_idx].setdefault(band_tuple, [])
            bucket.append(doc_id)

    def query_duplicates(self, text: str, threshold: float = 0.8, n: int = 13) -> List[Tuple[int, float]]:
        """
        Queries candidate near-duplicates matching Jaccard similarity >= threshold.
        """
        shingles = get_ngrams(text, n=n, level="word")
        query_sig = self.minhash.compute_signature(shingles)

        candidate_ids: Set[int] = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_tuple = tuple(query_sig[start:end])
            if band_tuple in self.band_tables[band_idx]:
                candidate_ids.update(self.band_tables[band_idx][band_tuple])

        results: List[Tuple[int, float]] = []
        for cid in candidate_ids:
            cand_sig = self.doc_signatures[cid]
            sim = MinHash.jaccard_similarity(query_sig, cand_sig)
            if sim >= threshold:
                results.append((cid, sim))

        return sorted(results, key=lambda x: x[1], reverse=True)


class DecontaminationScanner:
    """
    Test Set Contamination Scanner checking against benchmark question & solution sets.
    Targets 0% 13-gram leakage against GSM8K, MATH, and HumanEval.
    """

    def __init__(self, n: int = 13):
        self.n = n
        self.benchmark_ngrams: Dict[str, Set[str]] = {}
        self.benchmark_counts: Dict[str, int] = {}

    def register_benchmark(self, benchmark_name: str, test_samples: Sequence[str]) -> None:
        """
        Registers a benchmark test suite by indexing all its 13-grams.
        """
        ngram_set: Set[str] = set()
        for sample in test_samples:
            sample_ngrams = get_ngrams(sample, n=self.n, level="word")
            ngram_set.update(sample_ngrams)

        self.benchmark_ngrams[benchmark_name] = ngram_set
        self.benchmark_counts[benchmark_name] = len(test_samples)

    def check_contamination(self, text: str) -> Dict[str, Union[bool, int, List[str]]]:
        """
        Scans a training document for any 13-gram overlap with registered benchmarks.
        """
        doc_ngrams = get_ngrams(text, n=self.n, level="word")
        leakages: Dict[str, List[str]] = {}
        is_contaminated = False

        for bname, b_ngrams in self.benchmark_ngrams.items():
            intersection = doc_ngrams.intersection(b_ngrams)
            if len(intersection) > 0:
                is_contaminated = True
                leakages[bname] = list(intersection)

        return {
            "is_contaminated": is_contaminated,
            "total_matches": sum(len(v) for v in leakages.values()),
            "leakages": leakages
        }

    def filter_clean_corpus(
        self,
        corpus: Sequence[str]
    ) -> Tuple[List[str], List[Dict[str, Union[bool, int, List[str]]]]]:
        """
        Filters a corpus, discarding any document that exhibits test-set contamination.
        """
        clean_docs: List[str] = []
        contamination_reports: List[Dict[str, Union[bool, int, List[str]]]] = []

        for doc in corpus:
            report = self.check_contamination(doc)
            if not report["is_contaminated"]:
                clean_docs.append(doc)
            else:
                contamination_reports.append(report)

        return clean_docs, contamination_reports
