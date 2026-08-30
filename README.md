# Native Sparse Attention (NSA) Triton Kernel & Frontier ML Systems

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org/)
[![Triton Kernel](https://img.shields.io/badge/Kernel-OpenAI%20Triton-orange.svg)](https://github.com/openai/triton)
[![Architecture NSA](https://img.shields.io/badge/Architecture-Native%20Sparse%20Attention%20(ACL%202025%20Best%20Paper)-green.svg)](https://arxiv.org/abs/2502.11089)

> **Hardware-Efficient Trainable Sparse Attention & Pretraining-to-Serving Systems Core**.  
> High-performance OpenAI Triton implementation of **Native Sparse Attention (NSA)** (Yuan et al., DeepSeek + PKU, ACL 2025 Best Paper), featuring GQA-shared top-$n$ block selection, coarse-grained KV compression, sliding-window attention, and full backward recomputation kernels, evaluated with rigorous roofline performance benchmarks.

---

## 🏛️ System Architecture & Artifact Map

```
                                ┌──────────────────────────────────────────────────────────┐
                                │       NATIVE SPARSE ATTENTION (NSA) SYSTEM TOPOLOGY      │
                                └──────────────────────────────────────────────────────────┘
                                                              │
     ┌───────────────────────────┬────────────────────────────┴────────────────────────────┬───────────────────────────┐
     ▼                           ▼                                                         ▼                           ▼
[1. COMPRESSED BRANCH]   [2. SELECTION BRANCH]                                     [3. SLIDING BRANCH]         [4. SERVING & EVAL]
- Block-level Average    - GQA-Shared Block Importance Scoring                     - Sliding-Window Local      - Roofline Efficiency
  Pooling / Compression  - Top-n Blockwise KV Token Gather                         - Dense Adjacency Attention - FlashAttention-2 Baseline
- SRAM-budgeted Tiling   - Triton Forward & Backward Kernels                       - Flash-style Tiling        - Memory & Latency Sweeps
- Coarse Context Cache   - Deterministic Gradient Recomputation                    - Boundary Masking          - Long-Context (>64k)
```

---

## 🔬 Core Deliverables & Milestones

1. **Artifact 1 (Weeks 1–6)**: **NSA Selection-Branch Triton Kernel** (Target: 11 October 2026)
   - Triton forward & backward passes tested against float64 PyTorch exact reference.
   - Comprehensive benchmark sweeps vs `F.scaled_dot_product_attention` and FlashAttention-2 across $S \in \{1\text{k}, 4\text{k}, 8\text{k}, 16\text{k}, 32\text{k}, 64\text{k}\} \times d \in \{64, 128\} \times B \in \{1, 4\}$.
   - Roofline model analysis: achieved memory bandwidth ($\text{GB/s}$) and compute utilization ($\text{TFLOPS}$).
2. **Artifact 2 (Weeks 5–11)**: **Digit Tokenization Under Matched Compute** (Target: 15 November 2026)
   - Multi-scale matched-compute ablation across 4 tokenization schemes on arithmetic & reasoning.
3. **Artifact 3 (Weeks 12–16)**: **Muon & MuonClip Optimizer Scaling** (Target: 20 December 2026)
   - Newton–Schulz momentum orthogonalization and QK-Clip ($\tau=100$) stability study.
4. **Artifact 4 (Weeks 17–20)**: **GSPO vs GRPO Failure Modes**
   - Sequence-level importance ratio analysis and MoE stability verification.

---

## ⚡ Daily Drill Disciplines

Timed, blank-file implementation drills (30 min/day) covering:
- MHA with KV Cache, RoPE, RMSNorm, SwiGLU, MLA, and Online Softmax.
- Masked Cross-Entropy, AdamW, and Muon Newton-Schulz iterations.
- Paged KV-cache dynamic block management.
- Triton vector addition, fused softmax, and tiled matrix multiplication.
