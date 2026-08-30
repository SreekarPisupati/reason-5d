# Native Sparse Attention (NSA) Triton Kernel & Frontier ML Systems

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org/)
[![Triton Kernel](https://img.shields.io/badge/Kernel-OpenAI%20Triton-orange.svg)](https://github.com/openai/triton)
[![Architecture NSA](https://img.shields.io/badge/Architecture-Native%20Sparse%20Attention%20(ACL%202025%20Best%20Paper)-green.svg)](https://arxiv.org/abs/2502.11089)

High-performance OpenAI Triton implementation and systems benchmark of **Native Sparse Attention (NSA)** (Yuan et al., DeepSeek + PKU, ACL 2025 Best Paper). NSA introduces a natively trainable, hardware-aligned sparse attention mechanism designed to break quadratic compute and memory bottlenecks across long context windows (16k–64k+).

---

## 1. Overview and Core Motivation

Standard full attention exhibits quadratic computational complexity $O(S^2)$ and linear memory bandwidth scaling during autoregressive generation. Existing sparse-attention methods are predominantly inference-only approximations that drop gradients during training or suffer from poor GPU hardware utilization due to uncoalesced memory access.

Native Sparse Attention addresses this by decomposing the attention operator into three parallel, complementary branches:

1. **Compressed Coarse-Grained Branch**: Computes attention over blockwise average-pooled Key and Value states, capturing macroscopic global context with minimal compute overhead.
2. **Selected Fine-Grained Branch**: Dynamically routes queries to the top-$n$ most relevant Key/Value blocks using a GQA-shared similarity index, preserving high-importance non-local associations.
3. **Sliding-Window Local Branch**: Retains dense, high-frequency token interactions within an immediate receptive field ($w = 256\text{--}512$).
4. **Learned Fusion Gate**: A per-head gated linear projection that dynamically weights the three representations:
   $$\mathbf{O} = g_{\text{cmp}} \odot \mathbf{O}_{\text{cmp}} + g_{\text{sel}} \odot \mathbf{O}_{\text{sel}} + g_{\text{win}} \odot \mathbf{O}_{\text{win}}$$

---

## 2. Architectural Topology

```
+-------------------------------------------------------------------------------+
|                 NATIVE SPARSE ATTENTION (NSA) ARCHITECTURE                    |
+-------------------------------------------------------------------------------+
                                       |
    +----------------------------------+----------------------------------+
    |                                  |                                  |
    v                                  v                                  v
[1. COMPRESSED BRANCH]       [2. SELECTION BRANCH]              [3. SLIDING BRANCH]
- Block-level Average        - GQA-Shared Block Scoring         - Sliding Window
  Pooling / Compression      - Top-n Blockwise KV Gather        - Dense Local Tokens
- SRAM-budgeted Tiling       - Triton Forward + Backward        - Flash-style Tiling
- Global Context Cache       - Deterministic Recomputation      - Boundary Masking
    |                                  |                                  |
    +----------------------------------+----------------------------------+
                                       |
                                       v
                             [4. LEARNED GATE FUSION]
                             O = g_cmp*O_cmp + g_sel*O_sel + g_win*O_win
```

---

## 3. Mathematical Formulation

### 3.1 Online Softmax Recurrence
To avoid materializing $S \times S$ intermediate attention matrices in High Bandwidth Memory (HBM), the forward and backward kernels utilize the online softmax formulation:

$$\mathbf{m}_{\text{new}} = \max(\mathbf{m}_{\text{prev}}, \max(S_{\text{curr}}))$$

$$\alpha = \exp(\mathbf{m}_{\text{prev}} - \mathbf{m}_{\text{new}})$$

$$\mathbf{l}_{\text{new}} = \mathbf{l}_{\text{prev}} \cdot \alpha + \sum \exp(S_{\text{curr}} - \mathbf{m}_{\text{new}})$$

$$\mathbf{O}_{\text{new}} = \frac{\mathbf{O}_{\text{prev}} \cdot (\mathbf{l}_{\text{prev}} \cdot \alpha) + \exp(S_{\text{curr}} - \mathbf{m}_{\text{new}}) \mathbf{V}_{\text{curr}}}{\mathbf{l}_{\text{new}}}$$

### 3.2 GQA-Shared Selection Branch
For Grouped-Query Attention ($H_q$ query heads sharing $H_{kv}$ key/value heads), selection indices are shared across the query group to preserve memory coalescing and eliminate redundant gathers:

$$I_{\text{selected}} = \text{Top-}k \left( \frac{1}{|G_k|} \sum_{h \in G_k} \mathbf{Q}_{h} \mathbf{\bar{K}}_{k}^T \right)$$

---

## 4. Benchmark Harness & Roofline Profiling

The repository includes an automated profiling suite (`benchmarks/bench_attention.py`) providing:
- Forward and backward latency measurements (ms) via CUDA hardware events.
- Peak VRAM allocation tracking (MB).
- Effective TFLOPS achieved ($4 \times B \times H \times S^2 \times D$).
- Operational arithmetic intensity and memory bandwidth utilization ($\text{GB/s}$) across sequence sweeps.

### Baseline Benchmark Protocol
```bash
python benchmarks/bench_attention.py --seq_lens 1024 4096 8192 16384 32768
```

---

## 5. Repository Structure

```
Reason-5D/
|-- benchmarks/
|   `-- bench_attention.py          # Latency, memory, and roofline profiling suite
|-- reason_x/                       # Core engine package
|   |-- data/                       # Tokenization, memmap dataset streamer, and decontam
|   |-- distributed/                # Parallelism and fused GPU kernels
|   |-- models/                     # NSA, MLA, MoE, and MTP architectures
|   |-- post_training/              # Rule verifiers, GRPO, and GSPO RL
|   |-- serving/                    # Paged KV-cache and speculative serving
|   `-- training/                   # Scaling laws, Muon/MuonClip, stability suite
|-- tests/                          # Automated pytest suite (SLA verification)
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

---

## 6. Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/SreekarPisupati/reason_5d.git
cd reason_5d

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Verification Tests
```bash
pytest tests/
```

---

## 7. Research References

- Yuan et al., *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*, arXiv:2502.11089 (ACL 2025 Best Paper).
- Dao et al., *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, ICLR 2024.
- Jordan et al., *Muon: An Optimizer for Hidden Layers in Neural Networks*, 2024.
- Zheng et al., *Group Sequence Policy Optimization for LLM Reasoning*, Qwen Team, arXiv:2507.18071, 2025.
