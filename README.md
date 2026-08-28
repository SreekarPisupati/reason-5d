# Reason-X: Enterprise Foundation Model Distributed Pretraining, Reasoning RL & High-Throughput Serving Framework

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org/)
[![Distributed 5D](https://img.shields.io/badge/Distributed-TP%20%7C%20SP%20%7C%20CP%20%7C%20PP%20%7C%20EP-green.svg)](https://github.com/)
[![SLA Parity](https://img.shields.io/badge/Parity%20SLA-%3C10%5E--6-brightgreen.svg)](https://github.com/)

> **Engineered for Frontier AI Research & Engineering at Soket AI (IndiaAI Mission)**.
> A ground-up, production-grade foundation model systems architecture spanning arithmetic-preserving tokenization, 5D distributed training (TP/SP/CP/PP/EP), sparse MoE, scaling laws, DeepSeek-R1 style Group Relative Policy Optimization (GRPO), and PagedAttention serving.

---

## 🏛️ System Topology

```
                                ┌──────────────────────────────────────────────────────────┐
                                │             PROJECT REASON-X SYSTEM TOPOLOGY             │
                                └──────────────────────────────────────────────────────────┘
                                                              │
     ┌───────────────────────────┬────────────────────────────┴────────────────────────────┬───────────────────────────┐
     ▼                           ▼                                                         ▼                           ▼
[1. DATA & DECONTAM]     [2. 5D DISTRIBUTED & MOE CORE]                    [3. TRAINING DYNAMICS]      [4. REASONING POST-TRAINING]
- Digit-Split BPE        - TP: Column/Row Parallel Linear                  - Power-Law Scaling Laws    - Packed CoT SFT (Masked)
- 13-gram MinHash LSH    - SP: Sequence-Sliced Norms & Dropout             - WSD Learning Rate Curve   - SymPy / Code Verifiers
- Test Set Decontamination- CP: Ring-Attention / DeepSpeed-Ulysses         - Loss Spike Auto-Recovery  - GRPO (DeepSeek-R1 style)
- Zero-Copy mmap Stream  - PP: 1F1B Inter-Node Pipeline Schedules          - nsys Profiling (≥50% MFU) - PagedAttention Serving
                         - EP: Top-2 MoE (All-to-All Dispatch & Combine)   - Fused Triton Kernels      - Speculative Verification
                         - DP: FSDP-2 Sharded Data Parallel
```

---

## ⚡ Hardware Execution Tiers

| Tier | Environment | Capabilities & Workloads |
|---|---|---|
| **Tier 1: Local Rig** | RTX 4050 6GB (Ada CC 8.9) + i7-13700HX | All core algorithms, BPE Tokenizer, Binary Memmap, MinHash LSH, PagedAttention block manager, Gloo CPU distributed mock, Unit tests, PyTorch/Triton kernels |
| **Tier 2: Kaggle Cloud** | 2x NVIDIA T4 (32GB VRAM Free) | Real 2-GPU NCCL `torchrun` execution: TP/SP/CP/EP, `all_to_all_single` MoE, 1F1B Pipeline Parallelism, GRPO reasoning RL on SmolLM/Qwen |
| **Tier 3: Burst Credits** | 4x/8x GPU Cloud Instances | Multi-GPU Nsight Systems (`nsys`) profiling traces, NVTX markers, sustained $\ge 45\text{--}50\%$ Model FLOPs Utilization (MFU) graphs |

---

## 🔬 Core Modules & Verification SLAs

### Module 1: Data Ingestion, Tokenizer & Decontamination
- **1.1 Arithmetic-Preserving BPE Tokenizer**: 32,000 vocab, regex digit-split `\d`, discrete whitespace retention, byte fallback.
  - *SLA*: $0\%$ `<unk>`, `'10842'` $\rightarrow$ `['1', '0', '8', '4', '2']`.
- **1.2 Zero-Copy Binary Memmap Streamer & Packer**: Flat uint16 `.bin` + `.idx` table, document packing with 2D block-diagonal attention mask.
  - *SLA*: Zero memory growth over 10k steps, $<0.5\text{ ms}$ latency/batch.
- **1.3 13-Gram MinHash LSH Decontamination**: 128 hash permutations, Jaccard 0.8 clustering, 13-gram contamination scanner against GSM8K/MATH/HumanEval.
  - *SLA*: $0\%$ 13-gram test contamination overlap.

### Module 2: 5D Parallelism Distributed Core & Sparse MoE
- **2.1 Tensor & Sequence Parallelism (TP + SP)**: `ColumnParallelLinear` & `RowParallelLinear` with custom `torch.autograd.Function` NCCL hooks (`all_reduce`, `reduce_scatter`), sequence-sliced LayerNorm.
  - *SLA*: Forward/backward difference $< 10^{-6}$ vs `torch.nn.Linear`.
- **2.2 Context Parallelism (CP - Ring Attention)**: Ring-based attention, local Query, cyclic Key/Value passing with non-blocking `P2POp`, online softmax.
  - *SLA*: Output matches single-GPU scaled dot-product attention to $10^{-5}$ tolerance.
- **2.3 Pipeline Parallelism (PP - 1F1B Scheduling)**: Multi-stage layer partitioning with non-blocking P2P staging buffers.
  - *SLA*: Measured bubble matches theoretical $F_{\text{bubble}} = \frac{p-1}{m+p-1}$.
- **2.4 Sparse MoE with Expert Parallelism (EP)**: Top-2 gated sparse FFN, shared router experts, router z-loss, aux load-balancing loss, `all_to_all_single`.
  - *SLA*: Expert load imbalance coefficient of variation $\text{CV} < 0.05$.
- **2.5 Custom Fused Triton GPU Kernels**: Fused RMSNorm, RoPE, and SwiGLU kernels for Ada Lovelace / Ampere architectures.
  - *SLA*: $\ge 1.6\times$ speedup, $> 40\%$ activation memory reduction.

### Module 3: Scientific Scaling, Stability & Profiling
- **3.1 Empirical Scaling Laws**: Parametric power-law fitting $L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$ across model family $\{10\text{M}, 25\text{M}, 50\text{M}, 100\text{M}\}$.
  - *SLA*: Predicts $150\text{M}$ validation loss within $3\%$ relative error.
- **3.2 Warmup-Stable-Decay (WSD) Scheduler**: Linear warmup $\rightarrow$ flat stable plateau $\rightarrow$ rapid $1/\sqrt{t}$ or cosine decay.
- **3.3 Numerical Stability Suite & Auto-Recovery**: Per-head QK-Norm, Logit Z-loss, automated $\|\mathbf{g}\| > \tau$ spike detector, async checkpoint rollback & corrupted batch skipper.
  - *SLA*: Zero unrecoverable NaNs across gradient injection tests.
- **3.4 Hardware FLOPs & MFU Profiler**: NVTX instrumentation, exact MFU tracking ($6ND$), nsys trace visualizer.
  - *SLA*: Sustained $\text{MFU} \ge 45\text{--}50\%$.

### Module 4: Reasoning Post-Training & Reinforcement Learning (GRPO)
- **4.1 Packed CoT SFT**: Prompt loss masking with `<think> ... </think>` delimiters.
  - *SLA*: Exact $0.0$ loss and zero gradients on prompt/padding positions.
- **4.2 Dual Rule-Based & Symbolic Verifiers**: Sandboxed Python executor + SymPy/LaTeX symbolic equivalence normalizer.
  - *SLA*: $< 2\text{ ms}$ evaluation latency per math solution.
- **4.3 Group Relative Policy Optimization (GRPO)**: DeepSeek-R1 style RL from scratch without Critic network, group-normalized advantage $A_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}$.
  - *SLA*: $> 15\%$ accuracy gain on GSM8K reasoning benchmark.

### Module 5: Test-Time Compute & High-Throughput Serving
- **5.1 PagedAttention Dynamic Memory Manager**: Non-contiguous virtual memory block table manager (16-token physical blocks), dynamic ref-counting for copy-on-write branching.
  - *SLA*: $< 5\%$ memory waste due to fragmentation.
- **5.2 Continuous Batching & Chunked Prefill**: Iteration-level scheduling loop.
  - *SLA*: $\ge 3\times$ higher request throughput vs static batching.
- **5.3 Speculative Decoding Engine**: Compact draft model $K$-token proposal + target parallel verification.
  - *SLA*: $\ge 2\times$ wall-clock speedup, $\ge 70\%$ token acceptance rate.
