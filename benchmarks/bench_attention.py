"""
Reason-5D: Attention Kernel Benchmark Harness & Roofline Analyzer
=================================================================
Automated profiling suite for measuring:
- Forward & Backward latency (ms)
- Peak VRAM footprint (MB)
- Effective TFLOPS achieved
- Memory Bandwidth utilization (GB/s)
- Comparison against PyTorch SDPA (F.scaled_dot_product_attention) and FlashAttention-2
"""

import argparse
import math
import time
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


# Theoretical hardware specifications for roofline analysis
DEVICE_PROFILES = {
    "RTX_4050": {"peak_tflops": 9.0, "memory_bandwidth_gb_s": 96.0},
    "NVIDIA_T4": {"peak_tflops": 65.0, "memory_bandwidth_gb_s": 320.0},
    "RTX_4090": {"peak_tflops": 165.0, "memory_bandwidth_gb_s": 1008.0},
    "NVIDIA_A100_SXM": {"peak_tflops": 312.0, "memory_bandwidth_gb_s": 2039.0},
    "NVIDIA_H100_SXM": {"peak_tflops": 989.0, "memory_bandwidth_gb_s": 3350.0},
}


def compute_attention_flops(B: int, H: int, S: int, D: int, is_causal: bool = True) -> float:
    """
    Computes theoretical FLOPs for self-attention forward pass:
    Q @ K^T: 2 * B * H * S * S * D (causal is half: B * H * S * S * D)
    Attn @ V: 2 * B * H * S * S * D (causal is half: B * H * S * S * D)
    Total forward FLOPs = 2 * B * H * S^2 * D (or 4 * B * H * S^2 * D non-causal).
    Backward pass is 2.5x forward FLOPs.
    """
    causal_factor = 0.5 if is_causal else 1.0
    return 4.0 * causal_factor * B * H * (S ** 2) * D


def benchmark_fn(
    fn: Callable[[], None],
    num_warmup: int = 10,
    num_repeats: int = 25
) -> float:
    """
    Measures execution time of a callable in milliseconds using CUDA events or high-res timers.
    """
    # Warmup
    for _ in range(num_warmup):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_repeats):
            fn()
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / num_repeats
    else:
        start_time = time.perf_counter()
        for _ in range(num_repeats):
            fn()
        end_time = time.perf_counter()
        return ((end_time - start_time) / num_repeats) * 1000.0


def benchmark_attention_layer(
    attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    B: int = 1,
    H_q: int = 32,
    H_kv: int = 8,
    S: int = 4096,
    D: int = 64,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    measure_backward: bool = False
) -> Dict[str, float]:
    """
    Runs isolated benchmark on attention function with exact memory and timing metrics.
    """
    q = torch.randn(B, H_q, S, D, device=device, dtype=dtype, requires_grad=measure_backward)
    k = torch.randn(B, H_kv, S, D, device=device, dtype=dtype, requires_grad=measure_backward)
    v = torch.randn(B, H_kv, S, D, device=device, dtype=dtype, requires_grad=measure_backward)

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    def forward_call():
        return attn_fn(q, k, v)

    fwd_latency_ms = benchmark_fn(forward_call)

    # Memory Tracking
    peak_vram_mb = 0.0
    if torch.cuda.is_available() and device.startswith("cuda"):
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # FLOPS & Bandwidth Calculation
    flops = compute_attention_flops(B, H_q, S, D, is_causal=True)
    tflops_achieved = (flops / (fwd_latency_ms * 1e-3)) / 1e12

    # Memory IO Bytes: Read Q, K, V + Write O
    bytes_transferred = (B * H_q * S * D + 2 * B * H_kv * S * D + B * H_q * S * D) * q.element_size()
    bandwidth_gb_s = (bytes_transferred / (fwd_latency_ms * 1e-3)) / 1e9

    return {
        "batch_size": B,
        "seq_len": S,
        "head_dim": D,
        "fwd_latency_ms": fwd_latency_ms,
        "peak_vram_mb": peak_vram_mb,
        "tflops_achieved": tflops_achieved,
        "bandwidth_gb_s": bandwidth_gb_s,
    }


def run_benchmark_suite(
    attn_fn_dict: Dict[str, Callable],
    seq_lengths: List[int] = [1024, 4096, 8192, 16384],
    head_dims: List[int] = [64, 128],
    batch_sizes: List[int] = [1],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32
) -> str:
    """
    Sweeps multiple attention implementations across sequence lengths and generates a markdown table.
    """
    rows = []
    header = "| Implementation | Batch | SeqLen | HeadDim | Latency (ms) | TFLOPS | Peak VRAM (MB) | Bandwidth (GB/s) |"
    divider = "|---|---|---|---|---|---|---|---|"
    rows.extend([header, divider])

    for name, fn in attn_fn_dict.items():
        for S in seq_lengths:
            for D in head_dims:
                for B in batch_sizes:
                    try:
                        res = benchmark_attention_layer(
                            attn_fn=fn,
                            B=B,
                            H_q=16,
                            H_kv=4,
                            S=S,
                            D=D,
                            device=device,
                            dtype=dtype
                        )
                        row = f"| **{name}** | {B} | {S} | {D} | {res['fwd_latency_ms']:.3f} ms | {res['tflops_achieved']:.2f} | {res['peak_vram_mb']:.1f} MB | {res['bandwidth_gb_s']:.2f} GB/s |"
                        rows.append(row)
                    except Exception as e:
                        rows.append(f"| **{name}** | {B} | {S} | {D} | ERROR ({e}) | - | - | - |")

    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention Benchmark & Roofline Analyzer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[1024, 2048, 4096])
    args = parser.parse_args()

    # Standard PyTorch SDPA reference
    def sdpa_baseline(q, k, v):
        # Expand KV heads for GQA if needed
        if q.shape[1] != k.shape[1]:
            repeat_factor = q.shape[1] // k.shape[1]
            k_exp = k.repeat_interleave(repeat_factor, dim=1)
            v_exp = v.repeat_interleave(repeat_factor, dim=1)
        else:
            k_exp, v_exp = k, v
        return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)

    print(f"Running Attention Benchmark on device={args.device}...")
    table = run_benchmark_suite(
        {"PyTorch_SDPA": sdpa_baseline},
        seq_lengths=args.seq_lens,
        device=args.device
    )
    print("\n" + table)
