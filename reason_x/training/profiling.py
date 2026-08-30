"""
Reason-X: Module 3.4 - Hardware FLOPs & Model FLOPs Utilization (MFU) Profiler
=============================================================================
Architectural Specification:
- Calculates theoretical FLOPs per token: 6 * N (forward + backward) + attention quadratic FLOPs: 12 * L * H * S.
- Computes sustained Model FLOPs Utilization (MFU):
  MFU = (Tokens_per_sec * FLOPs_per_token) / (World_Size * Peak_Hardware_TFLOPs)
- Targets >= 48-52% MFU on modern GPU architectures.
"""

from typing import Dict, Optional
import time
import torch


# Peak theoretical TFLOPs per GPU
PEAK_TFLOPS = {
    "RTX_4050": 9.0,         # FP32/BF16 ~9-18 TFLOPS
    "NVIDIA_T4": 65.0,       # FP16 Tensor Cores
    "NVIDIA_A100_SXM": 312.0,# BF16/FP16 Tensor Cores
    "NVIDIA_H100_SXM": 989.0,# BF16/FP16 Tensor Cores (w/o sparsity)
}


class MFUProfiler:
    """
    Tracks step times and calculates exact MFU and Tokens/sec.
    """

    def __init__(
        self,
        num_params: int,
        num_layers: int,
        hidden_size: int,
        seq_len: int,
        device_name: str = "NVIDIA_T4",
        world_size: int = 1
    ):
        self.num_params = num_params
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.device_name = device_name
        self.world_size = world_size
        self.peak_tflops = PEAK_TFLOPS.get(device_name, 65.0)

        # FLOPs per token = 6N + 12 * L * H * S (attention)
        self.flops_per_token = 6 * num_params + 12 * num_layers * hidden_size * seq_len
        self.start_time = None
        self.total_tokens_processed = 0

    def start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def step(self, batch_size: int, seq_len: Optional[int] = None) -> Dict[str, float]:
        """
        Records step completion and returns throughput & MFU metrics.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        elapsed = end_time - (self.start_time or end_time)
        elapsed = max(elapsed, 1e-6)

        actual_seq = seq_len or self.seq_len
        step_tokens = batch_size * actual_seq * self.world_size
        self.total_tokens_processed += step_tokens

        tokens_per_sec = step_tokens / elapsed
        flops_achieved = tokens_per_sec * self.flops_per_token
        peak_system_flops = self.world_size * self.peak_tflops * 1e12

        mfu = flops_achieved / peak_system_flops

        # Restart timer for next step
        self.start_time = time.perf_counter()

        return {
            "step_time_sec": elapsed,
            "tokens_per_sec": tokens_per_sec,
            "tflops_achieved": flops_achieved / 1e12,
            "mfu": mfu,
            "mfu_percent": mfu * 100.0
        }
