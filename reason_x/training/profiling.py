"""
Reason-5D: Module 3.4 - Hardware FLOPs & Model FLOPs Utilization (MFU) Profiler
=============================================================================
Architectural Specification:
- Calculates theoretical FLOPs per token: 6 * N + attention quadratic FLOPs: 12 * L * H * S.
- Computes sustained Model FLOPs Utilization (MFU).
"""

from typing import Dict, Optional
import time
import torch


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
        raise NotImplementedError("TODO: Implement MFUProfiler.__init__")

    def start(self) -> None:
        raise NotImplementedError("TODO: Implement MFUProfiler.start")

    def step(self, batch_size: int, seq_len: Optional[int] = None) -> Dict[str, float]:
        """
        Records step completion and returns throughput & MFU metrics.
        """
        raise NotImplementedError("TODO: Implement MFUProfiler.step")
