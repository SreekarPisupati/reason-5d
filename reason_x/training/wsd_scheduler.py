"""
Reason-X: Module 3.2 - Warmup-Stable-Decay (WSD) Learning Rate Scheduler
========================================================================
Architectural Specification:
- WSD Schedule (MiniCPM, DeepSeek, Llama-3 style):
  1. Warmup Phase (0 <= step < T_warmup): Linear ramp from lr_min to lr_max.
  2. Stable Phase (T_warmup <= step < T_stable): Flat plateau at lr_max.
  3. Decay Phase (T_stable <= step < T_total): Rapid cosine or 1/sqrt(t) decay to lr_min.
- Decouples pretraining exploration from learning rate annealing, enabling continual training
  branching without loss restart penalties.
"""

import math
from typing import List
from torch.optim.lr_scheduler import _LRScheduler


class WSDScheduler(_LRScheduler):
    """
    Warmup-Stable-Decay (WSD) learning rate scheduler.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        min_lr_ratio: float = 0.05,
        decay_type: str = "cosine",
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.total_steps = warmup_steps + stable_steps + decay_steps
        self.min_lr_ratio = min_lr_ratio
        self.decay_type = decay_type

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch

        # 1. Warmup Phase
        if step < self.warmup_steps:
            pct = step / max(1, self.warmup_steps)
            return [base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * pct) for base_lr in self.base_lrs]

        # 2. Stable Plateau Phase
        elif step < self.warmup_steps + self.stable_steps:
            return [base_lr for base_lr in self.base_lrs]

        # 3. Decay Phase
        else:
            decay_step = step - (self.warmup_steps + self.stable_steps)
            decay_pct = min(1.0, decay_step / max(1, self.decay_steps))

            if self.decay_type == "cosine":
                decay_factor = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * decay_pct))
            else:  # Linear decay
                decay_factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * (1.0 - decay_pct)

            return [base_lr * decay_factor for base_lr in self.base_lrs]
