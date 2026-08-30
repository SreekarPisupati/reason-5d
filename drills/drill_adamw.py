"""
Reason-5D: Daily Drill B2 - AdamW From Scratch
==============================================
Mathematical Problem:
Implement the complete AdamW optimizer (Loshchilov & Hutter 2017) from scratch with:
1. First moment: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
2. Second moment: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
3. Bias correction: m_hat = m_t / (1 - beta1^t), v_hat = v_t / (1 - beta2^t)
4. Decoupled Weight Decay: theta_t = theta_{t-1} - lr * weight_decay * theta_{t-1}
5. Parameter Update: theta_t = theta_t - lr * m_hat / (sqrt(v_hat) + eps)

Your Goal:
Implement `CustomAdamW.step` and verify parameter parity with PyTorch's native `torch.optim.AdamW`.
"""

from typing import Callable, Iterable, Optional
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class CustomAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # TODO: Implement AdamW state initialization and update equations
                raise NotImplementedError("Implement CustomAdamW.step")

        return loss


def verify_custom_adamw():
    torch.manual_seed(42)
    # Test model
    model_custom = nn.Linear(8, 4)
    model_ref = nn.Linear(8, 4)
    model_ref.load_state_dict(model_custom.state_dict())

    opt_custom = CustomAdamW(model_custom.parameters(), lr=1e-2, weight_decay=0.05)
    opt_ref = torch.optim.AdamW(model_ref.parameters(), lr=1e-2, weight_decay=0.05)

    for step in range(5):
        x = torch.randn(2, 8)
        loss_c = model_custom(x).sum()
        loss_r = model_ref(x).sum()

        loss_c.backward()
        loss_r.backward()

        opt_custom.step()
        opt_ref.step()

        opt_custom.zero_grad()
        opt_ref.zero_grad()

    diff_w = (model_custom.weight - model_ref.weight).abs().max().item()
    diff_b = (model_custom.bias - model_ref.bias).abs().max().item()
    print(f"Max weight difference vs PyTorch AdamW: {diff_w:.6e}")
    print(f"Max bias difference vs PyTorch AdamW:   {diff_b:.6e}")
    assert diff_w < 1e-6 and diff_b < 1e-6, "CustomAdamW does not match PyTorch native AdamW!"
    print(" PASSED: CustomAdamW matches exact PyTorch AdamW implementation!")


if __name__ == "__main__":
    verify_custom_adamw()
