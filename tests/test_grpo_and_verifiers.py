"""
Reason-X: Unit Tests for Module 4 (SymPy & Code Verifiers, GRPO RL Algorithm)
=============================================================================
Verifies:
1. SymPy symbolic equivalence on LaTeX math fractions, exponents, and expressions.
2. Sandboxed Python code executor and AST safety checks.
3. Group Relative Policy Optimization (GRPO) advantage calculation and training step.
"""

import pytest
import torch
import torch.nn as nn

from reason_x.post_training.verifiers import SymbolicMathVerifier, SandboxedCodeExecutor
from reason_x.post_training.grpo import GRPOTrainer


def test_symbolic_math_verifier():
    # 1. LaTeX fraction equivalence: \frac{4}{8} == \frac{1}{2} == 0.5
    assert SymbolicMathVerifier.verify(r"\boxed{\frac{4}{8}}", r"\boxed{\frac{1}{2}}") == 1.0
    assert SymbolicMathVerifier.verify(r"\boxed{0.5}", r"\boxed{\frac{1}{2}}") == 1.0

    # 2. Algebraic simplification: (x + 1)^2 == x^2 + 2x + 1
    assert SymbolicMathVerifier.verify(r"<answer>(x + 1)^2</answer>", r"<answer>x^2 + 2*x + 1</answer>") == 1.0

    # 3. GSM8K format with #### tag
    assert SymbolicMathVerifier.verify("#### 42", "The answer is #### 42") == 1.0
    assert SymbolicMathVerifier.verify("#### 42", "The answer is #### 43") == 0.0


def test_sandboxed_code_executor():
    # Safe python code
    safe_script = """
def solution():
    fib = [0, 1]
    for i in range(8):
        fib.append(fib[-1] + fib[-2])
    return fib[-1]
"""
    assert SandboxedCodeExecutor.is_safe_code(safe_script) is True
    reward = SandboxedCodeExecutor.execute_and_verify(safe_script, expected_output=34)
    assert reward == 1.0

    # Unsafe python code (importing forbidden os)
    unsafe_script = """
import os
def solution():
    return os.listdir('.')
"""
    assert SandboxedCodeExecutor.is_safe_code(unsafe_script) is False
    assert SandboxedCodeExecutor.execute_and_verify(unsafe_script, expected_output="anything") == 0.0


class SimplePolicy(nn.Module):
    def __init__(self, vocab_size: int = 50, d_model: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.embed(x))


def test_grpo_training_step():
    torch.manual_seed(42)
    vocab_size = 50
    policy = SimplePolicy(vocab_size=vocab_size)
    ref_policy = SimplePolicy(vocab_size=vocab_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    trainer = GRPOTrainer(
        policy_model=policy,
        ref_model=ref_policy,
        optimizer=optimizer,
        group_size=4,
        clip_eps=0.2,
        kl_beta=0.04
    )

    B, G, S_prompt, S_gen = 2, 4, 8, 6
    prompt_ids = torch.randint(0, vocab_size, (B, S_prompt))
    generated_ids = torch.randint(0, vocab_size, (B, G, S_gen))
    old_log_probs = torch.randn(B, G, S_gen)

    # Mixed rewards within group
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]])

    metrics = trainer.step(prompt_ids, generated_ids, old_log_probs, rewards)
    assert "loss_grpo" in metrics
    assert "approx_kl" in metrics
    assert "mean_reward" in metrics
