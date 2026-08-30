"""
Reason-X: Unit Tests for Module 3 (Scaling Laws, WSD Scheduler, Stability, Profiling)
=====================================================================================
Verifies:
1. Power-Law empirical curve fitting and compute-optimal allocation.
2. Warmup-Stable-Decay (WSD) learning rate phases.
3. Logit Z-loss and QK-Norm stability.
4. Gradient spike detection & auto-recovery rollback.
5. MFU profiler FLOPs calculation.
"""

import pytest
import torch
import torch.nn as nn

from reason_x.training.scaling_laws import ScalingLawFitter
from reason_x.training.wsd_scheduler import WSDScheduler
from reason_x.training.stability import (
    QKNorm,
    compute_z_loss,
    GradientSpikeAutoRecovery
)
from reason_x.training.profiling import MFUProfiler


def test_scaling_law_fitter():
    fitter = ScalingLawFitter()

    # Synthetic scaling sweep data
    N_list = [10e6, 25e6, 50e6, 100e6]
    D_list = [200e6, 500e6, 1e9, 2e9]
    # L(N, D) = 1.5 + 400 / N^0.35 + 400 / D^0.35
    L_list = [
        1.5 + 400.0 / (n ** 0.35) + 400.0 / (d ** 0.35)
        for n, d in zip(N_list, D_list)
    ]

    params = fitter.fit(N_list, D_list, L_list)
    assert abs(params["E"] - 1.5) < 0.2
    assert abs(params["alpha"] - 0.35) < 0.1

    # Predict loss for unseen scale
    pred_loss = fitter.predict_loss(150e6, 3e9)
    assert 1.5 < pred_loss < 2.5

    # Compute optimal allocation for 1e18 FLOPs budget
    n_opt, d_opt = fitter.compute_optimal_allocation(1e18)
    assert n_opt > 0
    assert d_opt > 0
    assert abs(6 * n_opt * d_opt - 1e18) / 1e18 < 0.05


def test_wsd_scheduler():
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    warmup_steps = 10
    stable_steps = 20
    decay_steps = 10
    scheduler = WSDScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        min_lr_ratio=0.1
    )

    # 1. Warmup check: step 0 -> min_lr
    lr_0 = scheduler.get_lr()[0]
    assert abs(lr_0 - 1e-4) < 1e-6

    # Step to middle of stable phase (step 15)
    for _ in range(15):
        optimizer.step()
        scheduler.step()

    lr_stable = scheduler.get_lr()[0]
    assert abs(lr_stable - 1e-3) < 1e-6

    # Step to end of decay (step 40)
    for _ in range(25):
        optimizer.step()
        scheduler.step()

    lr_decayed = scheduler.get_lr()[0]
    assert abs(lr_decayed - 1e-4) < 1e-5


def test_stability_and_spike_recovery():
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    recovery = GradientSpikeAutoRecovery(
        model=model,
        optimizer=optimizer,
        spike_threshold=10.0
    )
    
    # Save clean baseline
    recovery.save_clean_checkpoint()
    
    # Simulate normal step
    x = torch.randn(2, 10)
    loss = model(x).mean()
    loss.backward()
    is_spike, norm = recovery.check_and_recover()
    assert not is_spike

    # Simulate gradient explosion
    for p in model.parameters():
        p.grad.data.fill_(100.0)

    is_spike, norm = recovery.check_and_recover()
    assert is_spike
    assert recovery.recovery_count == 1

    # Check Z-Loss
    logits = torch.randn(2, 10, 100)
    z_loss = compute_z_loss(logits, eps_z=1e-4)
    assert z_loss.item() >= 0.0

    # Check QK-Norm
    qk_norm = QKNorm(head_dim=16)
    q = torch.randn(2, 4, 8, 16)
    q_normed = qk_norm(q)
    assert q_normed.shape == q.shape


def test_mfu_profiler():
    profiler = MFUProfiler(
        num_params=100_000_000,
        num_layers=12,
        hidden_size=768,
        seq_len=2048,
        device_name="NVIDIA_T4",
        world_size=1
    )
    profiler.start()
    metrics = profiler.step(batch_size=4, seq_len=2048)
    assert "tokens_per_sec" in metrics
    assert "tflops_achieved" in metrics
    assert "mfu" in metrics
