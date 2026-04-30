"""Tests for src/nid_video/trainer/scheduler.py (M4 task 4.3).

Pin the closed-form formula at specific step values so a future regression
in the math (or in PyTorch's LambdaLR semantics) breaks loudly.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch
from torch import nn

from nid_video.trainer.scheduler import (
    cosine_with_warmup_factor,
    make_cosine_scheduler,
)

# The dummy SGD optimizers in this file never receive gradients, so PyTorch's
# defensive "lr_scheduler.step() called before optimizer.step()" check fires.
# The math is unaffected (verified by the closed-form pinning tests). Filter
# here to keep test output clean.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Detected call of `lr_scheduler.step\\(\\)`:UserWarning"
)


# ---------------------------------------------------------------------------
# Closed-form factor: pin specific step values
# ---------------------------------------------------------------------------


def test_warmup_linear_ramp_from_zero_to_base_lr_in_warmup_steps() -> None:
    """During warmup the multiplier is exactly step/warmup_steps."""
    f = lambda s: cosine_with_warmup_factor(s, warmup_steps=10, total_steps=100)
    assert f(0) == 0.0                  # step 0 → multiplier 0 (no lr)
    assert f(1) == pytest.approx(0.1)
    assert f(5) == pytest.approx(0.5)
    assert f(9) == pytest.approx(0.9)
    # At step=warmup_steps the formula switches to the cosine branch (progress=0)
    # which evaluates to 1.0 — i.e. the multiplier is continuous through the boundary.
    assert f(10) == pytest.approx(1.0)


def test_cosine_decay_to_one_percent_at_total_steps() -> None:
    """At step = total_steps the multiplier equals min_ratio (0.01 default).
    Beyond that it stays at min_ratio (no negative lr, no rebound)."""
    kwargs = dict(warmup_steps=10, total_steps=100, min_ratio=0.01)
    assert cosine_with_warmup_factor(100, **kwargs) == pytest.approx(0.01)
    assert cosine_with_warmup_factor(150, **kwargs) == pytest.approx(0.01)


def test_lr_curve_matches_closed_form() -> None:
    """Spot-check 0/10/55/100 against the math, per M4 task 4.3 prompt."""
    kwargs = dict(warmup_steps=10, total_steps=100, min_ratio=0.01)

    # step 0: warmup, multiplier = 0/10 = 0
    assert cosine_with_warmup_factor(0, **kwargs) == 0.0

    # step 10: cosine branch, progress = 0/90 = 0, cos(0) = 1, multiplier = 1.0
    assert cosine_with_warmup_factor(10, **kwargs) == pytest.approx(1.0)

    # step 55: cosine branch, progress = 45/90 = 0.5, cos(π/2) = 0,
    # multiplier = 0.01 + 0.99 * 0.5 = 0.505
    expected = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * 0.5))
    assert cosine_with_warmup_factor(55, **kwargs) == pytest.approx(expected)
    assert cosine_with_warmup_factor(55, **kwargs) == pytest.approx(0.505)

    # step 100: cosine branch hits the floor min_ratio
    assert cosine_with_warmup_factor(100, **kwargs) == pytest.approx(0.01)


def test_factor_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        cosine_with_warmup_factor(0, warmup_steps=-1, total_steps=100)
    with pytest.raises(ValueError, match="total_steps"):
        cosine_with_warmup_factor(0, warmup_steps=100, total_steps=50)
    with pytest.raises(ValueError, match="total_steps"):
        # equal counts are also rejected (would div-by-zero in cosine branch)
        cosine_with_warmup_factor(0, warmup_steps=100, total_steps=100)
    with pytest.raises(ValueError, match="min_ratio"):
        cosine_with_warmup_factor(0, warmup_steps=10, total_steps=100, min_ratio=-0.1)
    with pytest.raises(ValueError, match="min_ratio"):
        cosine_with_warmup_factor(0, warmup_steps=10, total_steps=100, min_ratio=1.1)


def test_zero_warmup_starts_at_full_lr() -> None:
    """warmup_steps=0 → no warmup, step 0 immediately at multiplier=1.0."""
    kwargs = dict(warmup_steps=0, total_steps=100)
    assert cosine_with_warmup_factor(0, **kwargs) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration with PyTorch LambdaLR
# ---------------------------------------------------------------------------


def _make_dummy_optimizer(lr: float = 1e-3) -> torch.optim.Optimizer:
    p = nn.Parameter(torch.zeros(1, requires_grad=True))
    return torch.optim.SGD([p], lr=lr)


def test_make_cosine_scheduler_applies_factor_to_optimizer_lr() -> None:
    """After scheduler.step() the optimizer's param_group lr equals base_lr × factor."""
    base_lr = 1e-3
    opt = _make_dummy_optimizer(lr=base_lr)
    sched = make_cosine_scheduler(opt, warmup_steps=10, total_steps=100)

    # step 0 (initial): multiplier 0, optimizer lr = 0
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)

    for _ in range(5):
        sched.step()                                    # advance to step 5
    # multiplier at step 5 = 5/10 = 0.5
    assert opt.param_groups[0]["lr"] == pytest.approx(base_lr * 0.5)

    for _ in range(5):
        sched.step()                                    # advance to step 10
    assert opt.param_groups[0]["lr"] == pytest.approx(base_lr * 1.0)


def test_scheduler_state_dict_roundtrip() -> None:
    """Save mid-training state, restore into a freshly-built scheduler:
    get_last_lr() and post-load step() output must match the original.

    Required by M4 task 4.4 (resume): if scheduler can't round-trip its
    internal step counter we lose lr continuity on resume."""
    opt = _make_dummy_optimizer()
    sched = make_cosine_scheduler(opt, warmup_steps=10, total_steps=100)

    # Step 15 times → into the cosine branch
    for _ in range(15):
        sched.step()
    saved_state = sched.state_dict()
    saved_lr = sched.get_last_lr()

    # Fresh optimizer + scheduler with the SAME config; state must transfer
    opt2 = _make_dummy_optimizer()
    sched2 = make_cosine_scheduler(opt2, warmup_steps=10, total_steps=100)
    sched2.load_state_dict(saved_state)
    assert sched2.get_last_lr() == saved_lr

    # And: stepping once more from each must agree element-wise
    sched.step()
    sched2.step()
    assert sched2.get_last_lr() == sched.get_last_lr()


def test_lr_curve_through_full_run_is_monotonic_after_warmup() -> None:
    """Full 100-step trajectory: rises through warmup then monotonically
    decreases until total_steps, then plateaus. Catches sign errors and
    bad cumulative drift."""
    opt = _make_dummy_optimizer()
    sched = make_cosine_scheduler(opt, warmup_steps=10, total_steps=100, min_ratio=0.01)
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(120):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])

    # Warmup is monotonically non-decreasing 0 → ~base
    assert all(lrs[i] <= lrs[i + 1] + 1e-12 for i in range(10))
    # Cosine branch is monotonically non-increasing 1.0 → 0.01
    assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(10, 100))
    # Past total_steps lr is flat at min_ratio × base_lr (= 1e-5)
    plateau = lrs[100:]
    assert all(abs(v - 1e-5) < 1e-9 for v in plateau)
