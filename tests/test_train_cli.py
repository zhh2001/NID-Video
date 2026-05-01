"""Tests for scripts/train.py CLI helpers (M5 task 5.1).

Pin the closed-form total_steps math against the M4.8 measured grad-step
counts so a future regression to the pre-M5.1 single-scale formula
(under-counts by ~2× under round_robin → cosine bottoms mid-epoch) fails
loudly here.

The function under test is the pure helper ``_total_steps_from_train_n``
(no IO, no argparse). The IO wrapper ``_compute_total_steps`` is exercised
indirectly via the data-pipeline tests; these tests pin the math contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# scripts/ is not a package and not on sys.path under pytest. Insert the repo
# root so ``scripts.train`` resolves as a top-level namespace-package import.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.train import _total_steps_from_train_n  # noqa: E402


# ---------------------------------------------------------------------------
# M5.1 fix: total_steps must reflect epoch_end_strategy
# ---------------------------------------------------------------------------


def test_total_steps_round_robin_correct() -> None:
    """Multi-scale round_robin (M4.8 default): the *fast* stream is the anchor.
    With mix_ratio=0.5, total samples per epoch ≈ 2 × train_n_fast — yielding
    ~4850 grad steps for the M4.8 measured train_n_fast≈77592 / B=32 / accum=1.

    Pin against M4.8 actual run (4853 grad steps) within ±10 tolerance to
    survive small splits.parquet changes.
    """
    total = _total_steps_from_train_n(
        train_n=77592,
        batch_size=32,
        grad_accumulation=1,
        num_epochs=1,
        mix_ratio=0.5,
        epoch_end_strategy="round_robin",
    )
    assert abs(total - 4850) <= 10, (
        f"round_robin total_steps drifted from M4.8 measured 4853 "
        f"(expected 4850 ±10): got {total}"
    )


def test_total_steps_slow_exhausted_legacy() -> None:
    """Multi-scale slow_exhausted (M4.2 legacy, retained as option since M4.8):
    the *slow* stream is the anchor. With mix_ratio=0.5 and train_n_slow≈7752,
    total grad steps per epoch ≈ 2 × train_n_slow / B ≈ 485.

    Pin the M4.8 first-run 479 grad steps (under slow_exhausted) within ±10
    tolerance — confirms the legacy formula is still selectable for debug runs.
    """
    total = _total_steps_from_train_n(
        train_n=7752,
        batch_size=32,
        grad_accumulation=1,
        num_epochs=1,
        mix_ratio=0.5,
        epoch_end_strategy="slow_exhausted",
    )
    assert abs(total - 485) <= 10, (
        f"slow_exhausted total_steps drifted from M4.8 first-run 479 "
        f"(expected 485 ±10): got {total}"
    )


def test_total_steps_single_scale_unchanged() -> None:
    """Single-scale path: ``mix_ratio=None`` selects the unmodified
    ``ceil(train_n / (B × accum)) × num_epochs`` formula. Regression pin:
    the M5.1 fix must not change single-scale behaviour.

    train_n=77615, B=32, accum=1, num_epochs=1 → 2426 (the M4.8 splits.parquet
    train count divided by batch size, exact ceiling).
    """
    total = _total_steps_from_train_n(
        train_n=77615,
        batch_size=32,
        grad_accumulation=1,
        num_epochs=1,
    )
    assert total == 2426, (
        f"single-scale total_steps regressed: expected 2426, got {total}"
    )


def test_total_steps_multi_epoch_scaling() -> None:
    """Multi-epoch budget = single-epoch budget × num_epochs (linear).
    Pin that the helper multiplies correctly without any per-epoch decay or
    warmup-aware adjustment (warmup is a separate scheduler concern).

    round_robin, train_n_fast=10000, B=32, mix_ratio=0.5, num_epochs=3:
      anchor_n = ceil(10000 / 0.5) = 20000
      steps/epoch = ceil(20000 / 32) = 625
      total = 625 × 3 = 1875
    """
    total = _total_steps_from_train_n(
        train_n=10000,
        batch_size=32,
        grad_accumulation=1,
        num_epochs=3,
        mix_ratio=0.5,
        epoch_end_strategy="round_robin",
    )
    assert total == 1875, (
        f"multi-epoch scaling broke: expected 1875, got {total}"
    )


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [
        "max_len",         # reserved for M5/M6, not yet implemented
        "bogus",           # typo
        "ROUND_ROBIN",     # case-sensitivity check
        "",                # empty string
        "round-robin",     # hyphen vs underscore
    ],
)
def test_total_steps_unknown_strategy_raises(strategy: str) -> None:
    """Strategy not in {round_robin, slow_exhausted} must raise — fail fast on
    typos / future-reserved values rather than silently falling back to a
    single-scale formula. Catches both implementation gaps (max_len) and
    user error (typos, case mismatch, hyphen-vs-underscore)."""
    with pytest.raises(ValueError, match="epoch_end_strategy"):
        _total_steps_from_train_n(
            train_n=1000,
            batch_size=32,
            grad_accumulation=1,
            num_epochs=1,
            mix_ratio=0.5,
            epoch_end_strategy=strategy,
        )
