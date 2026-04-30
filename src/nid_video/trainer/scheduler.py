"""LR scheduling: cosine decay with linear warmup.

Two reasons we roll our own instead of using ``transformers.get_cosine_schedule_with_warmup``:

  1. The transformers helper is just a thin closure over LambdaLR with the
     same formula — no value lost, plus we keep the dependency surface flat.
  2. Tests can pin the closed-form formula at specific step values; binding
     to a third-party schedule would couple our regression tests to that
     library's micro-versioning.

Schedule (Idea.md M4 task 4.3):

    step in [0, warmup_steps):
        lr / base_lr = step / warmup_steps      # linear from 0 → 1

    step in [warmup_steps, total_steps):
        progress = (step - warmup_steps) / (total_steps - warmup_steps)   # 0 → 1
        cos     = 0.5 * (1 + cos(π · progress))                           # 1 → 0
        lr / base_lr = min_ratio + (1 - min_ratio) · cos                  # 1 → min_ratio

    step >= total_steps:
        lr / base_lr = min_ratio                # floor; never goes lower

Default ``min_ratio = 0.01`` per the M4 prompt: cosine decays to 1% of base lr.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup_factor(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.01,
) -> float:
    """Compute the lr multiplier for a given step. Pure function, no torch dep.

    Used both as the LambdaLR lambda and as the closed-form reference for tests.
    """
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if total_steps <= warmup_steps:
        raise ValueError(
            f"total_steps ({total_steps}) must be > warmup_steps ({warmup_steps})"
        )
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError(f"min_ratio must be in [0, 1], got {min_ratio}")

    if step < warmup_steps:
        # Linear ramp 0 → 1 over [0, warmup_steps). At step=0 the multiplier
        # is 0 (and PyTorch's LambdaLR will multiply that by base_lr → effective
        # lr=0 at step 0); the first non-zero lr arrives at step 1.
        return float(step) / float(max(1, warmup_steps))

    if step >= total_steps:
        return min_ratio

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cos


def make_cosine_scheduler(
    optimizer: Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.01,
) -> LambdaLR:
    """Wrap :func:`cosine_with_warmup_factor` as a ``LambdaLR``.

    Returns a standard PyTorch scheduler. ``state_dict()`` / ``load_state_dict()``
    round-trip the internal step counter (LambdaLR's ``last_epoch``); the lambda
    closure itself is NOT persisted, so callers must reconstruct the scheduler
    with the same warmup/total/min_ratio config when resuming. The trainer
    persists those config values alongside the state_dict to keep this honest.
    """
    def lr_lambda(step: int) -> float:
        return cosine_with_warmup_factor(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_ratio=min_ratio,
        )
    return LambdaLR(optimizer, lr_lambda=lr_lambda)
