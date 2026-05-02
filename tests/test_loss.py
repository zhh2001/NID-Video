"""Tests for src/nid_video/trainer/loss.py (M5.4 task).

Pin: ``gamma=0`` reduces to vanilla cross-entropy, ``gamma>0`` down-weights
easy samples relative to hard ones, and the gradient mechanism actually
moves more signal toward the hard rows. Numerical-stability and
reduction-mode tests guard against future refactor regressions; the
alpha test pins the Phase 2 hook so the per-class-weighting wiring won't
silently drift.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nid_video.trainer.loss import (
    FocalLoss,
    build_criterion,
    compute_inverse_sqrt_alpha,
)
from nid_video.utils.config import TrainingConfig


# ---------------------------------------------------------------------------
# Headline identity: gamma=0 is exactly cross-entropy
# ---------------------------------------------------------------------------


def test_focal_loss_gamma_zero_equals_cross_entropy() -> None:
    """The defining mathematical identity: with gamma=0, the focal factor
    (1-p_t)^0 = 1 for all samples, so FocalLoss reduces to multi-class CE.
    Within 1e-6 of F.cross_entropy on a random batch."""
    torch.manual_seed(0)
    logits = torch.randn(8, 13)
    targets = torch.randint(0, 13, (8,))
    fl = FocalLoss(gamma=0.0, reduction="mean")
    ce = F.cross_entropy(logits, targets, reduction="mean")
    fl_val = fl(logits, targets)
    assert torch.allclose(fl_val, ce, atol=1e-6), (
        f"gamma=0 must equal CE; got fl={fl_val.item()}, ce={ce.item()}"
    )


# ---------------------------------------------------------------------------
# Easy / hard sample relative weighting
# ---------------------------------------------------------------------------


def test_focal_loss_easy_sample_loss_strictly_smaller_than_ce() -> None:
    """Easy samples (high-confidence correct prediction) should be
    down-weighted by the focal factor. Concretely: with gamma=2 and a
    well-separated logit, FL should be much smaller than CE for the same
    sample."""
    # Shape (1, 3); sample is "easy": logit 5 on the correct class, 0 elsewhere.
    logits = torch.tensor([[5.0, 0.0, 0.0]])
    targets = torch.tensor([0])
    fl_val = FocalLoss(gamma=2.0, reduction="mean")(logits, targets)
    ce_val = F.cross_entropy(logits, targets, reduction="mean")
    # Strictly less: focal factor (1-p_t)^2 with p_t≈0.99 makes FL ≈ 0.0001*CE.
    assert fl_val.item() < 0.1 * ce_val.item(), (
        f"easy sample FL should be << CE; got fl={fl_val.item()}, ce={ce_val.item()}"
    )


def test_focal_loss_hard_sample_loss_close_to_ce() -> None:
    """Hard samples (near-uniform softmax) should NOT be heavily
    down-weighted: the focal factor (1-p_t)^gamma stays close to 1 when
    p_t is small. Pinning the FL/CE ratio in [0.5, 1.0] for hard cases."""
    # Near-uniform logits on a 3-class problem; p_t ≈ 1/3.
    logits = torch.tensor([[0.1, 0.0, 0.0]])
    targets = torch.tensor([0])
    fl_val = FocalLoss(gamma=2.0, reduction="mean")(logits, targets)
    ce_val = F.cross_entropy(logits, targets, reduction="mean")
    ratio = fl_val.item() / ce_val.item()
    # p_t ≈ 0.36, focal factor ≈ 0.41. Allow 0.3-0.6 envelope.
    assert 0.3 < ratio < 0.6, (
        f"hard sample FL/CE ratio should be in (0.3, 0.6) for p_t≈0.36; "
        f"got fl/ce = {ratio:.3f}"
    )


# ---------------------------------------------------------------------------
# Numerical stability under extreme inputs
# ---------------------------------------------------------------------------


def test_focal_loss_no_nan_inf_on_extreme_logits() -> None:
    """log_softmax handles overflow/underflow internally — verify FL stays
    finite even at logit magnitude 1e6 (overflow risk on naive
    exp(softmax)) and 1e-6 (no risk but sanity)."""
    fl = FocalLoss(gamma=2.0, reduction="mean")
    for scale in (1e6, 1e-6, -1e6):
        logits = torch.full((4, 5), float(scale))
        # Set a different value on the correct class to make the problem
        # non-degenerate (otherwise all-equal logits give p_t = 1/C exactly).
        logits[:, 0] += 1.0
        targets = torch.zeros(4, dtype=torch.long)
        out = fl(logits, targets)
        assert torch.isfinite(out), f"loss not finite at scale={scale}: {out.item()}"


# ---------------------------------------------------------------------------
# Phase 2 hook: per-class alpha weighting
# ---------------------------------------------------------------------------


def test_focal_loss_alpha_per_class_weighting_correct() -> None:
    """alpha[target] should multiply each sample's loss before reduction.

    To test linearity cleanly, use logits where both target classes carry
    identical logit values, so p_t is equal across the two samples and
    each sample's pre-alpha loss is the same. Then with alpha=(0.1, 1.0),
    sum_with_alpha / sum_without_alpha = (0.1 + 1.0) / 2.0 = 0.55 — pure
    linearity, no logit asymmetry confounding the result.
    """
    logits = torch.tensor([
        [1.0, 1.0, 0.0],     # logit on classes 0 and 1 are equal
        [1.0, 1.0, 0.0],
    ])
    targets = torch.tensor([0, 1])     # different classes, same p_t each

    no_alpha = FocalLoss(gamma=2.0, reduction="sum")(logits, targets)
    alpha = torch.tensor([0.1, 1.0, 0.0])
    weighted = FocalLoss(gamma=2.0, alpha=alpha, reduction="sum")(logits, targets)

    # Each per-sample loss = L (identical p_t across the two rows).
    # Sum without alpha = 2L. Sum with alpha = 0.1*L + 1.0*L = 1.1*L.
    # → weighted / no_alpha = 1.1 / 2.0 = 0.55.
    ratio = weighted.item() / no_alpha.item()
    assert abs(ratio - 0.55) < 1e-4, (
        f"alpha weighting incorrect: weighted/no_alpha = {ratio} (expected 0.55)"
    )


# ---------------------------------------------------------------------------
# Reduction mode shapes
# ---------------------------------------------------------------------------


def test_focal_loss_reduction_modes_shape() -> None:
    """``mean`` and ``sum`` return scalars; ``none`` returns shape (B,)."""
    logits = torch.randn(7, 4)
    targets = torch.randint(0, 4, (7,))
    for r in ("mean", "sum"):
        out = FocalLoss(gamma=2.0, reduction=r)(logits, targets)
        assert out.dim() == 0, f"reduction={r} should give a scalar; got {out.shape}"
    out = FocalLoss(gamma=2.0, reduction="none")(logits, targets)
    assert out.shape == (7,), f"reduction='none' should give (B,); got {out.shape}"


# ---------------------------------------------------------------------------
# ignore_index parity with cross-entropy
# ---------------------------------------------------------------------------


def test_focal_loss_ignore_index_matches_cross_entropy() -> None:
    """ignore_index=-100 (CE default): with gamma=0, FL with some targets
    set to -100 must match F.cross_entropy(ignore_index=-100) exactly —
    same denominator (count of non-ignored), same exclusion semantics."""
    torch.manual_seed(0)
    logits = torch.randn(6, 5)
    targets = torch.tensor([0, -100, 2, -100, 4, 1])
    fl_val = FocalLoss(gamma=0.0, reduction="mean", ignore_index=-100)(logits, targets)
    ce_val = F.cross_entropy(logits, targets, reduction="mean", ignore_index=-100)
    assert torch.allclose(fl_val, ce_val, atol=1e-6), (
        f"gamma=0 with ignore_index should match CE: fl={fl_val.item()}, "
        f"ce={ce_val.item()}"
    )


# ---------------------------------------------------------------------------
# Gradient mechanism: focal must move more signal to hard samples
# ---------------------------------------------------------------------------


def test_focal_loss_gradient_focuses_on_hard_samples() -> None:
    """The defining mechanism of focal loss is that gradient magnitude is
    concentrated on hard samples (low p_t) and damped on easy samples
    (high p_t). Pin the relative magnitude: per-row grad-norm of the
    hard sample > per-row grad-norm of the easy sample under FL.

    This guards against accidental detach() / no_grad regressions where
    the forward output could still look right but training would behave
    like CE (or worse, like nothing)."""
    # Two-row batch: row 0 easy (logit 8 on correct), row 1 hard (logit 0.5
    # on correct, mild margin). Both target class 0.
    logits = torch.tensor([
        [8.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ], requires_grad=True)
    targets = torch.tensor([0, 0])

    loss = FocalLoss(gamma=2.0, reduction="sum")(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.shape == (2, 3)
    assert torch.isfinite(logits.grad).all()

    easy_grad_norm = logits.grad[0].abs().sum().item()
    hard_grad_norm = logits.grad[1].abs().sum().item()
    assert hard_grad_norm > easy_grad_norm, (
        f"focal must concentrate gradient on hard sample: "
        f"easy_grad_norm={easy_grad_norm:.4e}, "
        f"hard_grad_norm={hard_grad_norm:.4e}"
    )
    # Stronger pin: the ratio should be substantial (orders of magnitude),
    # not just slightly larger. Easy logit 8 → p_t ≈ 1.0 → focal_w ≈ 0;
    # hard logit 0.5 → p_t ≈ 0.5 → focal_w ≈ 0.25. Ratio of grad norms
    # should be in the hundreds-to-thousands range.
    assert hard_grad_norm > 100 * easy_grad_norm, (
        f"focal grad-norm ratio (hard/easy) too small: "
        f"{hard_grad_norm / max(easy_grad_norm, 1e-30):.2f}× (expected > 100×)"
    )


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


def test_build_criterion_dispatches_on_loss_fn_field() -> None:
    """``build_criterion`` returns the configured loss class."""
    # Default (loss_fn="ce")
    cfg_ce = TrainingConfig()
    crit_ce = build_criterion(cfg_ce)
    assert type(crit_ce).__name__ == "CrossEntropyLoss"

    # loss_fn="focal" → FocalLoss with the configured gamma
    cfg_focal = TrainingConfig(loss_fn="focal", focal_gamma=3.5)
    crit_focal = build_criterion(cfg_focal)
    assert isinstance(crit_focal, FocalLoss)
    assert crit_focal.gamma == 3.5


# ---------------------------------------------------------------------------
# M5.4 Phase 2: inverse-sqrt class reweighting (compute_inverse_sqrt_alpha)
# ---------------------------------------------------------------------------


def test_inverse_sqrt_alpha_correctness() -> None:
    """For counts [100, 25, 4]:
      raw = [1/10, 1/5, 1/2] = [0.1, 0.2, 0.5]
      mean(present) = (0.1 + 0.2 + 0.5) / 3 = 0.2667
      normalized = [0.375, 0.75, 1.875]
    Pin the formula end-to-end."""
    alpha = compute_inverse_sqrt_alpha([100, 25, 4], num_classes=3)
    assert alpha.shape == (3,)
    expected = torch.tensor([0.375, 0.75, 1.875])
    assert torch.allclose(alpha, expected, atol=1e-4), (
        f"inverse-sqrt alpha incorrect: got {alpha.tolist()}, expected {expected.tolist()}"
    )


def test_alpha_normalization_uses_only_present_classes() -> None:
    """A zero-count class must not enter the mean denominator. Counts
    [100, 25, 0, 4] → raw [0.1, 0.2, 0.0, 0.5] → mean over n>0 entries
    only (0.2667), normalized [0.375, 0.75, 0.0, 1.875]. The n=0 entry
    stays exactly 0; without this, the mean denominator would be wrong
    (n=4 instead of n=3) and all normalised values would be inflated."""
    alpha = compute_inverse_sqrt_alpha([100, 25, 0, 4], num_classes=4)
    expected = torch.tensor([0.375, 0.75, 0.0, 1.875])
    assert torch.allclose(alpha, expected, atol=1e-4), (
        f"alpha with n=0 entry incorrect: got {alpha.tolist()}, expected {expected.tolist()}"
    )
    # Defensive: explicit zero check (atol could mask a sub-1e-4 bug)
    assert alpha[2].item() == 0.0


def test_alpha_handles_all_zero_count_classes_safely() -> None:
    """Degenerate case: every class has n=0. compute_inverse_sqrt_alpha
    must return all-zeros without div-by-zero or NaN. This guards
    against a future split bug where the train partition is empty —
    fail loud at training time via downstream metric checks rather than
    crash inside the loss factory."""
    alpha = compute_inverse_sqrt_alpha([0, 0, 0, 0, 0], num_classes=5)
    assert alpha.shape == (5,)
    assert torch.isfinite(alpha).all(), f"all-zero counts produced NaN/Inf: {alpha}"
    assert (alpha == 0.0).all(), f"all-zero counts should give all-zero alpha: {alpha}"


def test_compute_inverse_sqrt_alpha_validates_inputs() -> None:
    """Length mismatch and negative counts must raise rather than be
    silently truncated/padded. Forensic-clarity safeguard against
    upstream split bugs."""
    with pytest.raises(ValueError, match="length"):
        compute_inverse_sqrt_alpha([1, 2, 3], num_classes=4)
    with pytest.raises(ValueError, match="non-negative"):
        compute_inverse_sqrt_alpha([1, -2, 3], num_classes=3)


def test_compute_inverse_sqrt_alpha_returns_float32_cpu_tensor() -> None:
    """Public-API contract: float32 tensor on CPU, regardless of input
    type. Caller (scripts/train.py) is responsible for moving it to the
    model's device."""
    alpha = compute_inverse_sqrt_alpha([10, 100, 1000], num_classes=3)
    assert alpha.dtype == torch.float32
    assert alpha.device.type == "cpu"
    # And from a torch.Tensor input, behaviour is identical
    alpha_t = compute_inverse_sqrt_alpha(torch.tensor([10, 100, 1000]), num_classes=3)
    assert torch.allclose(alpha, alpha_t)


def test_focal_loss_alpha_none_equals_phase1_unchanged() -> None:
    """Backward-compat pin: FocalLoss(gamma=2.0, alpha=None) (the Phase 1
    construction) is bit-identical to building via build_criterion with
    no alpha kwarg. Catches a regression where the new alpha-injection
    path silently changes default behaviour."""
    torch.manual_seed(0)
    logits = torch.randn(8, 13)
    targets = torch.randint(0, 13, (8,))

    cfg = TrainingConfig(loss_fn="focal", focal_gamma=2.0)
    crit_no_alpha = build_criterion(cfg)                  # alpha defaults to None
    crit_explicit = FocalLoss(gamma=2.0, alpha=None)

    a = crit_no_alpha(logits, targets)
    b = crit_explicit(logits, targets)
    assert torch.allclose(a, b, atol=1e-7)
