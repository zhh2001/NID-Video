"""Multi-class focal loss + a small factory that picks the criterion based on
the training config. Mirrors the standalone-module pattern used by
``scheduler.py`` (M4.3) and ``evaluator.py`` (M4.5) so loss math stays
unit-testable in isolation and the trainer just consumes an ``nn.Module``.

Focal loss formulation (Lin et al. 2017, ICCV — "Focal Loss for Dense
Object Detection"), multi-class form:

    log_p_t = log_softmax(logits, dim=-1).gather(targets)
    p_t     = exp(log_p_t)
    FL      = -(alpha[target]) * (1 - p_t) ** gamma * log_p_t

``gamma=0`` reduces this to standard cross-entropy (the focal factor
``(1-p_t)^0 = 1`` cancels). Pinned by ``test_loss``'s
``test_focal_loss_gamma_zero_equals_cross_entropy`` so a future refactor
that drifts the formula breaks loudly.

Numerical stability: log-softmax handles the under/overflow cases that
naive ``log(softmax(...))`` would mishandle. ``p_t = exp(log_p_t)`` is
well-defined for all log_p_t in (-inf, 0]. ``(1 - p_t) ** gamma`` is
well-defined on ``p_t ∈ [0, 1]`` for non-negative ``gamma``. No clamps,
no eps — the log_softmax pathway already covers the failure modes,
including under FP16 autocast.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from nid_video.utils.config import TrainingConfig


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weighting.

    Args:
      gamma: focusing parameter. ``gamma=0`` recovers standard
        multi-class cross-entropy (verified by unit test).
      alpha: optional per-class weight, shape ``(num_classes,)``.
        Registered as a buffer so ``.to(device)`` follows the model.
        Phase 2 hook for class-frequency reweighting; ``None`` disables
        it (Phase 1 default).
      reduction: ``"mean"`` (default), ``"sum"``, or ``"none"`` —
        same conventions as ``nn.CrossEntropyLoss``.
      ignore_index: targets with this value are excluded from the loss
        and from the ``"mean"`` denominator. Default ``-100`` matches
        ``nn.CrossEntropyLoss``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none'; got {reduction!r}"
            )
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = int(ignore_index)
        if alpha is not None:
            self.register_buffer("alpha", alpha.detach().clone(), persistent=True)
        else:
            self.alpha = None

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        # logits: (B, C) float, targets: (B,) long. Keep ignore-index
        # masking before any gather to avoid out-of-bounds indices.
        valid = targets != self.ignore_index
        if not valid.any():
            # Match nn.CrossEntropyLoss's behaviour: return a 0 scalar
            # connected to the graph so backward still works.
            return logits.sum() * 0.0

        valid_logits = logits[valid]
        valid_targets = targets[valid]

        log_probs = F.log_softmax(valid_logits, dim=-1)
        log_p_t = log_probs.gather(-1, valid_targets.unsqueeze(-1)).squeeze(-1)
        p_t = log_p_t.exp()
        focal_w = (1.0 - p_t) ** self.gamma
        per_sample = -focal_w * log_p_t                       # (B_valid,)

        if self.alpha is not None:
            alpha_t = self.alpha[valid_targets].to(per_sample.dtype)
            per_sample = alpha_t * per_sample

        if self.reduction == "none":
            # Reinflate to original (B,) shape with zeros at ignored slots
            # so the caller can index against the original target tensor.
            out = torch.zeros_like(targets, dtype=per_sample.dtype)
            out[valid] = per_sample
            return out
        if self.reduction == "sum":
            return per_sample.sum()
        # "mean" — averaged over valid elements only (CE convention).
        return per_sample.mean()

    def extra_repr(self) -> str:
        alpha_repr = "None" if self.alpha is None else f"tensor(shape={tuple(self.alpha.shape)})"
        return (
            f"gamma={self.gamma}, alpha={alpha_repr}, "
            f"reduction='{self.reduction}', ignore_index={self.ignore_index}"
        )


def build_criterion(cfg: TrainingConfig) -> nn.Module:
    """Pick the loss function based on ``cfg.loss_fn``. Default ``"ce"``
    yields ``nn.CrossEntropyLoss`` so any pre-M5.4 config or test that
    leaves ``loss_fn`` unset is byte-identical to today.

    For ``"focal"`` returns ``FocalLoss(gamma=cfg.focal_gamma)`` with no
    alpha (Phase 1 default; alpha is the Phase 2 hook).
    """
    if cfg.loss_fn == "focal":
        return FocalLoss(gamma=cfg.focal_gamma)
    return nn.CrossEntropyLoss()
