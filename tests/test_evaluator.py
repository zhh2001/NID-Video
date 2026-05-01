"""Tests for src/nid_video/trainer/evaluator.py (M4 task 4.5).

Pin torchmetrics' actual behavior on edge cases (zero-sample classes,
single-sample-per-class) so a future torchmetrics upgrade that silently
changes those semantics fails loudly here rather than in M4.8 real training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nid_video.trainer.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _LogitsDataset(Dataset):
    """Map-style dataset that yields a fixed logits/label sequence.

    The ``model`` we pair with this dataset is a stub that returns the
    sample's pre-baked logits as-is — lets the test pin metric values
    against hand-computed reference numbers.
    """

    def __init__(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        self.logits = logits.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return self.logits.size(0)

    def __getitem__(self, i: int) -> dict:
        return {
            "tensor": self.logits[i],
            "label": self.labels[i],
            "meta": {},
        }


class _PassthroughModel(nn.Module):
    """Returns ``x`` as logits (treats x's last dim as num_classes)."""

    def forward(self, x: torch.Tensor, *, scale_id: torch.Tensor) -> dict:
        return {"logits": x}


def _collate(batch: list[dict]) -> dict:
    return {
        "tensor": torch.stack([b["tensor"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "meta": [b["meta"] for b in batch],
    }


def _build_eval(logits: torch.Tensor, labels: torch.Tensor,
                num_classes: int) -> Evaluator:
    ds = _LogitsDataset(logits, labels)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=_collate)
    return Evaluator(model=_PassthroughModel(), val_loader=loader,
                     num_classes=num_classes, device="cpu")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_evaluator_returns_all_documented_metrics() -> None:
    """Output dict must carry the full key set from the docstring contract."""
    logits = torch.eye(5).float()                    # 5 perfect predictions
    labels = torch.arange(5)
    ev = _build_eval(logits, labels, num_classes=5)
    m = ev.evaluate()
    for k in ("accuracy", "macro_f1", "per_class_f1",
              "per_class_precision", "per_class_recall",
              "auroc_macro", "per_class_auroc",
              "confusion_matrix", "n_samples", "n_per_class"):
        assert k in m, f"missing key: {k}"
    assert m["per_class_f1"].shape == (5,)
    assert m["confusion_matrix"].shape == (5, 5)
    assert m["n_per_class"].shape == (5,)
    assert m["n_samples"] == 5


# ---------------------------------------------------------------------------
# Pin: torchmetrics' (asymmetric) treatment of zero-sample classes
# ---------------------------------------------------------------------------


def test_macro_f1_excludes_zero_sample_classes(caplog) -> None:
    """torchmetrics 1.9.0: macro_f1 is the mean over only the classes that
    appear in target. Five classes total, two appear, both perfectly
    predicted → macro = 1.0 (NOT 0.4 = mean of [1,1,0,0,0])."""
    # Five-class problem, only classes 0 and 1 appear in true labels.
    logits = torch.tensor([
        [0.9, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0, 0.0],
    ])
    labels = torch.tensor([0, 1, 0, 1])
    ev = _build_eval(logits, labels, num_classes=5)
    m = ev.evaluate()

    assert m["macro_f1"] == pytest.approx(1.0)
    np.testing.assert_allclose(m["per_class_f1"], [1.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(m["n_per_class"], [2, 2, 0, 0, 0])


def test_macro_auroc_includes_zero_positive_classes_as_zero() -> None:
    """torchmetrics 1.9.0: auroc_macro INCLUDES zero-sample classes scoring 0,
    so macro = mean([1, 1, 0, 0, 0]) = 0.4. Asymmetric vs F1 macro."""
    logits = torch.tensor([
        [0.9, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0, 0.0],
    ])
    labels = torch.tensor([0, 1, 0, 1])
    ev = _build_eval(logits, labels, num_classes=5)
    m = ev.evaluate()

    assert m["auroc_macro"] == pytest.approx(0.4, abs=1e-4)
    np.testing.assert_allclose(m["per_class_auroc"], [1.0, 1.0, 0.0, 0.0, 0.0],
                                atol=1e-6)


def test_per_class_metrics_zero_for_zero_sample_class() -> None:
    """A class with no true samples → F1 / Precision / Recall / AUROC all 0,
    silently — the caller is expected to consult ``n_per_class`` to know
    these are not meaningful."""
    # 4-class logit space (must match num_classes), but only classes 0/1/2
    # appear in target. Class 3 is fully zero-sample.
    logits = torch.zeros(3, 4).float()
    logits[0, 0] = 10.0
    logits[1, 1] = 10.0
    logits[2, 2] = 10.0
    labels = torch.tensor([0, 1, 2])
    ev = _build_eval(logits, labels, num_classes=4)
    m = ev.evaluate()

    # Classes 0..2: present; perfect prediction.
    np.testing.assert_allclose(m["per_class_f1"][:3], [1.0, 1.0, 1.0])
    # Class 3: zero true samples → all four metrics = 0
    assert m["per_class_f1"][3] == 0.0
    assert m["per_class_precision"][3] == 0.0
    assert m["per_class_recall"][3] == 0.0
    assert m["per_class_auroc"][3] == 0.0
    assert m["n_per_class"][3] == 0


def test_auroc_handles_single_sample_class() -> None:
    """If a class has exactly one sample (the Heartbleed-tier extreme),
    AUROC is computable (it's a degenerate binary case where the threshold
    that ranks the lone positive above all negatives gives AUC=1, otherwise 0)
    and does not raise. Pinning that the metric stays computable."""
    logits = torch.tensor([
        [10.0,  0.0,  0.0],   # confidently class 0
        [ 0.0, 10.0,  0.0],   # confidently class 1
        [ 0.0,  0.0, 10.0],   # confidently class 2 — lone sample of class 2
    ])
    labels = torch.tensor([0, 1, 2])
    ev = _build_eval(logits, labels, num_classes=3)
    m = ev.evaluate()
    # All three classes have ≥1 sample. AUROC computed without raising.
    assert m["per_class_auroc"].shape == (3,)
    assert all(np.isfinite(m["per_class_auroc"]))
    assert all(0.0 <= v <= 1.0 for v in m["per_class_auroc"])


# ---------------------------------------------------------------------------
# Numerical correctness on hand-computed cases
# ---------------------------------------------------------------------------


def test_per_class_metrics_correct_for_known_predictions() -> None:
    """Build a 3x3 case with known TP/FP/FN per class and verify F1/P/R."""
    # Class 0: true=[0,0,0,0], pred=[0,0,1,2]  -> TP=2, FP=0, FN=2
    #   -> P=1.0, R=0.5, F1=2*0.5/1.5 = 0.6667
    # Class 1: true=[1,1], pred=[1,1]          -> TP=2, FP=1, FN=0
    #   -> P=2/3, R=1.0, F1=2*(2/3)/(5/3) = 0.8
    # Class 2: true=[2], pred=[2]              -> TP=1, FP=1, FN=0
    #   -> P=0.5, R=1.0, F1=2*0.5/1.5 = 0.6667
    logits = torch.tensor([
        [10, 0, 0],   # true=0 pred=0
        [10, 0, 0],   # true=0 pred=0
        [0, 10, 0],   # true=0 pred=1
        [0, 0, 10],   # true=0 pred=2
        [0, 10, 0],   # true=1 pred=1
        [0, 10, 0],   # true=1 pred=1
        [0, 0, 10],   # true=2 pred=2
    ]).float()
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 2])
    ev = _build_eval(logits, labels, num_classes=3)
    m = ev.evaluate()

    np.testing.assert_allclose(m["per_class_precision"], [1.0, 2/3, 0.5], atol=1e-4)
    np.testing.assert_allclose(m["per_class_recall"],    [0.5, 1.0, 1.0], atol=1e-4)
    np.testing.assert_allclose(
        m["per_class_f1"], [2/3, 0.8, 2/3], atol=1e-4
    )


def test_confusion_matrix_correct_shape_and_values() -> None:
    """row=true, col=pred."""
    logits = torch.tensor([
        [10, 0, 0],   # true=0 pred=0
        [10, 0, 0],   # true=0 pred=0
        [0, 10, 0],   # true=0 pred=1   <- off-diagonal: row 0, col 1
        [0, 0, 10],   # true=0 pred=2   <- off-diagonal: row 0, col 2
        [0, 10, 0],   # true=1 pred=1
        [0, 10, 0],   # true=1 pred=1
        [0, 0, 10],   # true=2 pred=2
    ]).float()
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 2])
    ev = _build_eval(logits, labels, num_classes=3)
    m = ev.evaluate()

    assert m["confusion_matrix"].shape == (3, 3)
    expected = np.array([
        [2, 1, 1],   # true=0 → pred 0 (×2) / 1 (×1) / 2 (×1)
        [0, 2, 0],   # true=1 → all pred=1
        [0, 0, 1],   # true=2 → pred=2
    ])
    np.testing.assert_array_equal(m["confusion_matrix"], expected)


# ---------------------------------------------------------------------------
# Eval path properties
# ---------------------------------------------------------------------------


def test_evaluator_no_grad_during_eval() -> None:
    """torch.inference_mode() must wrap evaluate(): logits emitted by the
    model don't carry gradient state, no autograd graph is built."""
    logits = torch.eye(3).float()
    labels = torch.tensor([0, 1, 2])
    ev = _build_eval(logits, labels, num_classes=3)

    class _GradCheckModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.eye(3, dtype=torch.float32))

        def forward(self, x: torch.Tensor, *, scale_id: torch.Tensor) -> dict:
            return {"logits": x @ self.w}

    ev.model = _GradCheckModel()
    m = ev.evaluate()
    # Sanity: metrics still computed
    assert m["accuracy"] == pytest.approx(1.0)
    # Gradients of w were never built — under inference_mode, requires_grad
    # tensors don't accumulate grads. Check param grad is None.
    assert ev.model.w.grad is None


def test_evaluator_resets_metric_state_between_calls() -> None:
    """Calling evaluate() twice on the same Evaluator must not contaminate
    the second pass with first-pass state."""
    logits1 = torch.eye(3).float() * 10
    labels1 = torch.tensor([0, 1, 2])
    ev = _build_eval(logits1, labels1, num_classes=3)
    m1 = ev.evaluate()

    # Repoint the loader to a different (worse-performing) dataset
    bad_logits = torch.tensor([
        [0, 10, 0],   # true=0 pred=1
        [0, 0, 10],   # true=1 pred=2
        [10, 0, 0],   # true=2 pred=0
    ]).float()
    bad_labels = torch.tensor([0, 1, 2])
    ev.val_loader = DataLoader(
        _LogitsDataset(bad_logits, bad_labels), batch_size=2,
        shuffle=False, collate_fn=_collate,
    )
    m2 = ev.evaluate()

    # First pass was perfect; second pass should be 0% accurate (everything wrong).
    assert m1["accuracy"] == pytest.approx(1.0)
    assert m2["accuracy"] == pytest.approx(0.0)


def test_evaluator_handles_scale_id_in_batch() -> None:
    """When the dataloader attaches scale_id (multi-scale path), evaluator
    forwards it to the model. Single-scale path (no scale_id) gets a zeros
    fallback."""
    captured = []

    class _CaptureScaleModel(nn.Module):
        def forward(self, x: torch.Tensor, *, scale_id: torch.Tensor) -> dict:
            captured.append(scale_id.detach().clone())
            return {"logits": x}

    logits = torch.eye(3).float()
    labels = torch.tensor([0, 1, 2])

    class _DSWithScale(_LogitsDataset):
        def __getitem__(self, i: int) -> dict:
            d = super().__getitem__(i)
            d["scale_id"] = torch.tensor(1, dtype=torch.long)
            return d

    def _collate_with_scale(batch: list[dict]) -> dict:
        out = _collate(batch)
        out["scale_id"] = torch.stack([b["scale_id"] for b in batch], dim=0)
        return out

    ds = _DSWithScale(logits, labels)
    loader = DataLoader(ds, batch_size=3, shuffle=False, collate_fn=_collate_with_scale)
    ev = Evaluator(model=_CaptureScaleModel(), val_loader=loader,
                   num_classes=3, device="cpu")
    ev.evaluate()
    assert captured, "model was not called"
    assert captured[0].tolist() == [1, 1, 1]
