"""Evaluator: compute multi-class NID metrics on a validation set.

Designed for the (T=16, C=6, H=32, W=64) NID tensor + the 13-class collapsed
or 15-class raw label scheme. Idea.md M4 task 4.5.

Important behavior pinning (torchmetrics 1.9.0):

  * ``MulticlassF1Score(average="macro")`` **excludes** classes with zero
    true samples in the batch from the average. So with 13 classes but only
    10 appearing in val, macro_f1 averages over 10 — silently more lenient.
    Pinned by ``test_macro_f1_excludes_zero_sample_classes``; if a future
    torchmetrics version changes this, that test fails loudly. We expose
    ``per_class_f1`` (length-num_classes) so the caller can compute their
    own macro if needed.

  * ``MulticlassAUROC(average="macro")`` **includes** classes with zero
    positives, scoring them 0, which drags macro down. Asymmetric vs F1.
    Pinned by ``test_macro_auroc_includes_zero_positive_classes_as_zero``.

  * For both, per-class F1/precision/recall/AUROC return 0.0 when the class
    has no true samples; AUROC additionally emits a UserWarning.

We surface ``n_per_class`` in the result dict so the caller can mark such
classes ``N/A`` in pretty-print rather than reading 0.0 as "model is bad
at it" — which is the real failure mode this exposes.

Idea.md §3.5 + §5.4.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from nid_video.utils import logger


class Evaluator:
    """Run inference on a val/test loader and collect multi-class metrics.

    Args:
        model: any module whose ``forward(x, *, scale_id)`` returns
            ``{"logits": (B, num_classes), ...}``.
        val_loader: DataLoader yielding batches with ``tensor`` / ``label``
            (and optionally ``scale_id``) keys.
        num_classes: total label count (13 for collapsed, 15 for raw15).
        device: ``"cuda"`` or ``"cpu"``. Metrics live on this device.

    The returned ``evaluate()`` dict carries:

    +----------------------+-----------------------------------------+
    | key                  | type / shape                            |
    +======================+=========================================+
    | ``accuracy``         | float (micro)                           |
    | ``macro_f1``         | float (excludes zero-sample classes)    |
    | ``per_class_f1``     | np.ndarray (num_classes,)               |
    | ``per_class_precision``| np.ndarray (num_classes,)             |
    | ``per_class_recall`` | np.ndarray (num_classes,)               |
    | ``auroc_macro``      | float (includes zero-positive as 0)     |
    | ``per_class_auroc``  | np.ndarray (num_classes,)               |
    | ``confusion_matrix`` | np.ndarray (num_classes, num_classes)   |
    |                      |   row=true, col=pred                    |
    | ``n_samples``        | int                                     |
    | ``n_per_class``      | np.ndarray (num_classes,) int           |
    +----------------------+-----------------------------------------+
    """

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        num_classes: int,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.val_loader = val_loader
        self.num_classes = int(num_classes)
        self.device = device

        self._acc = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
        self._f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
        self._f1_per = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
        self._p_per = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
        self._r_per = MulticlassRecall(num_classes=num_classes, average=None).to(device)
        self._auroc_macro = MulticlassAUROC(num_classes=num_classes, average="macro").to(device)
        self._auroc_per = MulticlassAUROC(num_classes=num_classes, average=None).to(device)
        self._cm = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    def _all_metrics(self) -> list:
        return [self._acc, self._f1_macro, self._f1_per, self._p_per, self._r_per,
                self._auroc_macro, self._auroc_per, self._cm]

    @torch.inference_mode()
    def evaluate(self) -> dict:
        """Run a full pass over ``val_loader`` and return the metrics dict.

        Uses ``torch.inference_mode()`` (PyTorch 1.9+ default for inference —
        stricter than ``no_grad`` because it also disables view tracking,
        giving slightly faster math).
        """
        self.model.eval()
        for m in self._all_metrics():
            m.reset()

        n_per_class = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        n_total = 0

        for batch in self.val_loader:
            x = batch["tensor"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)
            if "scale_id" in batch:
                scale_id = batch["scale_id"].to(self.device, non_blocking=True)
            else:
                scale_id = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

            out = self.model(x, scale_id=scale_id)
            logits = out["logits"]
            preds = logits.argmax(dim=-1)
            probs = logits.softmax(dim=-1)

            # Class-prediction-based metrics
            self._acc.update(preds, y)
            self._f1_macro.update(preds, y)
            self._f1_per.update(preds, y)
            self._p_per.update(preds, y)
            self._r_per.update(preds, y)
            self._cm.update(preds, y)
            # Probability-based metrics
            with warnings.catch_warnings():
                # torchmetrics emits "No positive samples in targets" for any
                # class with zero positives in the batch — keep this visible
                # via per_class_auroc + n_per_class but suppress the spam.
                warnings.simplefilter("ignore", category=UserWarning)
                self._auroc_macro.update(probs, y)
                self._auroc_per.update(probs, y)

            for c in range(self.num_classes):
                n_per_class[c] += (y == c).sum()
            n_total += int(y.size(0))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            acc = float(self._acc.compute().item())
            macro_f1 = float(self._f1_macro.compute().item())
            per_class_f1 = self._f1_per.compute().cpu().numpy()
            per_class_p = self._p_per.compute().cpu().numpy()
            per_class_r = self._r_per.compute().cpu().numpy()
            auroc_macro = float(self._auroc_macro.compute().item())
            per_class_auroc = self._auroc_per.compute().cpu().numpy()
            cm = self._cm.compute().cpu().numpy()

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
            "per_class_precision": per_class_p,
            "per_class_recall": per_class_r,
            "auroc_macro": auroc_macro,
            "per_class_auroc": per_class_auroc,
            "confusion_matrix": cm,
            "n_samples": n_total,
            "n_per_class": n_per_class.cpu().numpy(),
        }

    def pretty_print(
        self,
        metrics: dict,
        class_names: Sequence[str] | None = None,
    ) -> None:
        """Print the metrics dict to loguru INFO in an aligned, debug-friendly form.

        Format:

          === Eval (n=N) ===
            accuracy     : <0.0000>
            macro_f1     : <0.0000>   (excludes zero-sample classes per torchmetrics)
            auroc_macro  : <0.0000>   (includes zero-sample as 0 per torchmetrics)
            per-class:
              class                       n      F1    Prec     Rec   AUROC
              BENIGN                   1000  0.9000  0.8500  0.9500  0.9200
              ...
            confusion_matrix (row=true, col=pred):
                       0    1    2 ...
                0:   950   30   10 ...
                ...
        """
        if class_names is None:
            class_names = [f"class_{i}" for i in range(self.num_classes)]
        if len(class_names) != self.num_classes:
            raise ValueError(
                f"class_names length {len(class_names)} != num_classes {self.num_classes}"
            )

        n_per_class = metrics["n_per_class"]
        logger.info(f"=== Eval (n={metrics['n_samples']}) ===")
        logger.info(f"  accuracy     : {metrics['accuracy']:.4f}")
        logger.info(f"  macro_f1     : {metrics['macro_f1']:.4f}   (excludes zero-sample classes)")
        logger.info(f"  auroc_macro  : {metrics['auroc_macro']:.4f}   (zero-sample classes scored 0)")
        logger.info("  per-class:")
        logger.info(
            f"    {'class':30s}{'n':>6s}  {'F1':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'AUROC':>6s}"
        )
        for c in range(self.num_classes):
            name = class_names[c]
            marker = "  N/A (zero val samples)" if n_per_class[c] == 0 else ""
            logger.info(
                f"    {name[:30]:30s}{int(n_per_class[c]):>6d}  "
                f"{metrics['per_class_f1'][c]:.4f}  "
                f"{metrics['per_class_precision'][c]:.4f}  "
                f"{metrics['per_class_recall'][c]:.4f}  "
                f"{metrics['per_class_auroc'][c]:.4f}{marker}"
            )

        cm = metrics["confusion_matrix"]
        logger.info("  confusion_matrix (row=true, col=pred):")
        header = "      pred:" + "".join(f" {i:>5d}" for i in range(self.num_classes))
        logger.info(header)
        for i, row in enumerate(cm):
            line = f"      {i:>3d}: " + "".join(f" {int(v):>5d}" for v in row)
            logger.info(line)
