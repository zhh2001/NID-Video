"""Metrics writer: forward-only training instrumentation for M5.10+.

M5 training runs persisted only ``best.pt`` + ``eval_metrics.json`` (the
final-epoch eval result for the best checkpoint). Per-step training
loss / lr / grad_norm and per-epoch eval trajectory were emitted to
loguru only and lost when the log was rotated. M5.10 ablation
experiments and the paper-figure pipeline both need the trajectory
data; this module is the forward-only writer that captures it.

For the M5-era checkpoint set, the same per-epoch shape is reconstructed
post-hoc by ``scripts/baseline_rerun.py --output-mode per_epoch_metrics``
(see Part 2 of the M5.10 prep plan); the file format is shared so figure
code reads forward-instrumented runs and retrofitted runs identically.

Output layout under ``<run_dir>/metrics/``:

  per_step.jsonl
      One JSON object per grad step. Default schema:
        {"step": int, "epoch": int, "loss": float, "lr": float,
         "wall_time_s": float}
      With ``collect_grad_norm=True`` an additional ``"grad_norm": float``
      field is appended on every step where a grad_norm is supplied.

  per_epoch.json
      Single JSON document. Schema:
        {"run_id": str, "config": dict, "epochs": [<epoch_record>, ...]}
      Each ``<epoch_record>`` is:
        {"epoch": int, "grad_steps": int, "wall_time_s": float,
         "metrics": {"combined": {...}, "fast": {...}, "slow": {...}}}
      Forward instrumentation only fills ``combined`` (the trainer's
      evaluator does not split predictions by ``scale_id``); the
      retrofit path fills ``fast`` and ``slow`` as well by partitioning
      accumulated predictions on ``scale_id``. Either shape is valid;
      figure code reads ``.get("fast")``/``.get("slow")`` and skips
      missing splits.

  confusion_per_epoch.npz
      ``np.savez`` with keys ``epoch_0``, ``epoch_1``, .... Each value
      is an ``(num_classes, num_classes)`` ``int64`` matrix (row=true,
      col=pred).

The writer is not multi-process safe; the trainer is single-process by
construction in this project and the single per_step.jsonl file is
opened in line-buffered append mode so a SIGKILL still leaves a
parseable trailing-line truncated JSONL on disk (jq / pandas can drop
the bad final line).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class MetricsWriter:
    """Per-run training-metrics writer.

    Args:
      run_dir: target run directory; ``<run_dir>/metrics/`` is created
        on construction. Existing files in that directory are
        overwritten — re-running training in the same run_dir starts
        a fresh trajectory.
      config: optional snapshot of the training config to record in
        ``per_epoch.json`` for forensic clarity. ``None`` writes an
        empty dict.
      collect_grad_norm: when True, ``log_step`` calls passing a
        non-None ``grad_norm`` write the value into per_step.jsonl;
        when False (default), the field is omitted from every row
        regardless. The flag does not affect grad_norm computation
        itself — that lives in the trainer (``clip_grad_norm_``
        already returns the value for free).
    """

    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any] | None = None,
        collect_grad_norm: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.metrics_dir = self.run_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.per_step_path = self.metrics_dir / "per_step.jsonl"
        self.per_epoch_path = self.metrics_dir / "per_epoch.json"
        self.confusion_path = self.metrics_dir / "confusion_per_epoch.npz"
        self.collect_grad_norm = bool(collect_grad_norm)

        self._run_id = self.run_dir.name
        self._config = dict(config) if config is not None else {}
        self._epoch_records: list[dict[str, Any]] = []
        self._confusion_dict: dict[str, np.ndarray] = {}

        # Truncate per_step.jsonl at construction so re-runs in the same
        # run_dir don't append onto a stale trajectory; subsequent
        # log_step calls reopen in append mode.
        self.per_step_path.write_text("")

        self._closed = False

    # ----- per-step ----------------------------------------------------

    def log_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        lr: float,
        wall_time_s: float,
        grad_norm: float | None = None,
    ) -> None:
        """Append one row to per_step.jsonl.

        Default schema: ``{step, epoch, loss, lr, wall_time_s}``. If
        ``self.collect_grad_norm`` is True AND ``grad_norm is not None``,
        the row also carries ``grad_norm``. Other ``None``/missing
        cases skip the field.
        """
        if self._closed:
            raise RuntimeError("MetricsWriter is closed; call before finalize()")
        row: dict[str, Any] = {
            "step": int(step),
            "epoch": int(epoch),
            "loss": float(loss),
            "lr": float(lr),
            "wall_time_s": float(wall_time_s),
        }
        if self.collect_grad_norm and grad_norm is not None:
            row["grad_norm"] = float(grad_norm)
        with self.per_step_path.open("a") as f:
            f.write(json.dumps(row, separators=(",", ":")))
            f.write("\n")

    # ----- per-epoch ---------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        grad_steps: int,
        wall_time_s: float,
        metrics: dict[str, Any],
        confusion: np.ndarray,
    ) -> None:
        """Append one epoch record + flush per_epoch.json + confusion npz.

        Args:
          metrics: dict shaped per per_epoch.json schema. The
            ``combined`` key is required; ``fast`` and ``slow`` are
            optional (forward instrumentation skips them, retrofit
            fills them).
          confusion: ``(num_classes, num_classes)`` int matrix.
        """
        if self._closed:
            raise RuntimeError("MetricsWriter is closed; call before finalize()")
        if "combined" not in metrics:
            raise ValueError(
                f"metrics dict must include 'combined' key; got keys={sorted(metrics.keys())}"
            )
        record = {
            "epoch": int(epoch),
            "grad_steps": int(grad_steps),
            "wall_time_s": float(wall_time_s),
            "metrics": _jsonify_metrics(metrics),
        }
        self._epoch_records.append(record)
        self._confusion_dict[f"epoch_{int(epoch)}"] = np.asarray(
            confusion, dtype=np.int64
        )
        self._flush_per_epoch()
        self._flush_confusion()

    # ----- finalize ----------------------------------------------------

    def finalize(self) -> None:
        """Idempotent flush + close. Safe to call multiple times."""
        if self._closed:
            return
        # per_step.jsonl is already on disk (line-buffered append mode);
        # nothing to flush there.
        self._flush_per_epoch()
        self._flush_confusion()
        self._closed = True

    # ----- internals ---------------------------------------------------

    def _flush_per_epoch(self) -> None:
        payload = {
            "run_id": self._run_id,
            "config": self._config,
            "epochs": self._epoch_records,
        }
        # Two-step write to avoid a half-written file if the process is
        # killed mid-write (the trainer is the only writer, so a stale
        # .tmp is safe to overwrite next run).
        tmp = self.per_epoch_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.per_epoch_path)

    def _flush_confusion(self) -> None:
        if not self._confusion_dict:
            return
        # Write to ``<base>.tmp.npz`` then rename. The ``.npz`` tail is
        # required because ``np.savez`` silently appends ``.npz`` when
        # the supplied path lacks it — so we keep ``.npz`` and tuck the
        # ``.tmp`` marker before it.
        tmp = self.confusion_path.with_name(
            self.confusion_path.stem + ".tmp.npz"
        )
        np.savez(tmp, **self._confusion_dict)
        tmp.replace(self.confusion_path)


# ----- helpers --------------------------------------------------------


def _jsonify_metrics(obj: Any) -> Any:
    """Recursively convert numpy scalar / ndarray values to JSON-safe
    native types. The evaluator returns a mix of float, int, np.float64,
    and np.ndarray; per_epoch.json must be plain JSON.
    """
    if isinstance(obj, dict):
        return {k: _jsonify_metrics(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify_metrics(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonify_metrics(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
