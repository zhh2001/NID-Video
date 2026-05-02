"""Re-evaluate a saved checkpoint under the no_cycle strategy and save
combined / fast-only / slow-only metrics to disk for later reference.

The combined number is what an eval-only run of scripts/train.py
(``--eval-only --eval-strategy no_cycle``) would also report; the
fast-only / slow-only splits are not exposed by the regular CLI and
require this script's per-stream partitioning of accumulated predictions
on each sample's ``scale_id``.

Use this whenever a saved checkpoint's reported metric was taken under
``round_robin`` eval (which functioned as unintended test-time
augmentation on the slow stream) and you need an apples-to-apples
noise-free number for cross-run or cross-method comparison.

Usage:
    uv run python scripts/baseline_rerun.py \\
        --resume outputs/run_<ts>/ckpt/best.pt \\
        --shard-pattern-fast "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \\
        --shard-pattern-slow "data/processed/cicids2017_dt1000ms_v2/*/shards/shard-*.tar" \\
        --splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet \\
        --output-dir outputs/run_<ts>/<rerun_dir> \\
        --task-label "<short label written into the README>"
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from torch import nn

from nid_video.data.dataset import build_multi_scale_dataloader, num_classes as label_num_classes
from nid_video.models.videomae_nid import VideoMAESmallForNID
from nid_video.utils import logger, setup_logger
from nid_video.utils.config import load_config


# Class names (collapsed-13). Mirrors LABEL_TO_ID_COLLAPSED from labeling.py.
COLLAPSED_NAMES = [
    "BENIGN", "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye", "FTP-Patator",
    "SSH-Patator", "DoS slowloris", "DoS Slowhttptest", "Bot",
    "Web Attack", "Infiltration", "Heartbleed",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=Path("configs/training_perf.yaml"))
    p.add_argument("--resume", type=Path, required=True,
                   help="Checkpoint produced by Trainer.save_checkpoint.")
    p.add_argument("--shard-pattern-fast", type=str, required=True)
    p.add_argument("--shard-pattern-slow", type=str, required=True)
    p.add_argument("--splits-path", type=Path, required=True)
    p.add_argument("--keep-split", choices=["val", "test"], default="val")
    p.add_argument("--mix-ratio", type=float, default=0.5)
    p.add_argument("--label-mode", choices=["raw15", "collapsed13"], default="collapsed13")
    p.add_argument("--shuffle-buffer", type=int, default=0,
                   help="Default 0 for deterministic eval ordering.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pretrained", default=None,
                   help="HF model id; default None — backbone weights come from --resume.")
    p.add_argument("--model",
                   choices=["videomae_small", "timesformer_small", "c3d_small",
                            "i3d", "r2plus1d_18", "convlstm"],
                   default="videomae_small",
                   help="Backbone selector — must match what the source training "
                        "run used. Default keeps the M3-onward main method; M5.5 "
                        "baselines override per-row.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory to write metrics JSON / CSV / README into.")
    p.add_argument("--source-train-macro-f1", type=float, default=None,
                   help="Reference macro_f1 from the source training run "
                        "(e.g. the best-epoch eval reported in the source "
                        "log); written into the JSON as a delta anchor.")
    p.add_argument("--task-label", type=str, default="noise-free baseline retrofit",
                   help="Short label written into eval_metrics.json and the "
                        "README title to identify which run/checkpoint this "
                        "set of artefacts corresponds to.")
    p.add_argument("--script-name", type=str, default="scripts/baseline_rerun.py",
                   help="Path printed in the README's reproduction block. "
                        "Override only when this script is invoked through a "
                        "thin wrapper that has a different on-disk name.")
    return p.parse_args(argv)


def _build_model(args: argparse.Namespace, n_classes: int) -> nn.Module:
    """Construct the backbone the source run trained. Dispatch mirrors
    scripts/train.py's ``_build_model`` so eval-time loading reads the
    same architectural keys the trainer wrote.
    """
    cfg = load_config(args.config)
    name = args.model
    if name == "videomae_small":
        return VideoMAESmallForNID(
            num_classes=n_classes,
            pretrained=args.pretrained,
            in_channels=cfg.data.num_channels,
            tube_patch=tuple(cfg.model.tube_patch),
            spatial_grid=(cfg.data.num_ip_buckets, cfg.data.num_port_buckets),
            gradient_checkpointing=False,
        )
    if name == "timesformer_small":
        from nid_video.models.timesformer_small_nid import TimeSformerSmallForNID
        return TimeSformerSmallForNID(
            num_classes=n_classes,
            in_channels=cfg.data.num_channels,
            target_image_size=64,
            num_frames=16,
            gradient_checkpointing=False,
        )
    raise SystemExit(
        f"--model {name!r} not yet implemented (Round 2+ adds c3d_small / "
        f"i3d / r2plus1d_18 / convlstm)"
    )


def _load_weights(model: nn.Module, ckpt_path: Path, device: str) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise RuntimeError(f"{ckpt_path}: not a Trainer-format checkpoint (no 'model' field)")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        logger.info(f"missing keys (likely scale-token additions): {list(missing)[:5]}")
    if unexpected:
        logger.warning(f"unexpected keys: {list(unexpected)[:5]}")
    model.to(device)
    model.eval()


def _accumulate_predictions(
    model: nn.Module,
    loader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pass once over the loader, collecting (probs, preds, labels, scale_ids)
    on CPU. Memory budget for ~18,000 samples × 13 classes × float32 ≈ 1 MB."""
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_scale_ids: list[torch.Tensor] = []
    n_seen = 0
    with torch.inference_mode():
        for batch in loader:
            x = batch["tensor"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            scale_id = batch["scale_id"].to(device, non_blocking=True)
            out = model(x, scale_id=scale_id)
            probs = out["logits"].softmax(dim=-1)
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())
            all_scale_ids.append(scale_id.cpu())
            n_seen += int(y.size(0))
            if n_seen % 2048 == 0:
                logger.info(f"accumulated {n_seen} samples")
    probs_cat = torch.cat(all_probs, dim=0)
    preds_cat = probs_cat.argmax(dim=-1)
    labels_cat = torch.cat(all_labels, dim=0)
    scale_ids_cat = torch.cat(all_scale_ids, dim=0)
    return probs_cat, preds_cat, labels_cat, scale_ids_cat


def _compute_metrics(
    probs: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> dict:
    """Run torchmetrics on a (probs, preds, labels) tuple. Mirrors the
    Evaluator class's metric set so combined numbers from this script are
    comparable to per-epoch eval numbers in the training log."""
    if labels.numel() == 0:
        # No samples for this stream — return all-zero metrics rather than
        # raising. Caller decides what to do with a missing stream.
        return {
            "n_samples": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "auroc_macro": 0.0,
            "per_class_f1": np.zeros(n_classes).tolist(),
            "per_class_precision": np.zeros(n_classes).tolist(),
            "per_class_recall": np.zeros(n_classes).tolist(),
            "per_class_auroc": np.zeros(n_classes).tolist(),
            "n_per_class": np.zeros(n_classes, dtype=int).tolist(),
            "confusion_matrix": np.zeros((n_classes, n_classes), dtype=int).tolist(),
        }
    acc = MulticlassAccuracy(num_classes=n_classes, average="micro")
    f1_macro = MulticlassF1Score(num_classes=n_classes, average="macro")
    f1_per = MulticlassF1Score(num_classes=n_classes, average=None)
    p_per = MulticlassPrecision(num_classes=n_classes, average=None)
    r_per = MulticlassRecall(num_classes=n_classes, average=None)
    auroc_macro = MulticlassAUROC(num_classes=n_classes, average="macro")
    auroc_per = MulticlassAUROC(num_classes=n_classes, average=None)
    cm = MulticlassConfusionMatrix(num_classes=n_classes)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        acc.update(preds, labels)
        f1_macro.update(preds, labels)
        f1_per.update(preds, labels)
        p_per.update(preds, labels)
        r_per.update(preds, labels)
        cm.update(preds, labels)
        auroc_macro.update(probs, labels)
        auroc_per.update(probs, labels)
        return {
            "n_samples": int(labels.size(0)),
            "accuracy": float(acc.compute().item()),
            "macro_f1": float(f1_macro.compute().item()),
            "auroc_macro": float(auroc_macro.compute().item()),
            "per_class_f1": f1_per.compute().cpu().numpy().tolist(),
            "per_class_precision": p_per.compute().cpu().numpy().tolist(),
            "per_class_recall": r_per.compute().cpu().numpy().tolist(),
            "per_class_auroc": auroc_per.compute().cpu().numpy().tolist(),
            "n_per_class": [int((labels == c).sum().item()) for c in range(n_classes)],
            "confusion_matrix": cm.compute().cpu().numpy().astype(int).tolist(),
        }


def _git_head_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    n_classes: int,
    combined: dict,
    fast_only: dict,
    slow_only: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "task_label": args.task_label,
        "script_name": args.script_name,
        "output_dir": str(output_dir),
        "ckpt_source": str(args.resume),
        "config": str(args.config),
        "shard_pattern_fast": args.shard_pattern_fast,
        "shard_pattern_slow": args.shard_pattern_slow,
        "splits_path": str(args.splits_path),
        "keep_split": args.keep_split,
        "mix_ratio": args.mix_ratio,
        "eval_strategy_used": "no_cycle",
        "label_mode": args.label_mode,
        "n_classes": n_classes,
        "class_names": COLLAPSED_NAMES if n_classes == 13 else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "commit_hash": _git_head_short(),
        "source_train_macro_f1": args.source_train_macro_f1,
        "val_sample_count_fast": fast_only["n_samples"],
        "val_sample_count_slow": slow_only["n_samples"],
        "val_sample_count_total": combined["n_samples"],
        "combined_metrics": combined,
        "fast_only_metrics": fast_only,
        "slow_only_metrics": slow_only,
    }
    if args.source_train_macro_f1 is not None:
        payload["combined_macro_f1_delta"] = round(
            combined["macro_f1"] - args.source_train_macro_f1, 5,
        )

    (output_dir / "eval_metrics.json").write_text(
        json.dumps(payload, indent=2)
    )
    (output_dir / "confusion_matrix.json").write_text(
        json.dumps({
            "schema_version": 1,
            "scope": "combined (no_cycle, fast+slow merged)",
            "row": "true label",
            "col": "predicted label",
            "n_classes": n_classes,
            "class_names": COLLAPSED_NAMES if n_classes == 13 else None,
            "matrix": combined["confusion_matrix"],
        }, indent=2)
    )
    csv_path = output_dir / "per_class_table.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "label_id", "label_name", "support",
            "precision", "recall", "f1", "auroc",
        ])
        for c in range(n_classes):
            writer.writerow([
                c,
                COLLAPSED_NAMES[c] if n_classes == 13 else f"class_{c}",
                combined["n_per_class"][c],
                round(combined["per_class_precision"][c], 4),
                round(combined["per_class_recall"][c], 4),
                round(combined["per_class_f1"][c], 4),
                round(combined["per_class_auroc"][c], 4),
            ])

    readme = output_dir / "README.md"
    readme.write_text(_render_readme(args, payload))


def _render_readme(args: argparse.Namespace, payload: dict) -> str:
    return f"""# {payload['task_label']} (no_cycle strategy)

This directory holds a one-shot re-evaluation of a saved checkpoint under
the ``no_cycle`` epoch-end strategy. The motivation: the eval numbers
recorded during the source training run used ``round_robin``, which
cycles the slow stream and introduces ±0.04-magnitude per-epoch metric
variance. ``no_cycle`` drains both streams exactly once with no
duplicates and yields a count that does not depend on ``mix_ratio`` —
the property needed for stable cross-run baseline numbers.

## Source

| | |
|---|---|
| Task label | {payload['task_label']} |
| Checkpoint | `{payload['ckpt_source']}` |
| Source training macro_f1 (round_robin eval) | {payload['source_train_macro_f1']!r} |
| Config | `{payload['config']}` |
| Splits | `{payload['splits_path']}` |
| Split evaluated | `{payload['keep_split']}` |
| Eval strategy | `{payload['eval_strategy_used']}` |
| Mix ratio | {payload['mix_ratio']} |
| Label mode | `{payload['label_mode']}` |
| Repo commit | `{payload['commit_hash']}` |
| Timestamp (UTC) | `{payload['timestamp']}` |

## Numbers at a glance (combined: fast + slow merged)

| Metric | Value |
|---|---:|
| val_sample_count_fast | {payload['val_sample_count_fast']:,} |
| val_sample_count_slow | {payload['val_sample_count_slow']:,} |
| val_sample_count_total | {payload['val_sample_count_total']:,} |
| accuracy | {payload['combined_metrics']['accuracy']:.4f} |
| **macro_f1** | **{payload['combined_metrics']['macro_f1']:.4f}** |
| auroc_macro | {payload['combined_metrics']['auroc_macro']:.4f} |
| Δ vs source train macro_f1 | {payload.get('combined_macro_f1_delta', 'n/a')} |

## Per-stream comparison

| | combined | fast-only | slow-only |
|---|---:|---:|---:|
| n_samples | {payload['val_sample_count_total']:,} | {payload['val_sample_count_fast']:,} | {payload['val_sample_count_slow']:,} |
| macro_f1 | {payload['combined_metrics']['macro_f1']:.4f} | {payload['fast_only_metrics']['macro_f1']:.4f} | {payload['slow_only_metrics']['macro_f1']:.4f} |
| accuracy | {payload['combined_metrics']['accuracy']:.4f} | {payload['fast_only_metrics']['accuracy']:.4f} | {payload['slow_only_metrics']['accuracy']:.4f} |
| auroc_macro | {payload['combined_metrics']['auroc_macro']:.4f} | {payload['fast_only_metrics']['auroc_macro']:.4f} | {payload['slow_only_metrics']['auroc_macro']:.4f} |

The fast-only and slow-only rows are computed by partitioning the
accumulated `(probs, preds, labels)` tuples on each sample's
`scale_id` value and recomputing torchmetrics on each subset. They are
not exposed by `scripts/train.py --eval-only`.

## Files

- `eval_metrics.json` — full metric payload (combined + per-stream),
  schema_version=1, machine-readable.
- `confusion_matrix.json` — combined confusion matrix; row=true,
  col=pred; raw counts.
- `per_class_table.csv` — combined per-class table (label_id /
  label_name / support / precision / recall / f1 / auroc) suitable for
  direct paste into a paper table.

## Reproducing

The combined number alone (no per-stream breakdown) can be reproduced
via the regular CLI:

```bash
uv run python scripts/train.py --eval-only \\
    --config {payload['config']} \\
    --resume {payload['ckpt_source']} \\
    --shard-pattern-fast "{payload['shard_pattern_fast']}" \\
    --shard-pattern-slow "{payload['shard_pattern_slow']}" \\
    --splits-path {payload['splits_path']} \\
    --keep-split {payload['keep_split']} \\
    --eval-strategy no_cycle
```

The full set of artefacts (this directory) is reproduced by re-running
this script:

```bash
uv run python {payload['script_name']} \\
    --resume {payload['ckpt_source']} \\
    --shard-pattern-fast "{payload['shard_pattern_fast']}" \\
    --shard-pattern-slow "{payload['shard_pattern_slow']}" \\
    --splits-path {payload['splits_path']} \\
    --output-dir {payload['output_dir']} \\
    --task-label "{payload['task_label']}"
```

## Reading the per-stream numbers

The fast stream (Δt=100ms) targets short-burst attacks (DDoS/SYN-flood
spikes); the slow stream (Δt=1s) targets longer-tempo behaviours
(slowloris, Heartbleed). Different per-stream macro-F1 numbers are
expected and informative: a class with high fast-only F1 but low
slow-only F1 has bursty-only signal, and vice versa. For
single-resolution baseline comparisons (M5.5+), the fast-only number
is the fair reference because the baselines' input granularity matches
the fast stream.
"""


def main(argv: Sequence[str] | None = None) -> int:
    setup_logger()
    args = parse_args(argv)

    if not args.resume.is_file():
        logger.error(f"--resume not found: {args.resume}")
        return 2
    if not args.splits_path.is_file():
        logger.error(f"--splits-path not found: {args.splits_path}")
        return 2

    n_classes = label_num_classes(args.label_mode)
    model = _build_model(args, n_classes)
    _load_weights(model, args.resume, args.device)

    loader = build_multi_scale_dataloader(
        fast_pattern=args.shard_pattern_fast,
        slow_pattern=args.shard_pattern_slow,
        batch_size=32,
        num_workers=args.num_workers,
        label_mode=args.label_mode,
        shuffle_buffer=args.shuffle_buffer,
        mix_ratio=args.mix_ratio,
        splits_path=args.splits_path,
        keep_split=args.keep_split,
        epoch_end_strategy="no_cycle",
        pin_memory=(args.device == "cuda"),
    )

    logger.info("accumulating predictions ...")
    probs, preds, labels, scale_ids = _accumulate_predictions(model, loader, args.device)
    logger.info(
        f"accumulated {labels.numel()} samples "
        f"(fast={int((scale_ids == 0).sum().item())}, "
        f"slow={int((scale_ids == 1).sum().item())})"
    )

    fast_mask = scale_ids == 0
    slow_mask = scale_ids == 1

    combined = _compute_metrics(probs, preds, labels, n_classes)
    fast_only = _compute_metrics(
        probs[fast_mask], preds[fast_mask], labels[fast_mask], n_classes,
    )
    slow_only = _compute_metrics(
        probs[slow_mask], preds[slow_mask], labels[slow_mask], n_classes,
    )

    logger.info(
        f"combined macro_f1={combined['macro_f1']:.4f} "
        f"(n={combined['n_samples']}); "
        f"fast macro_f1={fast_only['macro_f1']:.4f} "
        f"(n={fast_only['n_samples']}); "
        f"slow macro_f1={slow_only['macro_f1']:.4f} "
        f"(n={slow_only['n_samples']})"
    )

    _write_outputs(args.output_dir, args, n_classes, combined, fast_only, slow_only)
    logger.info(f"results written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
