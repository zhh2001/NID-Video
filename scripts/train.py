"""M4 training / eval entry point.

Three usage scenarios:

1. Real training (multi-scale, with splits + eval):

    uv run python scripts/train.py \\
        --shard-pattern-fast "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \\
        --shard-pattern-slow "data/processed/cicids2017_dt1000ms_v2/*/shards/shard-*.tar" \\
        --splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet

   Defaults to ``configs/training_perf.yaml`` (B=32 / accum=1 / workers=2).
   Builds a 50/50 fast/slow MultiScaleNidDataset, runs Trainer with
   per-epoch eval on the val split, and copies the best-macro-F1 ckpt to
   ``run_<ts>/ckpt/best.pt``.

2. Smoke test (single-scale, no splits, sanity baseline):

    uv run python scripts/train.py \\
        --config configs/base.yaml \\
        --shard-pattern "data/processed/.../shard-*.tar" \\
        --debug --max-steps 20

   M3 legacy path, kept for CI / quick verification. Single-scale dataset,
   no eval, no best-ckpt — just a "does it run" check.

3. Eval-only on a saved checkpoint:

    uv run python scripts/train.py \\
        --shard-pattern-fast ... --shard-pattern-slow ... \\
        --splits-path ... \\
        --eval-only --resume outputs/run_<ts>/ckpt/best.pt \\
        [--keep-split test]

   Loads model weights from --resume, builds Evaluator on the val (default)
   or test split, prints metrics, and exits without touching training state.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import torch

from nid_video.data.dataset import (
    build_dataloader,
    build_multi_scale_dataloader,
    num_classes as label_num_classes,
)
from nid_video.models.videomae_nid import VideoMAESmallForNID
from nid_video.trainer.evaluator import Evaluator
from nid_video.trainer.trainer import Trainer
from nid_video.utils import logger, setup_logger
from nid_video.utils.config import load_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # ----- config -----
    # M3 deferred decision: M4 default is the throughput-optimized config.
    # CI / smoke runs explicitly pass --config configs/base.yaml.
    p.add_argument("--config", type=Path,
                   default=Path("configs/training_perf.yaml"),
                   help="OmegaConf YAML. Default: configs/training_perf.yaml "
                        "(B=32 / accum=1 / workers=2). Pass configs/base.yaml "
                        "for CI / smoke (B=2 / accum=16).")

    # ----- data: single-scale OR multi-scale (mutually exclusive) -----
    g = p.add_argument_group(
        "data sources",
        description="Pass EITHER --shard-pattern (single-scale, M3 legacy) "
                    "OR (--shard-pattern-fast + --shard-pattern-slow) (multi-scale).",
    )
    g.add_argument("--shard-pattern", type=str, default=None,
                   help="Glob to single-scale shards. M3 legacy path.")
    g.add_argument("--shard-pattern-fast", type=str, default=None,
                   help="Glob to fast-scale (Δt=100ms) shards. Pair with --shard-pattern-slow.")
    g.add_argument("--shard-pattern-slow", type=str, default=None,
                   help="Glob to slow-scale (Δt=1s) shards. Pair with --shard-pattern-fast.")
    g.add_argument("--mix-ratio", type=float, default=0.5,
                   help="P(fast sample) for MultiScaleNidDataset. Default 0.5.")

    # ----- splits + eval -----
    p.add_argument("--splits-path", type=Path, default=None,
                   help="Path to splits.parquet (from scripts/run_split.py). "
                        "Without it, no train/val/test partition and no eval.")
    p.add_argument("--keep-split", choices=["val", "test"], default="val",
                   help="For --eval-only, which split to evaluate. Ignored in training mode "
                        "(training always uses 'train' split for train_loader and 'val' for evaluator).")

    # ----- mode -----
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; load model from --resume, run Evaluator, exit. "
                        "Requires --resume.")

    # ----- training hyperparams CLI overrides -----
    p.add_argument("--num-epochs", type=int, default=None,
                   help="override training.num_epochs from config")
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap micro-batch count (debug)")
    p.add_argument("--warmup-steps", type=int, default=500,
                   help="LR scheduler linear warmup steps. Default 500.")
    p.add_argument("--total-steps", type=int, default=None,
                   help="LR scheduler total grad steps. If unset and --splits-path "
                        "is provided, computed from train-split sample count. "
                        "Otherwise Trainer falls back to a heuristic with WARNING.")
    p.add_argument("--track-best", default="macro_f1",
                   help="Metric to maximize for best.pt selection. Default 'macro_f1'.")

    # ----- existing -----
    p.add_argument("--label-mode", choices=["raw15", "collapsed13"], default="collapsed13")
    p.add_argument("--num-workers", type=int, default=None,
                   help="override training.num_workers from config")
    p.add_argument("--shuffle-buffer", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--debug", action="store_true",
                   help="FP32 + vanilla AdamW + no grad checkpointing (sanity run)")
    p.add_argument("--pretrained", default="MCG-NJU/videomae-small-finetuned-kinetics",
                   help="HF model id, or '' / 'none' to skip pretraining")
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from a .pt checkpoint (or load weights when --eval-only).")
    p.add_argument("--export-safetensors", type=Path, default=None,
                   help="After training, export final model weights (only) to "
                        "this safetensors path for deployment.")
    return p.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    """Mutual-exclusivity + required-set checks. Raises SystemExit on invalid."""
    single = args.shard_pattern is not None
    multi_fast = args.shard_pattern_fast is not None
    multi_slow = args.shard_pattern_slow is not None
    multi = multi_fast or multi_slow

    if single and multi:
        raise SystemExit(
            "--shard-pattern is mutually exclusive with "
            "--shard-pattern-fast / --shard-pattern-slow"
        )
    if multi and not (multi_fast and multi_slow):
        raise SystemExit(
            "multi-scale requires BOTH --shard-pattern-fast AND --shard-pattern-slow"
        )
    if not single and not multi:
        raise SystemExit(
            "must set --shard-pattern OR (--shard-pattern-fast + --shard-pattern-slow)"
        )
    if args.eval_only and args.resume is None:
        raise SystemExit("--eval-only requires --resume <path>")
    if not 0.0 <= args.mix_ratio <= 1.0:
        raise SystemExit(f"--mix-ratio must be in [0, 1], got {args.mix_ratio}")


def _override_for_debug(training_cfg):
    return training_cfg.model_copy(update={
        "precision": "fp32",
        "optimizer": "adamw",
        "gradient_checkpointing": False,
    })


def _compute_total_steps(args: argparse.Namespace, training_cfg) -> int | None:
    """Compute LR scheduler total_steps. Priority:
       1. --total-steps explicit
       2. ceil(train_n / (B * accum)) * num_epochs from splits.parquet
       3. None → Trainer's own num_epochs * 1000 heuristic + WARNING
    """
    if args.total_steps is not None:
        return args.total_steps
    if args.splits_path is not None and args.splits_path.is_file():
        from nid_video.data.split import load_splits
        splits = load_splits(args.splits_path)
        train_n = sum(1 for v in splits.values() if v == "train")
        n_epochs = args.num_epochs if args.num_epochs is not None else training_cfg.num_epochs
        steps_per_epoch = max(
            1,
            math.ceil(train_n / max(1, training_cfg.batch_size * training_cfg.grad_accumulation))
        )
        total = steps_per_epoch * n_epochs
        logger.info(
            f"total_steps = {total} (computed: train_n={train_n}, "
            f"batch={training_cfg.batch_size}, accum={training_cfg.grad_accumulation}, "
            f"steps/epoch={steps_per_epoch}, epochs={n_epochs})"
        )
        return total
    return None


def _build_loaders(
    args: argparse.Namespace, training_cfg, *, for_eval_only: bool,
):
    """Return (train_loader, val_loader). For --eval-only, train is None and
    val carries the --keep-split selection."""
    is_multi = args.shard_pattern_fast is not None
    num_workers = args.num_workers if args.num_workers is not None else training_cfg.num_workers
    common = dict(
        batch_size=training_cfg.batch_size,
        num_workers=num_workers,
        label_mode=args.label_mode,
        shuffle_buffer=args.shuffle_buffer,
        pin_memory=(args.device == "cuda"),
    )

    if for_eval_only:
        # Pure eval — build only val (or test) loader on requested split.
        if is_multi:
            val_loader = build_multi_scale_dataloader(
                fast_pattern=args.shard_pattern_fast,
                slow_pattern=args.shard_pattern_slow,
                splits_path=args.splits_path,
                keep_split=args.keep_split,
                mix_ratio=args.mix_ratio,
                **common,
            )
        else:
            val_loader = build_dataloader(
                args.shard_pattern,
                splits_path=args.splits_path,
                keep_split=args.keep_split,
                **common,
            )
        return None, val_loader

    # Training mode
    if is_multi:
        train_loader = build_multi_scale_dataloader(
            fast_pattern=args.shard_pattern_fast,
            slow_pattern=args.shard_pattern_slow,
            splits_path=args.splits_path,
            keep_split="train" if args.splits_path else None,
            mix_ratio=args.mix_ratio,
            **common,
        )
        val_loader = None
        if args.splits_path is not None:
            val_loader = build_multi_scale_dataloader(
                fast_pattern=args.shard_pattern_fast,
                slow_pattern=args.shard_pattern_slow,
                splits_path=args.splits_path,
                keep_split="val",
                mix_ratio=args.mix_ratio,
                **common,
            )
    else:
        train_loader = build_dataloader(
            args.shard_pattern,
            splits_path=args.splits_path,
            keep_split="train" if args.splits_path else None,
            **common,
        )
        val_loader = None
        if args.splits_path is not None:
            val_loader = build_dataloader(
                args.shard_pattern,
                splits_path=args.splits_path,
                keep_split="val",
                **common,
            )
    return train_loader, val_loader


def _run_eval_only(args, model, val_loader, n_classes) -> int:
    """Pure eval path: load model from --resume, run Evaluator on val/test, exit."""
    if not args.resume.is_file():
        logger.error(f"--resume path does not exist: {args.resume}")
        return 2
    # Use the same forward-compat policy as Trainer.load_checkpoint, but only
    # touch the model state — no optimizer/scheduler/scaler/RNG to restore here.
    ckpt = torch.load(str(args.resume), map_location=args.device, weights_only=False)
    if "model" not in ckpt:
        logger.error(f"{args.resume}: no 'model' field — not a Trainer-format ckpt")
        return 2
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        logger.info(f"eval-only: missing keys {list(missing)[:5]}")
    if unexpected:
        logger.warning(f"eval-only: unexpected keys {list(unexpected)[:5]}")
    model.to(args.device)

    evaluator = Evaluator(
        model=model, val_loader=val_loader,
        num_classes=n_classes, device=args.device,
    )
    metrics = evaluator.evaluate()
    evaluator.pretty_print(metrics)
    logger.info(f"eval-only complete: split={args.keep_split!r}, "
                f"resume={args.resume.name}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    setup_logger()
    args = parse_args(argv)
    _validate_args(args)

    cfg = load_config(args.config)
    training = cfg.training
    if args.debug:
        training = _override_for_debug(training)
        logger.warning("--debug: FP32 + AdamW + no grad checkpointing")

    pretrained = None if args.pretrained.lower() in ("", "none") else args.pretrained
    n_classes = label_num_classes(args.label_mode)

    model = VideoMAESmallForNID(
        num_classes=n_classes,
        pretrained=pretrained,
        in_channels=cfg.data.num_channels,
        tube_patch=tuple(cfg.model.tube_patch),
        spatial_grid=(cfg.data.num_ip_buckets, cfg.data.num_port_buckets),
        gradient_checkpointing=training.gradient_checkpointing,
    )

    train_loader, val_loader = _build_loaders(args, training, for_eval_only=args.eval_only)

    if args.eval_only:
        return _run_eval_only(args, model, val_loader, n_classes)

    total_steps = _compute_total_steps(args, training)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=training,
        device=args.device,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        val_loader=val_loader,
        num_classes=n_classes,
        track_best_metric=args.track_best,
    )
    if args.resume is not None:
        if not args.resume.is_file():
            logger.error(f"--resume path does not exist: {args.resume}")
            return 2
        trainer.load_checkpoint(args.resume)

    result = trainer.train(num_epochs=args.num_epochs, max_steps=args.max_steps)

    if args.export_safetensors is not None:
        trainer.export_model_safetensors(args.export_safetensors)

    logger.info("=" * 78)
    logger.info(f"  Final avg loss : {result.final_avg_loss:.4f}")
    logger.info(f"  Epochs done    : {result.epochs_completed}")
    logger.info(f"  Micro steps    : {result.micro_steps}")
    logger.info(f"  Grad steps     : {result.grad_steps}")
    logger.info(f"  Peak GPU mem   : {result.peak_gpu_memory_mb:.0f} MB")
    logger.info(f"  Wall time      : {result.elapsed_seconds:.1f} s")
    logger.info(f"  Checkpoints    : {len(result.checkpoint_paths)}")
    for p in result.checkpoint_paths:
        logger.info(f"    {p}")
    if trainer._best_ckpt_path is not None:
        logger.info(f"  Best ckpt      : {trainer._best_ckpt_path} "
                    f"({trainer.track_best_metric}={trainer._best_metric_value:.4f})")
    logger.info("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
