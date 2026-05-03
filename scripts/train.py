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
    g.add_argument("--epoch-end-strategy",
                   choices=["round_robin", "slow_exhausted", "no_cycle"],
                   default="round_robin",
                   help="Train-side stream-termination policy. 'round_robin' "
                        "(default): fast is anchor, slow cycles — correct for "
                        "training where coverage matters more than uniqueness. "
                        "'slow_exhausted' (legacy): epoch ends on first stream "
                        "exhaustion. 'no_cycle': drain both streams exactly "
                        "once (typically used as an eval value via "
                        "--eval-strategy, not here). The choice changes "
                        "total_steps via _compute_total_steps.")
    g.add_argument("--eval-strategy",
                   choices=["round_robin", "slow_exhausted", "no_cycle"],
                   default="no_cycle",
                   help="Eval-side stream-termination policy, applied to the "
                        "val loader (and the eval-only loader). Default "
                        "'no_cycle' yields each unique val sample exactly "
                        "once and gives metric numbers that don't depend on "
                        "mix_ratio. Override only to reproduce a legacy run.")

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
    p.add_argument("--loss-fn", choices=["ce", "focal"], default=None,
                   help="Override training.loss_fn from the config. 'ce' is "
                        "vanilla cross-entropy (default). 'focal' uses "
                        "FocalLoss with --focal-gamma. M5.4 invocation: "
                        "--loss-fn focal --focal-gamma 2.0")
    p.add_argument("--focal-gamma", type=float, default=None,
                   help="Override training.focal_gamma. Effective only when "
                        "--loss-fn is (or resolves to) 'focal'. Default 2.0.")
    p.add_argument("--reweighting", choices=["none", "inverse_sqrt"], default=None,
                   help="Override training.reweighting (M5.4 Phase 2). "
                        "'inverse_sqrt' computes alpha = 1/sqrt(n_train), "
                        "normalised to mean=1 over present classes, and "
                        "injects it into FocalLoss. 'none' (default) keeps "
                        "loss class-uniform.")
    p.add_argument("--head-lr-multiplier", type=float, default=None,
                   help="Override training.head_lr_multiplier (M5.4 Phase 2). "
                        "Sets the classification head (classifier + "
                        "scale_token + scale_embedding) LR to the given "
                        "multiple of the backbone LR. Default 1.0.")

    # ----- existing -----
    p.add_argument("--label-mode", choices=["raw15", "collapsed13"], default="collapsed13")
    p.add_argument("--num-workers", type=int, default=None,
                   help="override training.num_workers from config")
    p.add_argument("--shuffle-buffer", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--debug", action="store_true",
                   help="FP32 + vanilla AdamW + no grad checkpointing (sanity run)")
    p.add_argument("--pretrained", default="MCG-NJU/videomae-small-finetuned-kinetics",
                   help="HF model id, or '' / 'none' to skip pretraining. "
                        "Default targets VideoMAE-Small; baselines override "
                        "via --model (which selects the right default ckpt).")
    p.add_argument("--model",
                   choices=["videomae_small", "timesformer_small", "c3d_small",
                            "i3d", "r2plus1d_18", "convlstm"],
                   default="videomae_small",
                   help="Backbone selector (M5.5). Default keeps the M3-onward "
                        "main method. Baselines build the same forward "
                        "signature (x, scale_id) → logits/features but ignore "
                        "scale_id when scale-agnostic.")
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


def _build_model(args, cfg, training_cfg, n_classes: int, pretrained: str | None):
    """Dispatch on ``args.model`` to construct the chosen backbone with
    the project's standard input contract (T=16, C=6, H=32, W=64). The
    main method is ``videomae_small`` (M3 onward); baselines are added
    in M5.5 Round 1+.

    All baselines expose ``forward(x, *, scale_id) → {logits, features}``
    even when they're scale-agnostic — the trainer's call site treats
    every model uniformly.
    """
    name = args.model
    if name == "videomae_small":
        return VideoMAESmallForNID(
            num_classes=n_classes,
            pretrained=pretrained,
            in_channels=cfg.data.num_channels,
            tube_patch=tuple(cfg.model.tube_patch),
            spatial_grid=(cfg.data.num_ip_buckets, cfg.data.num_port_buckets),
            gradient_checkpointing=training_cfg.gradient_checkpointing,
        )
    if name == "timesformer_small":
        from nid_video.models.timesformer_small_nid import TimeSformerSmallForNID
        # TimeSformer-Small runs random-init from scratch (no public
        # 22M Kinetics ckpt at this scale). Any --pretrained value the
        # user passes is silently ignored — the constructor takes none.
        return TimeSformerSmallForNID(
            num_classes=n_classes,
            in_channels=cfg.data.num_channels,
            target_image_size=64,
            num_frames=16,
            gradient_checkpointing=training_cfg.gradient_checkpointing,
        )
    if name == "c3d_small":
        from nid_video.models.c3d_small_nid import C3DSmallForNID
        # C3D-Small runs random-init from scratch (no public small
        # variant K400 ckpt at the project's input scale). --pretrained
        # is silently ignored — the constructor takes none.
        return C3DSmallForNID(
            num_classes=n_classes,
            in_channels=cfg.data.num_channels,
            gradient_checkpointing=training_cfg.gradient_checkpointing,
        )
    if name == "convlstm":
        from nid_video.models.convlstm_nid import ConvLSTMForNID
        # ConvLSTM runs random-init from scratch — no canonical
        # video-action pretrained ckpt at this hidden size.
        return ConvLSTMForNID(
            num_classes=n_classes,
            in_channels=cfg.data.num_channels,
            gradient_checkpointing=training_cfg.gradient_checkpointing,
        )
    raise SystemExit(
        f"--model {name!r} not yet implemented in M5.5 (Round 2+ adds "
        f"i3d / r2plus1d_18)"
    )


def _override_for_debug(training_cfg):
    return training_cfg.model_copy(update={
        "precision": "fp32",
        "optimizer": "adamw",
        "gradient_checkpointing": False,
    })


def _total_steps_from_train_n(
    train_n: int,
    *,
    batch_size: int,
    grad_accumulation: int,
    num_epochs: int,
    mix_ratio: float | None = None,
    epoch_end_strategy: str | None = None,
) -> int:
    """Pure-math LR scheduler total_steps from the *anchor stream*'s train count.

    The "anchor" is the stream whose exhaustion ends the epoch. Per-call
    semantics:

    * ``mix_ratio is None`` → single-scale. ``train_n`` is the only stream;
      ``anchor_n = train_n``.
    * ``mix_ratio`` set, ``epoch_end_strategy == "round_robin"`` (M4.8 default)
      → fast stream is anchor. Caller passes ``train_n_fast``.
      ``anchor_n = ceil(train_n_fast / mix_ratio)`` (with mix=0.5, the slow
      stream contributes the other half of yields, so total samples per epoch
      is ~2× the fast stream).
    * ``mix_ratio`` set, ``epoch_end_strategy == "slow_exhausted"`` (M4.2
      legacy) → slow stream is anchor. Caller passes ``train_n_slow``.
      ``anchor_n = ceil(train_n_slow / (1 - mix_ratio))``.

    Returns ``ceil(anchor_n / (batch_size × grad_accumulation)) × num_epochs``.

    The M5.1 fix this function ships: pre-M5.1, the caller passed
    ``train_n_fast`` (counted from splits.parquet) but used the single-scale
    formula, which under-counted total_steps by ``1/mix_ratio`` (~2× under
    round_robin), causing cosine decay to bottom out mid-epoch.
    """
    if mix_ratio is None:
        anchor_n = int(train_n)
    elif epoch_end_strategy == "round_robin":
        anchor_n = math.ceil(train_n / max(mix_ratio, 1e-9))
    elif epoch_end_strategy == "slow_exhausted":
        anchor_n = math.ceil(train_n / max(1.0 - mix_ratio, 1e-9))
    else:
        raise ValueError(
            f"epoch_end_strategy must be 'round_robin' or 'slow_exhausted' "
            f"when mix_ratio is set; got {epoch_end_strategy!r}"
        )
    steps_per_epoch = max(
        1, math.ceil(anchor_n / max(1, batch_size * grad_accumulation))
    )
    return steps_per_epoch * num_epochs


def _scan_train_class_counts(
    splits: dict, fast_pattern: str, n_classes: int, label_mode: str,
) -> list[int]:
    """One-pass scan of the fast (100ms) train shards to count per-class
    samples in the train split. Used by ``--reweighting inverse_sqrt`` to
    compute focal-loss alpha. Returns a length-``n_classes`` list of ints.

    The scan is on the fast shards because splits.parquet was built on
    them (M5.1 contract); slow-stream class counts are deterministic
    multiples (~1/10) of the fast counts and don't add information for
    inverse-sqrt frequency weighting.
    """
    from nid_video.data.labeling import collapse_to_13
    from nid_video.data.split import collect_window_keys_from_shards

    counts = [0] * n_classes
    for kvl in collect_window_keys_from_shards(fast_pattern):
        if splits.get(kvl.key) != "train":
            continue
        label_id = kvl.label_id
        if label_mode == "collapsed13":
            label_id = collapse_to_13(label_id)
        if 0 <= label_id < n_classes:
            counts[label_id] += 1
    return counts


def _build_reweighted_criterion(args: argparse.Namespace, training_cfg, n_classes: int):
    """Compute inverse-sqrt alpha from the train split and build the
    matching FocalLoss. Returns a fully-constructed nn.Module ready to
    inject into ``Trainer(criterion=...)``.
    """
    import torch as _torch

    from nid_video.data.split import load_splits
    from nid_video.trainer.loss import build_criterion, compute_inverse_sqrt_alpha

    if args.splits_path is None or not args.splits_path.is_file():
        raise SystemExit(
            "--reweighting inverse_sqrt requires --splits-path to a valid "
            "splits.parquet file"
        )
    fast_pattern = args.shard_pattern_fast or args.shard_pattern
    if fast_pattern is None:
        raise SystemExit(
            "--reweighting inverse_sqrt requires --shard-pattern-fast "
            "(multi-scale) or --shard-pattern (single-scale) to scan for "
            "train class counts"
        )

    logger.info(
        f"scanning train shards for class counts: {fast_pattern} "
        f"(label_mode={args.label_mode}, n_classes={n_classes}) ..."
    )
    splits = load_splits(args.splits_path)
    counts = _scan_train_class_counts(splits, fast_pattern, n_classes, args.label_mode)
    alpha_cpu = compute_inverse_sqrt_alpha(counts, n_classes)
    counts_pretty = ", ".join(f"{c}" for c in counts)
    alpha_pretty = ", ".join(f"{a:.4f}" for a in alpha_cpu.tolist())
    logger.info(
        f"train class counts: [{counts_pretty}] (total {sum(counts)})"
    )
    logger.info(f"inverse-sqrt alpha (mean=1 over n>0): [{alpha_pretty}]")

    alpha_dev = alpha_cpu.to(args.device) if args.device != "cpu" else alpha_cpu
    return build_criterion(training_cfg, alpha=alpha_dev)


def _count_train_in_slow_shards(slow_pattern: str, splits: dict) -> int:
    """Count slow-shard windows whose (pcap_source, start_time) maps to the
    'train' split. Required for ``slow_exhausted`` total_steps because the
    splits.parquet was built on fast (100ms) shards — the slow (1s) train
    count is not available without scanning.
    """
    from nid_video.data.split import collect_window_keys_from_shards
    n = 0
    for kvl in collect_window_keys_from_shards(slow_pattern):
        if splits.get(kvl.key) == "train":
            n += 1
    return n


def _compute_total_steps(args: argparse.Namespace, training_cfg) -> int | None:
    """Compute LR scheduler total_steps. Priority:
       1. --total-steps explicit
       2. ``_total_steps_from_train_n`` driven by splits.parquet, with the
          anchor stream picked by ``--epoch-end-strategy`` (M5.1 fix)
       3. None → Trainer's own num_epochs × 1000 heuristic + WARNING
    """
    if args.total_steps is not None:
        return args.total_steps
    if args.splits_path is None or not args.splits_path.is_file():
        return None

    from nid_video.data.split import load_splits
    splits = load_splits(args.splits_path)
    n_epochs = args.num_epochs if args.num_epochs is not None else training_cfg.num_epochs
    is_multi = args.shard_pattern_fast is not None

    if not is_multi:
        train_n = sum(1 for v in splits.values() if v == "train")
        total = _total_steps_from_train_n(
            train_n,
            batch_size=training_cfg.batch_size,
            grad_accumulation=training_cfg.grad_accumulation,
            num_epochs=n_epochs,
        )
        logger.info(
            f"total_steps = {total} (single-scale: train_n={train_n}, "
            f"batch={training_cfg.batch_size}, accum={training_cfg.grad_accumulation}, "
            f"epochs={n_epochs})"
        )
        return total

    # Multi-scale: anchor depends on epoch_end_strategy. M5.1 fix.
    if args.epoch_end_strategy == "round_robin":
        # splits.parquet was built on 100ms fast shards → train_n directly.
        train_n_fast = sum(1 for v in splits.values() if v == "train")
        total = _total_steps_from_train_n(
            train_n_fast,
            batch_size=training_cfg.batch_size,
            grad_accumulation=training_cfg.grad_accumulation,
            num_epochs=n_epochs,
            mix_ratio=args.mix_ratio,
            epoch_end_strategy="round_robin",
        )
        logger.info(
            f"total_steps = {total} (multi-scale round_robin: "
            f"train_n_fast={train_n_fast}, mix_ratio={args.mix_ratio}, "
            f"batch={training_cfg.batch_size}, accum={training_cfg.grad_accumulation}, "
            f"epochs={n_epochs})"
        )
        return total

    if args.epoch_end_strategy == "slow_exhausted":
        # Need to scan the slow shards to count train-tagged 1s windows.
        train_n_slow = _count_train_in_slow_shards(args.shard_pattern_slow, splits)
        total = _total_steps_from_train_n(
            train_n_slow,
            batch_size=training_cfg.batch_size,
            grad_accumulation=training_cfg.grad_accumulation,
            num_epochs=n_epochs,
            mix_ratio=args.mix_ratio,
            epoch_end_strategy="slow_exhausted",
        )
        logger.info(
            f"total_steps = {total} (multi-scale slow_exhausted: "
            f"train_n_slow={train_n_slow}, mix_ratio={args.mix_ratio}, "
            f"batch={training_cfg.batch_size}, accum={training_cfg.grad_accumulation}, "
            f"epochs={n_epochs})"
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
        # The eval-only path is, by definition, an eval-mode loader, so it
        # uses --eval-strategy regardless of --epoch-end-strategy.
        if is_multi:
            val_loader = build_multi_scale_dataloader(
                fast_pattern=args.shard_pattern_fast,
                slow_pattern=args.shard_pattern_slow,
                splits_path=args.splits_path,
                keep_split=args.keep_split,
                mix_ratio=args.mix_ratio,
                epoch_end_strategy=args.eval_strategy,
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
            epoch_end_strategy=args.epoch_end_strategy,
            **common,
        )
        val_loader = None
        if args.splits_path is not None:
            # Val loader uses --eval-strategy (decoupled from training-side
            # --epoch-end-strategy). Default no_cycle yields each unique
            # val sample exactly once for metric stability.
            val_loader = build_multi_scale_dataloader(
                fast_pattern=args.shard_pattern_fast,
                slow_pattern=args.shard_pattern_slow,
                splits_path=args.splits_path,
                keep_split="val",
                mix_ratio=args.mix_ratio,
                epoch_end_strategy=args.eval_strategy,
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
    if args.loss_fn is not None:
        training = training.model_copy(update={"loss_fn": args.loss_fn})
    if args.focal_gamma is not None:
        training = training.model_copy(update={"focal_gamma": args.focal_gamma})
    if args.reweighting is not None:
        training = training.model_copy(update={"reweighting": args.reweighting})
    if args.head_lr_multiplier is not None:
        training = training.model_copy(update={"head_lr_multiplier": args.head_lr_multiplier})

    pretrained = None if args.pretrained.lower() in ("", "none") else args.pretrained
    n_classes = label_num_classes(args.label_mode)

    model = _build_model(args, cfg, training, n_classes, pretrained)

    train_loader, val_loader = _build_loaders(args, training, for_eval_only=args.eval_only)

    if args.eval_only:
        return _run_eval_only(args, model, val_loader, n_classes)

    total_steps = _compute_total_steps(args, training)

    # M5.4 Phase 2: when class reweighting is requested, scan the train shards
    # once for class counts, compute alpha, and inject the resulting criterion
    # into the Trainer. The Trainer's own build_criterion(config) default does
    # not have data access by design, so the alpha plumbing lives here.
    criterion = None
    if training.reweighting == "inverse_sqrt":
        criterion = _build_reweighted_criterion(args, training, n_classes)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=training,
        device=args.device,
        criterion=criterion,
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
