"""M3 training entry point.

Usage:
    uv run python scripts/train.py \\
        --config configs/base.yaml \\
        --shard-pattern "data/processed/*/shards/shard-*.tar"

Debug runs:
    uv run python scripts/train.py --debug --max-steps 20 --shard-pattern ...
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from nid_video.data.dataset import build_dataloader, num_classes as label_num_classes
from nid_video.models.videomae_nid import VideoMAESmallForNID
from nid_video.trainer.trainer import Trainer
from nid_video.utils import logger, setup_logger
from nid_video.utils.config import load_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    p.add_argument("--shard-pattern", type=str, required=True,
                   help='glob, e.g. "data/processed/*/shards/shard-*.tar"')
    p.add_argument("--num-epochs", type=int, default=None,
                   help="override training.num_epochs from config")
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap micro-batch count (debug)")
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
                   help="Resume from a .pt checkpoint produced by save_checkpoint. "
                        "Restores model + optimizer + scheduler + scaler + RNG.")
    p.add_argument("--export-safetensors", type=Path, default=None,
                   help="After training, export final model weights (only) to "
                        "this safetensors path for deployment.")
    return p.parse_args(argv)


def _override_for_debug(training_cfg):
    return training_cfg.model_copy(update={
        "precision": "fp32",
        "optimizer": "adamw",
        "gradient_checkpointing": False,
    })


def main(argv: Sequence[str] | None = None) -> int:
    setup_logger()
    args = parse_args(argv)

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

    num_workers = args.num_workers if args.num_workers is not None else training.num_workers
    loader = build_dataloader(
        args.shard_pattern,
        batch_size=training.batch_size,
        num_workers=num_workers,
        label_mode=args.label_mode,
        shuffle_buffer=args.shuffle_buffer,
        pin_memory=(args.device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        train_loader=loader,
        config=training,
        device=args.device,
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
    logger.info("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
