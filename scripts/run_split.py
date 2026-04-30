"""Generate splits.parquet from M2 webdataset shards.

Reads ``(pcap_source, start_time, label_id)`` from each shard's
``meta.json`` and computes the per-window train/val/test split via
``compute_split_assignments`` (index-based partition per attack class
since the M4.7 second redesign — no CSV / LabelIndex needed at this
stage; the label_id is already in the shard meta).

Usage:
    uv run python scripts/run_split.py \\
        --shard-pattern "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \\
        --output        data/processed/cicids2017_dt100ms_v2/splits.parquet \\
        [--seed 42]
        [--ratios 0.7 0.15 0.15]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from nid_video.data.split import (
    collect_window_keys_from_shards,
    compute_split_assignments,
    verify_splits_complete,
    write_splits_parquet,
)
from nid_video.utils import logger, setup_logger


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--shard-pattern", required=True,
                   help="Glob over webdataset shards "
                        "(e.g. 'data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar').")
    p.add_argument("--output", type=Path, required=True,
                   help="Path to write splits.parquet to.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for hash-based BENIGN partitioning.")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="train/val/test ratios; must sum to 1.0.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logger()
    args = parse_args(argv)
    ratios = tuple(args.ratios)

    logger.info(f"scanning shards: {args.shard_pattern}")
    window_items = list(collect_window_keys_from_shards(args.shard_pattern))
    logger.info(f"found {len(window_items)} window(s)")

    assignments = compute_split_assignments(
        window_items, ratios=ratios, seed=args.seed,
    )

    # Sanity check before persistence: every window has a split. Unwrap to
    # bare WindowKey since the parquet schema doesn't carry label_id.
    verify_splits_complete(assignments, (wkl.key for wkl in window_items))

    counts = {"train": 0, "val": 0, "test": 0}
    for v in assignments.values():
        counts[v] += 1
    n = len(window_items)
    logger.info(
        f"split distribution: train={counts['train']} ({100*counts['train']/n:.1f}%), "
        f"val={counts['val']} ({100*counts['val']/n:.1f}%), "
        f"test={counts['test']} ({100*counts['test']/n:.1f}%)"
    )

    write_splits_parquet(assignments, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
