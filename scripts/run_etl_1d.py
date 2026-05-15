"""M6.1 — CLI for the 1D byte-sequence ETL (Phase 0 step 2).

Mirrors ``scripts/run_etl.py`` shape (--pcap-dir / --label-dir /
--output-dir / --csv-dayfirst) but invokes ``run_byte_etl`` which emits
shards with bytes.npy + mask.npy (K=16, N=128) + label.cls + meta.json
per sample.

Phase 0 default ``--output-dir`` lands at
``data/processed/cicids2017_dt100ms_v2_bytes/`` (parallel to v2, NOT
modifying v2). Pass ``--limit-windows N`` for smoke runs.

Phase 1 Δt = 100ms (fast stream only — M6.1 has no slow analogue).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from nid_video.data.byte_extraction import ByteEtlStats, run_byte_etl
from nid_video.utils import logger, setup_logger


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pcap-dir", type=Path,
                   default=Path("data/raw/cicids2017"))
    p.add_argument("--label-dir", type=Path,
                   default=Path("data/raw/cicids2017/TrafficLabelling"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/processed/cicids2017_dt100ms_v2_bytes"))
    p.add_argument("--limit-windows", type=int, default=None,
                   help="Cap total windows (debug/smoke). Default: no limit.")
    p.add_argument("--samples-per-shard", type=int, default=1000)
    p.add_argument("--csv-dayfirst", action="store_true", default=True)
    p.add_argument("--csv-tz", default="America/Halifax")
    p.add_argument("--no-twelve-hour-inference", action="store_true")
    return p.parse_args(argv)


def _print_stats(stats: ByteEtlStats) -> None:
    logger.info("=" * 78)
    logger.info(f"  Total windows: {stats.n_windows_emitted}")
    logger.info(f"  Total shards : {stats.n_shards}")
    logger.info(f"  Pcaps OK     : {stats.n_pcaps_processed}")
    logger.info(f"  Pcaps failed : {stats.n_pcaps_failed}")
    logger.info(f"  Packets yielded: {stats.n_packets_yielded:,}")
    logger.info(f"  Packets truncated to N=128: {stats.n_packets_truncated:,}")
    logger.info(f"  Elapsed      : {stats.elapsed_seconds:.1f}s")
    logger.info("  Label distribution:")
    for label, count in sorted(stats.label_counts.items(), key=lambda kv: -kv[1]):
        logger.info(f"    {label:30s} {count:>8d}")
    logger.info("=" * 78)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logger()
    args = parse_args(argv)

    pcaps = sorted(args.pcap_dir.glob("*.pcap"))
    csvs = sorted(args.label_dir.glob("*.csv"))
    if not pcaps:
        logger.error(f"no .pcap files in {args.pcap_dir}")
        return 2
    if not csvs:
        logger.error(f"no .csv files in {args.label_dir}")
        return 2

    logger.info(
        f"byte ETL: {len(pcaps)} pcap(s), {len(csvs)} CSV(s), output -> {args.output_dir}"
    )

    stats = run_byte_etl(
        pcap_paths=pcaps,
        label_csv_paths=csvs,
        output_dir=args.output_dir,
        samples_per_shard=args.samples_per_shard,
        limit_windows=args.limit_windows,
        csv_dayfirst=args.csv_dayfirst,
        csv_tz=args.csv_tz,
        csv_twelve_hour_pm_inference=not args.no_twelve_hour_inference,
    )
    _print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
