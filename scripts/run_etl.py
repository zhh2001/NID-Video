"""ETL CLI: pcap dir + label CSV dir → webdataset shards + manifest.parquet.

Usage:
    uv run python scripts/run_etl.py \\
        --pcap-dir data/raw/cicids2017/PCAPs \\
        --label-dir data/raw/cicids2017/CSVs \\
        --output-dir data/processed/cicids2017 \\
        --csv-dayfirst   # required for raw CIC-IDS-2017 timestamps
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from nid_video.data.etl_pipeline import EtlStats, run_etl
from nid_video.utils import logger, setup_logger
from nid_video.utils.config import load_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pcap-dir", type=Path, required=True)
    p.add_argument("--label-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    p.add_argument("--limit-windows", type=int, default=None,
                   help="Cap total emitted windows (debug/smoke). Default: no limit.")
    p.add_argument("--num-workers", type=int, default=1,
                   help="Parallel pcap workers (each pcap stays single-threaded inside).")
    p.add_argument("--samples-per-shard", type=int, default=1000)
    p.add_argument("--csv-dayfirst", action="store_true",
                   help="Required for raw CIC-IDS-2017 (timestamps are DD/MM/YYYY).")
    p.add_argument("--csv-tz", default="America/Halifax",
                   help="IANA tz of CSV wall-clock timestamps. Default "
                        "America/Halifax matches CIC-IDS-2017's site.")
    p.add_argument("--no-twelve-hour-inference", action="store_true",
                   help="Disable the CIC-IDS-2017 12h-without-AM/PM inference "
                        "(hours 1..7 shifted +12h). Use for datasets that already "
                        "store 24h timestamps or carry explicit AM/PM markers.")
    return p.parse_args(argv)


def _process_one_pcap(
    pcap: Path,
    *,
    csvs: list[Path],
    output_dir: Path,
    data_cfg: object,
    samples_per_shard: int,
    limit_windows: int | None,
    csv_dayfirst: bool,
    csv_tz: str,
    csv_twelve_hour_pm_inference: bool,
) -> EtlStats:
    sub_out = output_dir / pcap.stem
    return run_etl(
        [pcap],
        csvs,
        sub_out,
        data_cfg,                       # type: ignore[arg-type]
        samples_per_shard=samples_per_shard,
        limit_windows=limit_windows,
        csv_dayfirst=csv_dayfirst,
        csv_tz=csv_tz,
        csv_twelve_hour_pm_inference=csv_twelve_hour_pm_inference,
    )


def main(argv: Sequence[str] | None = None) -> int:
    setup_logger()
    args = parse_args(argv)

    cfg = load_config(args.config)
    pcaps = sorted(args.pcap_dir.glob("*.pcap"))
    csvs = sorted(args.label_dir.glob("*.csv"))

    if not pcaps:
        logger.error(f"no .pcap files in {args.pcap_dir}")
        return 2
    if not csvs:
        logger.error(f"no .csv files in {args.label_dir}")
        return 2

    logger.info(
        f"ETL: {len(pcaps)} pcap(s), {len(csvs)} CSV(s), "
        f"workers={args.num_workers}, output -> {args.output_dir}"
    )

    csv_twelve_hour = not args.no_twelve_hour_inference
    if args.num_workers > 1 and len(pcaps) > 1:
        worker = partial(
            _process_one_pcap,
            csvs=csvs,
            output_dir=args.output_dir,
            data_cfg=cfg.data,
            samples_per_shard=args.samples_per_shard,
            limit_windows=args.limit_windows,
            csv_dayfirst=args.csv_dayfirst,
            csv_tz=args.csv_tz,
            csv_twelve_hour_pm_inference=csv_twelve_hour,
        )
        with Pool(args.num_workers) as pool:
            sub_stats = pool.map(worker, pcaps)
        total = EtlStats()
        for s in sub_stats:
            total.merge(s)
        stats = total
    else:
        stats = run_etl(
            pcaps,
            csvs,
            args.output_dir,
            cfg.data,
            samples_per_shard=args.samples_per_shard,
            limit_windows=args.limit_windows,
            csv_dayfirst=args.csv_dayfirst,
            csv_tz=args.csv_tz,
            csv_twelve_hour_pm_inference=csv_twelve_hour,
        )

    _print_stats(stats)
    return 0


def _print_stats(stats: EtlStats) -> None:
    logger.info("=" * 78)
    logger.info(f"  Total windows: {stats.n_windows_emitted}")
    logger.info(f"  Total shards : {stats.n_shards}")
    logger.info(f"  Pcaps OK     : {stats.n_pcaps_processed}")
    logger.info(f"  Pcaps failed : {stats.n_pcaps_failed}")
    logger.info(f"  Unmatched pkt: {stats.n_unmatched_total}")
    logger.info(f"  Elapsed      : {stats.elapsed_seconds:.1f}s")
    logger.info("  Label distribution:")
    for label, count in sorted(stats.label_counts.items(), key=lambda kv: -kv[1]):
        logger.info(f"    {label:30s} {count:>8d}")
    logger.info("=" * 78)


if __name__ == "__main__":
    raise SystemExit(main())
