"""End-to-end ETL: pcap → (T,C,H,W) tensor → labeled webdataset shards.

Wires the per-stage modules together: pcap_parser → SlidingWindow →
cluster_ips_in_window → encode_window → label_window → webdataset ShardWriter,
plus a manifest.parquet summarizing class distribution per shard.

The shard files store the RAW 15-class label IDs (LABEL_TO_ID_RAW). Storing
raw 15-class IDs allows raw15-vs-collapsed13 ablations to share the exact same
ETL output, eliminating any preprocessing-related variance in the comparison.

Single-process within this module. Per-pcap parallelism (--num-workers > 1) is
the responsibility of scripts/run_etl.py; each worker re-uses run_etl().

Idea.md §3.1–§3.5.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds

from nid_video.data.channels import (
    ChannelConfig,
    build_port_mapping,
    encode_window,
)
from nid_video.data.ip_clustering import cluster_ips_in_window
from nid_video.data.labeling import (
    LabelIndex,
    label_window,
    warn_low_population_classes,
)
from nid_video.data.pcap_parser import parse_pcap
from nid_video.data.windowing import SlidingWindow
from nid_video.utils import logger
from nid_video.utils.config import DataConfig


@dataclass
class EtlStats:
    """Aggregated counts from one or more ETL invocations."""

    n_windows_emitted: int = 0
    n_pcaps_processed: int = 0
    n_pcaps_failed: int = 0
    label_counts: Counter[str] = field(default_factory=Counter)
    n_unmatched_total: int = 0
    n_shards: int = 0
    elapsed_seconds: float = 0.0

    @property
    def n_attack_windows(self) -> int:
        return sum(c for k, c in self.label_counts.items() if k != "BENIGN")

    def merge(self, other: "EtlStats") -> "EtlStats":
        self.n_windows_emitted += other.n_windows_emitted
        self.n_pcaps_processed += other.n_pcaps_processed
        self.n_pcaps_failed += other.n_pcaps_failed
        self.label_counts.update(other.label_counts)
        self.n_unmatched_total += other.n_unmatched_total
        self.n_shards += other.n_shards
        self.elapsed_seconds = max(self.elapsed_seconds, other.elapsed_seconds)
        return self


def run_etl(
    pcap_paths: Iterable[Path],
    label_csv_paths: Iterable[Path],
    output_dir: Path,
    data_config: DataConfig,
    *,
    samples_per_shard: int = 1000,
    limit_windows: int | None = None,
    csv_dayfirst: bool = True,
    csv_tz: str = "America/Halifax",
    label_index: LabelIndex | None = None,
) -> EtlStats:
    """Run single-process ETL across the provided pcaps.

    Args:
      pcap_paths: paths to .pcap files; processed in order.
      label_csv_paths: TrafficLabelling CSVs; combined into one LabelIndex.
      output_dir: shards land in <output_dir>/shards/, manifest at <output_dir>/manifest.parquet.
      data_config: DataConfig from configs/base.yaml (T, C, H, W, Δt, overlap).
      samples_per_shard: webdataset shard rolling target.
      limit_windows: cap the total emitted windows (debug/smoke).
      csv_dayfirst: pass True for raw CIC-IDS-2017 CSVs (DD/MM/YYYY timestamps).
      csv_tz: IANA tz of the wall-clock timestamps in the CSV. Default
        "America/Halifax" matches CIC-IDS-2017; pass "UTC" for synthetic
        fixtures whose timestamps are already UTC.
      label_index: optional pre-built index (test/multi-worker re-use).

    Failed pcaps are logged and skipped — they don't crash the run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)

    if label_index is None:
        csvs = [Path(p) for p in label_csv_paths]
        label_index = LabelIndex.from_csv(csvs, dayfirst=csv_dayfirst, csv_tz=csv_tz)

    port_map = build_port_mapping(data_config.num_port_buckets)
    channel_cfg = ChannelConfig.from_data_config(data_config)
    windower = SlidingWindow(
        delta_t_s=data_config.delta_t_ms / 1000.0,
        num_frames=data_config.num_frames,
        overlap=data_config.window_overlap,
    )

    stats = EtlStats()
    t0 = time.perf_counter()

    shard_pattern = str(shard_dir / "shard-%06d.tar")
    sample_idx = 0
    per_shard_counts: list[Counter[str]] = []
    current_shard_counts: Counter[str] = Counter()

    pcaps = list(pcap_paths)
    logger.info(f"ETL: {len(pcaps)} pcap(s), output -> {output_dir}")

    with wds.ShardWriter(shard_pattern, maxcount=samples_per_shard) as writer:
        for pcap_path in pcaps:
            pcap_path = Path(pcap_path)
            logger.info(f"ETL: starting {pcap_path.name}")
            try:
                packets = parse_pcap(pcap_path)
                stop = False
                windows_before = stats.n_windows_emitted
                for window in windower(packets, pcap_source=pcap_path.name):
                    if limit_windows is not None and stats.n_windows_emitted >= limit_windows:
                        logger.info(f"--limit-windows={limit_windows} reached")
                        stop = True
                        break

                    ip_to_row = cluster_ips_in_window(
                        window, n_clusters=data_config.num_ip_buckets,
                    )
                    tensor = encode_window(window, ip_to_row, port_map, channel_cfg)
                    win_label = label_window(window, label_index)

                    # Roll the per-shard counter at shard boundary
                    if sample_idx > 0 and sample_idx % samples_per_shard == 0:
                        per_shard_counts.append(current_shard_counts)
                        current_shard_counts = Counter()

                    writer.write({
                        "__key__": f"{sample_idx:010d}",
                        "tensor.npy": tensor,
                        "label.cls": int(win_label.label_id),
                        "meta.json": {
                            "start_time": float(window.start_time),
                            "pcap_source": window.pcap_source,
                            "label": win_label.label,
                            "label_id": int(win_label.label_id),
                            "dominant_attack_ratio": float(win_label.dominant_ratio),
                            "n_unmatched": int(win_label.n_unmatched),
                        },
                    })

                    sample_idx += 1
                    stats.n_windows_emitted += 1
                    stats.label_counts[win_label.label] += 1
                    current_shard_counts[win_label.label] += 1
                    stats.n_unmatched_total += win_label.n_unmatched

                # Distinguish "pcap parsed but produced 0 windows" (a real
                # failure mode — was being silently mis-counted as OK pre-M3)
                # from "limit_windows preempted before this pcap could yield"
                # (legitimate stop, not the pcap's fault).
                produced = stats.n_windows_emitted - windows_before
                if produced == 0 and not stop:
                    logger.warning(
                        f"{pcap_path.name}: produced 0 windows — counting as failed. "
                        "Likely causes: parse error, all-malformed packets, or "
                        "pcap span shorter than one window."
                    )
                    stats.n_pcaps_failed += 1
                else:
                    stats.n_pcaps_processed += 1
                if stop:
                    break
            except Exception as exc:                                # noqa: BLE001
                logger.error(f"ETL failed on {pcap_path}: {exc}")
                stats.n_pcaps_failed += 1
                continue

    if current_shard_counts:
        per_shard_counts.append(current_shard_counts)

    stats.n_shards = len(per_shard_counts)
    stats.elapsed_seconds = time.perf_counter() - t0

    # Class-imbalance warning (Heartbleed-class floors)
    warn_low_population_classes(dict(stats.label_counts), min_samples=50)

    _write_manifest(output_dir, per_shard_counts)

    logger.info(
        f"ETL done in {stats.elapsed_seconds:.1f}s: "
        f"{stats.n_windows_emitted} windows / {stats.n_shards} shards / "
        f"{stats.n_pcaps_processed} OK / {stats.n_pcaps_failed} failed"
    )
    return stats


def load_combined_manifest(output_dir: Path) -> pd.DataFrame:
    """Read every manifest.parquet under `output_dir` and concat into one DataFrame.

    Handles both layouts:
      Single-process: ``output_dir/manifest.parquet``
      Multi-worker:   ``output_dir/<pcap_stem>/manifest.parquet`` (one per pcap)

    Adds a ``source_dir`` column (relative to output_dir, "." for top-level) so
    callers can group by pcap when in multi-worker mode.

    Why this exists: M3/M5/M6 frequently inspect the full label distribution
    for sampling, balancing, and curriculum decisions. Hand-rolling the glob
    each time is bug-prone and inconsistent across single- vs multi-worker
    layouts.
    """
    output_dir = Path(output_dir)
    manifests = sorted(output_dir.rglob("manifest.parquet"))
    if not manifests:
        raise FileNotFoundError(f"no manifest.parquet found under {output_dir}")
    frames: list[pd.DataFrame] = []
    for m in manifests:
        df = pq.read_table(m).to_pandas()
        rel = m.parent.relative_to(output_dir).as_posix() or "."
        df["source_dir"] = rel
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _write_manifest(
    output_dir: Path,
    per_shard_counts: list[Counter[str]],
) -> None:
    """Persist per-shard label distribution as parquet (one row per shard)."""
    if not per_shard_counts:
        logger.warning("no shards written; manifest will be empty")
        rows: list[dict[str, object]] = [{
            "shard_idx": -1,
            "shard_name": "",
            "n_samples": 0,
            "labels_json": "{}",
        }]
    else:
        rows = [
            {
                "shard_idx": i,
                "shard_name": f"shard-{i:06d}.tar",
                "n_samples": int(sum(counts.values())),
                "labels_json": json.dumps(dict(counts), ensure_ascii=False),
            }
            for i, counts in enumerate(per_shard_counts)
        ]
    table = pa.Table.from_pylist(rows)
    out = output_dir / "manifest.parquet"
    pq.write_table(table, out)
    logger.info(f"manifest written: {out}")
