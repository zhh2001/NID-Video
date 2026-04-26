"""End-to-end tests for etl_pipeline.run_etl: synthetic pcaps → shards → readback."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import dpkt
import numpy as np
import pyarrow.parquet as pq
import pytest
import webdataset as wds

from nid_video.data.etl_pipeline import EtlStats, load_combined_manifest, run_etl
from nid_video.data.labeling import LABEL_TO_ID_RAW
from nid_video.utils import load_config, project_root


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


_BASE_TS = dt.datetime(2017, 7, 5, 9, 0, 0, tzinfo=dt.timezone.utc).timestamp()


def _build_pcap_with_packets(
    out: Path,
    n_seconds: float,
    rate_per_sec: int,
    src_ips: list[str],
    *,
    src_port: int = 12345,
    dst_ip: str = "10.0.0.99",
    dst_port: int = 80,
    base_ts: float = _BASE_TS,
) -> int:
    """Write a small pcap. Returns number of packets written."""
    from scapy.all import IP, TCP, Ether, Raw

    n_packets = int(n_seconds * rate_per_sec)
    interval = 1.0 / rate_per_sec
    n_ips = len(src_ips)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        writer = dpkt.pcap.Writer(fh, linktype=dpkt.pcap.DLT_EN10MB)
        for i in range(n_packets):
            ts = base_ts + i * interval
            sip = src_ips[i % n_ips]
            buf = bytes(
                Ether()
                / IP(src=sip, dst=dst_ip)
                / TCP(sport=src_port + (i % 16), dport=dst_port, flags="S")
                / Raw(load=b"X" * 40)
            )
            writer.writepkt(buf, ts=ts)
    return n_packets


def _write_csv(path: Path, rows: list[tuple]) -> None:
    cols = [" Source IP", " Source Port", " Destination IP", " Destination Port",
            " Protocol", " Timestamp", " Flow Duration", " Label"]
    header = ",".join(cols)
    body = "\n".join(",".join(str(v) for v in r) for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + body + "\n", encoding="latin-1")


def _iso_offset(seconds: float) -> str:
    t = dt.datetime.fromtimestamp(_BASE_TS + seconds, tz=dt.timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> tuple[list[Path], list[Path], Path]:
    """Three mini pcaps + matching CSV labels.

    pcap_a → BENIGN (32 source IPs, port 80)
    pcap_b → DDoS (single src 10.2.0.1, heavy)
    pcap_c → SSH-Patator (single src 10.3.0.1, port 22)
    """
    pcap_dir = tmp_path / "pcaps"
    label_dir = tmp_path / "labels"
    out_dir = tmp_path / "out"

    src_ips_a = [f"10.1.0.{i}" for i in range(32)]
    _build_pcap_with_packets(pcap_dir / "a.pcap", 3.0, 100, src_ips_a)
    _build_pcap_with_packets(pcap_dir / "b.pcap", 3.0, 200, ["10.2.0.1"])
    _build_pcap_with_packets(pcap_dir / "c.pcap", 3.0, 100, ["10.3.0.1"],
                             dst_port=22)

    # CSV: cover the whole 3-second span for each pcap
    _write_csv(label_dir / "labels.csv", [
        # All BENIGN flows from group A IPs to port 80
        *[
            (sip, 12345 + i % 16, "10.0.0.99", 80, 6,
             _iso_offset(0.0), int(3.5 * 1e6), "BENIGN")
            for i, sip in enumerate(src_ips_a)
        ],
        # DDoS flow from 10.2.0.1
        ("10.2.0.1", 12345, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "DDoS"),
        ("10.2.0.1", 12346, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "DDoS"),
        ("10.2.0.1", 12347, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "DDoS"),
        ("10.2.0.1", 12348, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "DDoS"),
        ("10.2.0.1", 12349, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "DDoS"),
        ("10.2.0.1", 12350, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "DDoS"),
        ("10.2.0.1", 12351, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "DDoS"),
        ("10.2.0.1", 12352, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "DDoS"),
        ("10.2.0.1", 12353, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "DDoS"),
        ("10.2.0.1", 12354, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "DDoS"),
        ("10.2.0.1", 12355, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "DDoS"),
        ("10.2.0.1", 12356, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "DDoS"),
        ("10.2.0.1", 12357, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "DDoS"),
        ("10.2.0.1", 12358, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "DDoS"),
        ("10.2.0.1", 12359, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "DDoS"),
        # SSH-Patator from 10.3.0.1
        *[
            ("10.3.0.1", 12345 + i, "10.0.0.99", 22, 6,
             _iso_offset(0.0), int(3.5 * 1e6), "SSH-Patator")
            for i in range(16)
        ],
    ])

    return (
        sorted(pcap_dir.glob("*.pcap")),
        sorted(label_dir.glob("*.csv")),
        out_dir,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_etl_emits_shards_and_manifest(synthetic_dataset) -> None:
    pcaps, csvs, out_dir = synthetic_dataset
    cfg = load_config(project_root() / "configs" / "base.yaml")

    stats = run_etl(
        pcaps, csvs, out_dir, cfg.data,
        samples_per_shard=4,
        csv_dayfirst=False,         # synth uses ISO timestamps
        csv_tz="UTC",               # interpret naive ISO as UTC, not ADT
    )

    assert isinstance(stats, EtlStats)
    assert stats.n_windows_emitted > 0, "expected at least one window per pcap"
    assert stats.n_pcaps_processed == 3
    assert stats.n_pcaps_failed == 0
    assert stats.n_shards >= 1

    shard_files = sorted((out_dir / "shards").glob("shard-*.tar"))
    assert len(shard_files) == stats.n_shards

    manifest = out_dir / "manifest.parquet"
    assert manifest.is_file()


@pytest.mark.slow
def test_shards_round_trip_through_webdataset(synthetic_dataset) -> None:
    """Read the written shards back; tensors and labels must match expectations."""
    pcaps, csvs, out_dir = synthetic_dataset
    cfg = load_config(project_root() / "configs" / "base.yaml")
    run_etl(pcaps, csvs, out_dir, cfg.data, samples_per_shard=4,
            csv_dayfirst=False, csv_tz="UTC")

    shard_files = sorted((out_dir / "shards").glob("shard-*.tar"))
    urls = [str(p) for p in shard_files]
    dataset = wds.WebDataset(urls).decode()

    n_seen = 0
    expected_label_ids = set(LABEL_TO_ID_RAW.values())
    seen_labels: set[int] = set()
    for sample in dataset:
        n_seen += 1
        # All required keys present
        assert "tensor.npy" in sample
        assert "label.cls" in sample
        assert "meta.json" in sample
        # Tensor shape and dtype contract
        tensor = sample["tensor.npy"]
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (16, 6, 32, 64), f"got shape {tensor.shape}"
        assert tensor.dtype == np.float32
        # Label is an int in the known label-ID set
        label_id = sample["label.cls"]
        assert isinstance(label_id, int)
        assert label_id in expected_label_ids
        seen_labels.add(label_id)
        # Meta has all the documented fields (M2 §2.7 contract)
        meta = sample["meta.json"]
        assert isinstance(meta, dict)
        for key in ("start_time", "pcap_source", "label", "label_id",
                    "dominant_attack_ratio", "n_unmatched"):
            assert key in meta, f"meta missing {key!r}: {meta}"
        assert meta["label_id"] == label_id

    # We should have emitted multiple windows
    assert n_seen >= 3
    # And both BENIGN and at least one attack should be represented
    assert LABEL_TO_ID_RAW["BENIGN"] in seen_labels
    assert any(lid != 0 for lid in seen_labels), \
        f"expected at least one attack window, got only {seen_labels}"


@pytest.mark.slow
def test_manifest_parquet_has_per_shard_rows(synthetic_dataset) -> None:
    pcaps, csvs, out_dir = synthetic_dataset
    cfg = load_config(project_root() / "configs" / "base.yaml")
    stats = run_etl(pcaps, csvs, out_dir, cfg.data, samples_per_shard=4,
                    csv_dayfirst=False, csv_tz="UTC")

    table = pq.read_table(out_dir / "manifest.parquet")
    assert table.num_rows == stats.n_shards
    df = table.to_pandas()
    for col in ("shard_idx", "shard_name", "n_samples", "labels_json"):
        assert col in df.columns

    # n_samples sum across shards should match total windows
    assert int(df["n_samples"].sum()) == stats.n_windows_emitted

    # Each labels_json is parseable and consistent with n_samples
    for _, row in df.iterrows():
        d = json.loads(row["labels_json"])
        assert sum(d.values()) == row["n_samples"]


@pytest.mark.slow
def test_limit_windows_respected(synthetic_dataset) -> None:
    pcaps, csvs, out_dir = synthetic_dataset
    cfg = load_config(project_root() / "configs" / "base.yaml")
    stats = run_etl(pcaps, csvs, out_dir, cfg.data, samples_per_shard=4,
                    csv_dayfirst=False, csv_tz="UTC", limit_windows=3)
    assert stats.n_windows_emitted <= 3


def test_load_combined_manifest_concats_all_subdirs(tmp_path: Path) -> None:
    """load_combined_manifest finds parquet files at any depth and adds source_dir."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Multi-worker layout: two pcap subdirs each with a manifest
    for stem, n in [("a", 4), ("b", 7)]:
        sub = tmp_path / stem
        sub.mkdir()
        rows = [{"shard_idx": 0, "shard_name": f"shard-{stem}.tar",
                 "n_samples": n, "labels_json": "{}"}]
        pq.write_table(pa.Table.from_pylist(rows), sub / "manifest.parquet")

    df = load_combined_manifest(tmp_path)
    assert len(df) == 2
    assert set(df["source_dir"]) == {"a", "b"}
    assert int(df["n_samples"].sum()) == 11


def test_load_combined_manifest_single_top_level(tmp_path: Path) -> None:
    """Single-process layout: a single top-level manifest.parquet, source_dir = '.'."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [{"shard_idx": 0, "shard_name": "shard-000000.tar",
             "n_samples": 13, "labels_json": "{}"}]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "manifest.parquet")

    df = load_combined_manifest(tmp_path)
    assert len(df) == 1
    assert df["source_dir"].iloc[0] == "."


def test_load_combined_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_combined_manifest(tmp_path)


@pytest.mark.slow
def test_failed_pcap_is_logged_not_raised(tmp_path) -> None:
    """A corrupt pcap should be reported in stats.n_pcaps_failed; the run continues."""
    # One bogus file, one real
    bogus = tmp_path / "bogus.pcap"
    bogus.write_bytes(b"not a real pcap header")
    src_ips_a = [f"10.1.0.{i}" for i in range(32)]
    real = tmp_path / "real.pcap"
    _build_pcap_with_packets(real, 3.0, 100, src_ips_a)

    csv = tmp_path / "labels.csv"
    _write_csv(csv, [
        (sip, 12345, "10.0.0.99", 80, 6,
         _iso_offset(0.0), int(3.5 * 1e6), "BENIGN")
        for sip in src_ips_a
    ])

    cfg = load_config(project_root() / "configs" / "base.yaml")
    stats = run_etl([bogus, real], [csv], tmp_path / "out", cfg.data,
                    samples_per_shard=4, csv_dayfirst=False, csv_tz="UTC")

    # Bogus pcap should fail or yield zero packets; real pcap should produce windows
    assert stats.n_windows_emitted > 0
    # Either it counts as failed (parse error during reader open) or simply yields
    # 0 packets. Both are acceptable; what's NOT acceptable is the run crashing.
    assert stats.n_pcaps_processed + stats.n_pcaps_failed == 2
