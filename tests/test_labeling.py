"""Tests for CIC-IDS-2017 label alignment + the CSV space-bug fix."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from nid_video.data.labeling import (
    BENIGN_ID,
    ID_TO_LABEL,
    LABEL_TO_ID,
    LABEL_TO_ID_COLLAPSED,
    LABEL_TO_ID_RAW,
    LabelIndex,
    WindowLabel,
    _load_label_csv,
    collapse_to_13,
    label_window,
    normalize_label_name,
    warn_low_population_classes,
)
from nid_video.data.pcap_parser import PacketRecord
from nid_video.data.windowing import Frame, Window


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pkt(
    *,
    ts: float,
    src_ip: str = "10.0.0.1",
    src_port: int = 12345,
    dst_ip: str = "10.0.0.99",
    dst_port: int = 80,
    proto: int = 6,
) -> PacketRecord:
    return PacketRecord(
        timestamp=ts, src_ip=src_ip, dst_ip=dst_ip,
        src_port=src_port, dst_port=dst_port, protocol=proto,
        pkt_size=100, tcp_flags=0x02, payload_len=60, direction=0,
    )


def _make_window(packets: list[PacketRecord]) -> Window:
    """One bin everything into frame 0; pad to T=16 with empties."""
    if packets:
        start = min(p.timestamp for p in packets)
    else:
        start = 0.0
    frames = [
        Frame(start_time=start + i * 0.1, end_time=start + (i + 1) * 0.1,
              packets=packets if i == 0 else [])
        for i in range(16)
    ]
    return Window(start_time=start, frames=frames, pcap_source="t")


def _write_csv(path: Path, rows: list[tuple], with_space_bug: bool = True) -> None:
    """Write a synthetic CIC-IDS-style CSV.

    rows: list of (sip, sport, dip, dport, proto, ts_iso, dur_us, label).
    with_space_bug: if True, prefix every column name with a literal space
                    (mimicking real CIC-IDS CSVs).
    """
    cols = ["Source IP", "Source Port", "Destination IP", "Destination Port",
            "Protocol", "Timestamp", "Flow Duration", "Label"]
    if with_space_bug:
        cols = [" " + c for c in cols]
    header = ",".join(cols)
    body = "\n".join(",".join(str(v) for v in r) for r in rows)
    path.write_text(header + "\n" + body + "\n", encoding="latin-1")


def _iso(year: int = 2017, month: int = 7, day: int = 5,
         hour: int = 9, minute: int = 0, second: int = 0) -> str:
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"


def _unix(year: int = 2017, month: int = 7, day: int = 5,
          hour: int = 9, minute: int = 0, second: int = 0) -> float:
    return dt.datetime(year, month, day, hour, minute, second,
                       tzinfo=dt.timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# CIC space-bug fix
# ---------------------------------------------------------------------------


def test_load_csv_strips_leading_space_in_column_names(tmp_path: Path) -> None:
    """The CIC space-bug: every column name has a leading space; we strip it."""
    csv = tmp_path / "fake.csv"
    _write_csv(csv, [("10.0.0.1", 12345, "10.0.0.2", 80, 6,
                      _iso(), 1_000_000, "BENIGN")], with_space_bug=True)
    df = _load_label_csv(csv)
    assert "Label" in df.columns
    assert " Label" not in df.columns
    for c in ("Source IP", "Source Port", "Destination IP",
              "Destination Port", "Protocol", "Timestamp", "Flow Duration"):
        assert c in df.columns, f"missing {c!r} after strip"


def test_load_csv_works_without_space_bug_too(tmp_path: Path) -> None:
    """Some redistribution copies don't have leading spaces — must still load."""
    csv = tmp_path / "fake.csv"
    _write_csv(csv, [("10.0.0.1", 12345, "10.0.0.2", 80, 6,
                      _iso(), 1_000_000, "BENIGN")], with_space_bug=False)
    df = _load_label_csv(csv)
    assert "Label" in df.columns


# ---------------------------------------------------------------------------
# Label name normalization
# ---------------------------------------------------------------------------


def test_normalize_label_strips_whitespace_and_canonicalizes_dash() -> None:
    assert normalize_label_name("BENIGN") == "BENIGN"
    assert normalize_label_name("  DDoS  ") == "DDoS"
    # ASCII hyphen variants → EN DASH
    assert normalize_label_name("Web Attack - Brute Force") == "Web Attack – Brute Force"
    assert normalize_label_name("Web Attack - XSS") == "Web Attack – XSS"
    # EN DASH already → unchanged
    assert normalize_label_name("Web Attack – XSS") == "Web Attack – XSS"


def test_label_table_id_zero_is_benign() -> None:
    assert LABEL_TO_ID["BENIGN"] == 0
    assert BENIGN_ID == 0
    assert ID_TO_LABEL[0] == "BENIGN"


# ---------------------------------------------------------------------------
# LabelIndex build + lookup
# ---------------------------------------------------------------------------


@pytest.fixture
def small_idx(tmp_path: Path) -> LabelIndex:
    csv = tmp_path / "labels.csv"
    _write_csv(csv, [
        ("10.0.0.1", 12345, "10.0.0.99", 80, 6, _iso(), 1_000_000, "DDoS"),
        ("10.0.0.2", 23456, "10.0.0.99", 80, 6, _iso(second=2), 500_000, "BENIGN"),
        ("10.0.0.3", 34567, "10.0.0.99", 22, 6, _iso(second=4), 750_000, "SSH-Patator"),
    ])
    return LabelIndex.from_csv(csv)


def test_label_index_indexes_all_flows(small_idx: LabelIndex) -> None:
    assert small_idx.n_flows == 3
    assert small_idx.n_keys == 3


def test_label_index_lookup_forward(small_idx: LabelIndex) -> None:
    ts = _unix() + 0.5  # within the first flow's 1.0s duration
    lid = small_idx.lookup("10.0.0.1", 12345, "10.0.0.99", 80, 6, ts)
    assert lid == LABEL_TO_ID["DDoS"]


def test_label_index_lookup_reverse_direction(small_idx: LabelIndex) -> None:
    ts = _unix() + 0.5
    lid = small_idx.lookup("10.0.0.99", 80, "10.0.0.1", 12345, 6, ts)
    assert lid == LABEL_TO_ID["DDoS"], "reverse-direction lookup must hit the same flow"


def test_label_index_lookup_outside_time_window(small_idx: LabelIndex) -> None:
    """Same 5-tuple but outside the labeled flow's [start, end] → no match."""
    ts = _unix() + 100.0
    lid = small_idx.lookup("10.0.0.1", 12345, "10.0.0.99", 80, 6, ts)
    assert lid is None


def test_label_index_lookup_unknown_5tuple(small_idx: LabelIndex) -> None:
    lid = small_idx.lookup("8.8.8.8", 53, "10.0.0.99", 80, 6, _unix())
    assert lid is None


# ---------------------------------------------------------------------------
# label_window
# ---------------------------------------------------------------------------


def test_label_window_all_benign(small_idx: LabelIndex) -> None:
    """Five packets all matching the BENIGN flow → label = BENIGN, ratio = 1.0."""
    ts0 = _unix(second=2) + 0.1   # inside the BENIGN flow's window
    pkts = [
        _pkt(ts=ts0 + i * 0.001,
             src_ip="10.0.0.2", src_port=23456,
             dst_ip="10.0.0.99", dst_port=80)
        for i in range(5)
    ]
    result = label_window(_make_window(pkts), small_idx)
    assert isinstance(result, WindowLabel)
    assert result.label == "BENIGN"
    assert result.label_id == 0
    assert result.dominant_ratio == 1.0
    assert result.n_unmatched == 0


def test_label_window_dominant_attack(small_idx: LabelIndex) -> None:
    """Mostly DDoS packets, no other attack → label = DDoS, ratio = 1.0."""
    ts0 = _unix() + 0.1
    pkts = [
        _pkt(ts=ts0 + i * 0.001,
             src_ip="10.0.0.1", src_port=12345,
             dst_ip="10.0.0.99", dst_port=80)
        for i in range(8)
    ]
    result = label_window(_make_window(pkts), small_idx)
    assert result.label == "DDoS"
    assert result.label_id == LABEL_TO_ID["DDoS"]
    assert result.dominant_ratio == 1.0
    assert result.n_unmatched == 0


def test_label_window_mixed_attacks_picks_dominant(tmp_path: Path) -> None:
    """3 DDoS + 1 SSH-Patator → DDoS wins, ratio = 0.75."""
    csv = tmp_path / "labels.csv"
    _write_csv(csv, [
        ("10.0.0.1", 12345, "10.0.0.99", 80, 6, _iso(), 5_000_000, "DDoS"),
        ("10.0.0.2", 23456, "10.0.0.99", 22, 6, _iso(), 5_000_000, "SSH-Patator"),
    ])
    idx = LabelIndex.from_csv(csv)

    ts0 = _unix() + 0.5
    pkts = [
        # 3 DDoS-flagged
        _pkt(ts=ts0, src_ip="10.0.0.1", src_port=12345,
             dst_ip="10.0.0.99", dst_port=80),
        _pkt(ts=ts0 + 0.001, src_ip="10.0.0.1", src_port=12345,
             dst_ip="10.0.0.99", dst_port=80),
        _pkt(ts=ts0 + 0.002, src_ip="10.0.0.1", src_port=12345,
             dst_ip="10.0.0.99", dst_port=80),
        # 1 SSH-Patator-flagged
        _pkt(ts=ts0 + 0.003, src_ip="10.0.0.2", src_port=23456,
             dst_ip="10.0.0.99", dst_port=22),
    ]
    result = label_window(_make_window(pkts), idx)
    assert result.label == "DDoS"
    assert result.dominant_ratio == pytest.approx(3 / 4)
    assert result.counts["DDoS"] == 3
    assert result.counts["SSH-Patator"] == 1


def test_label_window_unmatched_packet_defaults_to_benign(small_idx: LabelIndex) -> None:
    """A packet whose 5-tuple is not in the index counts as BENIGN, no crash."""
    pkts = [_pkt(ts=_unix(), src_ip="8.8.8.8", src_port=53,
                 dst_ip="10.0.0.99", dst_port=80)]
    result = label_window(_make_window(pkts), small_idx)
    assert result.label == "BENIGN"
    assert result.n_unmatched == 1


def test_label_window_empty_window_returns_benign(small_idx: LabelIndex) -> None:
    result = label_window(_make_window([]), small_idx)
    assert result.label == "BENIGN"
    assert result.n_unmatched == 0


def test_label_window_handles_dayfirst_timestamps(tmp_path: Path) -> None:
    """Real CIC-IDS uses '5/7/2017 9:00:00' = 5-July (DD/MM/YYYY).
    LabelIndex.from_csv must accept dayfirst=True for this to parse."""
    csv = tmp_path / "labels.csv"
    _write_csv(csv, [
        ("10.0.0.1", 12345, "10.0.0.99", 80, 6, "5/7/2017 9:00:00", 1_000_000, "DDoS"),
    ])
    idx = LabelIndex.from_csv(csv, dayfirst=True)
    assert idx.n_flows == 1

    ts = dt.datetime(2017, 7, 5, 9, 0, 0, tzinfo=dt.timezone.utc).timestamp() + 0.5
    lid = idx.lookup("10.0.0.1", 12345, "10.0.0.99", 80, 6, ts)
    assert lid == LABEL_TO_ID["DDoS"]


# ---------------------------------------------------------------------------
# EN-DASH foot-gun: ASCII-hyphen labels in a CSV must round-trip to the
# EN-DASH constant ID.   (M2 task 2.6 user requirement)
# ---------------------------------------------------------------------------


def test_csv_with_ascii_hyphen_web_attack_maps_to_en_dash_id(tmp_path: Path) -> None:
    """Some redistributed CIC CSVs replace the EN DASH (U+2013) with an ASCII
    hyphen. After load, the lookup must return the EN-DASH key's ID, NOT a
    fresh unknown-label fallback."""
    csv = tmp_path / "labels.csv"
    # Note the ASCII hyphen between "Attack" and "Brute"
    _write_csv(csv, [
        ("10.0.0.1", 12345, "10.0.0.99", 80, 6, _iso(),
         5_000_000, "Web Attack - Brute Force"),
    ])
    idx = LabelIndex.from_csv(csv)

    ts = _unix() + 0.5
    lid = idx.lookup("10.0.0.1", 12345, "10.0.0.99", 80, 6, ts)
    # Must match the EN-DASH constant, not be None or a new id
    assert lid == LABEL_TO_ID_RAW["Web Attack – Brute Force"] == 10


# ---------------------------------------------------------------------------
# 13-class collapse mapping
# ---------------------------------------------------------------------------


def test_collapse_to_13_web_attack_subtypes_merge_to_id_10() -> None:
    """Raw 10 / 11 / 12 (the three Web Attack subtypes) → collapsed 10."""
    assert collapse_to_13(10) == 10   # Brute Force
    assert collapse_to_13(11) == 10   # XSS
    assert collapse_to_13(12) == 10   # Sql Injection


def test_collapse_to_13_higher_ids_shift_down_by_two() -> None:
    """Raw 13 (Infiltration) → 11; raw 14 (Heartbleed) → 12."""
    assert collapse_to_13(13) == 11
    assert collapse_to_13(14) == 12


def test_collapse_to_13_low_ids_unchanged() -> None:
    """Raw 0..9 are stable across the collapse."""
    for raw in range(10):
        assert collapse_to_13(raw) == raw


def test_collapse_to_13_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        collapse_to_13(-1)
    with pytest.raises(ValueError):
        collapse_to_13(15)


def test_collapse_table_is_consistent_with_function() -> None:
    """The string→ID tables and the collapse function tell the same story."""
    # Web Attack subtypes from RAW must end at the same collapsed slot as
    # the explicit "Web Attack" entry in COLLAPSED.
    assert LABEL_TO_ID_COLLAPSED["Web Attack"] == 10
    for subtype in ("Web Attack – Brute Force",
                    "Web Attack – XSS",
                    "Web Attack – Sql Injection"):
        raw = LABEL_TO_ID_RAW[subtype]
        assert collapse_to_13(raw) == LABEL_TO_ID_COLLAPSED["Web Attack"]
    # Infiltration & Heartbleed cross-check
    assert collapse_to_13(LABEL_TO_ID_RAW["Infiltration"]) == LABEL_TO_ID_COLLAPSED["Infiltration"]
    assert collapse_to_13(LABEL_TO_ID_RAW["Heartbleed"]) == LABEL_TO_ID_COLLAPSED["Heartbleed"]
    # The two tables must have 15 vs 13 entries
    assert len(LABEL_TO_ID_RAW) == 15
    assert len(LABEL_TO_ID_COLLAPSED) == 13


# ---------------------------------------------------------------------------
# Class-imbalance warning (used by the ETL pipeline)
# ---------------------------------------------------------------------------


def test_warn_low_population_skips_benign_and_returns_low_classes(caplog) -> None:
    """warn_low_population_classes flags every non-BENIGN class with < min_samples;
    BENIGN is intentionally exempt (it's always abundant in CIC-IDS)."""
    counts = {
        "BENIGN": 100_000,
        "DoS Hulk": 200,
        "Heartbleed": 11,        # known CIC-IDS-2017 floor
        "Bot": 30,
    }
    low = warn_low_population_classes(counts, min_samples=50)
    assert "BENIGN" not in low
    assert "DoS Hulk" not in low
    assert set(low) == {"Heartbleed", "Bot"}


def test_label_to_id_alias_points_at_raw() -> None:
    """The default LABEL_TO_ID is the RAW table — keeps existing call sites stable."""
    assert LABEL_TO_ID is LABEL_TO_ID_RAW
