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
    # Synthetic timestamps in this file are UTC-naive on purpose; keep the
    # CSV interpretation aligned by passing csv_tz="UTC".
    return LabelIndex.from_csv(csv, csv_tz="UTC")


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
    idx = LabelIndex.from_csv(csv, csv_tz="UTC")

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
    idx = LabelIndex.from_csv(csv, dayfirst=True, csv_tz="UTC")
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
    idx = LabelIndex.from_csv(csv, csv_tz="UTC")

    ts = _unix() + 0.5
    lid = idx.lookup("10.0.0.1", 12345, "10.0.0.99", 80, 6, ts)
    # Must match the EN-DASH constant, not be None or a new id
    assert lid == LABEL_TO_ID_RAW["Web Attack – Brute Force"] == 10


def test_csv_drops_fully_empty_trailing_rows_with_distinct_warning(tmp_path: Path) -> None:
    """CIC's WebAttacks CSV ships with ~288k trailing all-empty rows (commas-only).
    These must be dropped up front so they aren't mis-reported as 'unparseable
    timestamps' and don't pollute the index. Discovered M3-to-M4 dry-run."""
    import io
    from nid_video.utils import logger as loguru_logger

    csv = tmp_path / "labels.csv"
    # Build the CSV at byte level so the trailing rows are exactly the
    # commas-only lines CIC's tool emits.
    header = (b" Source IP, Source Port, Destination IP, Destination Port,"
              b" Protocol, Timestamp, Flow Duration, Label\n")
    real_rows = (
        b"10.0.0.1,12345,10.0.0.99,80,6,2017-07-05 09:00:00,1000000,DDoS\n"
        b"10.0.0.2,23456,10.0.0.99,22,6,2017-07-05 09:00:01,1000000,SSH-Patator\n"
    )
    empty_rows = b",,,,,,,\n" * 5
    csv.write_bytes(header + real_rows + empty_rows)

    sink = io.StringIO()
    handler_id = loguru_logger.add(sink, level="WARNING")
    try:
        idx = LabelIndex.from_csv(csv, csv_tz="UTC")
    finally:
        loguru_logger.remove(handler_id)

    assert idx.n_flows == 2

    log = sink.getvalue()
    assert "5 fully-empty rows dropped" in log, log
    # No "unparseable timestamps" warning, because we dropped empties first.
    assert "unparseable timestamps" not in log, log


def test_csv_warns_separately_when_row_has_data_but_bad_timestamp(tmp_path: Path) -> None:
    """A row with real data but a malformed timestamp should still trigger
    the 'unparseable timestamps' warning (defensive — so future CSV format
    issues stay visible after the empty-row drop is in place)."""
    import io
    from nid_video.utils import logger as loguru_logger

    csv = tmp_path / "labels.csv"
    header = (b" Source IP, Source Port, Destination IP, Destination Port,"
              b" Protocol, Timestamp, Flow Duration, Label\n")
    body = (
        b"10.0.0.1,12345,10.0.0.99,80,6,2017-07-05 09:00:00,1000000,DDoS\n"
        b"10.0.0.2,23456,10.0.0.99,22,6,not-a-timestamp,1000000,SSH-Patator\n"
    )
    csv.write_bytes(header + body)

    sink = io.StringIO()
    handler_id = loguru_logger.add(sink, level="WARNING")
    try:
        idx = LabelIndex.from_csv(csv, csv_tz="UTC")
    finally:
        loguru_logger.remove(handler_id)

    assert idx.n_flows == 1   # only the parseable row makes it in
    log = sink.getvalue()
    assert "fully-empty" not in log, log     # nothing was fully empty
    assert "unparseable timestamps" in log, log
    assert "1 rows" in log, log


# ---------------------------------------------------------------------------
# Timezone handling (Finding 3c): wall-clock CSV → UTC unix epoch
# ---------------------------------------------------------------------------


def test_csv_tz_localizes_adt_summer_time_to_utc(tmp_path: Path) -> None:
    """CIC-IDS-2017 was captured 2017-07-{3..7} in Halifax (ADT, UTC-3 in DST).
    A CSV row reading '2017-07-06 09:00:00' must be stored as the unix epoch
    of 2017-07-06 12:00:00 UTC, not 09:00:00 UTC. The synthetic test fixtures
    that pre-date this fix used UTC strings, so they keep csv_tz='UTC'; real
    CIC uses csv_tz='America/Halifax' (the new default)."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    csv = tmp_path / "labels.csv"
    _write_csv(csv, [
        ("10.0.0.1", 12345, "10.0.0.99", 80, 6,
         "2017-07-06 09:00:00", 1_000_000, "DDoS"),
    ])
    idx = LabelIndex.from_csv(csv, csv_tz="America/Halifax")

    # Independent reference: 09:00 ADT = 12:00 UTC.
    expected_unix = dt.datetime(
        2017, 7, 6, 9, 0, 0, tzinfo=ZoneInfo("America/Halifax")
    ).timestamp()

    flows = next(iter(idx._index.values()))
    assert flows[0].start_ts == pytest.approx(expected_unix, abs=1.0)
    # And concretely: 12:00:00 UTC of the same day.
    expected_utc = dt.datetime(
        2017, 7, 6, 12, 0, 0, tzinfo=dt.timezone.utc
    ).timestamp()
    assert flows[0].start_ts == pytest.approx(expected_utc, abs=1.0)


def test_csv_tz_localizes_ast_winter_time_to_utc(tmp_path: Path) -> None:
    """DST regression: a January CIC-style timestamp must use AST (UTC-4),
    not ADT (UTC-3). CIC-IDS-2017 doesn't span winter, but cross-dataset users
    in M5 (e.g. UNSW-NB15 captures from January) need this correct.
    zoneinfo handles the DST transition by date — this test pins it down."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    csv = tmp_path / "labels.csv"
    _write_csv(csv, [
        ("10.0.0.1", 12345, "10.0.0.99", 80, 6,
         "2017-01-15 09:00:00", 1_000_000, "DDoS"),
    ])
    idx = LabelIndex.from_csv(csv, csv_tz="America/Halifax")

    # 09:00 AST = 13:00 UTC (UTC-4 in winter, NOT UTC-3).
    expected_utc = dt.datetime(
        2017, 1, 15, 13, 0, 0, tzinfo=dt.timezone.utc
    ).timestamp()
    expected_via_zoneinfo = dt.datetime(
        2017, 1, 15, 9, 0, 0, tzinfo=ZoneInfo("America/Halifax")
    ).timestamp()
    assert expected_utc == pytest.approx(expected_via_zoneinfo, abs=1.0)

    flows = next(iter(idx._index.values()))
    assert flows[0].start_ts == pytest.approx(expected_utc, abs=1.0)


def test_csv_tz_utc_passes_through_naive_timestamp(tmp_path: Path) -> None:
    """csv_tz='UTC' must give the same unix epoch as a naive UTC timestamp —
    this is what synthetic test fixtures rely on."""
    import datetime as dt

    csv = tmp_path / "labels.csv"
    _write_csv(csv, [
        ("10.0.0.1", 12345, "10.0.0.99", 80, 6,
         "2017-07-06 09:00:00", 1_000_000, "DDoS"),
    ])
    idx = LabelIndex.from_csv(csv, csv_tz="UTC")

    expected_utc = dt.datetime(
        2017, 7, 6, 9, 0, 0, tzinfo=dt.timezone.utc
    ).timestamp()
    flows = next(iter(idx._index.values()))
    assert flows[0].start_ts == pytest.approx(expected_utc, abs=1.0)


def test_csv_with_cp1252_endash_byte_maps_to_web_attack_subtypes(tmp_path: Path) -> None:
    """Real CIC WebAttacks CSVs store the EN DASH as the single CP-1252 byte
    0x96. Reading as latin-1 (the prior behaviour) silently dropped all three
    Web-Attack subtypes — they failed lookup against LABEL_TO_ID and fell back
    to BENIGN. Reading as cp1252 turns 0x96 into U+2013 and the keys match.

    Discovered in the M3-to-M4 dry-run: 2180 Thursday-morning Web Attack rows
    were silently mis-labelled BENIGN.
    """
    csv = tmp_path / "labels.csv"
    header = (b" Source IP, Source Port, Destination IP, Destination Port,"
              b" Protocol, Timestamp, Flow Duration, Label\n")
    # Each row's label cell contains the literal CP-1252 byte 0x96 between
    # "Attack" and the subtype, exactly as CIC ships them.
    body = (
        b"10.0.0.1,12345,10.0.0.99,80,6,2017-07-05 09:00:00,1000000,"
        b"Web Attack \x96 Brute Force\n"
        b"10.0.0.2,12346,10.0.0.99,80,6,2017-07-05 09:00:01,1000000,"
        b"Web Attack \x96 XSS\n"
        b"10.0.0.3,12347,10.0.0.99,80,6,2017-07-05 09:00:02,1000000,"
        b"Web Attack \x96 Sql Injection\n"
        b"10.0.0.4,12348,10.0.0.99,80,6,2017-07-05 09:00:03,1000000,BENIGN\n"
    )
    csv.write_bytes(header + body)

    idx = LabelIndex.from_csv(csv, csv_tz="UTC")

    ts0 = _unix(second=0) + 0.5
    ts1 = _unix(second=1) + 0.5
    ts2 = _unix(second=2) + 0.5
    ts3 = _unix(second=3) + 0.5
    assert idx.lookup("10.0.0.1", 12345, "10.0.0.99", 80, 6, ts0) == 10  # Brute Force
    assert idx.lookup("10.0.0.2", 12346, "10.0.0.99", 80, 6, ts1) == 11  # XSS
    assert idx.lookup("10.0.0.3", 12347, "10.0.0.99", 80, 6, ts2) == 12  # Sql Injection
    assert idx.lookup("10.0.0.4", 12348, "10.0.0.99", 80, 6, ts3) == 0   # BENIGN


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


def _make_single_row_csv(
    tmp_path: Path,
    timestamp_str: str,
    *,
    sip: str = "10.0.0.1", sport: int = 12345,
    dip: str = "10.0.0.99", dport: int = 80, proto: int = 6,
    duration_us: int = 1_000_000, label: str = "DDoS",
    name: str = "labels.csv",
) -> Path:
    """Build a one-row CSV containing the given timestamp string. Bytes-level
    write so the timestamp string is preserved exactly (no pandas reformatting)."""
    csv = tmp_path / name
    header = (b" Source IP, Source Port, Destination IP, Destination Port,"
              b" Protocol, Timestamp, Flow Duration, Label\n")
    line = (
        f"{sip},{sport},{dip},{dport},{proto},{timestamp_str},{duration_us},{label}\n"
    ).encode()
    csv.write_bytes(header + line)
    return csv


def _stored_unix_for(idx: LabelIndex, sip: str = "10.0.0.1",
                     sport: int = 12345, dip: str = "10.0.0.99",
                     dport: int = 80, proto: int = 6) -> float:
    """Pull back the stored start_ts for a single-flow LabelIndex."""
    flows = idx._index[(sip, sport, dip, dport, proto)]
    assert len(flows) == 1, f"expected exactly 1 flow, got {len(flows)}"
    return flows[0].start_ts


# ---------------------------------------------------------------------------
# 12h-without-AM/PM inference (M4 task 4.7, Finding M4-001)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hour", [1, 2, 3, 4, 5, 6, 7])
def test_hour_in_pm_range_gets_shifted_plus_twelve(tmp_path: Path, hour: int) -> None:
    """Hours 1..7 are CIC's afternoon (PM); inference must add 12h.

    Verifies via UTC unix offset: hour-shifted ADT vs unshifted ADT differ by
    exactly 12 hours."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    csv = _make_single_row_csv(tmp_path, f"7/7/2017 {hour}:30:00", name=f"h{hour}.csv")
    idx = LabelIndex.from_csv(csv, dayfirst=True, csv_tz="America/Halifax")
    got_unix = _stored_unix_for(idx)

    # Expected: shift +12 → hour+12 in 24h, then ADT→UTC
    expected_local = dt.datetime(
        2017, 7, 7, hour + 12, 30, 0, tzinfo=ZoneInfo("America/Halifax"),
    )
    expected_unix = expected_local.timestamp()
    assert got_unix == pytest.approx(expected_unix, abs=1.0), (
        f"hour={hour} (PM) should map to {hour+12}:30 ADT, "
        f"got unix={got_unix:.0f} vs expected {expected_unix:.0f}"
    )


@pytest.mark.parametrize("hour", [8, 9, 10, 11])
def test_hour_in_am_range_unchanged(tmp_path: Path, hour: int) -> None:
    """Hours 8..11 are CIC's morning (AM); inference leaves them alone."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    csv = _make_single_row_csv(tmp_path, f"7/7/2017 {hour}:30:00", name=f"h{hour}.csv")
    idx = LabelIndex.from_csv(csv, dayfirst=True, csv_tz="America/Halifax")
    got_unix = _stored_unix_for(idx)

    expected_local = dt.datetime(
        2017, 7, 7, hour, 30, 0, tzinfo=ZoneInfo("America/Halifax"),
    )
    expected_unix = expected_local.timestamp()
    assert got_unix == pytest.approx(expected_unix, abs=1.0), (
        f"hour={hour} (AM) must remain {hour}:30 ADT, "
        f"got unix={got_unix:.0f} vs expected {expected_unix:.0f}"
    )


def test_hour_12_unchanged_as_noon(tmp_path: Path) -> None:
    """12:xx in CIC is noon (12 PM = 12:00 24h), not midnight (12 AM = 0:00).
    Inference must leave 12:xx alone (hour 12 already represents 12:00 24h)."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    csv = _make_single_row_csv(tmp_path, "7/7/2017 12:30:00", name="h12.csv")
    idx = LabelIndex.from_csv(csv, dayfirst=True, csv_tz="America/Halifax")
    got_unix = _stored_unix_for(idx)

    expected_local = dt.datetime(
        2017, 7, 7, 12, 30, 0, tzinfo=ZoneInfo("America/Halifax"),
    )
    expected_unix = expected_local.timestamp()
    assert got_unix == pytest.approx(expected_unix, abs=1.0)


def test_hour_0_warns_but_unchanged(tmp_path: Path) -> None:
    """Hour 0 (midnight) is impossible per CIC working hours.
    Inference must warn AND leave the hour alone (we don't have a confident
    interpretation, so the safe default is to pass-through and surface)."""
    import datetime as dt
    import io
    from zoneinfo import ZoneInfo
    from nid_video.utils import logger as loguru_logger

    csv = _make_single_row_csv(tmp_path, "7/7/2017 0:30:00", name="h0.csv")
    sink = io.StringIO()
    handler_id = loguru_logger.add(sink, level="WARNING")
    try:
        idx = LabelIndex.from_csv(csv, dayfirst=True, csv_tz="America/Halifax")
    finally:
        loguru_logger.remove(handler_id)

    log = sink.getvalue()
    assert "hour=0" in log, log
    assert "midnight" in log.lower() or "working hours" in log.lower(), log

    got_unix = _stored_unix_for(idx)
    expected_local = dt.datetime(
        2017, 7, 7, 0, 30, 0, tzinfo=ZoneInfo("America/Halifax"),
    )
    assert got_unix == pytest.approx(expected_local.timestamp(), abs=1.0)


def test_inference_disabled_passes_through(tmp_path: Path) -> None:
    """csv_twelve_hour_pm_inference=False: even hour ∈ [1,7] is left alone.
    For datasets that don't share CIC's 12h-without-AM/PM quirk."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    # Hour 3 would normally shift to 15. With inference off, it stays 3.
    csv = _make_single_row_csv(tmp_path, "7/7/2017 3:30:00", name="h3_no_inf.csv")
    idx = LabelIndex.from_csv(
        csv, dayfirst=True, csv_tz="America/Halifax",
        csv_twelve_hour_pm_inference=False,
    )
    got_unix = _stored_unix_for(idx)

    expected_local = dt.datetime(
        2017, 7, 7, 3, 30, 0, tzinfo=ZoneInfo("America/Halifax"),
    )
    expected_unix = expected_local.timestamp()
    assert got_unix == pytest.approx(expected_unix, abs=1.0)


def test_real_csv_friday_afternoon_ddos_hour_range_after_fix(tmp_path: Path) -> None:
    """Synthetic CSV mimicking CIC Friday-Afternoon-DDos.csv pattern: every
    row has hour ∈ {3, 4} (the actual hour values seen in the real CIC file
    per the M4.7 diagnostic). After the 12h-inference fix, the stored
    UTC unix timestamps must correspond to ADT 15:xx-16:xx (the documented
    CIC DDoS attack window 15:56-16:16)."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    csv = tmp_path / "fake_friday_ddos.csv"
    header = (b" Source IP, Source Port, Destination IP, Destination Port,"
              b" Protocol, Timestamp, Flow Duration, Label\n")
    rows = b""
    # 4 rows with hour 3, 4 with hour 4 — matching the real CSV's pattern
    for i, h in enumerate([3, 3, 3, 3, 4, 4, 4, 4]):
        rows += (
            f"172.16.0.{i+1},5{i:04d},192.168.10.50,80,6,"
            f"7/7/2017 {h}:{(i*7)%60:02d}:00,1000000,DDoS\n"
        ).encode()
    csv.write_bytes(header + rows)

    idx = LabelIndex.from_csv(csv, dayfirst=True, csv_tz="America/Halifax")
    # Pull all unix timestamps and check ADT hour
    all_hours_adt = []
    for flows in idx._index.values():
        for f in flows:
            local = dt.datetime.fromtimestamp(f.start_ts, tz=dt.timezone.utc) \
                               .astimezone(ZoneInfo("America/Halifax"))
            all_hours_adt.append(local.hour)

    assert sorted(set(all_hours_adt)) == [15, 16], (
        f"After 12h fix, Friday-PM-DDoS hours should be in {{15, 16}}, "
        f"got {sorted(set(all_hours_adt))}"
    )
    assert all(h in (15, 16) for h in all_hours_adt)
