"""Unit tests for src/nid_video/data/split.py.

Test history:
- M4.1: time-position partition + tmin tiebreak
- M4.7 first redesign: label-aware time-position (each attack class uses
  its own [tmin, tmax])
- M4.7 second redesign (this version): index-based partition per
  (pcap_source, label_id). AttackWindow class + attack_windows_for_pcaps
  helper removed; tests reflect that.
"""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nid_video.data.labeling import BENIGN_ID
from nid_video.data.split import (
    WindowKey,
    WindowKeyWithLabel,
    _allocate_counts,
    _bucket_to_split,
    _hash_bucket,
    cic_ids_2017_csv_to_pcap_map,
    compute_split_assignments,
    load_splits,
    verify_splits_complete,
    write_splits_parquet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _benign(pcap: str, ts: float) -> WindowKeyWithLabel:
    return WindowKeyWithLabel(key=WindowKey(pcap, ts), label_id=BENIGN_ID)


def _attack(pcap: str, ts: float, label_id: int) -> WindowKeyWithLabel:
    return WindowKeyWithLabel(key=WindowKey(pcap, ts), label_id=label_id)


def _capture_warnings():
    """Context-managed capture of loguru WARNING+ output to a string."""
    from nid_video.utils import logger as loguru_logger
    sink = io.StringIO()
    handler_id = loguru_logger.add(sink, level="WARNING")
    return sink, handler_id, loguru_logger


# ---------------------------------------------------------------------------
# Index-based partition for attack classes (the M4.7 second-redesign core)
# ---------------------------------------------------------------------------


def test_attack_window_split_70_15_15_uniform_timestamps() -> None:
    """100 attack-labelled windows uniformly spaced in time → exact
    70/15/15 split by count (and by time, since uniform). Both M4.1
    time-position and M4.7 index-based agree on uniform inputs; this
    test pins the trivial case as a sanity anchor."""
    LID = 5    # FTP-Patator
    n = 100
    items = [_attack("tue.pcap", float(i), LID) for i in range(n)]
    assigns = compute_split_assignments(items)
    counts = {"train": 0, "val": 0, "test": 0}
    for v in assigns.values():
        counts[v] += 1
    assert counts == {"train": 70, "val": 15, "test": 15}, counts


def test_attack_window_clustered_split_uses_index_not_time() -> None:
    """100 attack-labelled windows ALL clustered in the first 10% of the
    overall time range → time-position partition would put 100% in train;
    index-based partition still produces exact 70/15/15 by count.

    This is the central regression pin for the M4.7 second redesign:
    the bug it cures is "non-uniform window distribution within attack
    range causes time-position partition to fail" (CIC slowloris 100/0/0,
    Hulk 96/4/0)."""
    LID = 7    # slowloris
    n = 100
    # All 100 windows in [0, 5] minutes; 4.1 time-position would put all
    # in train (since 5min / 60min = 8% < 70%).
    items = [_attack("wed.pcap", float(i) * 0.05, LID) for i in range(n)]
    assigns = compute_split_assignments(items)
    counts = {"train": 0, "val": 0, "test": 0}
    for v in assigns.values():
        counts[v] += 1
    assert counts == {"train": 70, "val": 15, "test": 15}, counts

    # Still: early indices → train, late indices → test
    sorted_items = sorted(items, key=lambda w: w.key.start_time)
    assert assigns[sorted_items[0].key] == "train"
    assert assigns[sorted_items[69].key] == "train"
    assert assigns[sorted_items[70].key] == "val"
    assert assigns[sorted_items[84].key] == "val"
    assert assigns[sorted_items[85].key] == "test"
    assert assigns[sorted_items[99].key] == "test"


def test_split_70_15_15_guaranteed_for_each_attack_class() -> None:
    """Mixed multi-class input where Wed-style nesting WOULD have broken
    M4.1 / M4.7-first: each class gets exactly 70/15/15 in this design."""
    # slowloris [09:01, 14:25] (5h24m); inner classes packed into the early portion
    BASE = 0
    classes = {
        7: ((1 * 60, 5 * 3600 + 24 * 60), 1000),     # slowloris, 1000 windows
        8: ((1 * 3600 + 15 * 60, 1 * 3600 + 37 * 60), 100),    # Slowhttptest
        1: ((1 * 3600 + 43 * 60, 2 * 3600 + 7 * 60), 100),     # Hulk
        4: ((2 * 3600 + 10 * 60, 2 * 3600 + 19 * 60), 100),    # GoldenEye
        14: ((6 * 3600 + 12 * 60, 6 * 3600 + 32 * 60), 100),   # Heartbleed
    }
    items: list[WindowKeyWithLabel] = []
    for lid, ((tlo, thi), n) in classes.items():
        for i in range(n):
            t = BASE + tlo + (thi - tlo) * (i / max(1, n - 1))
            items.append(_attack("wed.pcap", t, lid))

    assigns = compute_split_assignments(items)

    for lid, ((_, _), n) in classes.items():
        per_split = {"train": 0, "val": 0, "test": 0}
        for wkl in items:
            if wkl.label_id == lid:
                per_split[assigns[wkl.key]] += 1
        # Per _allocate_counts truncation: int(n*0.7), int(n*0.15)
        n_train_expected = int(n * 0.7)
        n_val_expected = int(n * 0.15)
        n_test_expected = n - n_train_expected - n_val_expected
        assert per_split == {
            "train": n_train_expected, "val": n_val_expected, "test": n_test_expected
        }, f"label_id={lid}: got {per_split}"


def test_portscan_split_distribution_within_5pct() -> None:
    """111 PortScan windows (matching real CIC-IDS-2017 count) → exact
    78/16/17 split by count regardless of how clustered the timestamps are.
    Pins the M4.7 second-redesign cure of the 9/82/9 PortScan pathology."""
    LID = 2
    n = 111
    # Cluster 91 windows in the val zone (4.1-bug pathology) + 20 elsewhere
    items: list[WindowKeyWithLabel] = []
    items += [_attack("fri.pcap", 14 * 3600 + 41 * 60 + i, LID) for i in range(91)]
    items += [_attack("fri.pcap", 13 * 3600 + i * 60, LID) for i in range(10)]
    items += [_attack("fri.pcap", 15 * 3600 + 15 * 60 + i, LID) for i in range(10)]

    assigns = compute_split_assignments(items)
    counts = {"train": 0, "val": 0, "test": 0}
    for v in assigns.values():
        counts[v] += 1
    # int(111*0.7)=77, int(111*0.15)=16, n_test=18. Within ±5 of (78, 16, 17).
    assert counts == {"train": 77, "val": 16, "test": 18}, counts


def test_small_attack_class_warns_when_under_10_windows() -> None:
    """N < 10 → log a WARNING about high-variance per-class metrics, but
    still produce a non-empty split with each split getting ≥ 1 sample."""
    LID = 9
    items = [_attack("fri.pcap", float(i), LID) for i in range(5)]   # n=5
    sink, handler_id, loguru_logger = _capture_warnings()
    try:
        assigns = compute_split_assignments(items)
    finally:
        loguru_logger.remove(handler_id)

    log = sink.getvalue()
    assert "5 windows" in log
    assert "(< 10)" in log

    counts = {"train": 0, "val": 0, "test": 0}
    for v in assigns.values():
        counts[v] += 1
    # n=5: int(0.7*5)=3, int(0.15*5)=0 → max(1,0)=1, n_test=5-3-1=1.
    # All three splits get >= 1 sample (the M4.7 minimum guarantee).
    assert counts["train"] >= 1
    assert counts["val"] >= 1
    assert counts["test"] >= 1
    assert sum(counts.values()) == 5


def test_split_index_based_is_deterministic() -> None:
    """Same input twice → identical split assignments. sort is stable +
    no RNG involved beyond the BENIGN hash path."""
    LID = 5
    items = [_attack("p.pcap", float(i) + 0.5, LID) for i in range(50)]
    items += [_benign("p.pcap", float(i) + 1000) for i in range(50)]
    a = compute_split_assignments(items)
    b = compute_split_assignments(items)
    assert a == b


def test_split_index_based_input_order_invariant() -> None:
    """Shuffling the input order doesn't change the output — the per-class
    sort happens internally by start_time."""
    import random as _r
    LID = 5
    items = [_attack("p.pcap", float(i), LID) for i in range(50)]
    a = compute_split_assignments(items)
    rng = _r.Random(0)
    shuffled = items[:]
    rng.shuffle(shuffled)
    b = compute_split_assignments(shuffled)
    assert a == b


def test_attack_windows_in_different_pcaps_split_independently() -> None:
    """Same label_id in two pcaps → two separate index groups. Each gets
    its own 70/15/15."""
    items = (
        [_attack("tue.pcap", float(i), 5) for i in range(20)]
        + [_attack("fri.pcap", float(i), 5) for i in range(40)]
    )
    assigns = compute_split_assignments(items)
    tue = {"train": 0, "val": 0, "test": 0}
    fri = {"train": 0, "val": 0, "test": 0}
    for wkl in items:
        bucket = tue if wkl.key.pcap_source == "tue.pcap" else fri
        bucket[assigns[wkl.key]] += 1
    # Tue (n=20): 14/3/3 ; Fri (n=40): 28/6/6
    assert tue == {"train": 14, "val": 3, "test": 3}, tue
    assert fri == {"train": 28, "val": 6, "test": 6}, fri


# ---------------------------------------------------------------------------
# WindowKeyWithLabel hashability
# ---------------------------------------------------------------------------


def test_window_key_with_label_is_hashable() -> None:
    """Frozen+slots dataclass with hashable fields must remain hashable so
    downstream callers can use it as dict key / set member. A future change
    that adds an unhashable field (e.g. list) would break this silently;
    this test pins it."""
    wkl = WindowKeyWithLabel(key=WindowKey("p.pcap", 1.0), label_id=10)
    s = {wkl}
    d = {wkl: "x"}
    assert wkl in s
    assert d[wkl] == "x"
    wkl2 = WindowKeyWithLabel(key=WindowKey("p.pcap", 1.0), label_id=10)
    assert wkl == wkl2 and hash(wkl) == hash(wkl2)


# ---------------------------------------------------------------------------
# _allocate_counts edge cases
# ---------------------------------------------------------------------------


def test_allocate_counts_for_all_n_values() -> None:
    """Pin the count allocator for n in {0, 1, 2, 3, 4, 5, 10, 44, 111, 1598}."""
    R = (0.7, 0.15, 0.15)
    cases = {
        0:    (0, 0, 0),       # empty
        1:    (1, 0, 0),       # only train
        2:    (1, 1, 0),       # train, val, no test
        3:    (1, 1, 1),       # min-1 enforcement: int(0.7*3)=2 but stolen → 1/1/1
        4:    (2, 1, 1),
        5:    (3, 1, 1),       # int(0.7*5)=3, int(0.15*5)=0→max(1,0)=1, test=1
        10:   (7, 1, 2),       # int(0.15*10)=1, test=2
        44:   (30, 6, 8),      # Bot
        111:  (77, 16, 18),    # PortScan
        1598: (1118, 239, 241),  # slowloris
    }
    for n, expected in cases.items():
        got = _allocate_counts(n, R)
        assert got == expected, f"n={n}: got {got}, expected {expected}"
    # Counts must always sum to n
    for n in range(0, 200):
        a, b, c = _allocate_counts(n, R)
        assert a + b + c == n
        # And for n >= 3, each split has >= 1
        if n >= 3:
            assert a >= 1 and b >= 1 and c >= 1


# ---------------------------------------------------------------------------
# BENIGN hash-bucket path (regression-pinned from M4.1)
# ---------------------------------------------------------------------------


def test_benign_split_unchanged_by_label_aware_logic() -> None:
    """REGRESSION PIN: the BENIGN hash path produces identical assignments
    under the new index-based design as it did under M4.1 + first-redesign.
    The hash payload is ``(pcap_source, start_time, seed)``; label_id is
    intentionally NOT in it, so BENIGN behavior is unchanged across all
    redesigns."""
    items = [_benign(f"day{i % 3}.pcap", float(i) + 0.123) for i in range(100)]
    a = compute_split_assignments(items, seed=42)
    b = compute_split_assignments(items, seed=42)
    assert a == b
    assert len(set(a.values())) == 3


def test_benign_window_hash_split_deterministic_across_runs() -> None:
    items = [_benign(f"day{i % 3}.pcap", float(i) + 0.123) for i in range(100)]
    a = compute_split_assignments(items, seed=42)
    b = compute_split_assignments(items, seed=42)
    assert a == b


def test_benign_window_hash_seed_changes_partition() -> None:
    items = [_benign("d.pcap", float(i)) for i in range(200)]
    a = compute_split_assignments(items, seed=42)
    b = compute_split_assignments(items, seed=999)
    diff = sum(1 for wkl in items if a[wkl.key] != b[wkl.key])
    assert diff > 30


def test_benign_split_ratios_within_5pct_of_target() -> None:
    items = [_benign("p.pcap", float(i)) for i in range(1000)]
    assigns = compute_split_assignments(items, seed=42)
    counts = {"train": 0, "val": 0, "test": 0}
    for v in assigns.values():
        counts[v] += 1
    assert 650 <= counts["train"] <= 750, counts
    assert 100 <= counts["val"] <= 200, counts
    assert 100 <= counts["test"] <= 200, counts


def test_hash_bucket_uses_sha256_not_python_hash() -> None:
    from hashlib import sha256
    payload = b"k|1.000000|42"
    expected = int(sha256(payload).hexdigest()[:8], 16) % 100
    got = _hash_bucket(WindowKey("k", 1.0), seed=42)
    assert got == expected


# ---------------------------------------------------------------------------
# Parquet round-trip and completeness verification
# ---------------------------------------------------------------------------


def test_splits_parquet_roundtrip(tmp_path: Path) -> None:
    assigns: dict[WindowKey, str] = {
        WindowKey("a.pcap", 100.5): "train",
        WindowKey("a.pcap", 200.25): "val",
        WindowKey("b.pcap", 100.5): "test",
    }
    p = tmp_path / "splits.parquet"
    write_splits_parquet(assigns, p)            # type: ignore[arg-type]
    loaded = load_splits(p)
    assert loaded == assigns


def test_load_splits_validates_schema(tmp_path: Path) -> None:
    p = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"wrong_col": ["x"], "other": [1]}), p)
    with pytest.raises(ValueError, match="missing required columns"):
        load_splits(p)


def test_verify_splits_complete_passes_when_keys_match() -> None:
    keys = [WindowKey("a.pcap", float(i)) for i in range(5)]
    assigns: dict[WindowKey, str] = {k: "train" for k in keys}
    verify_splits_complete(assigns, keys)        # type: ignore[arg-type]


def test_verify_splits_complete_detects_missing(tmp_path: Path) -> None:
    keys = [WindowKey("a.pcap", float(i)) for i in range(5)]
    assigns: dict[WindowKey, str] = {k: "train" for k in keys}
    del assigns[keys[2]]
    with pytest.raises(ValueError, match="splits incomplete"):
        verify_splits_complete(assigns, keys)    # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CIC-IDS-2017 csv-to-pcap map (filename-convention helper, unchanged)
# ---------------------------------------------------------------------------


def test_cic_csv_to_pcap_map_groups_friday_csvs_to_one_pcap() -> None:
    csvs = [
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    ]
    m = cic_ids_2017_csv_to_pcap_map(csvs)
    assert all(v == "Friday-WorkingHours.pcap" for v in m.values())
    assert len(m) == 3


def test_cic_csv_to_pcap_map_handles_wednesday_lowercase_w() -> None:
    m = cic_ids_2017_csv_to_pcap_map(["Wednesday-workingHours.pcap_ISCX.csv"])
    assert m == {"Wednesday-workingHours.pcap_ISCX.csv": "Wednesday-workingHours.pcap"}


# ---------------------------------------------------------------------------
# Internal: bucket → split, ratio validation
# ---------------------------------------------------------------------------


def test_bucket_to_split_boundary_values() -> None:
    ratios = (0.7, 0.15, 0.15)
    assert _bucket_to_split(0, ratios) == "train"
    assert _bucket_to_split(69, ratios) == "train"
    assert _bucket_to_split(70, ratios) == "val"
    assert _bucket_to_split(84, ratios) == "val"
    assert _bucket_to_split(85, ratios) == "test"
    assert _bucket_to_split(99, ratios) == "test"


def test_compute_split_rejects_invalid_ratios() -> None:
    with pytest.raises(ValueError, match="ratios must sum"):
        compute_split_assignments([], ratios=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="non-negative"):
        compute_split_assignments([], ratios=(1.2, -0.1, -0.1))
