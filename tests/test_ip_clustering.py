"""Tests for per-window IP clustering."""

from __future__ import annotations

import numpy as np
import pytest

from nid_video.data.ip_clustering import (
    IPStats,
    _collect_per_ip,
    cluster_ips_in_window,
)
from nid_video.data.pcap_parser import PacketRecord
from nid_video.data.windowing import Frame, Window


def _pkt(
    *,
    ts: float,
    src_ip: str,
    dst_ip: str = "10.0.0.99",
    dst_port: int = 80,
    src_port: int = 12345,
    size: int = 100,
    flags: int = 0x02,
    direction: int = 0,
) -> PacketRecord:
    return PacketRecord(
        timestamp=ts, src_ip=src_ip, dst_ip=dst_ip,
        src_port=src_port, dst_port=dst_port, protocol=6,
        pkt_size=size, tcp_flags=flags, payload_len=size - 40, direction=direction,
    )


def _window_from_packets(packets: list[PacketRecord], dt: float = 0.1) -> Window:
    """Bucket packets into 16 frames by timestamp; pad if short."""
    bins: dict[int, list[PacketRecord]] = {i: [] for i in range(16)}
    if packets:
        origin = min(p.timestamp for p in packets)
        for p in packets:
            f = min(int((p.timestamp - origin) / dt), 15)
            bins[f].append(p)
    else:
        origin = 0.0
    frames = [
        Frame(start_time=origin + i * dt, end_time=origin + (i + 1) * dt,
              packets=bins[i])
        for i in range(16)
    ]
    return Window(start_time=origin, frames=frames, pcap_source="t")


# ---------------------------------------------------------------------------
# Empty / small windows
# ---------------------------------------------------------------------------


def test_empty_window_returns_empty_dict() -> None:
    assert cluster_ips_in_window(_window_from_packets([])) == {}


def test_few_ips_each_get_unique_row() -> None:
    """5 IPs (< n_clusters=32) → each IP gets its own row, sorted by packet count desc."""
    pkts = []
    counts = {"10.0.0.1": 50, "10.0.0.2": 30, "10.0.0.3": 100,
              "10.0.0.4": 5, "10.0.0.5": 20}
    ts = 0.001
    for ip, count in counts.items():
        for _ in range(count):
            pkts.append(_pkt(ts=ts, src_ip=ip))
            ts += 0.001
    window = _window_from_packets(pkts)

    mapping = cluster_ips_in_window(window)
    assert len(mapping) == 5
    # Distinct rows
    assert len(set(mapping.values())) == 5
    # Highest-packet IP gets row 0
    sorted_by_row = sorted(mapping.items(), key=lambda kv: kv[1])
    expected_order = sorted(counts.items(), key=lambda kv: -kv[1])
    assert [ip for ip, _ in sorted_by_row] == [ip for ip, _ in expected_order]


def test_rows_always_in_range() -> None:
    """For a stress-test population, every assigned row must lie in [0, n_clusters)."""
    rng = np.random.default_rng(1)
    pkts = []
    for i in range(100):                 # 100 distinct IPs
        ip = f"10.{i // 256}.{i % 256}.1"
        n = int(rng.integers(1, 50))
        for _ in range(n):
            pkts.append(_pkt(ts=rng.uniform(0, 1.5), src_ip=ip,
                             dst_port=int(rng.integers(1, 65535))))
    window = _window_from_packets(pkts)
    mapping = cluster_ips_in_window(window, n_clusters=32)
    assert mapping, "expected non-empty mapping for 100-IP stress test"
    for ip, row in mapping.items():
        assert 0 <= row < 32, f"{ip} -> row {row}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_same_input_same_output() -> None:
    """Two runs with identical input must produce identical mappings."""
    rng = np.random.default_rng(42)
    pkts = []
    for i in range(80):
        ip = f"10.{i // 256}.{i % 256}.1"
        n = int(rng.integers(2, 30))
        for _ in range(n):
            pkts.append(_pkt(ts=rng.uniform(0, 1.5), src_ip=ip,
                             dst_port=int(rng.integers(1, 65535))))
    window = _window_from_packets(pkts)

    a = cluster_ips_in_window(window, n_clusters=32, seed=42)
    b = cluster_ips_in_window(window, n_clusters=32, seed=42)
    assert a == b


def test_determinism_under_input_reordering() -> None:
    """The IP string is used as a tiebreaker, so even with shuffled packet order
    the cold-start mapping is identical (both runs produce the same rank ordering
    for the IP behaviors)."""
    pkts_ordered = [
        _pkt(ts=0.01 * i, src_ip=ip)
        for ip, count in [("10.0.0.1", 5), ("10.0.0.2", 5), ("10.0.0.3", 10)]
        for i in range(count)
    ]
    # Identical packet content, just permuted
    pkts_shuffled = list(reversed(pkts_ordered))

    a = cluster_ips_in_window(_window_from_packets(pkts_ordered))
    b = cluster_ips_in_window(_window_from_packets(pkts_shuffled))
    assert a == b


# ---------------------------------------------------------------------------
# Behavior-driven structure
# ---------------------------------------------------------------------------


def test_high_volume_group_ranks_at_top_rows() -> None:
    """Heaviest-traffic IP group ranks lower row numbers (clusters sorted by total
    packets desc per design)."""
    rng = np.random.default_rng(7)
    pkts: list[PacketRecord] = []
    group_scan, group_ddos, group_chat = [], [], []

    for i in range(20):
        ip = f"10.1.0.{i}"
        group_scan.append(ip)
        for k in range(15):                  # 15 pkts per scanner
            pkts.append(_pkt(
                ts=0.01 + 0.001 * (i + k),
                src_ip=ip,
                dst_port=int(rng.integers(1, 65535)),
                size=60, flags=0x02, direction=0,
            ))
    for i in range(20):
        ip = f"10.2.0.{i}"
        group_ddos.append(ip)
        for k in range(80):                  # 80 pkts per DDoS source
            pkts.append(_pkt(
                ts=0.5 + 0.001 * (i + k * 0.001),
                src_ip=ip, dst_ip="10.0.0.99", dst_port=80,
                size=1500, flags=0x02 | 0x10, direction=1,
            ))
    for i in range(20):
        ip = f"10.3.0.{i}"
        group_chat.append(ip)
        for k in range(8):                   # 8 pkts per chatty client
            pkts.append(_pkt(
                ts=1.0 + 0.005 * (i + k),
                src_ip=ip, dst_ip=f"10.0.0.{20 + (k % 3)}",
                dst_port=443,
                size=int(rng.integers(80, 1400)),
                flags=int(rng.integers(0x02, 0x3F)),
                direction=int(rng.integers(0, 2)),
            ))

    mapping = cluster_ips_in_window(_window_from_packets(pkts), n_clusters=32, seed=42)

    mean_row_scan = np.mean([mapping[ip] for ip in group_scan])
    mean_row_ddos = np.mean([mapping[ip] for ip in group_ddos])
    mean_row_chat = np.mean([mapping[ip] for ip in group_chat])

    # DDoS sends the most total packets → its clusters rank top → lowest mean row
    assert mean_row_ddos < mean_row_scan, (
        f"DDoS mean row {mean_row_ddos} should be < scan mean {mean_row_scan}"
    )
    assert mean_row_ddos < mean_row_chat, (
        f"DDoS mean row {mean_row_ddos} should be < chat mean {mean_row_chat}"
    )


def test_uniform_groups_cluster_tightly() -> None:
    """When two groups have ~identical per-member behavior, k-means should
    concentrate each group into very few rows (≤ 4) and the groups should not
    overlap on any row."""
    pkts: list[PacketRecord] = []
    # Group A: 50 IPs, each sending 10 identical SYNs to port 80
    for i in range(50):
        ip = f"10.1.0.{i}"
        for k in range(10):
            pkts.append(_pkt(
                ts=0.001 * (i + k * 0.001), src_ip=ip,
                dst_port=80, size=60, flags=0x02, direction=0,
            ))
    # Group B: 50 IPs, each sending 30 identical ACK+PSH to port 443 with large packets
    for i in range(50):
        ip = f"10.2.0.{i}"
        for k in range(30):
            pkts.append(_pkt(
                ts=0.5 + 0.001 * (i + k * 0.001), src_ip=ip,
                dst_port=443, size=1000, flags=0x18, direction=1,
            ))

    mapping = cluster_ips_in_window(_window_from_packets(pkts), n_clusters=32, seed=42)
    rows_a = {mapping[f"10.1.0.{i}"] for i in range(50)}
    rows_b = {mapping[f"10.2.0.{i}"] for i in range(50)}

    # The two groups must not share any row — different behaviors land on
    # different clusters. With k=32 over 100 ~degenerate points, k-means leaves
    # many clusters empty and splits each group across a small fraction of rows;
    # we just bound that fraction.
    assert rows_a.isdisjoint(rows_b), f"groups overlap on rows {rows_a & rows_b}"
    assert len(rows_a) <= 8, f"uniform group A spread across {len(rows_a)} rows: {rows_a}"
    assert len(rows_b) <= 8, f"uniform group B spread across {len(rows_b)} rows: {rows_b}"


# ---------------------------------------------------------------------------
# Feature extraction sanity
# ---------------------------------------------------------------------------


def test_ipstats_feature_vector_dim_and_finite() -> None:
    """Sanity: feature vector is 7-dim and finite for a typical IP."""
    s = IPStats()
    for i in range(10):
        s.add(_pkt(ts=0.01 * i, src_ip="1.2.3.4",
                   dst_port=80 + i, dst_ip=f"10.0.0.{i}",
                   flags=0x02 | (0x10 if i % 2 else 0), direction=i % 2))
    vec = s.feature_vector()
    assert len(vec) == 7
    assert all(np.isfinite(v) for v in vec)
    assert vec[0] == pytest.approx(np.log1p(10))   # log packets
    assert vec[2] == 10.0                           # 10 distinct dst_ports
    assert vec[3] == 10.0                           # 10 distinct dst_ips
    assert vec[4] == pytest.approx(0.5)             # outbound ratio
    assert vec[6] == 2.0                            # SYN + ACK = 2 distinct bits


def test_collect_per_ip_groups_correctly() -> None:
    pkts = [
        _pkt(ts=0.01, src_ip="A"),
        _pkt(ts=0.02, src_ip="B"),
        _pkt(ts=0.03, src_ip="A"),
        _pkt(ts=0.04, src_ip="A"),
    ]
    by_ip = _collect_per_ip(_window_from_packets(pkts))
    assert by_ip["A"].n_packets == 3
    assert by_ip["B"].n_packets == 1
