"""Tests for channels.py — covers all 4 design decisions Q1..Q4 explicitly."""

from __future__ import annotations

import numpy as np
import pytest

from nid_video.data.channels import (
    COMMON_PORTS,
    NUM_HOT_PORTS,
    NUM_LOG_BUCKETS_DEFAULT,
    TCP_FLAG_ACK,
    TCP_FLAG_FIN,
    TCP_FLAG_PSH,
    TCP_FLAG_RST,
    TCP_FLAG_SYN,
    TCP_FLAG_URG,
    TCP_FLAGS_MASK,
    ChannelConfig,
    _log_bucket_for_port,
    build_port_mapping,
    canonical_port,
    encode_window,
)
from nid_video.data.pcap_parser import PacketRecord
from nid_video.data.windowing import Frame, Window


def _pkt(
    *,
    ts: float = 0.05,
    src_ip: str = "10.0.0.1",
    dst_ip: str = "10.0.0.99",
    src_port: int = 12345,
    dst_port: int = 80,
    proto: int = 6,
    size: int = 100,
    flags: int = 0,
    payload: int = 60,
    direction: int = 0,
) -> PacketRecord:
    return PacketRecord(
        timestamp=ts, src_ip=src_ip, dst_ip=dst_ip,
        src_port=src_port, dst_port=dst_port, protocol=proto,
        pkt_size=size, tcp_flags=flags, payload_len=payload, direction=direction,
    )


def _make_window(frame_packets: list[list[PacketRecord]]) -> Window:
    """Build a Window from a list of per-frame packet lists. Pads to T=16 with empties."""
    while len(frame_packets) < 16:
        frame_packets.append([])
    frames = [
        Frame(start_time=i * 0.1, end_time=(i + 1) * 0.1, packets=frame_packets[i])
        for i in range(16)
    ]
    return Window(start_time=0.0, frames=frames, pcap_source="t")


# ---------------------------------------------------------------------------
# Port mapping (Q4)
# ---------------------------------------------------------------------------


def test_q4_log_bucket_in_range_for_all_ports() -> None:
    """Decision Q4: every port in [0, 65535] must map to col in [16, 63]
    when not a hot port. Verify the floor=16 and ceiling=63 contract."""
    port_map = build_port_mapping()
    hot_set = set(COMMON_PORTS)
    for port in (0, 1, 2, 3, 1024, 12345, 32767, 32768, 65535):
        if port in hot_set:
            continue
        col = port_map[port]
        assert 16 <= col <= 63, f"port={port} mapped to col={col} (must be in [16, 63])"


def test_q4_hot_ports_get_canonical_columns() -> None:
    """Decision Q4: COMMON_PORTS map to cols 0..15 in declared order."""
    port_map = build_port_mapping()
    for expected_col, port in enumerate(COMMON_PORTS):
        assert port_map[port] == expected_col, f"port {port} should be col {expected_col}"


def test_q4_log_bucket_formula_boundaries() -> None:
    """Decision Q4: int(log2(p+1)*3) clamped to [0, 47]."""
    assert _log_bucket_for_port(0) == 0          # log2(1)*3 = 0
    assert _log_bucket_for_port(1) == 3          # log2(2)*3 = 3
    assert _log_bucket_for_port(7) == 9          # log2(8)*3 = 9
    assert _log_bucket_for_port(65535) == 47     # int(log2(65536)*3)=48 -> clamp to 47


def test_q4_log_bucket_coverage_acceptable() -> None:
    """Decision Q4: most of the 48 log columns are populated. A handful of low-end
    buckets (those whose intervals on log2(p+1)*3 contain no integer) are
    mathematically unreachable; everything from ~bucket 6 onward is dense."""
    port_map = build_port_mapping()
    hot = set(COMMON_PORTS)
    used: set[int] = {col for port, col in port_map.items() if port not in hot}
    # 45+ of 48 columns should be used; the unreachable buckets are ~ {1, 2, 5}.
    assert len(used) >= 44, f"only {len(used)} of 48 log cols populated"
    assert 16 in used    # bucket 0 reachable via port 0
    assert 63 in used    # bucket 47 reachable via port 65535


# ---------------------------------------------------------------------------
# canonical_port (Q3)
# ---------------------------------------------------------------------------


def test_q3_canonical_port_returns_min() -> None:
    assert canonical_port(12345, 80) == 80
    assert canonical_port(80, 12345) == 80
    assert canonical_port(443, 443) == 443


def test_q3_bidirectional_flow_lands_in_same_column() -> None:
    """Decision Q3: forward and reverse legs of an HTTP-style flow share W column."""
    port_map = build_port_mapping()
    ip_map = {"10.0.0.1": 0, "10.0.0.99": 1}

    forward = _pkt(src_ip="10.0.0.1", dst_ip="10.0.0.99",
                   src_port=44321, dst_port=80, direction=0)
    reverse = _pkt(src_ip="10.0.0.99", dst_ip="10.0.0.1",
                   src_port=80, dst_port=44321, direction=1)
    window = _make_window([[forward, reverse]])

    out = encode_window(window, ip_map, port_map)
    # Ch 0 at frame 0 should have packet count > 0 in exactly the column for port 80 (= col 0),
    # split across rows 0 and 1.
    nonzero_cols = np.unique(np.where(out[0, 0] > 0)[1])
    assert nonzero_cols.tolist() == [0]   # both packets share W col 0
    # And both rows participated
    nonzero_rows = np.unique(np.where(out[0, 0] > 0)[0])
    assert sorted(nonzero_rows.tolist()) == [0, 1]


# ---------------------------------------------------------------------------
# Channel encoding basics
# ---------------------------------------------------------------------------


def test_shape_and_dtype_locked() -> None:
    out = encode_window(_make_window([]), {}, build_port_mapping())
    assert out.shape == (16, 6, 32, 64)
    assert out.dtype == np.float32


def test_empty_window_is_all_zero() -> None:
    out = encode_window(_make_window([]), {"10.0.0.1": 0}, build_port_mapping())
    assert np.all(out == 0.0)


def test_motion_channels_zero_at_frame_0() -> None:
    """Hard requirement: Ch 4 and Ch 5 must be 0 in frame 0 even if frame 0 has packets."""
    pkt = _pkt(flags=TCP_FLAG_SYN, direction=1)
    window = _make_window([[pkt]])
    out = encode_window(window, {"10.0.0.1": 0}, build_port_mapping())
    assert np.all(out[0, 4] == 0.0), "Ch 4 must be 0 at frame 0"
    assert np.all(out[0, 5] == 0.0), "Ch 5 must be 0 at frame 0"


def test_handcomputed_single_packet_cell() -> None:
    """One SYN packet of 100 bytes outbound to port 80, in frame 0; frame 1 empty.
    Verify each channel value at (frame=0, ip_row=0, port_col=0)."""
    pkt = _pkt(size=100, flags=TCP_FLAG_SYN, direction=0)
    window = _make_window([[pkt]])
    out = encode_window(window, {"10.0.0.1": 0}, build_port_mapping())

    cell = out[0, :, 0, 0]
    assert cell[0] == pytest.approx(np.log1p(1))            # 1 packet
    assert cell[1] == pytest.approx(np.log1p(100))          # 100 bytes
    assert cell[2] == pytest.approx(100.0)                  # mean size
    assert cell[3] == pytest.approx(TCP_FLAG_SYN / float(TCP_FLAGS_MASK))   # 2/63
    assert cell[4] == 0.0                                   # frame 0 motion
    assert cell[5] == 0.0

    # In frame 1 (no packets), Ch 5 should be log1p(0) - log1p(1) = -ln(2)
    assert out[1, 5, 0, 0] == pytest.approx(-np.log1p(1))


def test_tcp_flags_or_aggregation() -> None:
    """OR-aggregation: 3 packets with SYN, ACK, FIN → flag_or = SYN|ACK|FIN."""
    p1 = _pkt(flags=TCP_FLAG_SYN)
    p2 = _pkt(flags=TCP_FLAG_ACK)
    p3 = _pkt(flags=TCP_FLAG_FIN)
    window = _make_window([[p1, p2, p3]])
    out = encode_window(window, {"10.0.0.1": 0}, build_port_mapping())

    expected_or = TCP_FLAG_SYN | TCP_FLAG_ACK | TCP_FLAG_FIN  # 0x13 = 19
    assert out[0, 3, 0, 0] == pytest.approx(expected_or / float(TCP_FLAGS_MASK))


# ---------------------------------------------------------------------------
# Q1: PSH/URG flags must be honored (master prompt typo asked to be fixed)
# ---------------------------------------------------------------------------


def test_q1_psh_and_urg_flags_encoded() -> None:
    """Decision Q1: Ch 3 covers all 6 flags; PSH and URG must affect the value."""
    p_psh = _pkt(flags=TCP_FLAG_PSH | TCP_FLAG_ACK)
    window = _make_window([[p_psh]])
    out = encode_window(window, {"10.0.0.1": 0}, build_port_mapping())
    assert out[0, 3, 0, 0] == pytest.approx((TCP_FLAG_PSH | TCP_FLAG_ACK) / 63.0)

    # All 6 flags set → ch 3 = 1.0 (max)
    p_all = _pkt(flags=TCP_FLAGS_MASK)
    window = _make_window([[p_all]])
    out = encode_window(window, {"10.0.0.1": 0}, build_port_mapping())
    assert out[0, 3, 0, 0] == pytest.approx(1.0)

    # Bits outside the 6-flag mask (e.g. ECE/CWR=0x40/0x80) must NOT affect Ch 3
    p_extra = _pkt(flags=TCP_FLAG_URG | 0x80)   # URG + CWR
    window = _make_window([[p_extra]])
    out = encode_window(window, {"10.0.0.1": 0}, build_port_mapping())
    assert out[0, 3, 0, 0] == pytest.approx(TCP_FLAG_URG / 63.0)


# ---------------------------------------------------------------------------
# Q2: log-delta keeps Ch 5 bounded under bursty traffic
# ---------------------------------------------------------------------------


def test_q2_ddos_burst_ch5_stays_bounded() -> None:
    """Decision Q2: 10 → 10000 packets across two frames produces Ch 5 ≈ ln(10001/11) ≈ 6.8.
    Raw count delta (9990) would blow up the channel; log-delta keeps it < 10."""
    burst_a = [_pkt(flags=TCP_FLAG_SYN) for _ in range(10)]
    burst_b = [_pkt(flags=TCP_FLAG_SYN) for _ in range(10_000)]
    window = _make_window([burst_a, burst_b])
    out = encode_window(window, {"10.0.0.1": 0}, build_port_mapping())

    ch5_at_frame1 = out[1, 5, 0, 0]
    expected = np.log1p(10_000) - np.log1p(10)   # ≈ ln(10001/11) ≈ 6.81
    assert ch5_at_frame1 == pytest.approx(expected, rel=1e-6)
    assert ch5_at_frame1 < 10.0, "Ch 5 must stay well below 10 even under DDoS bursts"


# ---------------------------------------------------------------------------
# Configuration plumbing
# ---------------------------------------------------------------------------


def test_channel_config_defaults_match_design() -> None:
    cfg = ChannelConfig()
    assert cfg.num_channels == 6
    assert cfg.num_ip_buckets == 32
    assert cfg.num_port_buckets == 64


def test_channel_config_from_data_config() -> None:
    from nid_video.utils import load_config, project_root

    full = load_config(project_root() / "configs" / "base.yaml")
    cfg = ChannelConfig.from_data_config(full.data)
    assert cfg.num_channels == 6
    assert cfg.num_ip_buckets == 32
    assert cfg.num_port_buckets == 64


def test_constants_sanity() -> None:
    assert TCP_FLAGS_MASK == 0x3F == 63
    assert NUM_HOT_PORTS == 16
    assert NUM_LOG_BUCKETS_DEFAULT == 48
