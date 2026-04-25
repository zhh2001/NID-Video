"""Unit tests for pcap_parser using scapy-built synthetic pcaps."""

from __future__ import annotations

from pathlib import Path

import pytest

from nid_video.data.pcap_parser import (
    PROTO_TCP,
    PROTO_UDP,
    PacketRecord,
    PacketStream,
    _direction,
    parse_pcap,
)


def _build_synth_pcap(out_path: Path) -> tuple[int, int, int]:
    """Write 10 TCP + 5 UDP + 2 malformed packets. Returns (n_tcp, n_udp, n_malformed).

    scapy is used only to build well-formed IP/TCP/UDP frames; the pcap file
    itself is written via dpkt so we can append malformed raw bytes without
    scapy refusing to serialize them.
    """
    import dpkt
    # scapy is heavy to import; lazy-load only when a test actually needs it.
    from scapy.all import IP, TCP, UDP, Ether, Raw

    base_ts = 1_000_000.0
    timestamped: list[tuple[float, bytes]] = []

    for i in range(10):
        p = (
            Ether(src="aa:bb:cc:dd:ee:01", dst="aa:bb:cc:dd:ee:02")
            / IP(src=f"10.0.0.{i + 1}", dst="10.0.0.99")
            / TCP(sport=12345 + i, dport=80, flags="S")
            / Raw(load=b"X" * 40)
        )
        timestamped.append((base_ts + i * 0.01, bytes(p)))

    for i in range(5):
        p = (
            Ether(src="aa:bb:cc:dd:ee:01", dst="aa:bb:cc:dd:ee:02")
            / IP(src=f"10.0.0.{20 + i}", dst="10.0.0.99")
            / UDP(sport=53000 + i, dport=53)
            / Raw(load=b"Y" * 30)
        )
        timestamped.append((base_ts + 1.0 + i * 0.01, bytes(p)))

    # 2 malformed: raw bytes too short to form an Ethernet header.
    timestamped.append((base_ts + 2.0, b"\x00\x01\x02\x03"))
    timestamped.append((base_ts + 2.1, b"\xff" * 5))

    with out_path.open("wb") as fh:
        writer = dpkt.pcap.Writer(fh, linktype=dpkt.pcap.DLT_EN10MB)
        for ts, buf in timestamped:
            writer.writepkt(buf, ts=ts)

    return 10, 5, 2


# --- direction heuristic ----------------------------------------------------


def test_direction_lower_port_is_server() -> None:
    assert _direction(12345, 80) == 0     # toward server
    assert _direction(80, 12345) == 1     # away from server
    assert _direction(443, 443) == 0      # equal port → default 0


# --- pcap parsing -----------------------------------------------------------


def test_parse_synthetic_pcap_yields_expected_records(tmp_path: Path) -> None:
    pcap = tmp_path / "synth.pcap"
    n_tcp, n_udp, n_mal = _build_synth_pcap(pcap)

    stream = PacketStream(pcap)
    records = list(stream)

    assert stream.stats["yielded"] == n_tcp + n_udp == 15
    assert stream.stats["malformed"] == n_mal == 2
    assert stream.stats["ipv6"] == 0
    assert stream.stats["non_tcpudp"] == 0
    assert all(isinstance(r, PacketRecord) for r in records)

    tcp = [r for r in records if r.protocol == PROTO_TCP]
    udp = [r for r in records if r.protocol == PROTO_UDP]
    assert len(tcp) == n_tcp
    assert len(udp) == n_udp


def test_parsed_fields_match_synth(tmp_path: Path) -> None:
    pcap = tmp_path / "synth.pcap"
    _build_synth_pcap(pcap)
    records = list(parse_pcap(pcap))
    tcp = [r for r in records if r.protocol == PROTO_TCP]
    udp = [r for r in records if r.protocol == PROTO_UDP]

    t0 = tcp[0]
    assert t0.dst_ip == "10.0.0.99"
    assert t0.src_ip == "10.0.0.1"
    assert t0.dst_port == 80
    assert t0.src_port == 12345
    assert t0.tcp_flags & 0x02       # SYN bit
    assert t0.direction == 0          # 80 < 12345 → toward server
    assert t0.payload_len == 40
    assert t0.timestamp == pytest.approx(1_000_000.0, abs=1e-3)

    u0 = udp[0]
    assert u0.dst_port == 53
    assert u0.src_port == 53000
    assert u0.tcp_flags == 0
    assert u0.payload_len == 30
    assert u0.direction == 0          # 53 < 53000 → toward server


def test_parse_pcap_helper_returns_iterator(tmp_path: Path) -> None:
    pcap = tmp_path / "synth.pcap"
    _build_synth_pcap(pcap)
    n = sum(1 for _ in parse_pcap(pcap))
    assert n == 15


def test_packet_stream_handles_only_malformed(tmp_path: Path) -> None:
    """A pcap with nothing but malformed records should yield nothing and not raise."""
    import dpkt

    pcap = tmp_path / "garbage.pcap"
    with pcap.open("wb") as fh:
        writer = dpkt.pcap.Writer(fh, linktype=dpkt.pcap.DLT_EN10MB)
        for i in range(7):
            writer.writepkt(b"\x00" * 3, ts=1000.0 + i)

    stream = PacketStream(pcap)
    assert list(stream) == []
    assert stream.stats["yielded"] == 0
    assert stream.stats["malformed"] == 7
