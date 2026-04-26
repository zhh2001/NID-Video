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


def _build_synth_pcap(out_path: Path, *, fmt: str = "pcap") -> tuple[int, int, int]:
    """Write 10 TCP + 5 UDP + 2 malformed packets. Returns (n_tcp, n_udp, n_malformed).

    scapy is used only to build well-formed IP/TCP/UDP frames; the pcap file
    itself is written via dpkt so we can append malformed raw bytes without
    scapy refusing to serialize them.

    ``fmt`` selects classic libpcap (``"pcap"``, default) or pcapng
    (``"pcapng"``, used to verify the pcapng dispatch path).
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
        if fmt == "pcapng":
            writer = dpkt.pcapng.Writer(fh, linktype=dpkt.pcap.DLT_EN10MB)
        elif fmt == "pcap":
            writer = dpkt.pcap.Writer(fh, linktype=dpkt.pcap.DLT_EN10MB)
        else:
            raise ValueError(f"unknown fmt={fmt!r}")
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


# --- pcapng + magic-byte dispatch -------------------------------------------


def test_packet_stream_parses_pcapng_same_as_classic(tmp_path: Path) -> None:
    """pcapng round-trip: same record fields as classic pcap (CIC-IDS-2017 ships pcapng)."""
    pcap_path = tmp_path / "synth.pcap"
    pcapng_path = tmp_path / "synth.pcapng"
    _build_synth_pcap(pcap_path, fmt="pcap")
    _build_synth_pcap(pcapng_path, fmt="pcapng")

    classic = list(PacketStream(pcap_path))
    new_fmt = list(PacketStream(pcapng_path))

    assert len(classic) == len(new_fmt) == 15
    # Same fields per packet (timestamps may differ in microsecond precision
    # between writer encodings, so compare everything except timestamp).
    for c, n in zip(classic, new_fmt):
        assert c._replace(timestamp=0.0) == n._replace(timestamp=0.0)


def test_packet_stream_unknown_magic_raises(tmp_path: Path) -> None:
    """A file with neither pcap nor pcapng magic must raise ValueError early —
    silently returning would mask real corruption (the M3 dry-run regression)."""
    bogus = tmp_path / "bogus.pcap"
    bogus.write_bytes(b"not a real pcap header")

    with pytest.raises(ValueError, match="unrecognized pcap magic"):
        list(PacketStream(bogus))


# --- truncated-L4 frames (Finding 4) ----------------------------------------


def _truncated_l4_frame(proto: int) -> bytes:
    """Build an Ethernet+IP frame where the L4 layer is sliced to 2 bytes.

    dpkt's IP parse succeeds (the IP header is well-formed) but it cannot
    construct a TCP/UDP header from 2 bytes, so ip.data falls back to raw
    bytes. Real CIC pcaps contain such truncated frames; the parser must
    skip them, not crash with AttributeError on l4.sport.
    """
    import struct
    ip_hdr = struct.pack(
        "!BBHHHBBHII",
        0x45,            # version + IHL=5
        0,               # TOS
        22,              # total length: 20 (IP) + 2 (truncated L4)
        1, 0,            # ID, flags+frag
        64, proto, 0,    # TTL, proto, checksum
        0x0a000001, 0x0a000063,  # src 10.0.0.1, dst 10.0.0.99
    )
    eth_hdr = struct.pack(
        "!6s6sH",
        b"\xaa\xbb\xcc\xdd\xee\x01",
        b"\xaa\xbb\xcc\xdd\xee\x02",
        0x0800,
    )
    return eth_hdr + ip_hdr + b"\x00\x50"   # 2 bytes ≪ TCP/UDP header size


def test_truncated_tcp_layer_is_counted_malformed_not_crashed(tmp_path: Path) -> None:
    """A frame whose IP parse succeeds but whose TCP layer is too short to
    parse must be counted as malformed and skipped — no AttributeError."""
    import dpkt
    pcap = tmp_path / "trunc.pcap"
    base_ts = 1_000_000.0
    with pcap.open("wb") as fh:
        w = dpkt.pcap.Writer(fh, linktype=dpkt.pcap.DLT_EN10MB)
        w.writepkt(_truncated_l4_frame(proto=6), ts=base_ts)

    stream = PacketStream(pcap)
    records = list(stream)                          # must not raise
    assert records == []
    assert stream.stats["yielded"] == 0
    assert stream.stats["malformed"] == 1


def test_truncated_udp_layer_is_counted_malformed_not_crashed(tmp_path: Path) -> None:
    """Same as the TCP case but for UDP — guard must cover both branches."""
    import dpkt
    pcap = tmp_path / "trunc_udp.pcap"
    with pcap.open("wb") as fh:
        w = dpkt.pcap.Writer(fh, linktype=dpkt.pcap.DLT_EN10MB)
        w.writepkt(_truncated_l4_frame(proto=17), ts=1_000_000.0)

    stream = PacketStream(pcap)
    records = list(stream)
    assert records == []
    assert stream.stats["yielded"] == 0
    assert stream.stats["malformed"] == 1


def test_well_formed_packet_after_truncated_still_parses(tmp_path: Path) -> None:
    """Truncated frames must NOT poison the iterator: well-formed packets
    that come after a truncated one must still be yielded. This is the real
    regression: in production, one bad frame crashed the whole pcap."""
    import dpkt
    from scapy.all import IP, TCP, Ether, Raw

    pcap = tmp_path / "mixed.pcap"
    base_ts = 1_000_000.0
    well_formed = bytes(
        Ether(src="aa:bb:cc:dd:ee:01", dst="aa:bb:cc:dd:ee:02")
        / IP(src="10.0.0.1", dst="10.0.0.99")
        / TCP(sport=12345, dport=80, flags="S")
        / Raw(load=b"X" * 40)
    )
    with pcap.open("wb") as fh:
        w = dpkt.pcap.Writer(fh, linktype=dpkt.pcap.DLT_EN10MB)
        w.writepkt(well_formed, ts=base_ts)                          # OK
        w.writepkt(_truncated_l4_frame(proto=6), ts=base_ts + 0.01)  # malformed
        w.writepkt(_truncated_l4_frame(proto=17), ts=base_ts + 0.02) # malformed
        w.writepkt(well_formed, ts=base_ts + 0.03)                   # OK

    stream = PacketStream(pcap)
    records = list(stream)
    assert len(records) == 2, "the two well-formed packets must still be yielded"
    assert stream.stats["yielded"] == 2
    assert stream.stats["malformed"] == 2
    assert records[0].timestamp == pytest.approx(base_ts, abs=1e-3)
    assert records[1].timestamp == pytest.approx(base_ts + 0.03, abs=1e-3)
