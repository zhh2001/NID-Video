"""pcap → PacketRecord stream.

dpkt is the production parser (C-backed, hundreds of thousands of packets per
second). scapy is reserved for test-pcap *construction* and never imported here.
See M2 design decisions for the rationale.

Idea.md §3.1 (Stage 1 · 数据摄入).
"""

from __future__ import annotations

import socket
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import dpkt

from nid_video.utils import logger


# IANA protocol numbers / Ethernet ethertype constants used in the hot loop.
PROTO_TCP = 6
PROTO_UDP = 17
ETH_TYPE_IP4 = 0x0800
ETH_TYPE_IP6 = 0x86DD


class PacketRecord(NamedTuple):
    """One IPv4 TCP/UDP packet observed on a pcap.

    NamedTuple (not dataclass): construction is C-level via tuple, ~2x faster
    than a slotted frozen dataclass in the hot per-packet loop.

    direction: low-port-as-server heuristic (documented assumption, not ground truth).
        0 = inbound  (toward the lower-port endpoint, treated as server)
        1 = outbound (away from the lower-port endpoint)
    """

    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    pkt_size: int
    tcp_flags: int
    payload_len: int
    direction: int


def _direction(src_port: int, dst_port: int) -> int:
    """Lower-port side is server; toward server = inbound (0)."""
    if dst_port < src_port:
        return 0
    if src_port < dst_port:
        return 1
    return 0  # equal-port edge case


def _ip_to_str(addr: bytes) -> str:
    return socket.inet_ntoa(addr)


class PacketStream:
    """Streaming IPv4 TCP/UDP packet iterator over one pcap file.

    Iterating consumes the file once; per-iteration counts live in `stats`.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._stats: dict[str, int] = {
            "yielded": 0,
            "malformed": 0,
            "ipv6": 0,
            "non_ip": 0,
            "non_tcpudp": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def __iter__(self) -> Iterator[PacketRecord]:
        # Local refs to module-level constants/functions: avoids global LOAD_GLOBAL
        # in the hot loop. Measured ~10-15% wall-time win on a 100k packet pcap.
        Ethernet = dpkt.ethernet.Ethernet
        UnpackError = dpkt.dpkt.UnpackError
        NeedData = dpkt.dpkt.NeedData
        inet_ntoa = socket.inet_ntoa
        eth_ip4 = ETH_TYPE_IP4
        eth_ip6 = ETH_TYPE_IP6
        proto_tcp = PROTO_TCP
        proto_udp = PROTO_UDP
        stats = self._stats
        path_name = self.path.name

        with self.path.open("rb") as fh:
            try:
                reader = dpkt.pcap.Reader(fh)
            except (ValueError, NeedData, UnpackError) as exc:
                logger.error(f"Cannot open pcap {self.path}: {exc}")
                return
            dlt = reader.datalink()
            if dlt != dpkt.pcap.DLT_EN10MB:
                logger.error(
                    f"{self.path}: unsupported datalink {dlt} (expected DLT_EN10MB=1)"
                )
                return

            for ts, buf in reader:
                try:
                    eth = Ethernet(buf)
                except (UnpackError, NeedData, struct.error, IndexError):
                    stats["malformed"] += 1
                    if stats["malformed"] == 1:
                        logger.warning(f"{path_name}: first malformed packet at ts={ts}")
                    continue

                eth_type = eth.type
                if eth_type == eth_ip6:
                    stats["ipv6"] += 1
                    continue
                if eth_type != eth_ip4:
                    stats["non_ip"] += 1
                    continue

                ip = eth.data
                proto = ip.p
                l4 = ip.data
                if proto == proto_tcp:
                    sport = l4.sport
                    dport = l4.dport
                    flags = l4.flags
                    payload_len = len(l4.data)
                elif proto == proto_udp:
                    sport = l4.sport
                    dport = l4.dport
                    flags = 0
                    payload_len = len(l4.data)
                else:
                    stats["non_tcpudp"] += 1
                    continue

                # direction: low-port side is server (toward server = 0).
                if sport < dport:
                    dir_ = 1
                elif dport < sport:
                    dir_ = 0
                else:
                    dir_ = 0

                stats["yielded"] += 1
                yield PacketRecord(
                    ts, inet_ntoa(ip.src), inet_ntoa(ip.dst),
                    sport, dport, proto, len(buf),
                    flags, payload_len, dir_,
                )

        s = self._stats
        logger.info(
            f"pcap {path_name}: yielded={s['yielded']} malformed={s['malformed']} "
            f"ipv6={s['ipv6']} non_ip={s['non_ip']} non_tcpudp={s['non_tcpudp']}"
        )


def parse_pcap(path: Path | str) -> Iterator[PacketRecord]:
    """Yield PacketRecord per IPv4 TCP/UDP packet from a pcap file. Idea.md §3.1."""
    return iter(PacketStream(path))
