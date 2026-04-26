"""pcap → PacketRecord stream.

dpkt is the production parser (C-backed, hundreds of thousands of packets per
second). scapy is reserved for test-pcap *construction* and never imported here.
See M2 design decisions for the rationale.

Both classic libpcap and pcapng are supported; the format is chosen per file by
inspecting the first 4 magic bytes. CIC-IDS-2017 ships pcapng; the synthetic
test fixtures use classic libpcap.

Idea.md §3.1 (Stage 1 · 数据摄入).
"""

from __future__ import annotations

import socket
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import IO, NamedTuple, Union

import dpkt

from nid_video.utils import logger


# IANA protocol numbers / Ethernet ethertype constants used in the hot loop.
PROTO_TCP = 6
PROTO_UDP = 17
ETH_TYPE_IP4 = 0x0800
ETH_TYPE_IP6 = 0x86DD

# Magic bytes for pcap format dispatch. pcapng uses a Section Header Block
# whose first 4 bytes are the Block Type (0x0a0d0d0a). Classic libpcap uses
# either 0xa1b2c3d4 (microsecond) or its byte-swapped form, depending on
# endianness of the host that wrote the file.
_MAGIC_PCAPNG = b"\x0a\x0d\x0d\x0a"
_MAGIC_PCAP_LE = b"\xd4\xc3\xb2\xa1"
_MAGIC_PCAP_BE = b"\xa1\xb2\xc3\xd4"

# Below this many yielded packets, log a WARNING after iteration ends. Real
# CIC-IDS pcaps contain millions of packets; a yield this small almost always
# means a parse problem (wrong datalink, all-malformed frames, partial file)
# and the user should be told before the rest of the pipeline silently produces
# zero windows.
_LOW_YIELD_WARN_THRESHOLD = 100

PcapReader = Union[dpkt.pcap.Reader, dpkt.pcapng.Reader]


def _detect_pcap_format(path: Path) -> str:
    """Return ``"pcap"`` or ``"pcapng"`` based on the file's magic bytes.

    Raises ``ValueError`` if the magic matches neither supported format. We do
    not fall through to ``dpkt.pcap.Reader`` for unknown magics because its
    own header validation produces a confusing low-level error message; an
    explicit early failure is easier to diagnose.
    """
    with path.open("rb") as fh:
        magic = fh.read(4)
    if magic == _MAGIC_PCAPNG:
        return "pcapng"
    if magic in (_MAGIC_PCAP_LE, _MAGIC_PCAP_BE):
        return "pcap"
    raise ValueError(
        f"unrecognized pcap magic {magic.hex()} in {path}; "
        f"expected pcap (d4c3b2a1 / a1b2c3d4) or pcapng (0a0d0d0a)"
    )


def _open_reader(fh: IO[bytes], fmt: str) -> PcapReader:
    if fmt == "pcapng":
        return dpkt.pcapng.Reader(fh)
    return dpkt.pcap.Reader(fh)


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

        # Detect format up front so the error message is honest: the caller
        # gets a ValueError that can be caught and counted as a failed pcap,
        # not a silent zero-yield iteration that looks like success.
        fmt = _detect_pcap_format(self.path)

        with self.path.open("rb") as fh:
            try:
                reader = _open_reader(fh, fmt)
            except (ValueError, NeedData, UnpackError) as exc:
                raise ValueError(
                    f"Cannot open {fmt} reader on {self.path}: {exc}"
                ) from exc
            dlt = reader.datalink()
            if dlt != dpkt.pcap.DLT_EN10MB:
                raise ValueError(
                    f"{self.path}: unsupported datalink {dlt} (expected DLT_EN10MB=1)"
                )

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
                # Real CIC pcaps contain truncated frames where dpkt's IP-layer
                # parse succeeds but the L4 layer is left as raw bytes (e.g. a
                # TCP header sliced before sport/dport). Without this guard the
                # next .sport access raises AttributeError and crashes the whole
                # iterator; the per-packet try/except above only wraps the
                # Ethernet parse, not the L4 attribute path.
                if not hasattr(l4, "sport") or not hasattr(l4, "dport"):
                    stats["malformed"] += 1
                    if stats["malformed"] == 1:
                        logger.warning(f"{path_name}: first malformed packet at ts={ts}")
                    continue
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
        # Visibility for the false-positive case caught in the M3 dry-run:
        # opening succeeded but iteration yielded nothing useful. This used to
        # bubble up as an "OK pcap with 0 packets" — silently wrong.
        if s["yielded"] == 0:
            logger.warning(
                f"{path_name}: yielded 0 IPv4 TCP/UDP packets — likely empty, "
                f"all-malformed, or non-Ethernet pcap "
                f"(malformed={s['malformed']}, ipv6={s['ipv6']}, "
                f"non_ip={s['non_ip']}, non_tcpudp={s['non_tcpudp']})"
            )
        elif s["yielded"] < _LOW_YIELD_WARN_THRESHOLD:
            logger.warning(
                f"{path_name}: only {s['yielded']} IPv4 TCP/UDP packets "
                f"(<{_LOW_YIELD_WARN_THRESHOLD}); verify this pcap is intact"
            )


def parse_pcap(path: Path | str) -> Iterator[PacketRecord]:
    """Yield PacketRecord per IPv4 TCP/UDP packet from a pcap file. Idea.md §3.1."""
    return iter(PacketStream(path))
