"""M6.1 — byte-sequence ETL (Phase 0 step 2).

New code path that produces per-window byte tensors for the 1D
Transformer paradigm. Runs in parallel to the existing v2 ETL —
neither modifies nor reads v2 shards. Output lands at
``data/processed/cicids2017_dt100ms_v2_bytes/`` by default, mirroring
the v2 shard layout (per-pcap subdirs + manifest.parquet).

Design source-of-truth: ``docs/m6_1_byte_transformer.md`` (Phase 0
design report). Locked decisions at write time:

  * A-i offline extraction → new parallel shards
  * B-i first K=16 packets in temporal order per window
  * C-i first N=128 bytes of L2 frame per packet
  * D-ii zero L2 bytes 0-11 (MAC dst+src) + 26-29, 30-33 (IPv4 src+dst)
  * E vocab 257 (bytes 0-255 + [PAD]=256); attention_mask via PAD

The output shard sample schema mirrors v2 except ``tensor.npy`` is
replaced by ``bytes.npy`` (uint8, shape (K, N)) and ``mask.npy``
(uint8, shape (K, N), 1 = real byte, 0 = pad). DataLoader converts to
int64 token ids + PAD at training time. Window keys ``(pcap_source,
start_time)`` are bit-identical with the v2 fast shards — verified by
the val_n=16,463 alignment test.
"""

from __future__ import annotations

import json
import socket
import struct
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import dpkt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds

from nid_video.data.labeling import LabelIndex, label_window
from nid_video.data.pcap_parser import (
    ETH_TYPE_IP4,
    ETH_TYPE_IP6,
    PROTO_TCP,
    PROTO_UDP,
    _detect_pcap_format,
    _open_reader,
)
from nid_video.data.windowing import SlidingWindow
from nid_video.utils import logger

# --- constants ---------------------------------------------------------------

K_PACKETS = 16          # M6.1 Phase 0 §B: K=16 packets per window
N_BYTES = 128           # M6.1 Phase 0 §C: N=128 bytes per packet (L2 frame)

# §D-ii byte ranges to mask (inclusive start, exclusive end) — relative to the
# start of the L2 Ethernet frame.
#   bytes 0..6   = Ethernet dst MAC
#   bytes 6..12  = Ethernet src MAC
#   bytes 26..30 = IPv4 src address (Ethernet 14 + IP header offset 12)
#   bytes 30..34 = IPv4 dst address (Ethernet 14 + IP header offset 16)
D_II_MASK_RANGES: tuple[tuple[int, int], ...] = (
    (0, 12),
    (26, 34),
)
# Total masked bytes per packet (out of N=128) = 12 + 8 = 20.

# Token vocabulary (E): bytes 0-255 → token ids 0-255; [PAD] = 256.
PAD_TOKEN_ID = 256
VOCAB_SIZE = 257


# --- packet byte stream ------------------------------------------------------


class PacketByteRecord(NamedTuple):
    """A packet record with L2 raw bytes attached. NamedTuple for duck-typing
    compatibility with ``nid_video.data.windowing.SlidingWindow`` (which uses
    only ``.timestamp``) and ``nid_video.data.labeling.label_window`` (which
    uses ``.src_ip, .src_port, .dst_ip, .dst_port, .protocol, .timestamp``).
    """

    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    pkt_size: int
    l2_bytes: bytes        # truncated to first N_BYTES; pre-mask


class PacketByteStream:
    """Streaming IPv4 TCP/UDP packet iterator that retains the raw L2 frame.

    Mirrors ``nid_video.data.pcap_parser.PacketStream`` loop semantics
    (same Ethernet / IP / L4 parse + same malformed-packet handling) but
    additionally retains the first N_BYTES of the L2 frame on each yield.
    The existing pcap_parser is NOT modified (forensic preservation).
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._stats: dict[str, int] = {
            "yielded": 0, "malformed": 0, "ipv6": 0,
            "non_ip": 0, "non_tcpudp": 0, "truncated": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def __iter__(self) -> Iterator[PacketByteRecord]:
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
                if not hasattr(l4, "sport") or not hasattr(l4, "dport"):
                    stats["malformed"] += 1
                    if stats["malformed"] == 1:
                        logger.warning(f"{path_name}: first malformed packet at ts={ts}")
                    continue

                if proto == proto_tcp:
                    sport = l4.sport
                    dport = l4.dport
                elif proto == proto_udp:
                    sport = l4.sport
                    dport = l4.dport
                else:
                    stats["non_tcpudp"] += 1
                    continue

                # Truncate L2 frame to first N_BYTES. Track truncation count.
                if len(buf) > N_BYTES:
                    l2 = buf[:N_BYTES]
                    stats["truncated"] += 1
                else:
                    l2 = buf

                stats["yielded"] += 1
                yield PacketByteRecord(
                    timestamp=float(ts),
                    src_ip=inet_ntoa(ip.src),
                    dst_ip=inet_ntoa(ip.dst),
                    src_port=int(sport),
                    dst_port=int(dport),
                    protocol=int(proto),
                    pkt_size=len(buf),
                    l2_bytes=bytes(l2),
                )


# --- byte tensor encoding ----------------------------------------------------


def encode_window_to_bytes(window) -> tuple[np.ndarray, np.ndarray]:
    """Encode a windowing.Window (of PacketByteRecord packets) into
    (bytes_tensor (K, N) uint8, mask_tensor (K, N) uint8).

    Phase 0 §B-i: takes the first K_PACKETS packets in temporal order
    across all frames; pads remaining packet slots with all-zero rows
    + mask=0.
    Phase 0 §C-i: per-packet first N_BYTES of L2 frame, right-pad with
    0x00 + mask=0.
    Phase 0 §D-ii: zero the L2 byte ranges in ``D_II_MASK_RANGES`` for
    every real packet (mask STAYS 1 — masked bytes are anonymized, not
    removed; attention still attends to those positions).
    """
    bytes_arr = np.zeros((K_PACKETS, N_BYTES), dtype=np.uint8)
    mask_arr = np.zeros((K_PACKETS, N_BYTES), dtype=np.uint8)

    flat_packets: list[PacketByteRecord] = []
    for frame in window.frames:
        for pkt in frame.packets:
            flat_packets.append(pkt)
            if len(flat_packets) >= K_PACKETS:
                break
        if len(flat_packets) >= K_PACKETS:
            break

    for k, pkt in enumerate(flat_packets[:K_PACKETS]):
        b = pkt.l2_bytes
        nb = min(len(b), N_BYTES)
        bytes_arr[k, :nb] = np.frombuffer(b[:nb], dtype=np.uint8)
        mask_arr[k, :nb] = 1
        # §D-ii: zero MAC + IPv4 src+dst bytes (only within real range).
        for (s, e) in D_II_MASK_RANGES:
            if s < nb:
                bytes_arr[k, s:min(e, nb)] = 0
                # Note: mask STAYS 1 — anonymized but positionally present.

    return bytes_arr, mask_arr


# --- ETL pipeline ------------------------------------------------------------


@dataclass
class ByteEtlStats:
    """Aggregated counts from one or more byte-ETL invocations."""

    n_windows_emitted: int = 0
    n_pcaps_processed: int = 0
    n_pcaps_failed: int = 0
    label_counts: Counter = field(default_factory=Counter)
    n_shards: int = 0
    elapsed_seconds: float = 0.0
    n_packets_truncated: int = 0
    n_packets_yielded: int = 0

    def merge(self, other: "ByteEtlStats") -> "ByteEtlStats":
        self.n_windows_emitted += other.n_windows_emitted
        self.n_pcaps_processed += other.n_pcaps_processed
        self.n_pcaps_failed += other.n_pcaps_failed
        self.label_counts.update(other.label_counts)
        self.n_shards += other.n_shards
        self.elapsed_seconds = max(self.elapsed_seconds, other.elapsed_seconds)
        self.n_packets_truncated += other.n_packets_truncated
        self.n_packets_yielded += other.n_packets_yielded
        return self


def run_byte_etl(
    pcap_paths: Iterable[Path],
    label_csv_paths: Iterable[Path],
    output_dir: Path,
    *,
    delta_t_ms: int = 100,
    num_frames: int = 16,
    window_overlap: float = 0.5,
    samples_per_shard: int = 1000,
    limit_windows: int | None = None,
    csv_dayfirst: bool = True,
    csv_tz: str = "America/Halifax",
    csv_twelve_hour_pm_inference: bool = True,
    label_index: LabelIndex | None = None,
) -> ByteEtlStats:
    """Run the byte ETL over the provided pcaps.

    Window keying matches v2 fast shards exactly: ``SlidingWindow(delta_t_s=
    0.1, num_frames=16, overlap=0.5)``. Same labeler (``label_window``) is
    used → labels match v2 fast windows. The shard emit yields ``bytes.npy``
    + ``mask.npy`` + ``label.cls`` + ``meta.json`` per sample.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)

    if label_index is None:
        csvs = [Path(p) for p in label_csv_paths]
        label_index = LabelIndex.from_csv(
            csvs, dayfirst=csv_dayfirst, csv_tz=csv_tz,
            csv_twelve_hour_pm_inference=csv_twelve_hour_pm_inference,
        )

    windower = SlidingWindow(
        delta_t_s=delta_t_ms / 1000.0,
        num_frames=num_frames,
        overlap=window_overlap,
    )

    stats = ByteEtlStats()
    t0 = time.perf_counter()

    shard_pattern = str(shard_dir / "shard-%06d.tar")
    sample_idx = 0
    per_shard_counts: list[Counter[str]] = []
    current_shard_counts: Counter[str] = Counter()

    pcaps = list(pcap_paths)
    logger.info(
        f"byte ETL: {len(pcaps)} pcap(s), output -> {output_dir} "
        f"K={K_PACKETS} N={N_BYTES} vocab={VOCAB_SIZE}"
    )

    with wds.ShardWriter(shard_pattern, maxcount=samples_per_shard) as writer:
        for pcap_path in pcaps:
            pcap_path = Path(pcap_path)
            logger.info(f"byte ETL: starting {pcap_path.name}")
            try:
                packets = PacketByteStream(pcap_path)
                stop = False
                windows_before = stats.n_windows_emitted
                pkts_yielded_before = stats.n_packets_yielded
                pkts_trunc_before = stats.n_packets_truncated

                for window in windower(packets, pcap_source=pcap_path.name):
                    if limit_windows is not None and stats.n_windows_emitted >= limit_windows:
                        logger.info(f"--limit-windows={limit_windows} reached")
                        stop = True
                        break

                    bytes_arr, mask_arr = encode_window_to_bytes(window)
                    win_label = label_window(window, label_index)

                    if sample_idx > 0 and sample_idx % samples_per_shard == 0:
                        per_shard_counts.append(current_shard_counts)
                        current_shard_counts = Counter()

                    writer.write({
                        "__key__": f"{sample_idx:010d}",
                        "bytes.npy": bytes_arr,
                        "mask.npy": mask_arr,
                        "label.cls": int(win_label.label_id),
                        "meta.json": {
                            "start_time": float(window.start_time),
                            "pcap_source": window.pcap_source,
                            "label": win_label.label,
                            "label_id": int(win_label.label_id),
                            "dominant_attack_ratio": float(win_label.dominant_ratio),
                            "n_unmatched": int(win_label.n_unmatched),
                        },
                    })

                    sample_idx += 1
                    stats.n_windows_emitted += 1
                    stats.label_counts[win_label.label] += 1
                    current_shard_counts[win_label.label] += 1

                # Pull packet counts from the stream after this pcap completes.
                pstats = packets.stats
                stats.n_packets_yielded += pstats["yielded"]
                stats.n_packets_truncated += pstats["truncated"]

                produced = stats.n_windows_emitted - windows_before
                if produced == 0 and not stop:
                    logger.warning(
                        f"{pcap_path.name}: produced 0 windows — counting as failed."
                    )
                    stats.n_pcaps_failed += 1
                else:
                    stats.n_pcaps_processed += 1
                if stop:
                    break
            except Exception as exc:                          # noqa: BLE001
                logger.error(f"byte ETL failed on {pcap_path}: {exc}")
                stats.n_pcaps_failed += 1
                continue

    if current_shard_counts:
        per_shard_counts.append(current_shard_counts)

    stats.n_shards = len(per_shard_counts)
    stats.elapsed_seconds = time.perf_counter() - t0

    _write_byte_manifest(output_dir, per_shard_counts)

    pct_trunc = (
        100.0 * stats.n_packets_truncated / max(stats.n_packets_yielded, 1)
    )
    logger.info(
        f"byte ETL done in {stats.elapsed_seconds:.1f}s: "
        f"{stats.n_windows_emitted} windows / {stats.n_shards} shards / "
        f"{stats.n_pcaps_processed} OK / {stats.n_pcaps_failed} failed; "
        f"packets yielded={stats.n_packets_yielded:,}, truncated to {N_BYTES} bytes="
        f"{stats.n_packets_truncated:,} ({pct_trunc:.2f}%)"
    )
    return stats


def _write_byte_manifest(
    output_dir: Path, per_shard_counts: list[Counter[str]]
) -> None:
    if not per_shard_counts:
        logger.warning("no shards written; manifest will be empty")
        rows = [{"shard_idx": -1, "shard_name": "", "n_samples": 0, "labels_json": "{}"}]
    else:
        rows = [
            {
                "shard_idx": i,
                "shard_name": f"shard-{i:06d}.tar",
                "n_samples": int(sum(counts.values())),
                "labels_json": json.dumps(dict(counts), ensure_ascii=False),
            }
            for i, counts in enumerate(per_shard_counts)
        ]
    table = pa.Table.from_pylist(rows)
    out = output_dir / "manifest.parquet"
    pq.write_table(table, out)
    logger.info(f"byte manifest written: {out}")
