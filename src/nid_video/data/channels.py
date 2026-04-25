"""Window → (T, C, H, W) tensor encoder.

This module owns *only* the encoding step. The H mapping (source IP → row) is
provided by the caller (see ip_clustering.py); the W mapping (port → column) is
built once via build_port_mapping() and reused for every window.

Channel semantics (Idea.md §3.2, locked at C=6):
  Ch 0  log(1 + packet count)
  Ch 1  log(1 + total bytes)
  Ch 2  mean packet size (0 when cell is empty)
  Ch 3  TCP flags OR-aggregated (6 bits) divided by 63   -- Decision: M2 task 2.4 Q1
  Ch 4  inter-frame delta of direction ratio              (motion)
  Ch 5  inter-frame delta of log(1 + packet count)        (motion)  -- Decision: M2 task 2.4 Q2

The motion channels (Ch 4, Ch 5) are zero at frame 0 (no prior frame).

Spatial layout:
  H = 32  source-IP rows; mapping supplied per-window by the clusterer.
  W = 64  port columns:
            cols  0..15: COMMON_PORTS in fixed order
            cols 16..63: 1/3-octave log2 buckets via 16 + clamp(int(log2(p+1)*3), 0, 47)
                                                                 -- Decision: M2 task 2.4 Q4
  Each packet's W column is looked up using min(src_port, dst_port) so that the
  request and reply legs of a connection share a column.    -- Decision: M2 task 2.4 Q3
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nid_video.data.windowing import Frame, Window
from nid_video.utils import logger


# 16 hot service ports → fixed columns 0..15. Order is part of the design.
COMMON_PORTS: tuple[int, ...] = (
    80, 443, 22, 53, 21, 3389, 25, 110,
    143, 3306, 5432, 8080, 8443, 445, 139, 23,
)

# TCP flag bit positions (RFC 793 + RFC 3168). Decision: M2 task 2.4 Q1.
TCP_FLAG_FIN = 0x01
TCP_FLAG_SYN = 0x02
TCP_FLAG_RST = 0x04
TCP_FLAG_PSH = 0x08
TCP_FLAG_ACK = 0x10
TCP_FLAG_URG = 0x20
TCP_FLAGS_MASK = (
    TCP_FLAG_FIN | TCP_FLAG_SYN | TCP_FLAG_RST
    | TCP_FLAG_PSH | TCP_FLAG_ACK | TCP_FLAG_URG
)  # 0x3F = 63

NUM_HOT_PORTS = len(COMMON_PORTS)              # 16
NUM_LOG_BUCKETS_DEFAULT = 64 - NUM_HOT_PORTS   # 48


@dataclass(frozen=True, slots=True)
class ChannelConfig:
    """Geometry knobs for the encoder. Locked to (C, H, W) = (6, 32, 64) by Idea.md §3.2."""

    num_channels: int = 6
    num_ip_buckets: int = 32   # H
    num_port_buckets: int = 64  # W

    @classmethod
    def from_data_config(cls, data_cfg: object) -> "ChannelConfig":
        """Bridge from utils.config.DataConfig (kept loose-typed to avoid cycle)."""
        return cls(
            num_channels=int(getattr(data_cfg, "num_channels")),
            num_ip_buckets=int(getattr(data_cfg, "num_ip_buckets")),
            num_port_buckets=int(getattr(data_cfg, "num_port_buckets")),
        )


# ---------------------------------------------------------------------------
# Port → column mapping
# ---------------------------------------------------------------------------


def _log_bucket_for_port(port: int, n_buckets: int = NUM_LOG_BUCKETS_DEFAULT) -> int:
    """1/3-octave log2 bucket for non-hot ports. Decision: M2 task 2.4 Q4.

    `int(log2(port+1) * 3)` clamped to [0, n_buckets - 1]. The +1 keeps log defined
    at port 0 and the *3 spreads 16 octaves of port space across all 48 buckets,
    avoiding the 31-empty-column waste of the integer-log2 alternative.
    """
    if port < 0 or port > 65535:
        raise ValueError(f"port out of range: {port}")
    raw = int(math.log2(port + 1) * 3)
    if raw < 0:
        raw = 0
    if raw >= n_buckets:
        raw = n_buckets - 1
    return raw


def build_port_mapping(num_port_buckets: int = 64) -> dict[int, int]:
    """Pre-fill a port→column map for every TCP/UDP port in [0, 65535].

    Cols 0..15 are COMMON_PORTS in order. Cols 16..(num_port_buckets-1) are the
    1/3-octave log2 buckets for any non-hot port. Idea.md §3.2.
    """
    if num_port_buckets <= NUM_HOT_PORTS:
        raise ValueError(
            f"num_port_buckets={num_port_buckets} must exceed hot-port count={NUM_HOT_PORTS}"
        )
    n_log = num_port_buckets - NUM_HOT_PORTS

    mapping: dict[int, int] = {}
    for col, port in enumerate(COMMON_PORTS):
        mapping[port] = col

    for port in range(65536):
        if port in mapping:
            continue
        mapping[port] = NUM_HOT_PORTS + _log_bucket_for_port(port, n_log)

    return mapping


def canonical_port(src_port: int, dst_port: int) -> int:
    """Return min(src, dst) — the service-side port. Decision: M2 task 2.4 Q3.

    Guarantees that the forward and reverse legs of a connection map to the same
    W column, so the model sees a coherent across-frame motion pattern instead of
    request and reply scattered across distant columns.
    """
    return src_port if src_port < dst_port else dst_port


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_window(
    window: Window,
    ip_to_row: dict[str, int],
    port_to_col: dict[int, int],
    config: ChannelConfig | None = None,
) -> np.ndarray:
    """Encode one Window as a (T, C, H, W) float32 tensor. Idea.md §3.2."""
    cfg = config or ChannelConfig()
    T = len(window.frames)
    C = cfg.num_channels
    H = cfg.num_ip_buckets
    W = cfg.num_port_buckets

    # --- per-cell scalar accumulators ---------------------------------------
    pkt_count = np.zeros((T, H, W), dtype=np.float64)
    byte_total = np.zeros((T, H, W), dtype=np.float64)
    flag_or = np.zeros((T, H, W), dtype=np.uint8)
    out_count = np.zeros((T, H, W), dtype=np.float64)  # outbound subset, for direction ratio

    n_dropped_no_ip = 0
    n_dropped_no_port = 0

    for t in range(T):
        frame: Frame = window.frames[t]
        for pkt in frame.packets:
            row = ip_to_row.get(pkt.src_ip)
            if row is None or row < 0 or row >= H:
                n_dropped_no_ip += 1
                continue
            col = port_to_col.get(canonical_port(pkt.src_port, pkt.dst_port))
            if col is None or col < 0 or col >= W:
                n_dropped_no_port += 1
                continue
            pkt_count[t, row, col] += 1.0
            byte_total[t, row, col] += pkt.pkt_size
            flag_or[t, row, col] |= np.uint8(pkt.tcp_flags & TCP_FLAGS_MASK)
            if pkt.direction == 1:
                out_count[t, row, col] += 1.0

    if n_dropped_no_ip or n_dropped_no_port:
        logger.debug(
            "encode_window: dropped {} pkts (no_ip_row), {} pkts (no_port_col)",
            n_dropped_no_ip, n_dropped_no_port,
        )

    # --- channel construction ----------------------------------------------
    out = np.zeros((T, C, H, W), dtype=np.float32)

    # Ch 0: log(1 + packets)
    log_count = np.log1p(pkt_count)
    out[:, 0] = log_count.astype(np.float32)

    # Ch 1: log(1 + total bytes)
    out[:, 1] = np.log1p(byte_total).astype(np.float32)

    # Ch 2: mean packet size, 0 where the cell is empty
    safe_count = np.where(pkt_count > 0, pkt_count, 1.0)
    mean_pkt = byte_total / safe_count
    mean_pkt[pkt_count == 0] = 0.0
    out[:, 2] = mean_pkt.astype(np.float32)

    # Ch 3: 6-bit TCP flags OR / 63   -- Decision: M2 task 2.4 Q1
    out[:, 3] = (flag_or.astype(np.float32) / float(TCP_FLAGS_MASK))

    # Ch 4: motion = direction-ratio inter-frame delta, frame 0 = 0
    dir_ratio = np.where(pkt_count > 0, out_count / safe_count, 0.0)
    dir_motion = np.zeros_like(dir_ratio)
    dir_motion[1:] = dir_ratio[1:] - dir_ratio[:-1]
    out[:, 4] = dir_motion.astype(np.float32)

    # Ch 5: motion = log(1+N(t)) - log(1+N(t-1)), frame 0 = 0
    # NOT raw count delta. Raw delta would dominate gradient under DDoS bursts
    # (peak ~10^3 vs other channels ~10^1).  Decision: M2 task 2.4 Q2
    count_motion = np.zeros_like(log_count)
    count_motion[1:] = log_count[1:] - log_count[:-1]
    out[:, 5] = count_motion.astype(np.float32)

    return out
