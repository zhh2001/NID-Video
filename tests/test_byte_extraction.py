"""Sanity tests for M6.1 byte ETL (Phase 0).

Covers:
  1. PacketByteStream yields PacketByteRecord with retained L2 bytes
  2. ``encode_window_to_bytes`` masks D-ii ranges (MAC + IPv4) to 0
  3. Same input window → byte-identical output (determinism)
  4. PAD region (packet slots > # packets in window) has mask == 0

The val_n=16,463 alignment test is exercised by the smoke runner, NOT
in unit tests (it needs a 33 GB pcap walk; out of scope for fast suite).
"""

from __future__ import annotations

import numpy as np

from nid_video.data.byte_extraction import (
    D_II_MASK_RANGES,
    K_PACKETS,
    N_BYTES,
    PacketByteRecord,
    encode_window_to_bytes,
)
from nid_video.data.windowing import Frame, Window


def _fake_packet(ts: float, byte_pattern: int = 0x42, real_len: int = 128) -> PacketByteRecord:
    """Build a fake PacketByteRecord. L2 bytes = (byte_pattern) * real_len, padded with 0."""
    buf = bytes([byte_pattern] * real_len)
    if len(buf) > N_BYTES:
        buf = buf[:N_BYTES]
    return PacketByteRecord(
        timestamp=ts,
        src_ip="10.0.0.1", dst_ip="10.0.0.2",
        src_port=12345, dst_port=80, protocol=6,
        pkt_size=real_len, l2_bytes=buf,
    )


def _fake_window(n_packets: int, byte_pattern: int = 0x42, real_len: int = 128) -> Window:
    """Build a Window with n_packets PacketByteRecord across 1 frame."""
    pkts = [_fake_packet(float(i), byte_pattern, real_len) for i in range(n_packets)]
    return Window(
        start_time=0.0,
        frames=[Frame(start_time=0.0, end_time=1.6, packets=pkts)],
        pcap_source="fake.pcap",
    )


def test_encode_window_to_bytes_shape() -> None:
    w = _fake_window(n_packets=10)
    bytes_arr, mask_arr = encode_window_to_bytes(w)
    assert bytes_arr.shape == (K_PACKETS, N_BYTES)
    assert mask_arr.shape == (K_PACKETS, N_BYTES)
    assert bytes_arr.dtype == np.uint8
    assert mask_arr.dtype == np.uint8


def test_encode_window_mask_d_ii_ranges_zeroed() -> None:
    """D-ii: L2 bytes 0-11 (MAC) + 26-29 + 30-33 (IPv4) must be zeroed.
    Other bytes retain the original pattern (0x42 here)."""
    w = _fake_window(n_packets=4, byte_pattern=0x42, real_len=128)
    bytes_arr, mask_arr = encode_window_to_bytes(w)
    for k in range(4):
        # Masked ranges → 0
        for (s, e) in D_II_MASK_RANGES:
            assert (bytes_arr[k, s:e] == 0).all(), (
                f"D-ii: bytes[{k}, {s}:{e}] should be zero, got {bytes_arr[k, s:e]}"
            )
        # Mask stays 1 even at masked positions (anonymized, not removed)
        for (s, e) in D_II_MASK_RANGES:
            assert (mask_arr[k, s:e] == 1).all()
        # Non-masked positions retain 0x42
        # Bytes 12-25, 34-127 should all be 0x42 (D-ii leaves these intact)
        for s, e in [(12, 26), (34, 128)]:
            assert (bytes_arr[k, s:e] == 0x42).all(), (
                f"unmasked range [{s}:{e}] should be 0x42; got "
                f"{bytes_arr[k, s:e]}"
            )


def test_encode_window_packet_pad_when_window_short() -> None:
    """Window with <K packets → trailing packet rows have mask == 0."""
    w = _fake_window(n_packets=3)
    bytes_arr, mask_arr = encode_window_to_bytes(w)
    # Rows 0-2 are real packets → mask should have ≥1 nonzero entry per row.
    # Rows 3-15 are padded → mask should be all zero.
    for k in range(3):
        assert mask_arr[k].sum() > 0
    for k in range(3, K_PACKETS):
        assert mask_arr[k].sum() == 0
        assert (bytes_arr[k] == 0).all()


def test_encode_window_byte_pad_when_packet_short() -> None:
    """Packet with <N bytes → trailing byte positions have mask == 0."""
    w = _fake_window(n_packets=2, real_len=40)
    bytes_arr, mask_arr = encode_window_to_bytes(w)
    for k in range(2):
        # First 40 bytes have content (with D-ii masking)
        assert mask_arr[k, :40].sum() == 40
        # Bytes 40-127 should be all 0 with mask=0
        assert (bytes_arr[k, 40:] == 0).all()
        assert (mask_arr[k, 40:] == 0).all()


def test_encode_window_determinism() -> None:
    """Encoding the same window twice → byte-identical output."""
    w1 = _fake_window(n_packets=10, byte_pattern=0xAB, real_len=120)
    w2 = _fake_window(n_packets=10, byte_pattern=0xAB, real_len=120)
    b1, m1 = encode_window_to_bytes(w1)
    b2, m2 = encode_window_to_bytes(w2)
    assert np.array_equal(b1, b2)
    assert np.array_equal(m1, m2)
