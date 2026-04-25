"""Unit tests for the sliding window splitter."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from nid_video.data.pcap_parser import PacketRecord
from nid_video.data.windowing import Frame, SlidingWindow, Window


def _pkts(timestamps: list[float]) -> Iterator[PacketRecord]:
    for ts in timestamps:
        yield PacketRecord(
            timestamp=ts,
            src_ip="1.1.1.1",
            dst_ip="2.2.2.2",
            src_port=12345,
            dst_port=80,
            protocol=6,
            pkt_size=100,
            tcp_flags=0,
            payload_len=60,
            direction=0,
        )


def test_uniform_50pps_5s_emits_5_complete_windows() -> None:
    """50 pps over 5 s with default (T=16, dt=0.1, overlap=0.5) → exactly 5 windows."""
    # offset by 0.01 to avoid land-on-frame-boundary float drift in this assertion
    timestamps = [0.01 + i * 0.02 for i in range(250)]
    windower = SlidingWindow()
    windows = list(windower(_pkts(timestamps), pcap_source="uniform.pcap"))

    assert len(windows) == 5
    # Each window is well-formed
    for w in windows:
        assert isinstance(w, Window)
        assert len(w.frames) == 16
        assert all(isinstance(f, Frame) for f in w.frames)
        assert w.pcap_source == "uniform.pcap"

    # First window starts at the first packet's timestamp
    assert windows[0].start_time == pytest.approx(0.01)
    # Window N starts step * dt = 0.8 s after the previous one
    deltas = [windows[i + 1].start_time - windows[i].start_time for i in range(4)]
    assert all(d == pytest.approx(0.8) for d in deltas)


def test_each_frame_has_dt_width_and_consecutive() -> None:
    timestamps = [0.01 + i * 0.02 for i in range(100)]
    windower = SlidingWindow()
    windows = list(windower(_pkts(timestamps)))

    w0 = windows[0]
    for i, f in enumerate(w0.frames):
        assert f.end_time - f.start_time == pytest.approx(0.1)
        if i > 0:
            assert f.start_time == pytest.approx(w0.frames[i - 1].end_time)


def test_silent_gap_preserved_as_empty_frames() -> None:
    """Insert a 1 s gap; verify the spanning windows have empty packet lists in the gap."""
    early = [0.01 + i * 0.02 for i in range(100)]   # 0.01 .. 1.99 (frames 0..19)
    late = [3.01 + i * 0.02 for i in range(100)]    # 3.01 .. 4.99 (frames 30..49)
    windower = SlidingWindow()
    windows = list(windower(_pkts(early + late)))

    # Find a frame whose midpoint sits in the silent gap [2.0, 3.0)
    gap_frames: list[Frame] = []
    for w in windows:
        for f in w.frames:
            mid = 0.5 * (f.start_time + f.end_time)
            if 2.0 <= mid < 3.0:
                gap_frames.append(f)

    assert len(gap_frames) > 0, "expected at least one frame to fall in the silent gap"
    for f in gap_frames:
        assert f.packets == [], f"silent frame at t={f.start_time} should be empty"


def test_window_right_edge_drives_emission_not_left_edge() -> None:
    """A window emits only when its right-edge frame has been observed.

    Packets only reach frame 15 → window F=0 (covers frames 0..15) emits, but
    F=8 (covers 8..23) does NOT, since frames 16..23 were never seen.
    """
    # 16 packets, one per frame, ending exactly at frame 15
    timestamps = [0.01 + i * 0.1 for i in range(16)]   # last at t=1.51 -> frame 15
    windower = SlidingWindow()
    windows = list(windower(_pkts(timestamps)))
    assert len(windows) == 1
    assert windows[0].start_time == pytest.approx(0.01)


def test_empty_packet_stream_emits_no_windows() -> None:
    windower = SlidingWindow()
    assert list(windower(iter([]))) == []


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        SlidingWindow(overlap=1.0)
    with pytest.raises(ValueError):
        SlidingWindow(overlap=-0.1)


def test_each_packet_appears_in_at_most_two_windows() -> None:
    """With overlap=0.5 a packet should land in at most 2 windows."""
    timestamps = [0.01 + i * 0.02 for i in range(250)]   # frames 0..49
    pkt_objs = list(_pkts(timestamps))
    windower = SlidingWindow()
    windows = list(windower(iter(pkt_objs)))

    counts: dict[int, int] = {id(p): 0 for p in pkt_objs}
    for w in windows:
        for f in w.frames:
            for p in f.packets:
                counts[id(p)] = counts.get(id(p), 0) + 1
    assert max(counts.values()) <= 2
    # Most packets should appear exactly twice (interior packets are always in 2 windows)
    assert sum(1 for c in counts.values() if c == 2) > 0
