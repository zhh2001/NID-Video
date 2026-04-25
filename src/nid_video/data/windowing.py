"""Sliding-window splitter: PacketRecord stream → fixed-length, overlapping Windows.

Each Window covers T·Δt seconds (default 1.6 s = 16 × 100 ms) and is divided into
T consecutive Frame bins. Windows slide by step = T·Δt·(1-overlap), default 0.8 s.

Streaming, not batch: only buffers the in-flight T frames. Empty frames are
preserved on output (silence is signal — Idea.md §3.2). Trailing partial windows
whose right edge was never reached are dropped at end-of-stream.

Idea.md §3.2 (Stage 2 · 帧构造).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from nid_video.data.pcap_parser import PacketRecord
from nid_video.utils import logger


@dataclass(slots=True)
class Frame:
    """One Δt-wide bin within a Window. Empty list of packets = silent frame."""

    start_time: float
    end_time: float
    packets: list[PacketRecord]


@dataclass(slots=True)
class Window:
    """T-frame contiguous slice of a packet stream."""

    start_time: float
    frames: list[Frame]
    pcap_source: str


class SlidingWindow:
    """Build overlapping fixed-length windows from a streaming packet iterator.

    Idea.md §3.2.

    The first window aligns to the first observed packet's timestamp; subsequent
    windows are spaced exactly `step * delta_t_s` apart. Float drift on the
    boundary check is absorbed by a tiny epsilon.
    """

    def __init__(
        self,
        delta_t_s: float = 0.1,
        num_frames: int = 16,
        overlap: float = 0.5,
    ) -> None:
        if delta_t_s <= 0:
            raise ValueError(f"delta_t_s must be > 0, got {delta_t_s}")
        if num_frames <= 0:
            raise ValueError(f"num_frames must be > 0, got {num_frames}")
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        step = int(round(num_frames * (1.0 - overlap)))
        if step <= 0 or step > num_frames:
            raise ValueError(
                f"derived step={step} from T={num_frames}, overlap={overlap} is invalid"
            )
        self.delta_t_s = delta_t_s
        self.num_frames = num_frames
        self.overlap = overlap
        self.step = step

    def __call__(
        self,
        packets: Iterable[PacketRecord],
        pcap_source: str = "",
    ) -> Iterator[Window]:
        return self._iter(packets, pcap_source)

    def _iter(
        self,
        packets: Iterable[PacketRecord],
        pcap_source: str,
    ) -> Iterator[Window]:
        T = self.num_frames
        dt = self.delta_t_s
        step = self.step
        eps = 1e-9  # absorb float drift at frame boundaries

        bins: dict[int, list[PacketRecord]] = defaultdict(list)
        origin: float | None = None
        next_window_frame = 0
        last_frame_seen = -1
        n_packets = 0
        n_emitted = 0

        for pkt in packets:
            n_packets += 1
            if origin is None:
                origin = pkt.timestamp

            elapsed = pkt.timestamp - origin
            frame_idx = math.floor(elapsed / dt + eps)
            if frame_idx < 0:
                logger.warning(
                    f"packet predates origin (ts={pkt.timestamp} < origin={origin}); skipped"
                )
                continue
            if frame_idx > last_frame_seen:
                last_frame_seen = frame_idx
            bins[frame_idx].append(pkt)

            # Emit every window whose right edge has been provably passed
            # (a packet at frame_idx >= F + T means no more packets will land
            # in frames [F, F+T-1] because pcap timestamps are monotonic).
            while last_frame_seen >= next_window_frame + T:
                yield self._build_window(bins, next_window_frame, origin, pcap_source)
                n_emitted += 1
                # Frames [next_window_frame, next_window_frame+step) drop out of all
                # future open windows; safe to free.
                for f in range(next_window_frame, next_window_frame + step):
                    bins.pop(f, None)
                next_window_frame += step

        # End of stream: emit windows whose right-edge frame has been observed.
        # Trailing windows that would be padded with phantom future frames are dropped.
        if origin is not None:
            while last_frame_seen >= next_window_frame + T - 1:
                yield self._build_window(bins, next_window_frame, origin, pcap_source)
                n_emitted += 1
                for f in range(next_window_frame, next_window_frame + step):
                    bins.pop(f, None)
                next_window_frame += step

        logger.info(
            f"sliding_window {pcap_source or '<stream>'}: "
            f"{n_packets} packets -> {n_emitted} windows "
            f"(T={T}, dt={dt}, overlap={self.overlap})"
        )

    def _build_window(
        self,
        bins: dict[int, list[PacketRecord]],
        start_frame: int,
        origin: float,
        pcap_source: str,
    ) -> Window:
        T = self.num_frames
        dt = self.delta_t_s
        frames = [
            Frame(
                start_time=origin + (start_frame + k) * dt,
                end_time=origin + (start_frame + k + 1) * dt,
                packets=bins.get(start_frame + k, []),
            )
            for k in range(T)
        ]
        return Window(
            start_time=origin + start_frame * dt,
            frames=frames,
            pcap_source=pcap_source,
        )
