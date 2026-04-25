"""Per-window source-IP clustering for the H=32 row dimension.

Idea.md §3.2 says the H axis groups source IPs by behavioral similarity. Each
window stands alone: k-means is fit on THIS window's active source IPs only.

DO NOT fit a global k-means over the dataset. Doing so would assign attacker IPs
to fixed rows because training saw them, leaking labels at inference and inflating
closed-world metrics. Per-window k-means has zero carry-over between windows.
(Decision recorded in M2 task design memo and feedback memory.)

Cluster → row assignment:
  * The active IPs in this window are clustered into n_clusters buckets.
  * Each bucket is ranked by the total packet count of its members (descending).
  * The bucket with the most traffic is assigned row 0; the least → row n_clusters-1.
  * All IPs in the same bucket map to the same row (many-to-one).

Cold-start fallback:
  * If active IPs < n_clusters, skip k-means entirely; each IP gets its own row,
    sorted by per-IP packet count descending. Empty rows are simply not present
    in the returned dict — the channel encoder treats absent rows as zero.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from nid_video.data.pcap_parser import PacketRecord
from nid_video.data.windowing import Window
from nid_video.utils import logger

TCP_FLAGS_MASK = 0x3F


@dataclass(slots=True)
class IPStats:
    """Behavioral signature of one source IP within one window."""

    n_packets: int = 0
    n_bytes: int = 0
    n_outbound: int = 0
    flag_or: int = 0
    min_ts: float = float("inf")
    max_ts: float = float("-inf")
    dst_ports: set[int] = field(default_factory=set)
    dst_ips: set[str] = field(default_factory=set)

    def add(self, pkt: PacketRecord) -> None:
        self.n_packets += 1
        self.n_bytes += pkt.pkt_size
        if pkt.direction == 1:
            self.n_outbound += 1
        self.flag_or |= pkt.tcp_flags & TCP_FLAGS_MASK
        if pkt.timestamp < self.min_ts:
            self.min_ts = pkt.timestamp
        if pkt.timestamp > self.max_ts:
            self.max_ts = pkt.timestamp
        self.dst_ports.add(pkt.dst_port)
        self.dst_ips.add(pkt.dst_ip)

    def feature_vector(self) -> tuple[float, float, float, float, float, float, float]:
        """7-dim vector: (log packets, log bytes, #dst_ports, #dst_ips,
        outbound_ratio, mean_iat, flag_diversity)."""
        log_pkts = math.log1p(self.n_packets)
        log_bytes = math.log1p(self.n_bytes)
        n_dst_ports = float(len(self.dst_ports))
        n_dst_ips = float(len(self.dst_ips))
        out_ratio = self.n_outbound / self.n_packets if self.n_packets > 0 else 0.0
        if self.n_packets >= 2 and math.isfinite(self.max_ts) and math.isfinite(self.min_ts):
            mean_iat = (self.max_ts - self.min_ts) / max(self.n_packets - 1, 1)
        else:
            mean_iat = 0.0
        flag_diversity = float(bin(self.flag_or).count("1"))
        return (log_pkts, log_bytes, n_dst_ports, n_dst_ips,
                out_ratio, mean_iat, flag_diversity)


def _collect_per_ip(window: Window) -> dict[str, IPStats]:
    """Walk all packets in the window, keyed by source IP."""
    by_ip: dict[str, IPStats] = {}
    for frame in window.frames:
        for pkt in frame.packets:
            stats = by_ip.get(pkt.src_ip)
            if stats is None:
                stats = IPStats()
                by_ip[pkt.src_ip] = stats
            stats.add(pkt)
    return by_ip


def cluster_ips_in_window(
    window: Window,
    n_clusters: int = 32,
    seed: int = 42,
) -> dict[str, int]:
    """Cluster source IPs in a single window into rows. Idea.md §3.2.

    Returns a dict mapping source-IP-string to row index in [0, n_clusters).
    Many-to-one when active IPs ≥ n_clusters; one-to-one (with empty rows
    omitted) when active IPs < n_clusters. Empty windows return {}.
    """
    if n_clusters <= 0:
        raise ValueError(f"n_clusters must be > 0, got {n_clusters}")

    by_ip = _collect_per_ip(window)
    if not by_ip:
        return {}

    n_active = len(by_ip)

    # Cold-start path: each IP gets its own row, ranked by packet count desc.
    # This subsumes the "< 5" extreme-cold-start case from the M2 prompt.
    if n_active < n_clusters:
        sorted_ips = sorted(
            by_ip.items(),
            key=lambda kv: (-kv[1].n_packets, kv[0]),  # tiebreak by ip string
        )
        return {ip: i for i, (ip, _) in enumerate(sorted_ips)}

    # k-means path: ≥ n_clusters active IPs.
    ip_list = list(by_ip.keys())
    feature_matrix = np.asarray(
        [by_ip[ip].feature_vector() for ip in ip_list],
        dtype=np.float64,
    )
    feat_norm = StandardScaler().fit_transform(feature_matrix)

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
        batch_size=min(n_active, 1024),
        max_iter=100,
    )
    cluster_ids = km.fit_predict(feat_norm)

    # Rank clusters by the total packets they own, descending.
    cluster_total_pkts: dict[int, int] = defaultdict(int)
    for ip, cid in zip(ip_list, cluster_ids, strict=True):
        cluster_total_pkts[int(cid)] += by_ip[ip].n_packets
    cluster_to_row: dict[int, int] = {
        cid: rank
        for rank, (cid, _) in enumerate(
            # secondary key (cid) breaks ties so two equal-sized clusters always
            # produce the same row order across runs.
            sorted(cluster_total_pkts.items(), key=lambda kv: (-kv[1], kv[0]))
        )
    }

    out = {ip: cluster_to_row[int(cid)]
           for ip, cid in zip(ip_list, cluster_ids, strict=True)}

    logger.debug(
        "cluster_ips: window {} active_ips={} -> {} rows used",
        window.start_time, n_active, len({v for v in out.values()}),
    )
    return out
