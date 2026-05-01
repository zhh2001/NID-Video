"""CIC-IDS-2017 flow-label alignment for ETL windows.

Three CIC foot-guns to keep in mind, all fixed in this module:

  1. The CIC space-bug: every column name in TrafficLabelling/*.csv carries a
     literal leading space (e.g. ` Label`, ` Source IP`). _load_label_csv strips
     whitespace from all columns once on load so downstream code never has to
     know about it.

  2. The CSVs are CP-1252 (Windows-1252) encoded — not latin-1, not UTF-8.
     The labelling tool that produced them ran on Windows. Web-Attack labels
     contain an EN DASH stored as the single byte ``0x96``. CP-1252 decodes
     ``0x96`` into U+2013 (EN DASH) directly; latin-1 decodes it into the
     U+0096 control character, which then never matches LABEL_TO_ID and would
     silently fall through to BENIGN. _load_label_csv reads with cp1252 to
     resolve this. Discovered in the M3-to-M4 dry-run on real Thursday data.

  3. Some redistribution copies replace the EN DASH with an ASCII hyphen.
     normalize_label_name() handles that fallback.

Idea.md §3.5 (multi-class output, CIC-IDS labels).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nid_video.data.windowing import Window
from nid_video.utils import logger

# CIC-IDS-2017 carries 14 distinct attack labels plus BENIGN. We expose two
# label schemes; the ETL stage stores the RAW 15-class ID in the shard meta
# (data-set fidelity), and the trainer can apply collapse_to_13() at load time
# for the main experiment (interoperable with CIC-IDS-2018, which doesn't split
# Web-Attack subtypes). Decision: M2 task 2.6.
LABEL_TO_ID_RAW: dict[str, int] = {
    "BENIGN": 0,
    "DoS Hulk": 1,
    "PortScan": 2,
    "DDoS": 3,
    "DoS GoldenEye": 4,
    "FTP-Patator": 5,
    "SSH-Patator": 6,
    "DoS slowloris": 7,
    "DoS Slowhttptest": 8,
    "Bot": 9,
    "Web Attack – Brute Force": 10,
    "Web Attack – XSS": 11,
    "Web Attack – Sql Injection": 12,
    "Infiltration": 13,
    "Heartbleed": 14,
}
# Backwards-compatible alias used by call sites that don't care about the scheme.
LABEL_TO_ID: dict[str, int] = LABEL_TO_ID_RAW
ID_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_ID_RAW.items()}
BENIGN_ID = 0

# Collapsed 13-class scheme: the 3 Web-Attack subtypes (raw 10/11/12) merge into
# one "Web Attack" class (collapsed 10). Infiltration and Heartbleed shift down
# by 2 to keep the IDs contiguous.
LABEL_TO_ID_COLLAPSED: dict[str, int] = {
    "BENIGN": 0,
    "DoS Hulk": 1,
    "PortScan": 2,
    "DDoS": 3,
    "DoS GoldenEye": 4,
    "FTP-Patator": 5,
    "SSH-Patator": 6,
    "DoS slowloris": 7,
    "DoS Slowhttptest": 8,
    "Bot": 9,
    "Web Attack": 10,
    "Infiltration": 11,
    "Heartbleed": 12,
}
ID_TO_LABEL_COLLAPSED: dict[int, str] = {v: k for k, v in LABEL_TO_ID_COLLAPSED.items()}


def collapse_to_13(raw_id: int) -> int:
    """Map a RAW 15-class label ID to the collapsed 13-class scheme.

    raw 10/11/12 (Web Attack subtypes) → collapsed 10 (Web Attack)
    raw 13 (Infiltration)              → collapsed 11
    raw 14 (Heartbleed)                → collapsed 12
    raw 0..9                           → unchanged
    """
    if raw_id in (10, 11, 12):
        return 10
    if raw_id == 13:
        return 11
    if raw_id == 14:
        return 12
    if 0 <= raw_id <= 9:
        return raw_id
    raise ValueError(f"raw_id {raw_id} is outside the 15-class range [0, 14]")


def warn_low_population_classes(
    counts: dict[str, int],
    min_samples: int = 50,
) -> list[str]:
    """Log a WARNING for any non-BENIGN class with fewer than `min_samples` samples.

    Returns the sorted list of low-population class names so the caller can
    record them.

    Heartbleed in CIC-IDS-2017 is known to ship with ~11 flows in the entire
    dataset; expect this warning on any partial or full Heartbleed run. This
    is **expected behaviour**, not a bug — the surface here is to make it
    explicit so reviewers understand why the head's Heartbleed accuracy is
    near-zero in main results.
    """
    low: list[str] = []
    for label, count in sorted(counts.items()):
        if label == "BENIGN":
            continue
        if count < min_samples:
            logger.warning(
                f"class {label!r} has only {count} samples (< {min_samples}); "
                f"a useful classifier head for it is not learnable from this run. "
                f"Heartbleed in CIC-IDS-2017 (~11 flows total) routinely triggers this."
            )
            low.append(label)
    return low


def normalize_label_name(raw: str) -> str:
    """Canonicalize raw CSV label strings to LABEL_TO_ID keys.

    Currently handles:
      * leading/trailing whitespace on the cell value
      * ASCII-hyphen variants of "Web Attack -" → EN DASH "Web Attack –"
    """
    s = raw.strip()
    if s.startswith("Web Attack -"):
        s = "Web Attack –" + s[len("Web Attack -"):]
    return s


@dataclass(frozen=True, slots=True)
class FlowLabel:
    """One labeled flow record from CIC-IDS CSV."""

    start_ts: float    # Unix epoch seconds
    end_ts: float      # start_ts + Flow Duration (seconds)
    label_id: int


@dataclass(frozen=True, slots=True)
class WindowLabel:
    """Aggregated label decision for a single Window."""

    label: str             # canonical CIC label string
    label_id: int          # LABEL_TO_ID[label]
    dominant_ratio: float  # in [0, 1]; fraction of attack packets matching `label`
    counts: dict[str, int] # raw per-label packet counts (incl. BENIGN, incl. unmatched)
    n_unmatched: int       # packets whose 5-tuple+ts had no flow in the index


def _load_label_csv(path: Path) -> pd.DataFrame:
    """Load CIC-IDS-2017 TrafficLabelling CSV with the space-bug stripped.

    Shipped CSVs prefix every column with one literal space; we strip whitespace
    from all column names once on load. DO NOT remove this call — failing to
    strip leaves ` Label` undetectable to consumers expecting `Label`.

    Encoding is cp1252 (Windows-1252), not latin-1: the labelling tool was
    Windows-native and stores the EN DASH in Web-Attack labels as byte 0x96.
    See module docstring footgun #2.
    """
    df = pd.read_csv(path, encoding="cp1252", low_memory=False)
    df.columns = df.columns.str.strip()  # CIC space-bug fix
    return df


class LabelIndex:
    """Pre-built index from 5-tuple → list of (start_ts, end_ts, label) flows.

    Built once at startup from one or more TrafficLabelling CSVs. Lookup is a
    dict-get (O(1)) plus a tiny linear scan over flows sharing that 5-tuple
    (typically 1–3 elements).
    """

    def __init__(self) -> None:
        self._index: dict[
            tuple[str, int, str, int, int], list[FlowLabel]
        ] = defaultdict(list)
        self._n_flows = 0
        # Populated during _absorb. csv_source -> {label_id: (tmin, tmax)}
        # (BENIGN excluded). Used to build per-(csv, attack_class) time
        # bounds for the M4 split module.
        self._csv_attack_summaries: dict[str, dict[int, tuple[float, float]]] = {}

    @classmethod
    def from_csv(
        cls,
        paths: Path | Iterable[Path],
        *,
        dayfirst: bool = False,
        csv_tz: str = "America/Halifax",
        csv_twelve_hour_pm_inference: bool = True,
    ) -> "LabelIndex":
        """Build from one or more CSV paths.

        Args:
          dayfirst: pass True for raw CIC-IDS CSVs, whose timestamps are like
            ``5/7/2017 9:00:13`` meaning 5 July 2017 (DD/MM/YYYY).
          csv_tz: IANA tz database name (e.g. ``"America/Halifax"`` for CIC-IDS,
            ``"UTC"`` for synthetic test fixtures, ``"Australia/Sydney"`` if
            ever needed for UNSW-NB15). Wall-clock CSV times are localized to
            this zone and converted to UTC. DST is handled by zoneinfo.
            Default Halifax matches CIC-IDS-2017's recording site.
          csv_twelve_hour_pm_inference: enable CIC-IDS-2017's 12h-without-AM/PM
            timestamp recovery (hours 1..7 shifted +12h, see ``_absorb`` for
            full reasoning). Default True. Disable for datasets that already
            use 24h or carry explicit AM/PM markers — without this flag the
            wrong-direction shift would silently break those data sources.
        """
        if isinstance(paths, Path):
            paths = [paths]
        idx = cls()
        for p in paths:
            p = Path(p)
            df = _load_label_csv(p)
            idx._absorb(
                df, source=p.name, dayfirst=dayfirst, csv_tz=csv_tz,
                csv_twelve_hour_pm_inference=csv_twelve_hour_pm_inference,
            )
        logger.info(
            f"LabelIndex built: {idx._n_flows} flows across {idx.n_keys} 5-tuples"
        )
        return idx

    @property
    def n_keys(self) -> int:
        return len(self._index)

    @property
    def n_flows(self) -> int:
        return self._n_flows

    def _absorb(self, df: pd.DataFrame, source: str, dayfirst: bool,
                csv_tz: str, csv_twelve_hour_pm_inference: bool) -> None:
        required = ["Source IP", "Source Port", "Destination IP", "Destination Port",
                    "Protocol", "Timestamp", "Flow Duration", "Label"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{source}: missing required columns {missing}")

        # CIC's WebAttacks CSV ships with ~288k trailing all-empty rows (a CIC
        # tooling bug discovered in the M3-to-M4 dry-run). Dropping them up
        # front gives an honest count and stops them from being mis-reported
        # as "unparseable timestamps".
        n_before = len(df)
        df = df.dropna(how="all").reset_index(drop=True)
        n_empty = n_before - len(df)
        if n_empty > 0:
            logger.warning(f"{source}: {n_empty} fully-empty rows dropped")

        ts = pd.to_datetime(df["Timestamp"], dayfirst=dayfirst,
                            format="mixed", errors="coerce")
        bad_mask = ts.isna()
        if bad_mask.any():
            logger.warning(
                f"{source}: {int(bad_mask.sum())} rows with unparseable timestamps "
                f"dropped (after empty-row removal — these have data but bad ts)"
            )
            df = df.loc[~bad_mask].reset_index(drop=True)
            ts = ts.loc[~bad_mask].reset_index(drop=True)

        # CIC-IDS-2017 CSVs use 12-hour format WITHOUT AM/PM markers, encoding
        # the period implicitly in the hour range. ``pd.to_datetime`` parses
        # "2:54" as 02:54 (2:54 AM), but in CIC it actually means 14:54
        # (2:54 PM); the 12h offset makes every PM CSV row miss its pcap
        # packet by 12 hours. Discovered M4 task 4.7 — see Findings M4-001.
        #
        # Boundary reasoning (CIC working hours 09:00–17:00 ADT, never midnight):
        #   hour ∈ [1, 7]  → afternoon (PM, → 13:00–19:00 in 24h): add 12 hours
        #   hour ∈ [8, 11] → morning   (AM,    08:00–11:00 in 24h): unchanged
        #   hour == 12     → noon      (PM,    12:00      in 24h): unchanged
        #                    (12:00 AM = midnight is excluded by working hours;
        #                     12:00 PM is noon, which equals 12:00 24h, so no shift)
        #   hour == 0      → unexpected (CIC never captures midnight); warn.
        #
        # The shift MUST happen on naive datetimes BEFORE tz_localize. Doing it
        # after timezone conversion would crisscross the offset with DST and
        # bury the bug in another layer.
        #
        # Configurable: set csv_twelve_hour_pm_inference=False on LabelIndex
        # build for datasets that already use 24h or carry explicit AM/PM.
        if csv_twelve_hour_pm_inference:
            hours = ts.dt.hour
            n_zero = int((hours == 0).sum())
            if n_zero > 0:
                logger.warning(
                    f"{source}: {n_zero} rows with hour=0 (CIC working hours "
                    f"never include midnight). Possible data quality issue or "
                    f"wrong 12h-inference assumption for this dataset."
                )
            needs_pm_shift = hours.between(1, 7)
            n_shift = int(needs_pm_shift.sum())
            if n_shift > 0:
                ts = ts + pd.to_timedelta(needs_pm_shift.astype(int) * 12, unit="h")
                logger.info(
                    f"{source}: {n_shift} rows with hour∈[1,7] shifted +12h "
                    f"(CIC 12h-without-AM/PM PM inference)"
                )

        # Convert wall-clock CSV strings to UTC unix epoch. CIC-IDS-2017
        # records local Atlantic time (ADT in July, DST-aware); pcap ts is
        # UTC unix epoch. Naive int64 conversion would treat the CSV as UTC
        # and silently mis-align by 3-4 hours, causing every label lookup to
        # miss. zoneinfo handles DST automatically so cross-dataset users in
        # M5 (UNSW-NB15 in January = AST UTC-4, etc.) just pass a different
        # csv_tz.
        tz = ZoneInfo(csv_tz)
        ts_utc = ts.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT") \
                   .dt.tz_convert("UTC")
        # Unix-epoch float seconds (tz-aware Timestamp.astype('int64') is the
        # underlying UTC nanosecond value, which is what we want).
        start_unix = (ts_utc.astype("int64") // 10**9).astype(np.float64)
        # CIC-IDS Flow Duration is in microseconds
        duration_s = pd.to_numeric(df["Flow Duration"], errors="coerce").fillna(0.0) / 1e6
        end_unix = start_unix + duration_s.astype(np.float64)

        canon_labels = df["Label"].astype(str).map(normalize_label_name)
        label_ids = canon_labels.map(LABEL_TO_ID)
        unknown_mask = label_ids.isna()
        if unknown_mask.any():
            unknowns = canon_labels.loc[unknown_mask].unique()
            logger.warning(
                f"{source}: {int(unknown_mask.sum())} rows with unknown labels "
                f"defaulted to BENIGN. Examples: {list(unknowns)[:5]}"
            )
            label_ids = label_ids.fillna(BENIGN_ID)
        label_ids = label_ids.astype(int)

        attack_summary: dict[int, tuple[float, float]] = {}
        for sip, sport, dip, dport, proto, s, e, lid in zip(
            df["Source IP"].astype(str),
            df["Source Port"].astype(int),
            df["Destination IP"].astype(str),
            df["Destination Port"].astype(int),
            df["Protocol"].astype(int),
            start_unix, end_unix, label_ids,
            strict=True,
        ):
            self._index[(sip, sport, dip, dport, proto)].append(
                FlowLabel(start_ts=float(s), end_ts=float(e), label_id=int(lid))
            )
            self._n_flows += 1
            lid_int = int(lid)
            if lid_int != BENIGN_ID:
                s_f = float(s)
                if lid_int in attack_summary:
                    omin, omax = attack_summary[lid_int]
                    if s_f < omin or s_f > omax:
                        attack_summary[lid_int] = (min(omin, s_f), max(omax, s_f))
                else:
                    attack_summary[lid_int] = (s_f, s_f)
        self._csv_attack_summaries[source] = attack_summary

    @cached_property
    def attack_windows_by_csv(self) -> dict[str, list[tuple[int, float, float]]]:
        """Per-CSV attack-class time bounds, used by the M4 split module.

        Returns ``{csv_source: [(label_id, tmin, tmax), ...]}`` with
        BENIGN excluded. Each ``(label_id, tmin, tmax)`` is the [min,max]
        ``start_ts`` over all flows in that CSV with that ``label_id``.

        Lazily computed on first access from ``_csv_attack_summaries``
        (which is populated during ``from_csv``). Cached per-instance —
        the index is treated as immutable after construction. Each list
        is sorted by ``tmin`` ascending so callers that need overlap
        tiebreak (split.py) get earliest-first ordering for free.
        """
        return {
            csv_src: sorted(
                [(lid, tmin, tmax) for lid, (tmin, tmax) in by_label.items()],
                key=lambda t: t[1],
            )
            for csv_src, by_label in self._csv_attack_summaries.items()
        }

    def lookup(
        self,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
        protocol: int,
        ts: float,
    ) -> int | None:
        """Return label_id for the matching flow, else None.

        Tries the forward 5-tuple first, then the reversed direction so the
        return leg of a connection picks up the same label as the request leg.
        """
        forward = (src_ip, src_port, dst_ip, dst_port, protocol)
        for flow in self._index.get(forward, ()):
            if flow.start_ts <= ts <= flow.end_ts:
                return flow.label_id
        reverse = (dst_ip, dst_port, src_ip, src_port, protocol)
        for flow in self._index.get(reverse, ()):
            if flow.start_ts <= ts <= flow.end_ts:
                return flow.label_id
        return None


def label_window(window: Window, label_index: LabelIndex) -> WindowLabel:
    """Aggregate per-packet labels and decide a single label for the window.

    Rules (Idea.md §3.5):
      * All packets BENIGN (or unmatched, default-BENIGN) → "BENIGN"
      * Any attack packets → dominant attack class (highest count)
      * Mixed attacks → log a warning with the breakdown
      * >50% unmatched packets → log a warning (suggests systematic CSV/PCAP
        time/index mismatch); ≤50% just emits a debug count.

    Returns dict (not str) to provide dominant_attack_ratio for webdataset
    meta.json (M2 §2.7). Decision: M2 task 2.6.
    """
    counts: Counter[str] = Counter()
    n_unmatched = 0

    for frame in window.frames:
        for pkt in frame.packets:
            lid = label_index.lookup(
                pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port,
                pkt.protocol, pkt.timestamp,
            )
            if lid is None:
                n_unmatched += 1
                counts[ID_TO_LABEL[BENIGN_ID]] += 1     # default to BENIGN
            else:
                counts[ID_TO_LABEL[lid]] += 1

    total = sum(counts.values())
    if total == 0:                                       # empty window
        return WindowLabel("BENIGN", BENIGN_ID, 1.0, {}, n_unmatched)

    if n_unmatched > 0:
        pct = n_unmatched / total
        msg = (
            f"window @ {window.start_time:.3f}: {n_unmatched}/{total} "
            f"packets unmatched ({pct * 100:.1f}%); defaulting to BENIGN"
        )
        if pct > 0.5:
            logger.warning(msg)
        else:
            logger.debug(msg)

    attack_counts = {k: v for k, v in counts.items() if LABEL_TO_ID[k] != BENIGN_ID}
    if not attack_counts:
        return WindowLabel(
            "BENIGN", BENIGN_ID,
            counts.get("BENIGN", 0) / total,
            dict(counts), n_unmatched,
        )

    if len(attack_counts) > 1:
        breakdown = ", ".join(
            f"{k}={v}" for k, v in
            sorted(attack_counts.items(), key=lambda kv: -kv[1])
        )
        logger.warning(
            f"window @ {window.start_time:.3f} has mixed attack labels: {breakdown}"
        )

    dominant_label, dominant_count = max(attack_counts.items(), key=lambda kv: kv[1])
    return WindowLabel(
        label=dominant_label,
        label_id=LABEL_TO_ID[dominant_label],
        dominant_ratio=dominant_count / sum(attack_counts.values()),
        counts=dict(counts),
        n_unmatched=n_unmatched,
    )
