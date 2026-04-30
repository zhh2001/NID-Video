"""Per-window train/val/test split assignment.

Two regimes, dispatched by the window's own label:

  1. **Attack-labelled window** (``label_id != BENIGN_ID``): index-based
     partition within ``(pcap_source, label_id)``. Sort the group's
     windows by ``start_time`` ascending, then take the first ``ratios[0]``
     fraction (by index, not time-position) → train, the next
     ``ratios[1]`` → val, the rest → test. Each attack class is
     **guaranteed** to split exactly 70/15/15 by sample count regardless
     of how clustered its windows are in time. The "no same-attack-
     session leakage" property is preserved because train windows still
     come earlier in time than val windows, which still come earlier
     than test windows.

  2. **BENIGN window** (``label_id == BENIGN_ID``): split by SHA-256 of
     ``(pcap_source, start_time, seed)`` → 0..99 bucket; ``< train_cutoff``
     train, ``< val_cutoff`` val, else test. Built-in ``hash()`` is
     intentionally NOT used — it is salted per-process by
     ``PYTHONHASHSEED`` and not stable across Python versions, which
     would silently break reproducibility between split.parquet
     generation and training-time filtering on a different machine.

Design history (4.1 → 4.7 first → 4.7 second)
---------------------------------------------
**M4.1 (initial)**: time-position partition with ``tmin``-tiebreak for
overlapping windows. Discovered in M4.7 to fail on **nested** attack
windows: CIC Wednesday DoS slowloris 09:01–14:25 contains Slowhttptest /
Hulk / GoldenEye entirely. With tmin-tiebreak, all inner-attack windows
used slowloris's 5h24min range → all fell into 0–70% → 100% in train.

**M4.7 first redesign (label-aware time-position)**: each attack-labelled
window uses its own label's ``(pcap_source, label_id) → AttackWindow``
range for position-based partition. Removed nested-window interference,
but exposed a deeper assumption error: **flow-level ``[tmin, tmax]`` from
CSV 5-tuple aggregation does not equal window-level dominant-rule active
period**. Empirically, slowloris-dominant windows clustered in the early
70% of slowloris's flow range → still 100/0/0; PortScan-dominant
windows clustered in the 70–85% slice → 9/82/9. The "uniform window
distribution within ``[tmin, tmax]``" assumption underlying time-position
partition is violated by every CIC attack class except a few.

**M4.7 second redesign (this version, index-based)**: replaced
time-position with index-based per-class partition. Sort each
``(pcap_source, label_id)`` group's windows by ``start_time``, take the
first 70% / middle 15% / last 15% **by index**. Mathematically guarantees
each class hits 70/15/15 by count. ``AttackWindow`` and
``attack_windows_for_pcaps`` are removed entirely — index-based partition
needs only the windows themselves, no externally-derived flow ranges.

Idea.md §3.5 + M4 tasks 4.1 / 4.7.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from nid_video.data.labeling import BENIGN_ID
from nid_video.utils import logger

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class WindowKey:
    """Stable identity of a window across shards/manifest/splits sidecars."""

    pcap_source: str   # filename, e.g. "Tuesday-WorkingHours.pcap"
    start_time: float  # window start (UTC unix epoch seconds)


@dataclass(frozen=True, slots=True)
class WindowKeyWithLabel:
    """A WindowKey paired with its raw15 label_id.

    Used as input to ``compute_split_assignments`` so the split logic can
    look up each window's OWN attack range (M4.7 redesign — see module
    docstring). ``label_id`` is the raw15 ID; ``BENIGN_ID = 0`` triggers
    the hash-bucket path, anything else triggers the per-label position
    path.

    Frozen+slots to remain hashable and usable as dict keys.
    """

    key: WindowKey
    label_id: int


# ---------------------------------------------------------------------------
# Internals: hashing, ratio bucketing, count allocation
# ---------------------------------------------------------------------------


def _hash_bucket(key: WindowKey, seed: int) -> int:
    """Deterministic 0..99 bucket via SHA-256.

    NB: Built-in ``hash()`` is intentionally NOT used — it is salted per
    process by ``PYTHONHASHSEED`` and not stable across Python versions,
    which would silently break reproducibility between the split.parquet
    generator and the training-time filter on a different machine.
    """
    payload = f"{key.pcap_source}|{key.start_time:.6f}|{seed}".encode()
    return int(sha256(payload).hexdigest()[:8], 16) % 100


def _bucket_to_split(bucket: int, ratios: tuple[float, float, float]) -> SplitName:
    """0..99 bucket → split via cumulative ratios."""
    train_cut = int(round(ratios[0] * 100))
    val_cut = int(round((ratios[0] + ratios[1]) * 100))
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def _position_to_split(
    pos_pct: float, ratios: tuple[float, float, float]
) -> SplitName:
    """Position within ``[tmin, tmax]`` (0..1) → split."""
    if pos_pct < ratios[0]:
        return "train"
    if pos_pct < ratios[0] + ratios[1]:
        return "val"
    return "test"


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {ratios}")
    if not (0.99 < sum(ratios) < 1.01):
        raise ValueError(f"ratios must sum to ~1.0, got {ratios} (sum={sum(ratios)})")


def _allocate_counts(
    n: int, ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Allocate (n_train, n_val, n_test) for an attack class with N windows.

    Truncated proportions, then minimum-1 enforcement: when ``n >= 3`` each
    split gets at least one sample (steal from train→val→test as needed
    to keep the count exact). For ``n < 3`` graceful degradation:
        n=1 → (1, 0, 0);  n=2 → (1, 1, 0)
    Caller should warn loudly when ``n < 10`` since splits this small
    yield high-variance per-class metrics.
    """
    if n <= 0:
        return (0, 0, 0)
    if n == 1:
        return (1, 0, 0)
    if n == 2:
        return (1, 1, 0)
    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1]))
    n_test = n - n_train - n_val
    if n_test < 1:
        # Steal from train (preferred) then val to ensure test gets >= 1.
        if n_train > 1:
            n_train -= 1
        else:
            n_val -= 1
        n_test = n - n_train - n_val
    return (n_train, n_val, n_test)


# ---------------------------------------------------------------------------
# Core: split assignment
# ---------------------------------------------------------------------------


def compute_split_assignments(
    window_keys_with_labels: Iterable[WindowKeyWithLabel],
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[WindowKey, SplitName]:
    """Compute per-window train/val/test assignment (M4.7 second redesign).

    Each input is a ``WindowKeyWithLabel`` — the window's identity plus
    its raw15 ``label_id``. Two-regime dispatch by label:

    * ``label_id == BENIGN_ID``: hash-bucket path. Deterministic 70/15/15
      via SHA-256 of ``(pcap_source, start_time, seed)``.

    * ``label_id != BENIGN_ID``: index-based partition within the
      ``(pcap_source, label_id)`` group. Sort by ``start_time``
      ascending, take first ``int(n * 0.7)`` → train, next ``int(n *
      0.15)`` → val (with min-1 enforcement), rest → test. Mathematical
      guarantee that each class hits 70/15/15 by count, regardless of
      how clustered its windows are in time. Earlier windows always go
      to train, later to test — preserving the "no same-session leakage"
      property of the original 4.1 design.

    Output is deterministic for a fixed
    ``(window_keys_with_labels, ratios, seed)``. The order in which
    ``window_keys_with_labels`` is iterated does not matter — sorting
    happens internally per group.
    """
    _validate_ratios(ratios)

    out: dict[WindowKey, SplitName] = {}
    attack_groups: dict[tuple[str, int], list[WindowKeyWithLabel]] = {}

    for wkl in window_keys_with_labels:
        if wkl.label_id == BENIGN_ID:
            # BENIGN raw15 ID is 0 — see labeling.LABEL_TO_ID_RAW.
            out[wkl.key] = _bucket_to_split(_hash_bucket(wkl.key, seed), ratios)
        else:
            attack_groups.setdefault((wkl.key.pcap_source, wkl.label_id), []).append(wkl)

    for (pcap_source, label_id), group in attack_groups.items():
        n = len(group)
        if n < 10:
            logger.warning(
                f"split: attack class label_id={label_id} in {pcap_source!r} "
                f"has only {n} windows (< 10). Index-based partition will "
                f"yield very small val/test sets; per-class F1 metrics will "
                f"be high-variance for this class."
            )
        n_train, n_val, _ = _allocate_counts(n, ratios)
        # Stable sort by start_time. Groups built per (pcap, label_id) so
        # all entries share the same pcap_source — start_time alone orders.
        group.sort(key=lambda w: w.key.start_time)
        for i, wkl in enumerate(group):
            if i < n_train:
                out[wkl.key] = "train"
            elif i < n_train + n_val:
                out[wkl.key] = "val"
            else:
                out[wkl.key] = "test"

    return out


# ---------------------------------------------------------------------------
# Persistence: parquet sidecar
# ---------------------------------------------------------------------------


_SPLITS_SCHEMA = pa.schema([
    ("pcap_source", pa.string()),
    ("start_time", pa.float64()),
    ("split", pa.string()),
])


def write_splits_parquet(
    assigns: dict[WindowKey, SplitName], path: Path,
) -> None:
    """Persist split assignments as parquet.

    Schema: ``(pcap_source: string, start_time: float64, split: string)``.
    """
    rows = [
        {"pcap_source": k.pcap_source, "start_time": k.start_time, "split": v}
        for k, v in assigns.items()
    ]
    table = pa.Table.from_pylist(rows, schema=_SPLITS_SCHEMA)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    logger.info(f"splits parquet written: {path} ({len(rows)} rows)")


def load_splits(path: Path) -> dict[WindowKey, SplitName]:
    """Load splits parquet into a ``WindowKey → split`` dict.

    Validates the documented schema. Raises ``ValueError`` on schema
    mismatch — silent column drift could let stale sidecar files match
    only a subset of windows.
    """
    table = pq.read_table(path)
    actual_cols = set(table.schema.names)
    expected_cols = {"pcap_source", "start_time", "split"}
    if not expected_cols.issubset(actual_cols):
        raise ValueError(
            f"{path}: missing required columns (need {expected_cols}, "
            f"have {sorted(actual_cols)})"
        )
    df = table.to_pandas()
    out: dict[WindowKey, SplitName] = {}
    for _, row in df.iterrows():
        key = WindowKey(pcap_source=str(row["pcap_source"]),
                        start_time=float(row["start_time"]))
        split = str(row["split"])
        if split not in ("train", "val", "test"):
            raise ValueError(f"{path}: bad split value {split!r} for {key}")
        out[key] = split  # type: ignore[assignment]
    return out


def verify_splits_complete(
    splits: dict[WindowKey, SplitName],
    manifest_keys: Iterable[WindowKey],
) -> None:
    """Raise ``ValueError`` if any manifest window has no split assignment.

    Catches drift between ETL output and the splits sidecar (e.g. ETL
    re-run produced new windows but the splits.parquet was not regenerated).
    """
    manifest_set = set(manifest_keys)
    missing = manifest_set - set(splits.keys())
    if missing:
        sample = list(missing)[:5]
        raise ValueError(
            f"splits incomplete: {len(missing)} of {len(manifest_set)} "
            f"manifest windows have no split assignment. Examples: {sample}"
        )


# ---------------------------------------------------------------------------
# CIC-IDS-2017 specific: csv-filename → pcap-filename mapping
# ---------------------------------------------------------------------------

# CIC's CSV naming has a day-of-week prefix; multiple CSVs share one pcap day.
# The actual pcap filenames come from the unb.ca distribution: note the
# lowercase 'w' in "Wednesday-workingHours.pcap" (sic).
_CIC_IDS_2017_DAY_TO_PCAP: dict[str, str] = {
    "monday":    "Monday-WorkingHours.pcap",
    "tuesday":   "Tuesday-WorkingHours.pcap",
    "wednesday": "Wednesday-workingHours.pcap",
    "thursday":  "Thursday-WorkingHours.pcap",   # not in the standard subset
    "friday":    "Friday-WorkingHours.pcap",
}


def cic_ids_2017_csv_to_pcap_map(
    csv_filenames: Iterable[str],
) -> dict[str, str]:
    """Map CIC-IDS-2017 CSV filenames to their parent pcap filenames.

    Uses the day-of-week prefix convention. CSVs without a recognised
    prefix are logged as WARNING and skipped — a hand-supplied override
    dict can be merged on top by the caller for non-standard names.
    """
    out: dict[str, str] = {}
    for fn in csv_filenames:
        lower = fn.lower()
        for day, pcap in _CIC_IDS_2017_DAY_TO_PCAP.items():
            if lower.startswith(day):
                out[fn] = pcap
                break
        else:
            logger.warning(
                f"split: csv {fn!r} matches no known day prefix "
                f"({sorted(_CIC_IDS_2017_DAY_TO_PCAP)}); skipping"
            )
    return out


# ---------------------------------------------------------------------------
# Manifest helper: collect WindowKeys from M2 webdataset shards
# ---------------------------------------------------------------------------


def collect_window_keys_from_shards(
    shard_pattern: str | list[str],
) -> Iterator[WindowKeyWithLabel]:
    """Yield ``WindowKeyWithLabel`` from M2 webdataset shards.

    Reads ``pcap_source`` / ``start_time`` / ``label_id`` from each
    sample's ``meta.json``. The ``label_id`` is the raw15 ID assigned by
    ETL's ``label_window`` (BENIGN_ID for non-attack-dominant windows).
    The split logic uses it to anchor each attack-labelled window to its
    own attack range (M4.7 redesign).

    Glob expansion is delegated to ``_resolve_shard_urls`` in
    ``nid_video.data.dataset`` — webdataset itself does not expand ``*``
    globs and would otherwise raise ``FileNotFoundError`` on the literal
    pattern string.
    """
    import webdataset as wds

    from nid_video.data.dataset import _resolve_shard_urls

    urls = _resolve_shard_urls(shard_pattern)
    dataset = wds.WebDataset(urls, shardshuffle=False).decode()
    for sample in dataset:
        meta = sample["meta.json"]
        yield WindowKeyWithLabel(
            key=WindowKey(
                pcap_source=str(meta["pcap_source"]),
                start_time=float(meta["start_time"]),
            ),
            label_id=int(meta["label_id"]),
        )
