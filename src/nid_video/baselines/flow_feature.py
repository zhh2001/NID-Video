"""M6.2 — flow-feature paradigm baseline (RF + XGBoost on CICFlowMeter CSVs).

Implements the "per-flow train, per-window max-confidence aggregate" pipeline
(Phase 0 option B):

  1. Parse 8 TrafficLabelling CSVs into a unified per-flow table:
       columns = 80 numeric flow features + 5-tuple + (start_ts, end_ts)
                 + label_id (raw15) + label_id_collapsed (13).
     Reuses ``nid_video.data.labeling._load_label_csv`` + the cp1252 /
     dayfirst / 12h-PM-inference / America/Halifax-tz conversion logic
     anchored in M4-001.

  2. Enumerate the cross-cell-comparable val window set (val_n=18,156)
     by union-ing fast val keys from ``splits.parquet`` with the slow
     shard keys whose (pcap_source, start_time) appears in the fast val
     set. Each enumerated window carries its own duration_s (fast=1.6s,
     slow=16s) and stream tag (``"fast"`` / ``"slow"``).

  3. Per-flow train/val/test split: each flow is assigned to the fast
     window whose ``start_time`` is *closest* to ``flow.start_ts``. The
     flow's split = that window's split. Deterministic; ties broken by
     lowest start_time. Flows that fall outside any window's [-60s, +60s]
     coverage of any same-pcap fast window are dropped (rare; CIC's
     minute-rounded timestamps imply at-worst 60s offset from the true
     packet timestamps).

  4. Per-window max-confidence aggregation at eval: for each val window
     find all flows whose interval [start_ts, end_ts] intersects
     [w_start, w_start + duration_s]; predict per-flow via
     ``model.predict_proba``; the per-window prediction is the argmax
     of the highest-confidence flow's probability vector. Windows with
     0 matching flows default to BENIGN with one-hot probability.

Outputs (under ``--output-dir``): the canonical 6-deliverable schema
matching M5.10 video-cell eval bundles —

  ``model.pkl``                  trained sklearn / xgboost classifier
  ``eval_metrics.json``          combined / fast / slow metrics + per-class
  ``per_class_table.csv``        13-class P/R/F1/AUROC + n
  ``confusion_matrix.json``      13×13 int counts for combined
  ``training.log``               stdout/stderr capture of the training run
  ``feature_importances.csv``    80 features × importance (RF / XGB native)
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from nid_video.data.labeling import (
    BENIGN_ID,
    LABEL_TO_ID_COLLAPSED,
    LABEL_TO_ID_RAW,
    _load_label_csv,
    collapse_to_13,
    normalize_label_name,
)
from nid_video.data.split import load_splits
from nid_video.utils import logger

# --- constants ---------------------------------------------------------------

# CIC-IDS-2017 official tz + offset rules (mirrors LabelIndex._absorb).
CSV_TZ = "America/Halifax"
CSV_DAYFIRST = True
CSV_TWELVE_HOUR_PM_INFERENCE = True

# Non-feature columns excluded from RF/XGB input. Flow ID is a CIC hash;
# Source/Destination IP would leak attacker-IP identity into the model
# (closed-world contamination); Timestamp is a date string; Label is the
# target. Source/Destination Port are kept as numeric features (standard
# CICFlowMeter baseline practice).
NON_FEATURE_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label",
}

# Window duration by stream, in seconds. fast = T × Δt = 16 × 0.1 = 1.6 s;
# slow = T × Δt = 16 × 1.0 = 16 s. These match the SlidingWindow construction
# in the video ETL.
WINDOW_DURATION_S = {"fast": 16 * 0.1, "slow": 16 * 1.0}

# Per-pcap CSV mapping (which CSVs cover which pcap, by CIC convention).
PCAP_TO_CSVS: dict[str, tuple[str, ...]] = {
    "Tuesday-WorkingHours.pcap": ("Tuesday-WorkingHours.pcap_ISCX.csv",),
    "Wednesday-workingHours.pcap": ("Wednesday-workingHours.pcap_ISCX.csv",),
    "Friday-WorkingHours.pcap": (
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    ),
}


# --- step 1: flow-table parsing ---------------------------------------------


def _parse_csv_timestamps(ts_str: pd.Series) -> pd.Series:
    """Parse CIC CSV Timestamp column into UTC-unix-epoch float seconds.

    Mirrors ``LabelIndex._absorb`` step-for-step so flow timestamps are
    comparable to webdataset window keys (which are UTC unix epoch).
    """
    from zoneinfo import ZoneInfo

    ts = pd.to_datetime(ts_str, dayfirst=CSV_DAYFIRST, format="mixed",
                        errors="coerce")
    bad_mask = ts.isna()
    if bad_mask.any():
        logger.warning(
            f"flow-table CSV: {int(bad_mask.sum())} rows with unparseable "
            f"timestamps will produce NaT (dropped downstream)"
        )

    if CSV_TWELVE_HOUR_PM_INFERENCE:
        hours = ts.dt.hour
        needs_pm_shift = hours.between(1, 7)
        n_shift = int(needs_pm_shift.sum())
        if n_shift > 0:
            ts = ts + pd.to_timedelta(needs_pm_shift.astype(int) * 12, unit="h")

    tz = ZoneInfo(CSV_TZ)
    ts_utc = ts.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT") \
               .dt.tz_convert("UTC")
    return (ts_utc.astype("int64") // 10**9).astype(np.float64)


def load_flow_table(csv_dir: Path) -> pd.DataFrame:
    """Parse all 8 TrafficLabelling CSVs into a unified per-flow DataFrame.

    Returned columns:
      ``pcap_source``      : str — which pcap the flow belongs to
      ``start_ts``         : float — UTC unix epoch seconds (CIC tz-converted)
      ``end_ts``           : float — start_ts + Flow_Duration / 1e6
      ``label_id``         : int — raw 15-class ID
      ``label_id_collapsed`` : int — collapsed 13-class ID
      ``Source Port``, ``Destination Port``, ``Protocol``, …,
      ``Idle Min`` — 80 numeric features (CICFlowMeter standard).
    """
    csv_dir = Path(csv_dir)
    frames: list[pd.DataFrame] = []
    for pcap_source, csv_names in PCAP_TO_CSVS.items():
        for name in csv_names:
            path = csv_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"missing CSV: {path}")
            df = _load_label_csv(path)
            n_before = len(df)
            df = df.dropna(how="all").reset_index(drop=True)
            n_empty = n_before - len(df)
            if n_empty > 0:
                logger.warning(
                    f"{name}: {n_empty} fully-empty rows dropped"
                )

            df["pcap_source"] = pcap_source

            df["start_ts"] = _parse_csv_timestamps(df["Timestamp"])
            duration_s = (
                pd.to_numeric(df["Flow Duration"], errors="coerce").fillna(0.0)
                / 1e6
            )
            df["end_ts"] = df["start_ts"] + duration_s.astype(np.float64)

            canon = df["Label"].astype(str).map(normalize_label_name)
            df["label_id"] = canon.map(LABEL_TO_ID_RAW).fillna(BENIGN_ID).astype(int)
            df["label_id_collapsed"] = df["label_id"].map(collapse_to_13).astype(int)

            # Drop rows with unparseable timestamps (NaT-derived NaN
            # propagated through; cannot be assigned to any window).
            n_bad_ts = int(df["start_ts"].isna().sum())
            if n_bad_ts > 0:
                logger.warning(
                    f"{name}: {n_bad_ts} rows with NaT timestamps dropped"
                )
                df = df.dropna(subset=["start_ts"]).reset_index(drop=True)
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        f"flow table: {len(combined):,} flows across {combined.pcap_source.nunique()}"
        f" pcaps, label distribution (collapsed): "
        f"{dict(combined.label_id_collapsed.value_counts().sort_index())}"
    )
    return combined


# --- step 2: window enumeration ---------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowInfo:
    """A single val/train/test window with its time interval + stream tag."""

    pcap_source: str
    start_time: float
    split: str          # "train" | "val" | "test"
    stream: str         # "fast" | "slow"

    @property
    def end_time(self) -> float:
        return self.start_time + WINDOW_DURATION_S[self.stream]


def enumerate_val_windows(
    splits_path: Path,
    slow_shard_pattern: str | None,
) -> list[WindowInfo]:
    """Build the val_n=18,156 window list = (fast val from splits) ∪
    (slow keys in splits's val set).

    The slow side requires scanning slow shards' meta.json to find keys;
    if ``slow_shard_pattern`` is None, returns fast-only (val_n=16,463).
    """
    splits = load_splits(splits_path)
    fast_val: list[WindowInfo] = [
        WindowInfo(
            pcap_source=k.pcap_source,
            start_time=k.start_time,
            split=v,
            stream="fast",
        )
        for k, v in splits.items() if v == "val"
    ]

    slow_val: list[WindowInfo] = []
    if slow_shard_pattern is not None:
        from nid_video.data.split import collect_window_keys_from_shards

        for kvl in collect_window_keys_from_shards(slow_shard_pattern):
            if splits.get(kvl.key) == "val":
                slow_val.append(WindowInfo(
                    pcap_source=kvl.key.pcap_source,
                    start_time=kvl.key.start_time,
                    split="val",
                    stream="slow",
                ))

    logger.info(
        f"val window enumeration: fast={len(fast_val)} slow={len(slow_val)} "
        f"total={len(fast_val) + len(slow_val)}"
    )
    return fast_val + slow_val


# --- step 3: per-flow split assignment ---------------------------------------


def _fast_split_lookup_table(splits_path: Path) -> dict[str, np.ndarray]:
    """Return per-pcap arrays of (start_time, split_int) sorted by start_time.

    split_int: 0=train, 1=val, 2=test (matches sklearn label convention).
    Built once for the assign_flow_splits inner loop.
    """
    splits = load_splits(splits_path)
    by_pcap: dict[str, list[tuple[float, int]]] = {}
    split_to_int = {"train": 0, "val": 1, "test": 2}
    for k, v in splits.items():
        by_pcap.setdefault(k.pcap_source, []).append(
            (k.start_time, split_to_int[v])
        )
    out: dict[str, np.ndarray] = {}
    for pcap, rows in by_pcap.items():
        arr = np.array(rows, dtype=[("st", "f8"), ("sp", "i4")])
        arr.sort(order="st")
        out[pcap] = arr
    return out


def assign_flow_splits(flow_table: pd.DataFrame, splits_path: Path) -> pd.DataFrame:
    """Add a ``split`` column ("train"/"val"/"test"/"none") to ``flow_table``.

    Each flow is assigned to the fast window whose ``start_time`` is
    *closest* to ``flow.start_ts`` (within ±60s; outside that range the
    flow gets ``"none"`` and is excluded from training). Deterministic;
    ties broken by lower start_time (earlier window wins).
    """
    tables = _fast_split_lookup_table(splits_path)
    int_to_split = {0: "train", 1: "val", 2: "test"}

    splits_out = np.full(len(flow_table), "none", dtype=object)
    by_pcap_groups = flow_table.groupby("pcap_source", sort=False).groups
    for pcap_source, idx in by_pcap_groups.items():
        if pcap_source not in tables:
            logger.warning(f"no splits.parquet entries for pcap {pcap_source}")
            continue
        arr = tables[pcap_source]
        start_times = arr["st"]
        flow_ts = flow_table.loc[idx, "start_ts"].to_numpy()

        # For each flow, find the nearest fast window start_time.
        # np.searchsorted returns insertion index; check left/right neighbours.
        pos = np.searchsorted(start_times, flow_ts)
        pos_l = np.clip(pos - 1, 0, len(start_times) - 1)
        pos_r = np.clip(pos, 0, len(start_times) - 1)
        d_l = np.abs(flow_ts - start_times[pos_l])
        d_r = np.abs(flow_ts - start_times[pos_r])
        pick = np.where(d_l <= d_r, pos_l, pos_r)
        d_pick = np.where(d_l <= d_r, d_l, d_r)

        # Only accept the assignment if within ±60s (CIC minute-rounded
        # timestamps imply at most 60s offset from the real packet ts).
        ok = d_pick <= 60.0
        sp_int = arr["sp"][pick]
        for j, (flow_idx, accepted) in enumerate(zip(idx.tolist(), ok.tolist())):
            if accepted:
                splits_out[flow_idx] = int_to_split[int(sp_int[j])]

    flow_table = flow_table.copy()
    flow_table["split"] = splits_out
    counts = flow_table["split"].value_counts().to_dict()
    logger.info(f"flow split assignment: {counts}")
    return flow_table


# --- step 4: features + clean ------------------------------------------------


def feature_columns(flow_table: pd.DataFrame) -> list[str]:
    """Return the 80 CICFlowMeter feature column names (numeric, in CSV order)."""
    return [
        c for c in flow_table.columns
        if c not in NON_FEATURE_COLUMNS
        and c not in ("pcap_source", "start_ts", "end_ts",
                      "label_id", "label_id_collapsed", "split")
    ]


def prepare_xy(
    flow_table: pd.DataFrame,
    split: str,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y_collapsed) for the requested split.

    Replaces ±Inf with NaN, then NaN with 0.0 (RF can't handle NaN
    natively in sklearn < 1.4; XGB handles natively but we normalise
    both for cross-model parity).
    """
    sub = flow_table[flow_table["split"] == split]
    X = sub[feat_cols].to_numpy(dtype=np.float64, copy=True)
    X[~np.isfinite(X)] = np.nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = sub["label_id_collapsed"].to_numpy(dtype=np.int64)
    return X, y


# --- step 5: per-window aggregation -----------------------------------------


def _sort_flows_for_lookup(flow_table: pd.DataFrame) -> dict[str, dict]:
    """Index flows by pcap_source, sorted by start_ts. Returns
    {pcap: {"start_ts": np.ndarray, "end_ts": np.ndarray,
             "row_idx": np.ndarray}} for fast interval search.
    """
    out: dict[str, dict] = {}
    for pcap, df in flow_table.groupby("pcap_source", sort=False):
        order = np.argsort(df["start_ts"].to_numpy(), kind="stable")
        out[pcap] = {
            "start_ts": df["start_ts"].to_numpy()[order],
            "end_ts": df["end_ts"].to_numpy()[order],
            "row_idx": df.index.to_numpy()[order],
        }
    return out


def find_active_flows(
    window: WindowInfo,
    flow_index: dict[str, dict],
) -> np.ndarray:
    """Return the indices (into the original flow_table) of flows that
    intersect the given window's [start_time, end_time] interval.
    """
    info = flow_index.get(window.pcap_source)
    if info is None:
        return np.empty(0, dtype=np.int64)
    w_start = window.start_time
    w_end = window.end_time

    # A flow [f_start, f_end] intersects [w_start, w_end] iff
    #   f_start <= w_end  AND  f_end >= w_start.
    # We binary-search f_start <= w_end (left bound), then linear filter
    # by f_end >= w_start. Most windows match few flows.
    starts = info["start_ts"]
    ends = info["end_ts"]
    rows = info["row_idx"]
    cutoff = np.searchsorted(starts, w_end, side="right")  # f_start <= w_end
    if cutoff == 0:
        return np.empty(0, dtype=np.int64)
    pre_starts = starts[:cutoff]
    pre_ends = ends[:cutoff]
    pre_rows = rows[:cutoff]
    keep = pre_ends >= w_start
    return pre_rows[keep]


def aggregate_per_window_predictions(
    windows: list[WindowInfo],
    flow_table: pd.DataFrame,
    proba: np.ndarray,
    feat_cols: list[str],
    flow_idx_to_proba_row: dict[int, int],
    n_classes: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window prediction via max-confidence aggregation.

    For each window, look up active flows; pick the one with highest
    ``max(proba_row)`` and use its argmax as the window prediction. The
    full proba vector of the winning flow is stored too (for AUROC).
    Windows with 0 matching flows default to BENIGN with a one-hot
    probability vector.

    Returns (window_preds[N], window_proba[N, n_classes]).
    """
    flow_index = _sort_flows_for_lookup(flow_table)
    N = len(windows)
    preds = np.full(N, BENIGN_ID, dtype=np.int64)
    out_proba = np.zeros((N, n_classes), dtype=np.float64)
    out_proba[:, BENIGN_ID] = 1.0  # default one-hot BENIGN

    n_zero_flow = 0
    for i, w in enumerate(windows):
        active = find_active_flows(w, flow_index)
        if len(active) == 0:
            n_zero_flow += 1
            continue
        # Map raw flow_table indices to proba rows (only flows in the predict
        # set have a proba row; flows in train split do not — but our predict
        # set should cover all val + test + train flows of interest at eval
        # time, so this lookup must succeed for any active flow used here).
        proba_rows = np.array(
            [flow_idx_to_proba_row[int(j)] for j in active], dtype=np.int64
        )
        active_proba = proba[proba_rows]
        max_conf = active_proba.max(axis=1)
        winner = int(np.argmax(max_conf))
        preds[i] = int(np.argmax(active_proba[winner]))
        out_proba[i] = active_proba[winner]

    logger.info(
        f"per-window aggregation: {N} windows; "
        f"{n_zero_flow} windows had 0 matching flows → default BENIGN"
    )
    return preds, out_proba


# --- step 6: model training -------------------------------------------------


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray,
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """sklearn RandomForestClassifier — phase 1 spec hp."""
    t0 = time.perf_counter()
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )
    clf.fit(X_train, y_train)
    dt = time.perf_counter() - t0
    logger.info(
        f"RF trained: n_estimators={n_estimators} max_depth={max_depth} "
        f"on {len(y_train):,} flows in {dt:.1f}s"
    )
    return clf


class _XGBWith13Classes:
    """Thin wrapper around XGBClassifier that fits on a remapped contiguous
    label space and exposes a ``predict_proba`` returning the full 13-class
    probability matrix (zero-filling missing classes).

    XGBClassifier requires ``y`` values in ``[0, num_class-1]`` contiguous.
    Our M6.2 train set (Tues + Wed + Fri pcaps only, no Thursday) has
    ``y_unique = [0..9, 12]`` — Web Attack (10) and Infiltration (11) are
    absent. This wrapper:

      * Builds an ``orig→compact`` map at fit time,
      * Trains XGB on the compact labels,
      * At predict_proba reshapes the (N, n_present) output back to
        (N, 13) with zero columns for absent classes.

    Mimics the sklearn ``RandomForestClassifier`` API enough that the
    downstream ``run_pipeline`` code path doesn't need to branch on
    model type. Exposes ``feature_importances_`` and ``classes_``.
    """

    def __init__(self, n_classes: int = 13, **xgb_kwargs):
        import xgboost as xgb
        self._n_classes = int(n_classes)
        self._present_classes: np.ndarray | None = None  # set at fit time
        # We pass num_class via the fitted compact label count, not 13.
        self._xgb_kwargs = xgb_kwargs
        self._clf: xgb.XGBClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_XGBWith13Classes":
        import xgboost as xgb
        present = np.unique(y)
        n_present = int(len(present))
        self._present_classes = present
        orig_to_compact = {int(c): i for i, c in enumerate(present)}
        y_compact = np.array(
            [orig_to_compact[int(v)] for v in y], dtype=np.int64,
        )
        self._clf = xgb.XGBClassifier(
            num_class=n_present, **self._xgb_kwargs,
        )
        self._clf.fit(X, y_compact)
        return self

    @property
    def classes_(self) -> np.ndarray:
        # Report the full 13-class space; downstream code already handles
        # zero-filled missing columns via the predict_proba shape check.
        return np.arange(self._n_classes, dtype=np.int64)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._clf.feature_importances_                # type: ignore[union-attr]

    @property
    def n_jobs(self) -> int | None:
        return getattr(self._clf, "n_jobs", None)

    @n_jobs.setter
    def n_jobs(self, value: int) -> None:
        if self._clf is not None:
            self._clf.n_jobs = value

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        compact = self._clf.predict_proba(X)                  # type: ignore[union-attr]
        full = np.zeros((compact.shape[0], self._n_classes), dtype=compact.dtype)
        assert self._present_classes is not None
        for j, c in enumerate(self._present_classes):
            full[:, int(c)] = compact[:, j]
        return full

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    *,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    n_classes: int = 13,
):
    """xgboost.XGBClassifier wrapped in _XGBWith13Classes so absent classes
    (e.g. Web Attack and Infiltration when training from Tues+Wed+Fri only)
    don't trip XGB's contiguous-label requirement.

    ``tree_method='hist'`` is the fast modern default. CPU because the
    train set fits comfortably in RAM (~250 MB) and GPU init cost for a
    one-shot fit doesn't amortise.
    """
    t0 = time.perf_counter()
    clf = _XGBWith13Classes(
        n_classes=n_classes,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method="hist",
        objective="multi:softprob",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(X_train, y_train)
    dt = time.perf_counter() - t0
    n_present = len(clf._present_classes) if clf._present_classes is not None else 0
    logger.info(
        f"XGB trained: n_estimators={n_estimators} max_depth={max_depth} "
        f"lr={learning_rate} on {len(y_train):,} flows ({n_present} of "
        f"{n_classes} classes present) in {dt:.1f}s"
    )
    return clf


# --- step 7: metrics + deliverables -----------------------------------------


def _collapsed_class_names() -> list[str]:
    """Return the 13 collapsed-class names in label-id order."""
    inv = {v: k for k, v in LABEL_TO_ID_COLLAPSED.items()}
    return [inv[i] for i in range(13)]


def compute_split_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
    n_classes: int = 13,
) -> dict:
    """Compute the metric dict matching the M5.10 eval_metrics.json schema
    for a single (combined/fast/slow) split.
    """
    classes = list(range(n_classes))
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0,
    )
    macro_f1 = float(f1_score(y_true, y_pred, labels=classes,
                              average="macro", zero_division=0))

    # AUROC needs each present class to have at least one positive sample.
    # Use one-vs-rest with the proba slice; if a class is absent in y_true,
    # report np.nan for that class (matches video-cell behaviour) and
    # compute auroc_macro on present classes only.
    per_class_auroc = np.full(n_classes, np.nan, dtype=np.float64)
    present_classes = [c for c in classes if (y_true == c).sum() > 0]
    for c in present_classes:
        try:
            per_class_auroc[c] = float(roc_auc_score((y_true == c).astype(int),
                                                     y_proba[:, c]))
        except ValueError:
            per_class_auroc[c] = np.nan
    valid_auc = per_class_auroc[~np.isnan(per_class_auroc)]
    auroc_macro = float(valid_auc.mean()) if len(valid_auc) else 0.0

    n_per_class = np.array(
        [int((y_true == c).sum()) for c in classes], dtype=np.int64,
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes).astype(np.int64)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "auroc_macro": auroc_macro,
        "per_class_f1": [float(x) for x in f1.tolist()],
        "per_class_precision": [float(x) for x in p.tolist()],
        "per_class_recall": [float(x) for x in r.tolist()],
        "per_class_auroc": [
            None if np.isnan(v) else float(v) for v in per_class_auroc.tolist()
        ],
        "n_per_class": [int(x) for x in n_per_class.tolist()],
        "confusion_matrix": cm.tolist(),
    }


def build_eval_metrics_payload(
    windows: list[WindowInfo],
    window_labels: np.ndarray,
    window_preds: np.ndarray,
    window_proba: np.ndarray,
    *,
    task_label: str,
    script_name: str,
    config_path: str,
    splits_path: str,
    output_dir: str,
    n_classes: int = 13,
) -> dict:
    """Build the eval_metrics.json payload mirroring M5.10 video-cell schema."""
    streams = np.array([w.stream for w in windows])
    fast_mask = streams == "fast"
    slow_mask = streams == "slow"

    combined = compute_split_metrics(
        window_labels, window_preds, window_proba, n_classes,
    )
    fast = compute_split_metrics(
        window_labels[fast_mask], window_preds[fast_mask],
        window_proba[fast_mask], n_classes,
    )
    slow = compute_split_metrics(
        window_labels[slow_mask], window_preds[slow_mask],
        window_proba[slow_mask], n_classes,
    )

    return {
        "schema_version": "m6.2-v1",
        "task_label": task_label,
        "script_name": script_name,
        "output_dir": output_dir,
        "config": config_path,
        "splits_path": splits_path,
        "keep_split": "val",
        "label_mode": "collapsed13",
        "n_classes": n_classes,
        "class_names": _collapsed_class_names(),
        "val_sample_count_fast": int(fast_mask.sum()),
        "val_sample_count_slow": int(slow_mask.sum()),
        "val_sample_count_total": int(len(windows)),
        "combined_metrics": combined,
        "fast_only_metrics": fast,
        "slow_only_metrics": slow,
    }


def write_per_class_table(metrics: dict, out_path: Path) -> None:
    """13-row CSV: class, n_combined, P, R, F1, AUROC (combined split)."""
    classes = metrics["class_names"]
    c = metrics["combined_metrics"]
    rows = []
    for i, name in enumerate(classes):
        rows.append({
            "class": name,
            "n": c["n_per_class"][i],
            "precision": c["per_class_precision"][i],
            "recall": c["per_class_recall"][i],
            "f1": c["per_class_f1"][i],
            "auroc": c["per_class_auroc"][i],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_confusion_matrix_json(metrics: dict, out_path: Path) -> None:
    classes = metrics["class_names"]
    payload = {
        "class_names": classes,
        "matrix": metrics["combined_metrics"]["confusion_matrix"],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def write_feature_importances(
    feat_cols: list[str], importances: np.ndarray, out_path: Path,
) -> None:
    df = pd.DataFrame({"feature": feat_cols, "importance": importances})
    df.sort_values("importance", ascending=False).to_csv(out_path, index=False)


def write_val_flow_coverage(
    windows: list[WindowInfo],
    coverage_counts: np.ndarray,
    out_path: Path,
) -> None:
    """Per-window flow-count distribution diagnostic.

    Format matches the spec call-out:
    ``{histogram, zero_flow_windows, mean, max}``.
    """
    streams = [w.stream for w in windows]
    payload = {
        "n_windows_total": int(len(windows)),
        "n_zero_flow_windows": int((coverage_counts == 0).sum()),
        "mean_flows_per_window": float(coverage_counts.mean()),
        "max_flows_per_window": int(coverage_counts.max()),
        "median_flows_per_window": float(np.median(coverage_counts)),
        "histogram_clipped_to_50": np.histogram(
            np.clip(coverage_counts, 0, 50), bins=51, range=(-0.5, 50.5),
        )[0].tolist(),
        "by_stream": {
            "fast": {
                "n": int(sum(s == "fast" for s in streams)),
                "n_zero_flow": int(((coverage_counts == 0) &
                                    (np.array(streams) == "fast")).sum()),
            },
            "slow": {
                "n": int(sum(s == "slow" for s in streams)),
                "n_zero_flow": int(((coverage_counts == 0) &
                                    (np.array(streams) == "slow")).sum()),
            },
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


# --- step 8: end-to-end pipeline --------------------------------------------


def run_pipeline(
    csv_dir: Path,
    splits_path: Path,
    slow_shard_pattern: str | None,
    output_dir: Path,
    *,
    model_name: str,                       # "rf" | "xgb"
    rf_n_estimators: int = 200,
    rf_max_depth: int | None = None,
    xgb_n_estimators: int = 500,
    xgb_max_depth: int = 6,
    xgb_lr: float = 0.1,
    config_path: str = "configs/base.yaml",
    task_label: str | None = None,
    script_name: str = "scripts/train_m6_flow_baseline.py",
) -> dict:
    """Full M6.2 pipeline: parse CSVs → split assign → train → predict
    → aggregate per-window → write deliverables.

    Returns the eval_metrics.json payload as a dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if task_label is None:
        task_label = (
            f"M6.2 flow-feature baseline — "
            f"{'RF' if model_name == 'rf' else 'XGBoost'} "
            f"with option B max-confidence aggregation"
        )

    # 1. flow table
    logger.info(f"=== M6.2 stage 1: parse {csv_dir} ===")
    flow_table = load_flow_table(csv_dir)

    # 2. assign splits
    logger.info(f"=== M6.2 stage 2: assign per-flow splits ===")
    flow_table = assign_flow_splits(flow_table, splits_path)

    # 3. features
    feat_cols = feature_columns(flow_table)
    if len(feat_cols) != 80:
        logger.warning(
            f"expected 80 feature cols; got {len(feat_cols)}: {feat_cols[:5]}…"
        )

    X_train, y_train = prepare_xy(flow_table, "train", feat_cols)
    logger.info(f"train shape: X={X_train.shape} y unique={np.unique(y_train)}")

    # 4. train
    logger.info(f"=== M6.2 stage 3: train {model_name.upper()} ===")
    if model_name == "rf":
        model = train_random_forest(
            X_train, y_train,
            n_estimators=rf_n_estimators, max_depth=rf_max_depth,
        )
    elif model_name == "xgb":
        model = train_xgboost(
            X_train, y_train,
            n_estimators=xgb_n_estimators, max_depth=xgb_max_depth,
            learning_rate=xgb_lr,
        )
    else:
        raise ValueError(f"unknown model {model_name}; want rf or xgb")

    # 5. predict per-flow on ALL flows (train+val+test) for aggregation lookup.
    #    We need proba for any flow that might appear in any val window.
    #    Use float32 (not float64) and chunked predict_proba with n_jobs=1
    #    to avoid OOM on 8 GB RAM machines — RF with unbounded depth +
    #    n_jobs=-1 duplicates the feature matrix across CPU workers and the
    #    proba accumulator across trees, easily crossing 6+ GB RSS at this
    #    scale.
    feat_all = flow_table[feat_cols].to_numpy(dtype=np.float32, copy=True)
    feat_all[~np.isfinite(feat_all)] = np.nan
    feat_all = np.nan_to_num(feat_all, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"=== M6.2 stage 4: predict on {len(feat_all):,} flows ===")
    # Force single-threaded predict for the RF case (XGB's predict is cheap).
    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = 1
        except AttributeError:
            pass
    chunk = 50_000
    proba_chunks: list[np.ndarray] = []
    for s in range(0, len(feat_all), chunk):
        e = min(s + chunk, len(feat_all))
        proba_chunks.append(model.predict_proba(feat_all[s:e]).astype(np.float32))
        if s % (chunk * 10) == 0:
            logger.info(f"  predict_proba progress: {e:,}/{len(feat_all):,}")
    proba_all = np.concatenate(proba_chunks, axis=0)
    del proba_chunks
    # XGB.predict_proba returns a numpy array but the class column order
    # follows model.classes_; validate that ordering = 0..12 (it will be
    # because we trained with y in [0, 12]; if missing classes in train
    # they wouldn't appear here).
    if hasattr(model, "classes_"):
        cls = model.classes_
        if not np.array_equal(cls, np.arange(13)):
            # Re-shape proba to 13-col by filling missing classes with 0.
            logger.warning(
                f"model.classes_ = {cls}; reshaping proba to 13 cols (filling 0)"
            )
            reshaped = np.zeros((proba_all.shape[0], 13), dtype=np.float64)
            for j, c in enumerate(cls):
                reshaped[:, int(c)] = proba_all[:, j]
            proba_all = reshaped

    flow_idx_to_proba_row = {
        int(orig): row_i for row_i, orig in enumerate(flow_table.index.tolist())
    }

    # 6. enumerate val windows
    logger.info(f"=== M6.2 stage 5: enumerate val windows ===")
    val_windows = enumerate_val_windows(splits_path, slow_shard_pattern)
    if len(val_windows) != 18156:
        raise SystemExit(
            f"val_n mismatch: expected 18156, got {len(val_windows)}; "
            "splits.parquet or slow shard pattern drift — stop-and-report"
        )

    # 7. window labels (ground truth)
    logger.info(f"=== M6.2 stage 6: load window labels ===")
    window_labels = _load_window_labels(val_windows, csv_dir)

    # 8. aggregate per-window predictions
    logger.info(f"=== M6.2 stage 7: aggregate predictions ===")
    flow_index = _sort_flows_for_lookup(flow_table)
    coverage_counts = np.zeros(len(val_windows), dtype=np.int64)
    for i, w in enumerate(val_windows):
        coverage_counts[i] = len(find_active_flows(w, flow_index))
    window_preds, window_proba = aggregate_per_window_predictions(
        val_windows, flow_table, proba_all, feat_cols,
        flow_idx_to_proba_row, n_classes=13,
    )

    # 9. metrics + deliverables
    logger.info(f"=== M6.2 stage 8: compute metrics + write deliverables ===")
    payload = build_eval_metrics_payload(
        val_windows, window_labels, window_preds, window_proba,
        task_label=task_label,
        script_name=script_name,
        config_path=config_path,
        splits_path=str(splits_path),
        output_dir=str(output_dir),
        n_classes=13,
    )
    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=float)
    write_per_class_table(payload, output_dir / "per_class_table.csv")
    write_confusion_matrix_json(payload, output_dir / "confusion_matrix.json")

    importances = model.feature_importances_
    write_feature_importances(feat_cols, importances,
                              output_dir / "feature_importances.csv")
    write_val_flow_coverage(val_windows, coverage_counts,
                            output_dir / "val_window_flow_coverage.json")

    # 10. save model
    import pickle
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    logger.info(
        f"M6.2 done: model={model_name} "
        f"combined macro_f1={payload['combined_metrics']['macro_f1']:.4f} "
        f"fast={payload['fast_only_metrics']['macro_f1']:.4f} "
        f"slow={payload['slow_only_metrics']['macro_f1']:.4f}"
    )
    return payload


def _load_window_labels(
    windows: list[WindowInfo],
    csv_dir: Path,
) -> np.ndarray:
    """Load per-window ground-truth labels from the existing labeled shards.

    For each window in ``windows`` we need the same label that the video
    cells used for that window. The video cells get this from the
    pre-computed shard meta.json's ``label_id``. The fast and slow shards
    live in ``data/processed/cicids2017_dt100ms_v2`` and
    ``data/processed/cicids2017_dt1000ms_v2`` respectively; we read the
    shard manifests then look up each window by (pcap_source, start_time).
    """
    import tarfile
    import io as _io
    from collections import defaultdict

    proc_root = Path("data/processed")
    shard_dirs = {
        "fast": proc_root / "cicids2017_dt100ms_v2",
        "slow": proc_root / "cicids2017_dt1000ms_v2",
    }
    label_lookup: dict[tuple[str, float, str], int] = {}
    for stream, root in shard_dirs.items():
        if not root.is_dir():
            continue
        shards = sorted(root.rglob("shard-*.tar"))
        logger.info(f"loading {stream} window labels from {len(shards)} shards")
        for sh in shards:
            with tarfile.open(sh, "r") as tf:
                for m in tf.getmembers():
                    if m.name.endswith(".meta.json"):
                        d = json.load(_io.BytesIO(tf.extractfile(m).read()))
                        key = (
                            str(d["pcap_source"]),
                            float(d["start_time"]),
                            stream,
                        )
                        # Convert raw15 label_id to collapsed13
                        label_lookup[key] = collapse_to_13(int(d["label_id"]))

    out = np.full(len(windows), BENIGN_ID, dtype=np.int64)
    missing = 0
    for i, w in enumerate(windows):
        key = (w.pcap_source, w.start_time, w.stream)
        if key in label_lookup:
            out[i] = label_lookup[key]
        else:
            missing += 1
    if missing > 0:
        logger.warning(
            f"_load_window_labels: {missing}/{len(windows)} windows had no "
            f"shard meta match — defaulted to BENIGN. Stream/key drift?"
        )
    return out
