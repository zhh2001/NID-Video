"""Diagnose why CIC TrafficLabelling CSV rows have unparseable timestamps.

Built during the M3-to-M4 dry-run when LabelIndex reported 288,602 dropped
rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv. The verdict
turned out to be "trailing all-empty rows", not a date format / timezone issue
— this script is preserved so the same diagnosis can be reproduced quickly on
new CSVs (e.g., when CIC-IDS-2018 lands in M5).

Usage:
    uv run python scripts/diagnose_unparseable_timestamps.py \
        --csv data/raw/cicids2017/TrafficLabelling/<some>.pcap_ISCX.csv \
        [--pcap data/raw/cicids2017/<matching>.pcap]   # optional: TZ cross-check

The pcap argument is optional. When provided, the script prints the first
packet's wall-clock interpretation under both UTC and AST/ADT, so you can
sanity-check against the CSV's timestamp string format.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
from pathlib import Path

import pandas as pd

# Local time zone CIC ran the captures in (Atlantic Standard Time / Daylight
# Time, UTC-4 in summer). This is what the CSV strings *appear* to be in,
# given the 9-5 working-hours convention CIC documents.
_CIC_TZ_OFFSET_HOURS_DST = -3  # ADT during July 2017


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True,
                   help="A TrafficLabelling/*.csv file to diagnose.")
    p.add_argument("--pcap", type=Path, default=None,
                   help="Optional matching pcap; used for TZ cross-check via first packet ts.")
    p.add_argument("--n-samples", type=int, default=10,
                   help="Sample size at each of (head, middle, tail) of bad rows.")
    return p.parse_args()


def _diagnose_csv(csv: Path, n: int) -> dict[str, object]:
    """Load the CSV with cp1252 (the M4 fix) and characterize the bad-timestamp rows."""
    df = pd.read_csv(csv, encoding="cp1252", low_memory=False)
    df.columns = df.columns.str.strip()

    if "Timestamp" not in df.columns:
        raise ValueError(f"{csv}: no Timestamp column (got {list(df.columns)[:5]}…)")

    bad_mask = df["Timestamp"].isna() | (
        pd.to_datetime(df["Timestamp"], dayfirst=True, format="mixed", errors="coerce").isna()
    )
    bad_idx = df.index[bad_mask].tolist()
    good_idx = df.index[~bad_mask].tolist()

    print("=" * 78)
    print(f"CSV: {csv.name}")
    print(f"  total rows           : {len(df):>10d}")
    print(f"  parseable timestamps : {len(good_idx):>10d}  ({100 * len(good_idx) / len(df):5.1f}%)")
    print(f"  bad timestamps       : {len(bad_idx):>10d}  ({100 * len(bad_idx) / len(df):5.1f}%)")
    if bad_idx:
        is_contig = bad_idx == list(range(bad_idx[0], bad_idx[-1] + 1))
        print(f"  bad-row idx range    : {bad_idx[0]}..{bad_idx[-1]} (contiguous={is_contig})")
    print("=" * 78)

    # Collect raw byte content of timestamp cell + neighborhood for samples.
    # Re-read line-oriented so we can print actual bytes (csv splitting + pandas
    # NaN coercion would hide whether the cell is empty vs a malformed string).
    with csv.open("rb") as fh:
        lines = fh.readlines()
    # df row i corresponds to lines[i+1] (lines[0] is header)
    return {
        "df": df,
        "bad_idx": bad_idx,
        "good_idx": good_idx,
        "lines": lines,
        "n": n,
    }


def _print_samples(result: dict[str, object]) -> None:
    """30 bad samples (10 head / 10 mid / 10 tail) + 10 good for contrast."""
    bad: list[int] = result["bad_idx"]   # type: ignore[assignment]
    good: list[int] = result["good_idx"]   # type: ignore[assignment]
    lines: list[bytes] = result["lines"]   # type: ignore[assignment]
    n: int = result["n"]                   # type: ignore[assignment]

    if not bad:
        print("\n(no bad rows — nothing to sample)")
        return

    head = bad[:n]
    tail = bad[-n:]
    mid_start = max(n, len(bad) // 2 - n // 2)
    middle = bad[mid_start:mid_start + n]

    def _show(label: str, idxs: list[int]) -> None:
        print(f"\n--- {label} ({len(idxs)} samples) ---")
        for df_i in idxs:
            line = lines[df_i + 1]                        # +1 to skip header
            stripped = line.rstrip(b"\r\n")
            preview = stripped[:80] + (b"..." if len(stripped) > 80 else b"")
            print(f"  df_idx={df_i:>7d}  bytes_len={len(line):>4d}  "
                  f"raw={preview!r}")

    _show("BAD rows: HEAD", head)
    _show("BAD rows: MIDDLE", middle)
    _show("BAD rows: TAIL", tail)

    # Good contrast: first n good rows.
    _show("GOOD rows for contrast", good[:n])


def _tz_crosscheck(csv: Path, pcap: Path | None, df: pd.DataFrame) -> None:
    """Cross-reference the CSV's timestamp string convention against the
    pcap's first packet (which is unambiguously UTC unix epoch)."""
    if pcap is None:
        return

    print("\n" + "=" * 78)
    print(f"Timezone cross-check vs pcap: {pcap.name}")
    print("=" * 78)

    # Get first packet timestamp from the pcap. Use our own parser so we
    # exercise the production code path (and inherit pcapng support).
    from nid_video.data.pcap_parser import PacketStream
    first_ts = next(iter(PacketStream(pcap))).timestamp
    utc = dt.datetime.fromtimestamp(first_ts, tz=dt.timezone.utc)
    adt = utc + dt.timedelta(hours=_CIC_TZ_OFFSET_HOURS_DST)
    print(f"  first packet unix_ts : {first_ts:.3f}")
    print(f"    if interpreted UTC : {utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"    if interpreted ADT : {adt.strftime('%Y-%m-%d %H:%M:%S')} (UTC{_CIC_TZ_OFFSET_HOURS_DST:+d})")

    # Show the CSV's earliest and latest parseable timestamp strings as raw
    # text — does its hour-of-day match UTC reading or ADT reading?
    ts_parsed = pd.to_datetime(df["Timestamp"], dayfirst=True, format="mixed", errors="coerce")
    valid = df.loc[ts_parsed.notna()].copy()
    valid["_ts_parsed"] = ts_parsed[ts_parsed.notna()].values
    if not valid.empty:
        earliest = valid["_ts_parsed"].min()
        latest = valid["_ts_parsed"].max()
        # Get the original raw strings at those positions
        ear_raw = valid.loc[valid["_ts_parsed"] == earliest, "Timestamp"].iloc[0]
        lat_raw = valid.loc[valid["_ts_parsed"] == latest, "Timestamp"].iloc[0]
        print(f"  CSV earliest ts (raw): {ear_raw!r}  → parsed {earliest}")
        print(f"  CSV latest   ts (raw): {lat_raw!r}  → parsed {latest}")

    print("\n  Interpretation:")
    print(f"   - CIC documents working-hours captures (~09:00–17:00 local).")
    print(f"   - If CSV hours match the ADT row above (working hours), CSV is local ADT.")
    print(f"   - If CSV hours match the UTC row above, CSV is UTC.")
    print(f"   - Mismatch between CSV hour and pcap-as-ADT hour by ~3h → CSV is in UTC.")


def main() -> None:
    args = _parse_args()
    result = _diagnose_csv(args.csv, args.n_samples)
    _print_samples(result)
    _tz_crosscheck(args.csv, args.pcap, result["df"])  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
