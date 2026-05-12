"""M6.2 — train+eval RF or XGBoost on CICFlowMeter flow features.

Usage:
    uv run python scripts/train_m6_flow_baseline.py \\
        --model rf \\
        --output-dir outputs/m6_rf/
    uv run python scripts/train_m6_flow_baseline.py \\
        --model xgb \\
        --output-dir outputs/m6_xgb/

The pipeline lives in nid_video.baselines.flow_feature. Phase 0 option B
mapping (per-flow train, per-window max-confidence aggregate) — see the
module docstring for the full design.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from nid_video.baselines.flow_feature import run_pipeline
from nid_video.utils import setup_logger


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", choices=["rf", "xgb"], required=True,
                   help="rf = sklearn RandomForestClassifier, "
                        "xgb = xgboost.XGBClassifier.")
    p.add_argument("--csv-dir", type=Path,
                   default=Path("data/raw/cicids2017/TrafficLabelling"),
                   help="Directory containing the 8 CICFlowMeter CSVs.")
    p.add_argument("--splits-path", type=Path,
                   default=Path("data/processed/cicids2017_dt100ms_v2/splits.parquet"),
                   help="splits.parquet anchor (M5.3).")
    p.add_argument("--slow-shard-pattern", type=str,
                   default="data/processed/cicids2017_dt1000ms_v2/*/shards/shard-*.tar",
                   help="Glob for slow-stream shards (used to enumerate slow val "
                        "keys). Pass empty string to skip slow side.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory for 6-deliverable bundle.")
    p.add_argument("--rf-n-estimators", type=int, default=200)
    p.add_argument("--rf-max-depth", type=int, default=None)
    p.add_argument("--xgb-n-estimators", type=int, default=500)
    p.add_argument("--xgb-max-depth", type=int, default=6)
    p.add_argument("--xgb-lr", type=float, default=0.1)
    p.add_argument("--task-label", type=str, default=None,
                   help="Override the task_label string in eval_metrics.json.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logger()
    args = parse_args(argv)

    slow_pattern = args.slow_shard_pattern if args.slow_shard_pattern else None

    run_pipeline(
        csv_dir=args.csv_dir,
        splits_path=args.splits_path,
        slow_shard_pattern=slow_pattern,
        output_dir=args.output_dir,
        model_name=args.model,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_lr=args.xgb_lr,
        task_label=args.task_label,
        script_name="scripts/train_m6_flow_baseline.py",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
