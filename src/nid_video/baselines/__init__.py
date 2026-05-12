"""Cross-paradigm baselines for M6 (paper Table 1 row expansion).

Currently implemented: ``flow_feature`` (M6.2 — RF + XGBoost on
CICFlowMeter per-flow features under option B max-confidence per-window
aggregation). M6.1 (1D byte Transformer) and M6.3 (2D snapshot ResNet-18)
are scheduled separately.
"""
