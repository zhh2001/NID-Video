# Paper Figures: Data Source Schema

This document is the data anchor for the paper-writing figure pipeline.
It catalogues the figures that the M5-era trajectory data + the M5.10
forward-instrumented data collectively support, with explicit data
paths, finding cross-references, and a plotting-skeleton convention.

The list is intentionally NOT a paper-figure layout plan — the actual
paper figures will be selected, ordered, and captioned during paper
writing. This list catalogues the **available raw data** so figure
scripts can be authored without re-deriving the data sources.

## Figure typography discipline (apply to every figure)

- **No figure title** — captions belong in the paper LaTeX, not in the
  figure raster. ``configure_paper_style()`` zeros ``axes.titlesize``
  as a guardrail so any accidental ``ax.set_title("...")`` call emits
  the layout but the text is invisible.
- **Times New Roman serif** for all text (axis labels, ticks, legend,
  annotations). Math expressions render via STIX (Times-equivalent).
- **dpi=300** for both display and savefig.
- **PDF + PNG dual save** via ``save_figure()``. PDF is the canonical
  paper artefact; PNG is a convenience copy for online review.
- **Domain naming** — figure names use domain terms (``fig_bot_auroc_collapse``,
  ``fig_k400_head_start``), not engineering placeholders (``fig_main``,
  ``ax_0``). The figure name in this schema doc is the contract.
- **Color discipline** — colorblind-friendly palette (avoid red/green
  pairs). Default to ``mpl.cm.viridis`` for sequential / ``tab10`` for
  categorical. Random-init group vs K400 group distinction should be
  one of: line style (solid vs dashed) OR marker style, not just color.

All conventions are locked into ``scripts/plot_helper.py::configure_paper_style()``
so individual figure scripts can't drift.

## Available figure candidates (post-M5 + post-retrofit)

### 1. Silent failure detection chain — v1 vs v2 conservation laws

- **Source**: ``docs/v1_vs_v2_comparison.md`` "v1 → v2 delta" table
- **Data**: 11 classes × {v1 windows, v2 windows, delta, sum_check}; markdown table; extract via Python pandas ``read_html`` or hand-extract.
- **Type**: Two-bar grouped barplot (v1 in muted color, v2 in primary color) per class, with a delta column annotated above each pair. 11 class rows; horizontal layout for readability.
- **Findings**: TRANSITION-005, M4-001, M4-002, M4-010a — the silent failure detection chain flagship.
- **Figure name**: ``fig_v1_v2_conservation``

### 2. Bot AUROC step-collapse trajectory (M5.2 vanilla CE)

- **Source**: ``outputs/run_20260501_162117/metrics/per_epoch.json`` (M5.2 retrofit)
- **Data path**: ``epochs[*].metrics.combined.per_class.Bot.auroc`` over ``epochs[*].epoch`` (10 epochs)
- **Type**: Single-line line plot, x = epoch, y = Bot per-class AUROC, with horizontal dashed line at y=0.5 (random baseline). The within-M5.2 trajectory is the cleanest single-run evidence of the step-collapse: 0.6728 (epoch 0) → 0.4077 (epoch 9), with the sharpest drop in epochs 1–3.
- **Findings**: M5-002 (HIGHEST priority).
- **Figure name**: ``fig_bot_auroc_collapse``
- **Verified by Part 3 implementation**: see "Verification example" below.

### 3. Loss × time-budget trajectory comparison (CE / focal P1 / focal P2)

- **Source**:
  - ``outputs/run_20260501_162117/metrics/per_epoch.json`` (M5.2 vanilla CE)
  - ``outputs/run_20260502_134735/metrics/per_epoch.json`` (M5.4 P1 focal γ=2)
  - ``outputs/run_20260502_184512/metrics/per_epoch.json`` (M5.4 P2 focal+α+head_lr×5)
- **Data path**: 3 runs × 10 epochs × ``combined.macro_f1``.
- **Type**: 3-line line plot, x = epoch, y = combined macro_f1. Horizontal dashed line at y=0.4677 (vanilla CE noise-free ceiling per M5-002). Shows that focal+α+head_lr×5 climbs above the CE ceiling only in the final 1–2 epochs; focal alone (P1) lands below CE at epoch 9.
- **Findings**: M5-002, M5-004.
- **Figure name**: ``fig_loss_trajectory_comparison``

### 4. Cross-architecture learning curves (6-row baseline suite)

- **Source**: 6 retrofitted runs:
  - ``outputs/run_20260502_184512/metrics/per_epoch.json`` (M5.4 P2 main, VideoMAE+K400)
  - ``outputs/run_20260502_232207/metrics/per_epoch.json`` (M5.5 R1 TimeSformer-S, random + head_lr×1)
  - ``outputs/run_20260503_180604/metrics/per_epoch.json`` (M5.5 R2 C3D-Small, random)
  - ``outputs/run_20260503_202832/metrics/per_epoch.json`` (M5.5 R2 ConvLSTM, random)
  - ``outputs/run_20260504_015958/metrics/per_epoch.json`` (M5.5 R2 I3D, K400)
  - ``outputs/run_20260504_051632/metrics/per_epoch.json`` (M5.5 R2 R(2+1)D-18, K400)
- **Data path**: 6 runs × 10 epochs × {combined, fast, slow} × ``macro_f1``.
- **Type**: 3-panel figure (combined / fast / slow), 6 lines per panel, x = epoch, y = macro_f1. Distinguish K400 group (main / I3D / R(2+1)D-18) and random-init group (TimeSformer-S / C3D / ConvLSTM) via line style (solid vs dashed). Slow panel shows the K400 prior 5-epoch head start (epoch 0 K400 ≈ random epoch 5).
- **Findings**: M5-007 (HIGHEST priority — paper Table 1's plotted version).
- **Figure name**: ``fig_cross_arch_learning_curves``

### 5. Per-class F1 epoch trajectory (selected attack classes)

- **Source**: same 6 retrofitted runs as figure 4
- **Selected classes**: Bot (n=12), DoS GoldenEye (n=61), PortScan (n=22), DDoS (n=228), SSH-Patator (n=175), Heartbleed (n=248) — chosen to span the support range from extreme-rare to high-support.
- **Data path**: 6 classes × 6 baselines × 10 epochs × ``per_class_f1``.
- **Type**: 6-panel grid (one panel per class, 2×3), 6 lines per panel. Reveals that:
  - Bot F1 is non-zero in only R1, R1.5, P2 (early epochs), I3D (early), and **sustained in R(2+1)D-18 only**
  - GoldenEye F1 oscillates in all 6 runs (the noisy-attractor finding documented across 8/8 retrofits)
  - DDoS F1 plateau-then-jump pattern visible in C3D / ConvLSTM (random-init), gradual climb in I3D / R(2+1)D-18 (K400), sustained climb in TimeSformer-S R1, late jump in M5.4 P2.
- **Findings**: M5-004, M5-007 (Bot F1 sub-finding), M5-002 (Bot per-class)
- **Figure name**: ``fig_per_class_f1_trajectory``

### 6. Confusion matrix evolution (selected ckpt × selected baselines)

- **Source**: ``confusion_per_epoch.npz`` from key runs (load via ``np.load(...)['epoch_<N>']``).
- **Type**: Heatmap grid, e.g. M5.4 P2 epoch 0/3/9 OR cross-baseline epoch 9 (6 baselines side-by-side). Annotate diagonal (correct predictions) and major off-diagonal (most common confusion).
- **Findings**: M4-010a (per-class threshold calibration), M5-007 (cross-arch confusion patterns).
- **Figure name**: ``fig_confusion_evolution``

### 7. ETL throughput stage breakdown

- **Source**: ``docs/etl_performance.md`` L1-L4 + per-stage breakdown + M2-009 finding.
- **Type**: Horizontal bar chart, 4 stages (PCAP read, packet parse, window construct, shard write), throughput in pps + each stage's runtime as a fraction of total.
- **Findings**: M2-009, TRANSITION-002.
- **Figure name**: ``fig_etl_throughput_breakdown``

### 8. Memory abundance reframing

- **Source**: ``docs/m3_perf.md`` Phase 1 batch sweep table + M4.8 measured 485 MB peak.
- **Type**: Bar chart with 8 GB ceiling line, B=2/8/32/128/512/M4.8-actual = 6 bars, peak GPU memory in MiB on y-axis.
- **Findings**: M3-007, M3-009, M4-010a.
- **Figure name**: ``fig_memory_abundance``

### 9. (deferred — M5.10 ablation completed)

Reserved for the M5.10 ablation suite figures once the ablation
experiments are executed and metrics retrofitted:
- Pretrained ablation (random vs K400 vs SSv2)
- Spatial layout ablation (semantic-cluster vs hash vs shuffle)
- Motion channel ablation (C=4 vs C=6)
- Scale token ablation (single-scale vs multi-scale)

These figures will reuse ``fig_cross_arch_learning_curves`` style
(3-panel combined/fast/slow) with the ablation dimensions as the
multi-line variable.

### 10. K400 prior loss-level inductive evidence (Part 2 surprise finding)

The retrofit data revealed that the K400 prior provides an instant
~5-epoch head start across **all** training metrics — not just final
ckpt advantage. This figure makes the head-start visually explicit.

- **Source**: 9 retrofitted runs (skip M4.8 1-epoch and M5.1 partial 3-epoch which are not full 10-epoch trajectories):
  - K400 group: M5.4 P2 main, M5.5 R2 I3D, M5.5 R2 R(2+1)D-18
  - Random-init group: M5.5 R1 TimeSformer-S, M5.5 R1.5 TimeSformer-S, M5.5 R2 C3D-Small, M5.5 R2 ConvLSTM
  - Vanilla CE M5.2 (random + CE for context — pre-focal-loss baseline)
  - M5.4 P1 (random + focal — bridge between CE and the focal+α main method)
- **Data path**: 9 runs × ``epochs[0].metrics.{combined, fast, slow}.macro_f1``.
- **Type**: 3-panel grouped bar chart (combined / fast / slow), one panel per metric. Bars colored by group (K400 = solid, random = hatched OR different color). The K400 bars (left side of each panel) sit ~0.05–0.15 above the random-init bars on the metric axis at epoch 0 alone. Annotate above the K400 cluster with "K400 epoch 0 ≈ random-init epoch 5+" with an arrow pointing to a small inset showing the random-init epoch-5 value (which matches the K400 epoch-0 value for slow stream).
- **Findings**: M5-007 (the K400 inductive-prior sub-finding); supplementary to figure 4.
- **Figure name**: ``fig_k400_head_start``

## Data path conventions

| Source | Path | Schema |
|---|---|---|
| per-epoch retrofit / forward | ``<run_dir>/metrics/per_epoch.json`` | run_id + config + epochs[*].metrics.{combined,fast?,slow?}.{macro_f1,accuracy,auroc_macro,per_class.<name>.{f1,p,r,auroc,n}} |
| per-epoch confusion | ``<run_dir>/metrics/confusion_per_epoch.npz`` | np.savez keyed by ``epoch_<N>``, each (13, 13) int64 |
| per-step training trace | ``<run_dir>/metrics/per_step.jsonl`` | M5.10 forward only — line-per-grad-step JSONL ``{step, epoch, loss, lr, wall_time_s, grad_norm?}`` |
| best ckpt eval (legacy) | ``<run_dir>/m*_eval/eval_metrics.json`` | bit-identical to ``per_epoch.json`` epoch 9 combined for the 11 retrofitted runs (verified during Part 2) |

## Plotting helper

Three functions in ``scripts/plot_helper.py``:

```python
from scripts.plot_helper import configure_paper_style, load_per_epoch, save_figure

configure_paper_style()   # MUST be first; sets Times New Roman + dpi=300 + no-title
data = load_per_epoch(Path("outputs/run_<ts>"))   # returns parsed per_epoch.json
# ... build fig ...
save_figure(fig, "fig_name")    # writes figures/fig_name.{pdf,png} and closes the figure
```

## Verification example: figure 2 (Bot AUROC step-collapse)

The M5.10 Part 3 verification renders figure 2 from the M5.2 retrofit
data as a smoke test of the helper + style discipline. Reference
implementation:

```python
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.plot_helper import configure_paper_style, load_per_epoch, save_figure


def main() -> None:
    configure_paper_style()

    run_dir = Path("outputs/run_20260501_162117")    # M5.2 vanilla CE 10ep
    data = load_per_epoch(run_dir)
    epochs = [e["epoch"] for e in data["epochs"]]
    bot_auroc = [e["metrics"]["combined"]["per_class"]["Bot"]["auroc"]
                 for e in data["epochs"]]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(epochs, bot_auroc, marker="o", linewidth=1.5,
            color="#1f77b4", label="M5.2 (vanilla CE)")
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1,
               label="Random-baseline floor")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bot per-class AUROC")
    ax.set_xticks(epochs)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, linestyle=":", alpha=0.4)

    save_figure(fig, "fig_bot_auroc_collapse")


if __name__ == "__main__":
    main()
```

The verification artefacts ship under ``figures/``:
- ``fig_bot_auroc_collapse.pdf`` (~5 KB; vector)
- ``fig_bot_auroc_collapse.png`` (~50–80 KB; 300 dpi)

Both files have NO figure title; both render text in Times New Roman
(verifiable via ``pdfinfo`` / ``pdffonts`` for PDF; visual inspection
for PNG). The PDF is suitable for direct paper inclusion.
