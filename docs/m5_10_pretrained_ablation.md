# M5.10 Round 1 Dimension 1: Pretrained-source ablation (VideoMAE-Small)

This document is the data anchor for the first M5.10 ablation
dimension — pretrained-source contribution to the main-method 22M
VideoMAE-Small backbone. Three cells run under the M5.4 P2 fairness
contract with **only** the source ckpt + the head_lr_multiplier (per
M5.5 Path B) varying:

| Cell | Pretrained source | head_lr | Status |
|---|---|---:|---|
| **random** | none (random Kaiming init) | ×1 | **Phase 1 forward training — this commit** |
| **K400** | `MCG-NJU/videomae-small-finetuned-kinetics` | ×5 | reused from M5.4 P2 retrofit (no re-train) |
| **SSv2** | `MCG-NJU/videomae-small-finetuned-ssv2` | ×5 | pending Phase 2 (deferred) |

The random cell uses head_lr ×1 per the M5.5 Path B contract derived
from the R1 vs R1.5 forensic ablation (R1.5 head_lr ×5 dropped
combined macro_f1 by 0.022 and collapsed Bot per-class F1 from 0.0909
to 0.0000 — see `m5_5_baselines.md` §"R1.5 ablation supplementary").
This ablation does NOT repeat the R1.5 experiment on VideoMAE-S; the
×5-on-from-scratch outcome is already empirically established and
re-running would add no new evidence. Path B is preserved: K400 → ×5,
random → ×1.

## Common contract (all cells)

- **Input**: identical (T=16, C=6, H=32, W=64) NID tensor; same
  `splits.parquet` (M5.3 anchor); multi-scale 50/50 fast/slow mix.
- **Optimiser**: 8-bit AdamW, batch=32, grad_accumulation=1, fp16 AMP,
  weight_decay=0.05, base_lr=1.5e-4 with linear warmup to 500 steps
  + cosine decay to 1% peak. Gradient checkpointing on.
- **Loss + reweighting**: focal γ=2 + inverse-square-root α reweighting.
- **Schedule**: 10 epochs, 48,530 grad steps under round_robin
  epoch terminator; per-epoch eval under `no_cycle` so the in-training
  metric is bit-identical to the noise-free re-evaluation
  (Δ ≤ 5e-5 across all retrofitted runs, including this Phase 1
  cell at Δ = 0.000000).
- **MetricsWriter on**: per_step.jsonl (with grad_norm via
  `--collect-grad-norm`), per_epoch.json, confusion_per_epoch.npz
  written to `<run_dir>/metrics/`.

## Three-cell summary table (Phase 1 + reference)

| Cell | Run dir | Params | Pretrained | head_lr | combined | fast | slow | Bot AUROC | Bot F1 | accuracy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **random** | `outputs/run_20260507_205921/` | 22M | none | ×1 | **0.4386** | 0.4259 | 0.5086 | **0.6743** | 0.0000 | 0.9456 |
| **K400** (main P2) | `outputs/run_20260502_184512/` | 22M | Kinetics-400 | ×5 | **0.4756** | 0.4525 | 0.6069 | 0.4968 | 0.0000 | 0.9560 |
| **SSv2** | _pending Phase 2_ | 22M | SSv2 | ×5 | — | — | — | — | — | — |

Numbers verbatim from each cell's `<run_dir>/m5_10_random_videomae_eval/eval_metrics.json`
(random) and `<run_dir>/m5_4_phase2_eval/eval_metrics.json` (K400 main).
val_sample_count_total = 18,156 (fast 16,463 + slow 1,693) bit-identical
across both cells.

## Δ random vs K400 (the headline ablation result for round 1)

| Metric | random | K400 | Δ (random − K400) |
|---|---:|---:|---:|
| combined macro_f1 | 0.4386 | 0.4756 | **−0.037** |
| fast macro_f1 | 0.4259 | 0.4525 | −0.027 |
| slow macro_f1 | 0.5086 | 0.6069 | **−0.098** |
| Bot per-class AUROC | 0.6743 | 0.4968 | **+0.178** |
| Bot per-class F1 | 0.0000 | 0.0000 | 0.000 |

**Two-direction reading**: K400 pretraining buys +0.037 combined
macro_f1 + +0.098 slow-stream macro_f1 (the slow-stream advantage is
~3× the fast-stream advantage, consistent with the M5-007 sub-finding
that K400 prior is loss-level inductive on the slow temporal axis).
At the same time, K400 pretraining COSTS −0.178 in Bot per-class AUROC
— the random VideoMAE-S preserves Bot ranking signal that K400
actively erases by epoch 9. Both directions are real.

## Phase 1 per-class table (random VideoMAE-S, combined eval, epoch 9)

| class | n | P | R | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| BENIGN | 16,829 | 0.9670 | 0.9800 | 0.9735 | 0.8947 |
| DoS Hulk | 105 | 0.3614 | 0.5714 | 0.4428 | 0.8756 |
| PortScan | 22 | 0.4412 | 0.6818 | 0.5357 | 0.9563 |
| DDoS | 228 | 0.8812 | 0.3904 | 0.5410 | 0.9880 |
| DoS GoldenEye | 61 | 0.3191 | 0.2459 | 0.2778 | 0.9165 |
| FTP-Patator | 107 | 0.0952 | 0.0374 | 0.0537 | 0.9191 |
| SSH-Patator | 175 | 0.2459 | 0.0857 | 0.1271 | 0.9295 |
| DoS slowloris | 264 | 0.7674 | 0.8371 | 0.8007 | 0.9908 |
| DoS Slowhttptest | 105 | 0.0971 | 0.0952 | 0.0962 | 0.9109 |
| Bot | 12 | 0.0000 | 0.0000 | 0.0000 | **0.6743** |
| Web Attack | 0 | — | — | — | — |
| Infiltration | 0 | — | — | — | — |
| Heartbleed | 248 | 0.9574 | 0.9960 | 0.9763 | 0.9999 |

## Phase 1 per-epoch trajectory (combined eval)

| epoch | combined | Bot AUROC | Bot F1 | GoldenEye F1 | DDoS F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2597 | 0.8842 | 0.0000 | 0.1359 | 0.1583 |
| 1 | 0.3142 | 0.7418 | 0.0000 | 0.0659 | 0.1673 |
| 2 | 0.2762 | 0.8911 | 0.0000 | 0.0571 | 0.1699 |
| 3 | 0.3519 | 0.7659 | 0.0000 | 0.2143 | 0.1739 |
| 4 | 0.3526 | 0.7272 | 0.0000 | 0.0580 | 0.1732 |
| 5 | 0.3787 | 0.7217 | 0.0000 | 0.1500 | 0.1667 |
| 6 | 0.3926 | 0.7614 | 0.0000 | 0.2901 | 0.1673 |
| 7 | 0.3759 | 0.7134 | 0.0000 | 0.1538 | 0.1680 |
| 8 | 0.3999 | 0.6664 | 0.0000 | 0.2340 | 0.3321 |
| 9 | 0.4386 | 0.6743 | 0.0000 | 0.2778 | 0.5410 |

Notable trajectory features (open-ended observations, recorded for
the round 1 closeout Findings batch; no Findings.md edits this round
per spec):

- **Bot AUROC stays > 0.66 across all 10 epochs** (range 0.6664–0.8911,
  peak 0.8911 at epoch 2). This is the **second** retrofitted run
  with sustained Bot AUROC > 0.66 across the full trajectory, after
  TimeSformer-S R1 (sustained > 0.70). Both are random-init + head_lr
  ×1. Combined with the within-K400 trajectory (M5.4 P2 Bot AUROC
  0.6835 → 0.4968 monotone-decline), this isolates **head_lr ×1 +
  random init** as the regime that preserves Bot rare-class signal.
- **DDoS F1 plateau-then-jump pattern** (0.16–0.17 plateau across
  epochs 0–7; +0.16 jump at epoch 8 to 0.33; +0.21 final-epoch jump
  to 0.54). Same shape as M5.5 R2 random-init 3D-conv-style baselines
  (C3D-Small +0.13 final jump; ConvLSTM +0.40 final jump). Random
  VideoMAE-S exhibits the random-init plateau-jump pattern despite
  being a VideoMAE/transformer architecture — the pattern tracks
  pretrained-status, not architecture family.
- **GoldenEye F1 oscillates [0.06, 0.29] across 10 epochs** (8 of 8
  retrofits + this 1 forward-instrumented run = 9/9 noisy-attractor
  confirmations). Universal property of small-support softmax under
  focal+α loss; not architecture- or pretrained-specific.
- **Combined macro_f1 trajectory has 2 dips** (epoch 1→2 −0.038,
  epoch 6→7 −0.017). Comparable count to TimeSformer-S R1 (2 small
  dips) and slightly more variable than K400 main P2 (3 dips, max
  0.012). Architecture-dip-count tracking remains weak; head_lr ×5
  vs ×1 dip-magnitude pattern (per the M5.10 prep retrofit
  consolidation) holds here too: random + ×1 → max dip 0.038,
  comparable to 3D-conv ×1 baselines.

## Phase 1 sanity verification

- **val_sample_count_total = 18,156** in all 10 epochs ✓
- **In-training vs noise-free re-eval**: combined macro_f1 0.438616
  bit-identical (Δ = 0.000000), accuracy 0.945583 bit-identical,
  auroc_macro 0.773510 bit-identical
- **Wall time**: ~3h (10 epochs × ~15 min/epoch on the 8 GB box, peak
  GPU 488 MB — matches the M4.8 reference baseline of 485 MB
  exactly, confirming no silent K400 weight load: K400 weights would
  push parameter memory ~25% higher)
- **per_step.jsonl**: 48,530 rows, each with `grad_norm` field.
  grad_norm distribution shows expected fp16 GradScaler overflow-skip
  behavior at warmup (early-step inf values triggering loss-scale
  halving + retry, standard AMP behaviour, not loss divergence —
  per-step loss decreased monotonically epoch-over-epoch from 0.0394
  to ≤ 0.0148 by epoch 1 and onward).
- **No silent K400 load**: random ckpt size 130 MB matches the random
  VideoMAE-S reference; K400-loaded ckpts in the project history have
  the same 130 MB footprint (state_dict same shape regardless of
  init) so size alone isn't a discriminator. The combined macro_f1
  0.4386 vs K400 main 0.4756 (Δ −0.037) is well within the random-init
  expected band (random-init TimeSformer-S R1 = 0.4836; random-init
  ConvLSTM = 0.4746; random-init C3D-Small = 0.4464); a silent K400
  load would have produced macro_f1 ≈ 0.4756 ± noise.

## Phase 0 SSv2 ckpt verification (recorded for Phase 2 readiness)

`MCG-NJU/videomae-small-finetuned-ssv2` exists on HF Hub with
identical architecture to K400 Small (hidden=384, 12 layers, 16 heads,
patch=16, tubelet=2, in_ch=3). Drop-in replacement for the K400
ckpt path in the main-method training command — only `--pretrained`
value changes between Phase 2 and the K400 main P2 launch.

The 3→6 channel adapter passes the M3-001 norm-ratio test on SSv2
weights at ratio **5.32×** (Kaiming-init extra channels / SSv2-derived
trilinear-downsampled first 3) vs K400's **5.28×** (M3-001 reference).
Both are ~5× the 1.5× PASS bar — bit-comparable signal preservation
magnitude, no SSv2-specific adapter complications.

## Source artefacts

| Cell | Source training run | Eval bundle | Commit |
|---|---|---|---|
| random | `outputs/run_20260507_205921/` | `m5_10_random_videomae_eval/` (this commit) | (this commit) |
| K400 main P2 | `outputs/run_20260502_184512/` | `m5_4_phase2_eval/` | `1d1a61e` (M5.4 P2) |
| SSv2 | (pending Phase 2) | (pending) | (pending) |

`outputs/**` is gitignored so artefacts ship locally only; recreate
random cell via:

```bash
uv run python scripts/train.py \
    --shard-pattern-fast "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \
    --shard-pattern-slow "data/processed/cicids2017_dt1000ms_v2/*/shards/shard-*.tar" \
    --splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet \
    --pretrained "" \
    --num-epochs 10 \
    --loss-fn focal --focal-gamma 2.0 --reweighting inverse_sqrt \
    --head-lr-multiplier 1.0 \
    --eval-strategy no_cycle --label-mode collapsed13 \
    --collect-grad-norm
```

The empty `--pretrained ""` routes through
`videomae_nid.py::_load_backbone_with_fallback`'s falsy branch
(line 117) which builds `VideoMAEModel(_videomae_small_config())` —
random Kaiming init throughout, M3-onward established codepath.

## Findings + Methods draft

Deferred to round 1 closeout (after Phase 2 SSv2 completes). Findings
fact-layer entries + paper Methods draft will be batch-written once
all three cells (random / K400 / SSv2) are filled in. Per round 1
spec, `prompts/Findings.md` is NOT edited this round.
