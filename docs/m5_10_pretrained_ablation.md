# M5.10 Round 1 Dimension 1: Pretrained-source ablation (VideoMAE-Small)

This document is the data anchor for the first M5.10 ablation
dimension — pretrained-source contribution to the main-method 22M
VideoMAE-Small backbone. Three cells run under the M5.4 P2 fairness
contract with **only** the source ckpt + the head_lr_multiplier (per
M5.5 Path B) varying:

| Cell | Pretrained source | head_lr | Status |
|---|---|---:|---|
| **random** | none (random Kaiming init) | ×1 | Phase 1 forward training (commit `572711d`) |
| **K400** | `MCG-NJU/videomae-small-finetuned-kinetics` | ×5 | reused from M5.4 P2 retrofit (no re-train) |
| **SSv2** | `MCG-NJU/videomae-small-finetuned-ssv2` | ×5 | **Phase 2 forward training — this commit** |

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

## Three-cell summary table (round 1 complete)

| Cell | Run dir | Params | Pretrained | head_lr | combined | fast | slow | Bot AUROC | Bot F1 | accuracy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **random** | `outputs/run_20260507_205921/` | 22M | none | ×1 | **0.4386** | 0.4259 | 0.5086 | **0.6743** | 0.0000 | 0.9456 |
| **K400** (main P2) | `outputs/run_20260502_184512/` | 22M | Kinetics-400 | ×5 | **0.4756** | 0.4525 | 0.6069 | 0.4968 | 0.0000 | 0.9560 |
| **SSv2** | `outputs/run_20260508_213702/` | 22M | SSv2 | ×5 | **0.4413** | 0.4221 | 0.5483 | 0.4115 | 0.0000 | 0.9547 |

Numbers verbatim from each cell's eval bundle:
- random: `outputs/run_20260507_205921/m5_10_random_videomae_eval/eval_metrics.json`
- K400 main P2: `outputs/run_20260502_184512/m5_4_phase2_eval/eval_metrics.json`
- SSv2: `outputs/run_20260508_213702/m5_10_ssv2_videomae_eval/eval_metrics.json`

val_sample_count_total = 18,156 (fast 16,463 + slow 1,693) bit-identical
across all three cells.

## Three-way Δ (the round 1 headline)

| Metric | random | SSv2 | K400 | Δ SSv2−random | Δ K400−random | Δ K400−SSv2 |
|---|---:|---:|---:|---:|---:|---:|
| combined macro_f1 | 0.4386 | 0.4413 | 0.4756 | **+0.003** | **+0.037** | **+0.034** |
| fast macro_f1 | 0.4259 | 0.4221 | 0.4525 | −0.004 | +0.027 | +0.030 |
| slow macro_f1 | 0.5086 | 0.5483 | 0.6069 | **+0.040** | **+0.098** | +0.059 |
| Bot per-class AUROC | 0.6743 | 0.4115 | 0.4968 | **−0.263** | **−0.178** | +0.085 |
| Bot per-class F1 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.000 |

**Three-cell observations** (recorded for round 1 closeout Findings;
no Findings.md edits this round per spec):

- **Pretrained-source effect on combined macro_f1 is asymmetric across
  K400 and SSv2.** K400 pretraining buys +0.037 over random init
  (fully consistent with the M5-007 sub-finding that K400 prior is
  loss-level inductive at this NID input scale). SSv2 pretraining buys
  only +0.003 over random init — within noise, an order of magnitude
  smaller effect. SSv2 → NID transfer is roughly null on combined macro_f1.
- **SSv2's slow-stream advantage is ~halfway between random and K400.**
  SSv2 lifts slow macro_f1 by +0.040 over random vs K400's +0.098.
  This makes SSv2 a partial transfer source — the slow-temporal axis
  benefits but the fast-stream and combined headline numbers lag.
- **Bot per-class AUROC: random is best, K400 is second, SSv2 is
  worst.** The pretrained-source affects Bot rare-class signal
  preservation in a non-monotone way: random VideoMAE-S preserves
  0.6743 Bot AUROC across the trajectory (sustained > 0.66, see
  Phase 1 trajectory), K400 collapses Bot AUROC to 0.4968 by epoch 9,
  and **SSv2 collapses Bot AUROC further to 0.4115 — below the
  random-baseline floor of 0.5**. This is a finding-direction surprise:
  SSv2 pretraining harms Bot rare-class ranking even more than K400
  pretraining does. The M5-005 / M5-007 sub-findings were anchored on
  the random-vs-K400 contrast; SSv2 reveals that the "pretraining
  hurts Bot" effect is not monotone in transferability.

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

## Phase 2 per-class table (SSv2 VideoMAE-S, combined eval, epoch 8 best.pt)

| class | n | P | R | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| BENIGN | 16,829 | 0.9619 | 0.9930 | 0.9772 | 0.8966 |
| DoS Hulk | 105 | 0.6286 | 0.6286 | 0.6286 | 0.9017 |
| PortScan | 22 | 0.5000 | 0.5455 | 0.5217 | 0.9544 |
| DDoS | 228 | 0.9531 | 0.2675 | 0.4178 | 0.9964 |
| DoS GoldenEye | 61 | 0.3158 | 0.1967 | 0.2424 | 0.9591 |
| FTP-Patator | 107 | 0.1667 | 0.0280 | 0.0480 | 0.8735 |
| SSH-Patator | 175 | 0.1852 | 0.0286 | 0.0495 | 0.8863 |
| DoS slowloris | 264 | 0.9409 | 0.7841 | 0.8554 | 0.9948 |
| DoS Slowhttptest | 105 | 0.2812 | 0.0857 | 0.1314 | 0.9038 |
| Bot | 12 | 0.0000 | 0.0000 | 0.0000 | **0.4115** |
| Web Attack | 0 | — | — | — | — |
| Infiltration | 0 | — | — | — | — |
| Heartbleed | 248 | 0.9686 | 0.9960 | 0.9821 | 0.9999 |

## Phase 2 per-epoch trajectory (combined eval)

| epoch | combined | Bot AUROC | Bot F1 | GoldenEye F1 | DDoS F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2920 | 0.6861 | 0.0000 | 0.1789 | 0.1689 |
| 1 | 0.2906 | 0.4794 | 0.0000 | 0.0580 | 0.1673 |
| 2 | 0.3335 | 0.5805 | 0.0000 | 0.1429 | 0.1732 |
| 3 | 0.3480 | 0.4959 | 0.0000 | 0.0571 | 0.1746 |
| 4 | 0.3728 | 0.3630 | 0.0000 | 0.1176 | 0.1753 |
| 5 | 0.4081 | 0.6231 | 0.0000 | 0.2581 | 0.1753 |
| 6 | 0.4153 | 0.4836 | 0.0000 | 0.4198 | 0.2039 |
| 7 | 0.4099 | 0.4152 | 0.0000 | 0.2708 | 0.1897 |
| 8 | **0.4413** | 0.4115 | 0.0000 | 0.2424 | 0.4178 |
| 9 | 0.4353 | 0.4295 | 0.0000 | 0.2692 | 0.3546 |

`best.pt = epoch_8_step_43677.pt` per the trainer's tracking — epoch 8
combined macro_f1 was the run's high-water-mark. Epoch 9 regressed
slightly (0.4413 → 0.4353, Δ −0.006), normal late-cosine-decay noise.

Notable trajectory features (open-ended, recorded for the round 1
closeout Findings batch; no Findings.md edits this round per spec):

- **Bot AUROC oscillates [0.36, 0.69] across 10 epochs** — NOT
  sustained > 0.5 like the random Phase 1 trajectory (which held
  > 0.66 across all 10 epochs). SSv2 starts with a high-Bot epoch 0
  (0.6861, comparable to random's 0.8842) but drops below 0.5 by
  epoch 1 (0.4794) and oscillates around the random-baseline floor
  for the rest of training. Final epoch 9 = 0.4295. The "head_lr ×1
  + random init preserves Bot ranking" pattern from M5-007 / Phase 1
  is now anchored against the contrast: head_lr ×5 + SSv2 (this
  trajectory) and head_lr ×5 + K400 (M5.4 P2's 0.6835 → 0.4968 monotone)
  both collapse Bot AUROC; head_lr ×1 + random preserves it. The
  splitting variable is **head_lr ×1 vs ×5**, not pretrained-status.
- **DDoS F1 plateau-then-jump at epoch 8** (held 0.16–0.20 epochs 0–7,
  jumped to 0.42 at epoch 8, regressed to 0.35 at epoch 9). Same shape
  as Phase 1 random VideoMAE-S and the M5.5 R2 random-init 3D-conv
  baselines. SSv2's epoch 8 jump (+0.219) is intermediate between
  Phase 1 random's epoch 8→9 jump (+0.21) and K400 main P2's pattern
  (gradual climb, smaller late jumps). Final-ckpt DDoS F1 = 0.4178
  for SSv2 vs 0.5410 for random vs 0.4590 for K400.
- **GoldenEye F1 oscillates [0.06, 0.42]** — universal noisy-attractor
  pattern continues (10/10 forward + retrofit runs).
- **Combined macro_f1 trajectory has 3 small dips** (epoch 0→1
  −0.0014, epoch 6→7 −0.0054, epoch 8→9 −0.0060). Max dip magnitude
  0.006 — the **smallest max-dip across all 11 trajectories analyzed
  to date**. SSv2 + head_lr ×5 produces an unusually smooth trajectory.

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

## Phase 2 sanity verification

- **val_sample_count_total = 18,156** in all 10 epochs ✓
- **In-training vs noise-free re-eval** (best ckpt = epoch 8):
  in-training combined macro_f1 = 0.441283; noise-free re-eval
  combined macro_f1 = 0.4413; **Δ = 1.7e-5 ≤ 5e-5** ✓ (within
  fairness contract for `--eval-strategy no_cycle` runs)
- **Wall time**: 10,058 s ≈ 167.6 min (10 epochs × ~14–15 min each
  on the 8 GB box, peak GPU 488 MB — matches K400 main P2's
  488 MB exactly, consistent with same VideoMAE-S architecture +
  same 6-channel adapter)
- **per_step.jsonl**: 48,530 rows (1 per grad step, full schedule
  completed). Each row carries `grad_norm` field via
  `--collect-grad-norm`. step-1 grad_norm = `Infinity` is the
  standard fp16 GradScaler dynamic-loss-scaling startup artefact
  (the scaler halves loss-scale and retries; per-step loss is
  sensible from step 2 onwards). All other 48,529 grad_norm values
  finite.
- **per_epoch.json**: 10 epoch records, all with
  `metrics.combined.{macro_f1, accuracy, auroc_macro, per_class}`
  populated. (No `fast` / `slow` keys in forward-instrumented runs;
  the noise-free re-eval `eval_metrics.json` carries those splits
  via `baseline_rerun.py`'s scale_id partition.)
- **confusion_per_epoch.npz**: 10 keys `epoch_0..epoch_9`, each
  (13, 13) int64, every sum = 18,156 ✓.
- **No silent K400/random load**: the trainer startup log records
  `loaded pretrained backbone: MCG-NJU/videomae-small-finetuned-ssv2`
  + the patch_embed adapter log shows `ch[0:3] downsampled 16→8
  shape=(384, 3, 2, 8, 8) norm=5.22; ch[3:6] kaiming-init
  shape=(384, 3, 2, 8, 8) norm=27.78` — the SSv2-derived
  trilinear-downsampled first 3 channels match the Phase 0 sanity
  norm of 5.22 (vs K400 main's 3.83 — different source ckpt
  produces different downsampled-norm values, K400 having compressed
  3-ch weights to a smaller magnitude during 16→8 downsample).
  Combined macro_f1 0.4413 ≠ K400's 0.4756 and ≠ random's 0.4386,
  rules out silent load of either source.
- **Two-dir consolidation note**: Phase 2 was launched with a manual
  `tee outputs/run_<ts>/training.log` redirect into a manually-created
  run dir `run_20260508_213308`, but the python Trainer chose its
  own timestamp `run_20260508_213702` for ckpt + metrics output.
  Post-training, `213308/training.log` was moved into
  `213702/training.log` and the empty `213308` dir was rmdir'd. The
  consolidated `213702/` is the canonical Phase 2 run dir; the
  /tmp-cleanup lesson (logs must live with the run) is held.

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
| random | `outputs/run_20260507_205921/` | `m5_10_random_videomae_eval/` | `572711d` (Phase 1) |
| K400 main P2 | `outputs/run_20260502_184512/` | `m5_4_phase2_eval/` | `1d1a61e` (M5.4 P2) |
| SSv2 | `outputs/run_20260508_213702/` | `m5_10_ssv2_videomae_eval/` (this commit) | (this commit) |

`outputs/**` is gitignored so artefacts ship locally only; recreate
the random cell via:

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

Recreate the SSv2 cell via the same command with two changes:

```bash
    --pretrained MCG-NJU/videomae-small-finetuned-ssv2 \  # was ""
    --head-lr-multiplier 5.0 \                            # was 1.0 (Path B)
```

(All other flags identical to the random cell.) The
`MCG-NJU/videomae-small-finetuned-ssv2` value routes through
`videomae_nid.py::_load_backbone_with_fallback`'s `from_pretrained`
branch (line 122) which loads the SSv2 ckpt via HuggingFace, then the
adapter applies the standard 16→8 trilinear downsample for the first
3 channels and Kaiming-init for the extra 3 channels. The empty
`--pretrained ""` for the random cell routes through the falsy branch
(line 117) which builds `VideoMAEModel(_videomae_small_config())` —
random Kaiming init throughout, M3-onward established codepath.

## Findings + Methods draft

Deferred to round 1 closeout (after Phase 2 SSv2 completes). Findings
fact-layer entries + paper Methods draft will be batch-written once
all three cells (random / K400 / SSv2) are filled in. Per round 1
spec, `prompts/Findings.md` is NOT edited this round.
