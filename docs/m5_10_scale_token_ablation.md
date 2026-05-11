# M5.10 Round 1 Dimension 4: Scale token + multi-scale ablation (VideoMAE-Small)

This document is the data anchor for the fourth M5.10 ablation
dimension — the joint contribution of the **scale token** + the
**multi-scale training data** to the main-method 22M VideoMAE-Small
backbone (Idea.md §3.4 / M4 task 4.2 / contribution point #4). Four
cells run under the M5.4 P2 fairness contract in a 2×2 factorial:

|  | scale token = on | scale token = off |
|---|---|---|
| **multi-scale train** | A (main P2 reuse) | **B** (this commit) |
| **single-scale train** | (n/a — single-scale + token has no valid scale_id signal) | **C** (fast-only, planned), **D** (slow-only, planned) |

| Cell | Pretrained | dataloader | scale token | head_lr | Status |
|---|---|---|---|---:|---|
| **A: main P2** | K400 | multi-scale 50/50 | on (use_scale_token=True) | ×5 | reused from M5.4 P2 retrofit (no re-train) |
| **B: ms+notoken** | K400 | multi-scale 50/50 | **off** (use_scale_token=False) | ×5 | Phase 1 forward training (commit `5bc6b32`) |
| **C: fast-only** | K400 | single-scale fast (Δt=100ms) | off | ×5 | Phase 1 forward training (commit `3d830af`) |
| **D: slow-only** | K400 | single-scale slow (Δt=1s) | off | ×5 | **Phase 1 forward training — this commit** |

Path B is preserved across all 4 cells: K400 + head_lr ×5. The
splitting variables are the scale token (A vs B) and the dataloader
stream count (B vs C, B vs D). A vs C and A vs D mix both factors;
B vs C and B vs D isolate the multi-scale-training contribution
holding the no-token regime fixed.

## Common contract (all cells)

- **Input**: identical (T=16, C=6, H=32, W=64) NID tensor; same
  `splits.parquet` (M5.3 anchor); for multi-scale cells (A, B), 50/50
  fast/slow mix.
- **Optimiser**: 8-bit AdamW, batch=32, grad_accumulation=1, fp16 AMP,
  weight_decay=0.05, base_lr=1.5e-4 with linear warmup to 500 steps
  + cosine decay to 1% peak. Gradient checkpointing on.
- **Loss + reweighting**: focal γ=2 + inverse-square-root α reweighting.
- **Schedule**: 10 epochs.
- **Pretrained source**: `MCG-NJU/videomae-small-finetuned-kinetics`
  (K400) for all 4 cells.
- **MetricsWriter on**: per_step.jsonl (with grad_norm via
  `--collect-grad-norm`), per_epoch.json, confusion_per_epoch.npz
  written to `<run_dir>/metrics/`.
- **Eval policy**: all 4 cells use **double-stream `no_cycle` eval**
  via `baseline_rerun.py` for the canonical cross-cell summary
  (val_n=18,156 = 16,463 fast + 1,693 slow). Cell C/D in-training
  per-epoch eval is single-stream (val_n=16,463 or 1,693) by trainer
  design — the canonical 18,156-sample number for the main table
  comes from the post-training noise-free re-eval, not the in-training
  log. This makes Cell C/D **OOD generalization** evaluations: the
  model sees only one stream during training, then is evaluated on
  the other stream too.

## Four-cell summary table

| Cell | Run dir | Params | scale token | streams | combined | fast | slow | Bot AUROC | Bot F1 | accuracy |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| **A: main P2** | `outputs/run_20260502_184512/` | 22M | on | ms (50/50) | **0.4756** | 0.4525 | 0.6069 | 0.4968 | 0.0000 | 0.9560 |
| **B: ms+notoken** | `outputs/run_20260510_154227/` | 22M | **off** | ms (50/50) | **0.4760** | 0.4555 | 0.5999 | 0.3224 | 0.0000 | 0.9528 |
| **C: fast-only** | `outputs/run_20260510_183624/` | 22M | off | fast only | **0.4341** | 0.4604 | **0.2895** | **0.6715** | 0.0000 | 0.9408 |
| **D: slow-only** | `outputs/run_20260510_201129/` | 22M | off | slow only | **0.2471** | **0.1683** | 0.5637 | **0.6931** | 0.0000 | 0.9160 |

Numbers verbatim from each cell's eval bundle:
- Cell A: `outputs/run_20260502_184512/m5_4_phase2_eval/eval_metrics.json`
- Cell B: `outputs/run_20260510_154227/m5_10_b_videomae_eval/eval_metrics.json`
- Cell C: `outputs/run_20260510_183624/m5_10_c_videomae_eval/eval_metrics.json`
- Cell D: `outputs/run_20260510_201129/m5_10_d_videomae_eval/eval_metrics.json`

val_sample_count_total = 18,156 (fast 16,463 + slow 1,693) bit-identical
across all four cells. Cell C's slow stream is OOD-evaluated (model
trained only on fast shards); Cell D's fast stream is OOD-evaluated
(model trained only on slow shards). The combined macro_f1 for C and D
includes the OOD half of the val set.

## Pair-wise Δ (full 4-cell)

### Headline metrics across all 6 pairs

| Metric | Δ B − A | Δ C − A | Δ D − A | Δ C − B | Δ D − B | Δ D − C |
|---|---:|---:|---:|---:|---:|---:|
| combined macro_f1 | +0.0004 | **−0.042** | **−0.229** | **−0.042** | **−0.229** | **−0.187** |
| fast macro_f1 | +0.003 | +0.008 | **−0.284** | +0.005 | **−0.287** | **−0.292** |
| slow macro_f1 | −0.007 | **−0.317** | −0.043 | **−0.310** | −0.036 | **+0.274** |
| Bot per-class AUROC | **−0.175** | **+0.175** | **+0.196** | **+0.349** | **+0.371** | +0.022 |
| accuracy | −0.003 | −0.015 | −0.040 | −0.012 | −0.037 | −0.025 |

### 2×2 factorial isolation

The 2×2 design isolates the **scale token** factor (A vs B) and the
**multi-scale training** factor (B vs {C, D}) cleanly:

| Factor | Δ on combined macro_f1 | Δ on Bot AUROC |
|---|---:|---:|
| **Scale token alone** (A vs B; multi-scale fixed) | **+0.0004** (null) | **+0.175** (token preserves Bot ranking) |
| **Multi-scale training** (B vs C, no-token side) | **+0.042** | **−0.349** (multi-scale collapses Bot) |
| **Multi-scale training** (B vs D, no-token side) | **+0.229** | **−0.371** (multi-scale collapses Bot harder vs slow-only) |

**Joint interpretation**: The contribution-point-#4 lift (multi-scale
+ scale token vs single-stream + no-token) decomposes cleanly. The
**scale token contributes ≈ 0** on combined macro_f1 — within
rounding noise of the multi-scale-training side it is paired with.
The **multi-scale training data** contributes the +0.042 (or +0.229
vs slow-only) headline lift. The token's load-bearing role is
elsewhere: it stabilises Bot rare-class AUROC ranking under the
head_lr ×5 + multi-scale regime; without it, Bot AUROC collapses
from 0.4968 (Cell A) to 0.3224 (Cell B).

### OOD asymmetry: two framings (macro_f1 Δ)

The single-stream cells (C, D) evaluate on both streams, so OOD
asymmetry can be measured under two distinct framings. Both are valid
macro_f1 Δ on the val split (val_n=18,156 for combined; 16,463 fast +
1,693 slow); they measure different physical quantities and disagree
on direction.

#### Framing A: within-cell in-distribution vs OOD

For each single-stream cell, compare the same model's macro_f1 on its
trained stream (in-distribution) vs the held-out stream (OOD):

| Cell | in-distribution stream | OOD stream | Δ = OOD − in-distribution |
|---|---:|---:|---:|
| C (fast-only training) | fast 0.460445 | slow 0.289536 | **−0.171** |
| D (slow-only training) | slow 0.563750 | fast 0.168269 | **−0.395** |

Direction: slow→fast (Cell D, −0.395) is heavier than fast→slow
(Cell C, −0.171). Measures the OOD-generalization vulnerability of
each single-stream-trained model on its own.

#### Framing B: cross-cell vs Cell A same-stream reference

Each single-stream cell's OOD stream macro_f1 is compared to Cell A's
macro_f1 on the same stream. Cell A was trained on the multi-scale
50/50 mix, so it provides a "best-attainable" reference under the same
fairness contract:

| OOD stream | reference (Cell A same stream) | Δ |
|---|---:|---:|
| Cell C slow (fast→slow) | A slow 0.606877 | **−0.317** |
| Cell D fast (slow→fast) | A fast 0.452541 | **−0.284** |

Direction: fast→slow (−0.317) is heavier than slow→fast (−0.284).
Measures the deployment-side penalty for single-stream training when
evaluated against the multi-scale benchmark.

#### Reconciliation

The two framings disagree on direction because they measure different
quantities. Framing A asks "how badly does a single-stream-trained
model degrade on the OOD stream vs its own trained stream"; Framing B
asks "how far is the single-stream OOD-eval from what a multi-scale
model achieves on the same stream". Cell D under Framing A loses 0.395
because its in-distribution slow score (0.564) is high — the OOD
collapse is steep but starts from a high reference. Cell C under
Framing B loses 0.317 because the multi-scale Cell A slow reference
(0.607) is much higher than Cell A's fast reference (0.453) — slow
stream multi-scale benchmark is harder to reach with OOD training.

Both framings should be cited in paper Discussion. Recording both
preserves the honest interpretation; collapsing to one direction
selects a deployment-narrative bias.

(See also the **Training budget caveat** section below: Cell C/D ran
at native data budget — Cell C 24,260 grad_steps vs A's 48,530 (0.5×),
Cell D 2,420 (0.05×) — so the within-cell in-distribution macro_f1
that anchors Framing A is itself partially compute-limited, especially
for Cell D. Both framings inherit this caveat.)

## Training budget caveat (cross-cell macro_f1 interpretation)

| Cell | grad_steps | × Cell A | wall time |
|---|---:|---:|---:|
| A (reuse) | 48,530 | 1.0× | (reused from M5.4 P2) |
| B | 48,530 | 1.0× | 154 min |
| C (fast-only) | 24,260 | 0.5× | 81.6 min |
| D (slow-only) | 2,420 | 0.05× | 9.0 min |

Single-stream cells (C, D) train at the native data budget of their
respective stream; this is not an iso-grad-step contract. Cell C ran
roughly half the optimizer updates of Cell A/B, and Cell D ran roughly
1/20. Consequently, the macro_f1 differences D < C < A/B should be
interpreted as **compound effects of OOD generalization degradation
and training compute disparity**, not isolated OOD generalization.

For context: the M5 vanilla CE trajectory (M4.8 ep1 / M5.1 ep3 / M5.2
ep10 = 4,853 / 14,559 / 48,530 grad_steps → noise-free combined
0.3324 / 0.4230 / 0.4677) suggests that at the 24,260 grad_step budget
the CE-extrapolated ceiling is roughly 0.44–0.45; Cell C's 0.4341
under focal+α at half the budget approaches but does not exceed this
extrapolation, consistent with focal+α providing a modest lift over CE
at matched grad_step count. Cell D's 0.2471 is far below any
multi-epoch reference at 2,420 grad_steps and reflects both severe
undertraining and the heaviest OOD stream mismatch.

This caveat is recorded for honest cross-cell interpretation and is
the source for a paper Limitations entry: "single-stream ablation
cells trained at native data budget, not iso-grad-step; combined
macro_f1 reflects compound effects."

## Pair-wise observations

Recorded for round 1 closeout Findings; no Findings.md edits this
round per spec.

- **A vs B (scale token alone, holding multi-scale fixed)**: token
  contribution to combined is null (+0.0004); token preserves Bot
  AUROC (Δ −0.175 without token).
- **A vs C / A vs D (full main method vs single-stream)**: combined
  drops −0.042 / −0.229; the multi-scale + token combination provides
  this margin. The slow-stream and fast-stream macro_f1 deltas split
  by direction (see OOD asymmetry section above for the two framings);
  in-distribution macro_f1 is comparable to or higher than Cell A's
  stream-specific number in both cases. Caveat: Cells C and D ran at
  0.5× / 0.05× Cell A's grad_step budget — the compound interpretation
  applies (see Training budget caveat above).
- **B vs C / B vs D (multi-scale training alone, no-token side)**:
  same combined Δ −0.042 / −0.229 and same OOD penalties. Holding
  the no-token regime fixed, multi-scale training is worth +0.042
  to +0.229 on combined macro_f1 (depending on which single-stream
  baseline). The narrative shorthand "multi-scale training data is
  the contribution-point-#4 lift" holds at this magnitude.
- **Bot AUROC ranking across cells**: D (0.6931) > C (0.6715) >
  A (0.4968) > B (0.3224). The pattern is now clear: head_lr ×5 +
  multi-scale + no-token (Cell B) is the worst regime for Bot rare-
  class ranking; both single-stream cells preserve it (best);
  scale token recovers some of the multi-scale-induced collapse
  (Cell A 0.4968, between B and the single-stream pair). This
  refines M5-007: head_lr ×5 by itself does NOT collapse Bot —
  head_lr ×5 + multi-scale training does. The scale token mitigates
  but does not fully eliminate the multi-scale collapse.
- **Cell D fast OOD Bot AUROC = 0.8056** (highest in dim 4): a
  surprising result — Cell D's model never saw fast during training,
  yet ranks Bot on fast better than any cell that did see fast. The
  Bot signature appears to be a stable beaconing-rate pattern that's
  visible at any temporal granularity; training on slow's aggregated
  view leaves the binary discrimination boundary cleaner than training
  on fast's fine-grained noise. (Note: Bot F1 = 0 across all cells —
  the AUROC discrimination doesn't translate to a usable threshold
  under focal+α reweighting at n=12 support.)

**Two-cell observations** (recorded for round 1 closeout Findings;
no Findings.md edits this round per spec):

- **Scale token contribution to combined macro_f1 is essentially null**:
  removing the scale token (Cell B) costs Δ +0.0004 — within rounding
  noise. The token is **not** the load-bearing component for the
  headline metric; the multi-scale training data alone (Cell B keeps
  it) recovers nearly all of the main-method's combined macro_f1.
  Direction-of-effect surprise: the no-token cell is marginally above
  Cell A (+0.0004), not below. Within-noise null result.
- **Scale token preserves Bot rare-class signal at the head_lr ×5
  regime**: removing the scale token collapses Bot AUROC from 0.4968
  to 0.3224 (Δ −0.175), well below the random-baseline floor of 0.5.
  This is striking because the Bot collapse signature was previously
  attributed to head_lr ×5 (M5-007 finding); Cell B is also head_lr
  ×5 yet drops Bot AUROC further. The scale token appears to act as
  a stabilising regulariser specifically for rare-class ranking under
  the ×5 + K400 regime — without it, the model is more aggressive at
  pushing Bot mass into the dominant attack head, dropping Bot AUROC
  to 0.3224 (the lowest Bot AUROC in any K400-pretrained cell across
  M5.10 round 1 dim 1, 2, and 4 to date).
- **Slow stream slightly degrades, fast stream slightly improves**
  without the scale token. fast: +0.003 (Cell B better), slow: −0.007
  (Cell B worse). The scale_id signal (which the token's scale_embedding
  consumes) appears to mainly help the slow stream — consistent with
  the slow stream having the more distinctive temporal signature that
  benefits from explicit scale conditioning.
- **DDoS F1 final-epoch jump pattern is more pronounced without the
  scale token**: Cell B trajectory shows DDoS F1 0.378 → 0.6685 (epoch
  8 → 9, +0.29 jump). Cell A's DDoS F1 final value is 0.4590 (no jump
  pattern recorded). The no-token regime shows the random-init / 3D-conv
  jump-on-final-epoch shape more strongly even though it's K400-pretrained.

## Cell B per-class table (combined eval, epoch 9 best.pt)

| class | n | P | R | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| BENIGN | 16,829 | 0.9666 | 0.9856 | 0.9760 | 0.8767 |
| DoS Hulk | 105 | 0.5856 | 0.6190 | 0.6019 | 0.8939 |
| PortScan | 22 | 0.5000 | 0.5909 | 0.5417 | 0.9657 |
| DDoS | 228 | 0.9030 | 0.5307 | **0.6685** | 0.9912 |
| DoS GoldenEye | 61 | 0.4783 | 0.1803 | 0.2619 | 0.9563 |
| FTP-Patator | 107 | 0.0943 | 0.0467 | 0.0625 | 0.8438 |
| SSH-Patator | 175 | 0.1625 | 0.0743 | 0.1020 | 0.8675 |
| DoS slowloris | 264 | 0.8321 | 0.8447 | 0.8383 | 0.9951 |
| DoS Slowhttptest | 105 | 0.3659 | 0.1429 | 0.2055 | 0.8527 |
| Bot | 12 | 0.0000 | 0.0000 | 0.0000 | **0.3224** |
| Web Attack | 0 | — | — | — | — |
| Infiltration | 0 | — | — | — | — |
| Heartbleed | 248 | 0.9611 | 0.9960 | 0.9782 | 0.9999 |

## Cell B per-epoch trajectory (combined eval)

| epoch | combined | Bot AUROC | Bot F1 | GoldenEye F1 | DDoS F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.3133 | 0.7926 | 0.0000 | 0.1560 | 0.2738 |
| 1 | 0.3445 | 0.7392 | 0.0000 | 0.1471 | 0.1769 |
| 2 | 0.3681 | 0.7085 | 0.0000 | 0.1600 | 0.1790 |
| 3 | 0.3787 | 0.7071 | 0.0000 | 0.2114 | 0.2383 |
| 4 | 0.4223 | 0.6769 | 0.0000 | 0.3371 | 0.2162 |
| 5 | 0.3783 | 0.6408 | 0.0000 | 0.0571 | 0.2162 |
| 6 | 0.4110 | 0.4415 | 0.0000 | 0.1519 | 0.2772 |
| 7 | 0.4017 | 0.3924 | 0.0000 | 0.1013 | 0.2772 |
| 8 | 0.4388 | 0.3009 | 0.0000 | 0.2376 | 0.3780 |
| 9 | **0.4760** | 0.3224 | 0.0000 | 0.2619 | **0.6685** |

`best.pt = epoch_9_step_48530.pt` — epoch 9 combined macro_f1 was the
high-water-mark. Trajectory has 2 dips (epoch 4→5 −0.044, epoch 6→7
−0.009); max dip 0.044 is comparable to dim 1 random's 0.038 and
larger than dim 1 SSv2's 0.006.

Notable trajectory features (open-ended observations, recorded for the
round 1 closeout Findings batch; no Findings.md edits this round per
spec):

- **Bot AUROC monotone-decline pattern across epochs**: 0.7926 (epoch
  0) → 0.3009 (epoch 8) → 0.3224 (epoch 9). Steady decline with one
  small rebound at epoch 9. This is the **most pronounced Bot AUROC
  collapse trajectory across all M5.10 round 1 cells to date** — Cell
  A had a similar shape but bottomed at 0.4968; Cell B bottoms at
  0.3009 then ends at 0.3224. Removing the scale token amplifies the
  head_lr ×5 → Bot collapse pattern that M5-007 first identified.
- **DDoS F1 plateau-then-jump pattern is the most extreme observed**:
  DDoS F1 held 0.18–0.28 across epochs 0–7, jumped to 0.378 at epoch 8,
  then jumped again to **0.6685** at epoch 9 (+0.29 over a single
  epoch). The final value 0.6685 is the **highest DDoS F1 across all
  M5.10 round 1 cells** (vs Cell A 0.459, dim 1 random 0.5410, dim 1
  SSv2 0.4178, dim 2 C=4 0.5231). The no-token + multi-scale regime
  appears to lock in DDoS pattern detection particularly well at the
  final-cosine-decay epoch — direction-of-effect surprise vs the
  expectation that removing the token would hurt all classes uniformly.
- **GoldenEye F1 oscillates [0.057, 0.337]**: universal noisy-attractor
  pattern continues (12/12 forward + retrofit runs across rounds 1+2).
- **Combined macro_f1 trajectory has 2 dips** (epoch 4→5 −0.044,
  epoch 6→7 −0.009). Max dip 0.044 is comparable to dim 1 random's
  0.038 and larger than dim 1 SSv2's 0.006 + dim 2 C=4's 0.022. Cell B
  + head_lr ×5 + no token sits at the high end of dip variance.

## Cell C per-class table (combined eval, epoch 9 best.pt)

| class | n | P | R | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| BENIGN | 16,829 | 0.9645 | 0.9760 | 0.9702 | 0.9086 |
| DoS Hulk | 105 | 0.5102 | 0.4762 | 0.4926 | 0.7878 |
| PortScan | 22 | 0.6875 | 0.5000 | 0.5789 | 0.8880 |
| DDoS | 228 | 0.8916 | 0.3246 | 0.4759 | 0.9388 |
| DoS GoldenEye | 61 | 0.3770 | 0.3770 | 0.3770 | 0.9480 |
| FTP-Patator | 107 | 0.1698 | 0.0841 | 0.1125 | 0.8730 |
| SSH-Patator | 175 | 0.1000 | 0.0171 | 0.0293 | 0.9222 |
| DoS slowloris | 264 | 0.5561 | 0.8636 | 0.6766 | 0.9928 |
| DoS Slowhttptest | 105 | 0.1200 | 0.1143 | 0.1171 | 0.9227 |
| Bot | 12 | 0.0000 | 0.0000 | 0.0000 | **0.6715** |
| Web Attack | 0 | — | — | — | — |
| Infiltration | 0 | — | — | — | — |
| Heartbleed | 248 | 0.8982 | 0.9960 | 0.9446 | 0.9999 |

## Cell C per-epoch trajectory (in-training fast-only eval, n=16,463)

| epoch | macro_f1 | Bot AUROC | Bot F1 | GoldenEye F1 | DDoS F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.1680 | 0.4325 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 0.2193 | 0.4053 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 0.2713 | 0.4682 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 0.3049 | 0.4219 | 0.0000 | 0.2727 | 0.0000 |
| 4 | 0.3342 | 0.4488 | 0.0000 | 0.3235 | 0.0000 |
| 5 | 0.3366 | 0.4939 | 0.0000 | 0.1587 | 0.0000 |
| 6 | 0.3670 | 0.5321 | 0.0000 | 0.2619 | 0.0000 |
| 7 | 0.3945 | 0.5531 | 0.0000 | 0.4000 | 0.1096 |
| 8 | 0.4108 | 0.6064 | 0.0000 | 0.4130 | 0.1339 |
| 9 | **0.4604** | 0.6370 | 0.0000 | 0.4167 | 0.5121 |

`best.pt = epoch_9_step_24260.pt` — epoch 9 fast-only macro_f1 was the
high-water-mark. Trajectory has **0 dips** (perfect monotone) — the
smoothest trajectory across all M5.10 round 1 cells; max-dip 0 is
strictly better than dim 1 SSv2's 0.006 and Cell A's 0.012.

Cell C in-training fast-only macro_f1 = 0.460445 matches the noise-free
re-eval fast_only_metrics.macro_f1 = 0.460445 **bit-identically** (Δ =
0.000000). The combined and slow numbers are NEW at re-eval time
(model never saw slow during training); see the three-cell summary
table for those.

Notable Cell C trajectory features (open-ended; recorded for round 1
closeout Findings batch):

- **Bot AUROC monotone CLIMB across epochs**: 0.4325 (epoch 0) →
  0.6370 (epoch 9). This is the **opposite** direction-of-effect vs
  Cell A (0.6835 → 0.4968) and Cell B (0.7926 → 0.3224) which both
  collapse Bot AUROC. Fast-only training under K400 + ×5 preserves
  AND improves Bot rare-class ranking. The combined-eval Bot AUROC
  0.6715 (slightly higher than the in-training fast-only 0.6370 — the
  slow-stream Bot OOD evaluation contributes weakly-discriminating
  but-on-average-favourable scores) is the **highest Bot AUROC across
  all M5.10 round 1 K400 cells** (vs Cell A 0.4968, dim 2 C=4 0.5233,
  dim 1 SSv2 0.4115; dim 1 random 0.6743 is comparable but uses
  random init + head_lr ×1, a different regime).
- **DDoS F1 = 0 across epochs 0–6, then jumps**: 0 / 0 / 0 / 0 / 0 /
  0 / 0 / 0.1096 / 0.1339 / 0.5121. The first 7 epochs the model
  cannot predict DDoS at all (it sits in the cosine-decay warmup
  regime under loss too high); epochs 7–9 jump-then-jump from 0.11
  to 0.51. Compared to Cell B's 0.27 → 0.67 +0.39 jump and Cell A's
  smaller continuous climb, Cell C's DDoS F1 trajectory is the most
  delayed (later jump-onset epoch).
- **GoldenEye F1 oscillates [0, 0.42]**: 0 / 0 / 0 / 0.27 / 0.32 /
  0.16 / 0.26 / 0.40 / 0.41 / 0.42 — appears to be the noisy-attractor
  pattern (universal 13/13 across rounds 1+2) with the addition of
  3 zero epochs at the start (model can't predict GoldenEye until
  epoch 3 — possibly because its temporal signature requires longer
  windowed observations than Δt=100ms fast frames provide reliably).
- **Combined macro_f1 trajectory has 0 dips** (perfect monotone climb
  across all 10 epochs). Strictly better than every other M5.10 round
  1 cell's dip count. Single-stream training under fast may be
  inherently more stable than multi-scale training.
- **Slow-stream OOD penalty is dramatic**: noise-free re-eval slow
  macro_f1 = 0.2895 vs in-training fast 0.4604 → Δ −0.171 between
  splits within the same eval. vs Cell A's slow 0.6069 → Δ −0.317
  cross-cell. Cell C's model has never seen Δt=1s windowed inputs;
  expectation is that the patches at Δt=1s look statistically
  different (more aggregated traffic per bucket) and the K400-pretrained
  + fast-only-fine-tuned model cannot bridge that gap.

## Cell D per-class table (combined eval, epoch 8 best.pt)

| class | n | P | R | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| BENIGN | 16,829 | 0.9414 | 0.9738 | 0.9573 | 0.8175 |
| DoS Hulk | 105 | 0.4925 | 0.3143 | 0.3837 | 0.8802 |
| PortScan | 22 | 0.3333 | 0.1364 | 0.1935 | 0.9410 |
| DDoS | 228 | 0.4634 | 0.0833 | 0.1413 | 0.9670 |
| DoS GoldenEye | 61 | 0.0000 | 0.0000 | 0.0000 | 0.7693 |
| FTP-Patator | 107 | 0.1868 | 0.1589 | 0.1717 | 0.8888 |
| SSH-Patator | 175 | 0.1688 | 0.3829 | 0.2343 | 0.9260 |
| DoS slowloris | 264 | 0.8229 | 0.2992 | 0.4389 | 0.8999 |
| DoS Slowhttptest | 105 | 0.2000 | 0.0190 | 0.0348 | 0.8385 |
| Bot | 12 | 0.0000 | 0.0000 | 0.0000 | **0.6931** |
| Web Attack | 0 | — | — | — | — |
| Infiltration | 0 | — | — | — | — |
| Heartbleed | 248 | 1.0000 | 0.0887 | 0.1630 | 0.9630 |

## Cell D per-epoch trajectory (in-training slow-only eval, n=1,693)

| epoch | macro_f1 | Bot AUROC | Bot F1 | GoldenEye F1 | DDoS F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2652 | 0.5070 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 0.4744 | 0.4667 | 0.0000 | 0.0000 | 0.9545 |
| 2 | 0.4828 | 0.5127 | 0.0000 | 0.0000 | 0.9362 |
| 3 | 0.4817 | 0.5863 | 0.0000 | 0.0000 | 0.8837 |
| 4 | 0.5300 | 0.6502 | 0.0000 | 0.0000 | 0.9333 |
| 5 | 0.5383 | 0.6721 | 0.0000 | 0.0000 | 0.9778 |
| 6 | 0.5471 | 0.7108 | 0.0000 | 0.0000 | 0.9767 |
| 7 | 0.5095 | 0.6926 | 0.0000 | 0.0000 | 0.9362 |
| 8 | **0.5637** | 0.6643 | 0.0000 | 0.0000 | 0.9048 |
| 9 | 0.5543 | 0.6950 | 0.0000 | 0.0000 | 0.9362 |

`best.pt = epoch_8_step_2178.pt` — epoch 8 slow-only macro_f1 was the
high-water-mark; epoch 9 regressed slightly (Δ −0.009, normal late-
cosine-decay noise). Trajectory has 3 dips (epoch 2→3, 6→7, 8→9), max
dip 0.038. Cell D's slow-only training reaches a higher
in-distribution macro_f1 (0.5637) than Cell C's fast-only training
reaches on its in-distribution (0.4604) — the slow stream's coarser
temporal aggregation may carry more discriminative signal per frame.

Cell D in-training slow-only macro_f1 = 0.563750 (best at epoch 8)
matches the noise-free re-eval slow_only_metrics.macro_f1 = 0.563750
**bit-identically** (Δ = 0.000000). The combined and fast OOD numbers
are NEW at re-eval time; see the four-cell summary table.

Notable Cell D trajectory features (open-ended; recorded for round 1
closeout Findings batch):

- **Bot AUROC monotone CLIMB** (with one small mid-trajectory dip):
  0.5070 (epoch 0) → 0.6950 (epoch 9). Same direction-of-effect as
  Cell C and opposite to Cell A/B. Single-stream training (either
  fast OR slow) under K400 + ×5 preserves Bot rare-class ranking,
  while multi-scale training collapses it.
- **DDoS F1 jumps to 0.95+ at epoch 1**: 0 / 0.9545 / 0.9362 / 0.8837
  / 0.9333 / 0.9778 / 0.9767 / 0.9362 / 0.9048 / 0.9362. The slow
  stream has an extremely clean DDoS signature visible from the very
  first epoch — far cleaner than fast (Cell C's DDoS F1 stayed 0
  through epoch 6 and only reached 0.5121 by epoch 9). Slow
  Δt=1s windows aggregate enough packets that DDoS pattern
  recognition is essentially saturated within 1 epoch.
- **GoldenEye F1 = 0 across all 10 epochs**: Cell D never predicts
  GoldenEye correctly during training. The slow stream may have very
  few GoldenEye train samples, OR the GoldenEye temporal signature
  requires fine-grained Δt=100ms observation. Compare to Cell C
  (fast-only) which reached GoldenEye F1 0.42 by epoch 9. The slow→
  fast OOD penalty for GoldenEye is structural — without per-frame
  temporal resolution, GoldenEye is invisible to Cell D.
- **Combined macro_f1 trajectory has 3 dips** (0.4828 → 0.4817 −0.001;
  0.5471 → 0.5095 −0.038; 0.5637 → 0.5543 −0.009). Max dip 0.038
  comparable to Cell B's 0.044 and dim 1 random's 0.038; smaller than
  the deepest dim-2 dips. Cell D dips occur on a small val_n=1,693
  baseline, so per-epoch noise has higher relative magnitude.

## Cell D sanity verification

- **val_sample_count_total**: in-training slow-only val_n = 1,693 in
  all 10 epochs ✓ (single-scale training path, model only sees slow
  shards). Noise-free re-eval double-stream val_n = 18,156 ✓.
- **In-training vs noise-free re-eval slow-only sub-metric** (best
  ckpt = epoch 8): in-training slow-only macro_f1 = 0.563750;
  noise-free re-eval slow_only_metrics.macro_f1 = 0.563750; **Δ =
  0.000000** (bit-identical, within ≤ 5e-5 fairness contract).
  Combined and fast noise-free numbers are new at re-eval time and
  have no in-training counterpart (Cell D trained slow-only, the
  fast stream is OOD); per the spec, this is expected for the
  single-stream-train + double-stream-eval ablation cell, not a
  Δ-sanity violation.
- **Wall time**: 539.6 s ≈ 9.0 min (10 epochs × ~36-41 s/epoch on
  the 8 GB box, peak GPU 426 MB throughout). Slow-stream-only training
  is ~1.5× faster wall-clock than Cell C's 81.6 min and ~17× faster
  than Cell B's 154 min because grad_steps/epoch ≈ 242 (vs Cell C's
  2,426 vs Cell B's 4,853). Total grad_steps = 2,420 (one tenth of
  Cell C, one twentieth of Cell B). Slow train_n = 7,752 — the
  smallest training set in the dim-4 suite.
- **per_step.jsonl**: 2,420 rows (1 per grad step, full schedule
  completed). Each row carries `grad_norm` field via
  `--collect-grad-norm`. Standard fp16 GradScaler dynamic-loss-scaling
  startup Inf pattern in the first few steps; remainder all finite.
- **per_epoch.json**: 10 epoch records with `metrics.combined.{macro_f1,
  accuracy, auroc_macro, per_class}` populated; "combined" here =
  single-stream slow-only (the trainer naming convention; the noise-
  free re-eval applies the multi-stream split).
- **confusion_per_epoch.npz**: 10 keys `epoch_0..epoch_9`, each
  (13, 13) int64, every sum = 1,693 ✓ (slow-only val_n).
- **K400 source verified**: trainer log records `loaded pretrained
  backbone: MCG-NJU/videomae-small-finetuned-kinetics` and the
  patch_embed adapter log shows `ch[0:3] downsampled 16→8
  shape=(384, 3, 2, 8, 8) norm=5.24` — same K400 source as Cells A/B/C.
- **Single-stream slow dataloader**: trainer log shows `NidShardDataset:
  12 url(s), label_mode=collapsed13, shuffle_buffer=1000,
  keep_split=train` — 12 shards from the slow Δt=1s set; vs Cell B's
  `MultiScaleNidDataset: ... mix_ratio=0.5 ...` covering 113 fast +
  12 slow shards. Cell D is single-stream by trainer dispatch,
  scale_id passed as all-zeros (irrelevant — token off).
- **Two-dir consolidation**: tee `outputs/run_20260510_201036/`
  consolidated into Trainer-internal `outputs/run_20260510_201129/`
  and rmdir'd. Same pattern as Cells B/C + dim 2 + dim 1 SSv2.

## Cross-dimension cross-validation (round 1 dim 1 + dim 2 + dim 4)

Recorded for round 1 closeout Findings batch (no Findings.md edits this
round per spec). The Bot AUROC pattern across all 8 forward-trained
cells in M5.10 round 1 lets us re-state the M5-007 finding:

| Cell | dim | regime | head_lr | Bot AUROC |
|---|---|---|---:|---:|
| dim 1 random | 1 | none + ms | ×1 | **0.6743** |
| dim 1 SSv2 | 1 | SSv2 + ms | ×5 | 0.4115 |
| dim 2 C=4 | 2 | K400 + ms (4ch) | ×5 | 0.5233 |
| **A: main P2** | (anchor) | K400 + ms + token | ×5 | 0.4968 |
| **B: ms+notoken** | 4 | K400 + ms − token | ×5 | **0.3224** |
| **C: fast-only** | 4 | K400 + fast only − token | ×5 | **0.6715** |
| **D: slow-only** | 4 | K400 + slow only − token | ×5 | **0.6931** |

The splitting variable for Bot AUROC preservation is now **(head_lr
×5) ∧ (multi-scale training)** = the malicious regime that collapses
Bot ranking. Removing either factor (×1 in dim 1 random; single-stream
in dim 4 C/D) preserves Bot ≥ 0.67. The scale token (dim 4 A vs B)
provides a partial mitigation when both malicious factors are present:
0.4968 with token vs 0.3224 without. Pretrained source (dim 1
K400 vs SSv2) and channel count (dim 2 C=4 vs C=6) are second-order
modulators within the malicious regime, not splitting variables.

This refines M5-007 from "head_lr ×5 collapses Bot AUROC" to:
**"head_lr ×5 collapses Bot AUROC under multi-scale training; the
scale token mitigates partially; single-stream training preserves
Bot."**

## Cell B sanity verification

- **val_sample_count_total = 18,156** in all 10 epochs ✓
- **In-training vs noise-free re-eval** (best ckpt = epoch 9):
  in-training combined macro_f1 = 0.476036; noise-free re-eval
  combined macro_f1 = 0.476036; **Δ = 0.000000** (bit-identical,
  within ≤ 5e-5 fairness contract for `--eval-strategy no_cycle`).
- **Wall time**: 9,243.0 s ≈ 154.0 min (10 epochs × ~12–13 min each
  on the 8 GB box, peak GPU 426–429 MB throughout — well under the
  485 MB threshold per the dim 2-revised stop-and-report contract.
  vs Cell A's 488 MB and dim 2 C=4's 480-482 MB, the 426-429 MB
  peak GPU is consistent with the use_scale_token=False codepath
  saving the per-batch scale_token cat allocation + 1 token slot in
  pos_emb broadcasting).
- **per_step.jsonl**: 48,530 rows (1 per grad step, full schedule
  completed). Each row carries `grad_norm` field via
  `--collect-grad-norm`. Standard fp16 GradScaler dynamic-loss-scaling
  startup Inf pattern in the first few steps; remainder all finite.
- **per_epoch.json**: 10 epoch records, all with
  `metrics.combined.{macro_f1, accuracy, auroc_macro, per_class}`
  populated. (No `fast` / `slow` keys in forward-instrumented runs;
  the noise-free re-eval `eval_metrics.json` carries those splits via
  `baseline_rerun.py`'s scale_id partition.)
- **confusion_per_epoch.npz**: 10 keys `epoch_0..epoch_9`, each
  (13, 13) int64, every sum = 18,156 ✓.
- **K400 source verified**: trainer log records `loaded pretrained
  backbone: MCG-NJU/videomae-small-finetuned-kinetics` and the
  patch_embed adapter log shows `ch[0:3] downsampled 16→8
  shape=(384, 3, 2, 8, 8) norm=5.24; ch[3:6] kaiming-init
  shape=(384, 3, 2, 8, 8) norm=27.77`. The K400-derived ch[0:3] norm
  5.24 is the deterministic K400 + (16→8) trilinear-downsample value
  (verified offline: `adapt_conv3d_to_6ch(K400_proj, target=(2,8,8),
  n_extra=3)` reproduces 5.2375 byte-for-byte; SSv2 produces 5.22 with
  the same pipeline). **Documentation correction**: prior dim 1 SSv2
  doc + dim 2 C=4 doc cited K400 main P2 ch[0:3] norm as "3.83" /
  "3.85" — those numbers actually came from the noise-free re-eval
  logs (which default `--pretrained=None` and load a random-init
  backbone with norm ≈ 3.84), not from the K400 training startup
  logs. The correct K400-derived ch[0:3] norm under the project's
  (2, 8, 8) adapter is 5.24. The prior docs do not need amending
  because the bit-identity contract (C=4 ch[0:3] = C=6 ch[0:3] under
  the same K400 source + same downsample) holds at norm 5.24 just as
  it would have at any common value; the SSv2 vs K400 discrimination
  argument from dim 1 SSv2 doc remains valid (SSv2 5.22 vs K400 5.24
  are still distinct enough to discriminate, and the combined macro_f1
  numbers 0.4413 vs 0.4756 rule out silent cross-load).
- **No silent token-on load**: `position_embedding rebuilt: shape=(1,
  256, 384) (256 patches (no scale token))` in the trainer log; the
  saved best.pt's `backbone.embeddings.position_embeddings` has shape
  `[1, 256, 384]` (verified at re-eval load time — a token-on ckpt
  would refuse to load into a token-off model with a (1, 257, 384) vs
  (1, 256, 384) mismatch error, and the noise-free re-eval succeeded
  only after `--use-scale-token false` was passed). Forward-hook test
  in `tests/test_videomae_nid.py::test_videomae_forward_skips_scale_token_when_disabled`
  also bit-precision verifies the 256-token encoder seq_len.

## Cell C sanity verification

- **val_sample_count_total**: in-training fast-only val_n = 16,463 in
  all 10 epochs ✓ (single-scale training path, model only sees fast
  shards). Noise-free re-eval double-stream val_n = 18,156 ✓
  (canonical cross-cell number; combines fast 16,463 + slow 1,693 OOD).
- **In-training vs noise-free re-eval fast-only sub-metric** (best ckpt
  = epoch 9): in-training fast-only macro_f1 = 0.460445; noise-free
  re-eval fast_only_metrics.macro_f1 = 0.460445; **Δ = 0.000000**
  (bit-identical, within ≤ 5e-5 fairness contract for `--eval-strategy
  no_cycle`). Combined and slow noise-free numbers are new at re-eval
  time and have no in-training counterpart (Cell C trained
  fast-only, the slow stream is OOD); per the spec, this does NOT
  trigger Δ-sanity violation — it is the expected behaviour for the
  single-stream-train + double-stream-eval ablation cell.
- **Wall time**: 4,895.2 s ≈ 81.6 min (10 epochs × ~5.9 min/epoch on
  the 8 GB box, peak GPU 426–428 MB throughout — well under the 485 MB
  threshold). Single-stream training is ~1.9× faster wall-clock than
  multi-scale Cell B (154 min) because grad_steps/epoch = 2,426 vs
  4,853 (half — single fast train_n only, no slow added).
- **per_step.jsonl**: 24,260 rows (1 per grad step, full schedule
  completed). Each row carries `grad_norm` field via
  `--collect-grad-norm`. Standard fp16 GradScaler dynamic-loss-scaling
  startup Inf pattern in the first few steps; remainder all finite.
  **Note**: 24,260 grad_steps ≠ Cell A/B's 48,530 — the multi-scale
  vs single-scale distinction halves the per-epoch grad step count.
- **per_epoch.json**: 10 epoch records with `metrics.combined.{macro_f1,
  accuracy, auroc_macro, per_class}` populated. Cell C's "combined"
  refers to the single-stream fast-only eval; there is no fast/slow
  split in the in-training records (forward-instrumentation physical
  fact). The split comes from `baseline_rerun.py`'s scale_id partition
  at re-eval time.
- **confusion_per_epoch.npz**: 10 keys `epoch_0..epoch_9`, each
  (13, 13) int64, every sum = 16,463 ✓ (fast-only val_n, not 18,156).
- **K400 source verified**: trainer log records `loaded pretrained
  backbone: MCG-NJU/videomae-small-finetuned-kinetics` and the
  patch_embed adapter log shows `ch[0:3] downsampled 16→8
  shape=(384, 3, 2, 8, 8) norm=5.24; ch[3:6] kaiming-init
  shape=(384, 3, 2, 8, 8) norm≈27.7` — the deterministic K400 + (16→8)
  trilinear-downsample value (matches Cell B startup norm 5.24, Cell A
  startup norm — although Cell A's actual training startup log is not
  in the M5.4 P2 retrofit dir, the deterministic offline reproduction
  yields 5.24 from the same K400 source).
- **No silent multi-scale dataloader**: `MultiScaleNidDataset` log NOT
  present in Cell C training startup; instead `NidShardDataset:
  113 url(s), label_mode=collapsed13, shuffle_buffer=1000,
  keep_split=train` for the single-stream path (113 shards from the
  fast Δt=100ms set; vs Cell B's `MultiScaleNidDataset:
  mix_ratio=0.5, seed=42, epoch_end_strategy=round_robin` covering
  113 fast + 12 slow shards). Cell C is single-stream by trainer
  dispatch, scale_id passed as all-zeros (irrelevant — token off).
- **Two-dir consolidation**: Cell C tee dir `outputs/run_20260510_183234/`
  consolidated into Trainer-internal `outputs/run_20260510_183624/` and
  rmdir'd post-training. Same pattern as Cell B + dim 2 + dim 1 SSv2.

## Adapter ch[0:3] norm forensic correction (cross-doc)

Cell B's training startup log records ch[0:3] norm = 5.24, verified
offline as the deterministic output of
``adapt_conv3d_to_6ch(K400_pretrained_proj, target=(2,8,8), n_extra=3)``
at 5.2375 byte-for-byte.

This contradicts the ch[0:3] norm citations in two prior round 1 docs:
- `docs/m5_10_pretrained_ablation.md` cites K400 main P2 ch[0:3]
  norm = 3.83 and SSv2 ch[0:3] norm = 5.22
- `docs/m5_10_motion_channel_ablation.md` cites K400-derived ch[0:3]
  norm = 3.85

The forensic explanation: the prior 3.83/3.85 values appear to have
been pulled from `--pretrained=""` (random init) re-eval logs rather
than from K400-loaded training startup logs. Offline verification this
commit (reproduction: load `MCG-NJU/videomae-small-finetuned-kinetics`
or `MCG-NJU/videomae-small-finetuned-ssv2` via
`_load_backbone_with_fallback`, then call
`adapt_conv3d_to_6ch(b.embeddings.patch_embeddings.projection,
target_kernel=(2,8,8), target_stride=(2,8,8), n_extra=3)` and take
`new.weight.data[:, :3].norm().item()`):

- K400 source + adapter:  norm = **5.237500** (verified deterministic)
- SSv2 source + adapter:  norm = **5.219768** (verified deterministic)
- |Δ| (SSv2 − K400)      = 0.017732

The SSv2 verified norm 5.220 matches the dim 1 SSv2 doc's cited "5.22"
to three decimal places — **only the K400 citation was the documentation
error**; the SSv2 citation was correct.

Implications:
- Round 1 functional findings remain valid. The bit-identity contract
  (C=4 ch[0:3] = C=6 ch[0:3] given same source) is enforced by
  ``test_videomae_c4_adapter_consistency_with_c6_on_pretrained`` and
  does not depend on the cited norm magnitude. The SSv2 vs K400
  downstream-functional discrimination is anchored at the macro_f1
  level (0.4413 vs 0.4756, Δ −0.034) and at the silent-load rule-out
  level (different source ckpts produce distinguishable downstream
  behaviour), not at the startup-time adapter-norm magnitude.
- The dim 1 SSv2 doc's "SSv2 vs K400 norm discrimination" narrative
  is now invalid: SSv2 norm 5.220 vs K400 norm 5.238 differ by only
  0.018, which is **within rounding noise** of the cited two-decimal
  precision (both round to "5.24" / "5.22"). The norm magnitude cannot
  discriminate between SSv2 and K400 silent-load. A different
  forensic anchor (e.g. macro_f1 Δ −0.034, or a per-layer state_dict
  weight signature) must replace it during round 1 closeout.

Per forensic-record discipline, this section records the correction
and routes the dim 1 / dim 2 doc edits to the round 1 closeout batch
update; the dim 1 / dim 2 doc text is **not** modified in this commit.

## Phase 0 sanity verification (recorded for Phase 1 readiness)

`--use-scale-token` flag implementation (from the Phase 0 review):
- `ModelConfig.use_scale_token: bool = True` field added to
  `src/nid_video/utils/config.py`. Default = main-method behaviour.
- `VideoMAESmallForNID(__init__, use_scale_token: bool = True)` saved
  to `self.use_scale_token`; passed through to
  `_adapt_position_embedding` which builds 256-pos pos_emb when False
  (vs 257 with token); `forward` conditionally skips the
  `scale_token + scale_embedding(scale_id)` cat when False. The
  `scale_token` Parameter and `scale_embedding` Module are always
  built in `__init__` (ckpt-shape stability across hyperparam-sweep
  tooling).
- `--use-scale-token {true,false}` CLI flag added to `scripts/train.py`
  (mirrors the `--num-channels` post-load cfg override pattern from
  dim 2) and to `scripts/baseline_rerun.py` (required for re-eval to
  load Cell B/C/D ckpts; mismatch yields a fail-loud (1,257) vs
  (1,256) shape error on `position_embeddings`).
- 4 new dim-4 tests added to `tests/test_videomae_nid.py`:
  `test_videomae_use_scale_token_false_pos_emb_is_256`,
  `test_videomae_forward_skips_scale_token_when_disabled` (forward-
  hook verifies encoder seq_len=256 vs 257 bit-precision),
  `test_videomae_use_scale_token_false_keeps_scale_params_for_ckpt_compat`,
  `test_model_config_use_scale_token_default_is_true`. Fast suite
  regression: 259 passed (was 255), 22 deselected (slow tests),
  0 failed.

Cell A K400 main P2 anchor verbatim verify (no re-train, reused from
M5.4 P2 retrofit `outputs/run_20260502_184512/metrics/per_epoch.json`):
- epoch 9 combined macro_f1 = **0.475576** ✓
- epoch 9 combined accuracy = **0.955993** ✓
- epoch 9 combined auroc_macro = **0.764068** ✓
- val_sample_count_total = 18,156 across all 10 epochs ✓

Three Phase 0 dry-runs (5 grad steps each, all peak GPU 426 MB,
all forward token count verified correct):
- Cell B config (multi-scale + use_scale_token=False) → token=256 ✓
- Cell C config (single-scale fast + use_scale_token=False) → token=256 ✓,
  in-training val_n=16,463
- Cell D config (single-scale slow + use_scale_token=False) → token=256 ✓,
  in-training val_n=1,693

## Source artefacts

| Cell | Source training run | Eval bundle | Commit |
|---|---|---|---|
| A: main P2 | `outputs/run_20260502_184512/` | `m5_4_phase2_eval/` | `1d1a61e` (M5.4 P2) |
| B: ms+notoken | `outputs/run_20260510_154227/` | `m5_10_b_videomae_eval/` | `5bc6b32` (Phase 0 + Cell B) |
| C: fast-only | `outputs/run_20260510_183624/` | `m5_10_c_videomae_eval/` | `3d830af` (Cell C) |
| D: slow-only | `outputs/run_20260510_201129/` | `m5_10_d_videomae_eval/` (this commit) | (this commit) |

`outputs/**` is gitignored so artefacts ship locally only; recreate
the Cell B run via:

```bash
uv run python scripts/train.py \
    --shard-pattern-fast "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \
    --shard-pattern-slow "data/processed/cicids2017_dt1000ms_v2/*/shards/shard-*.tar" \
    --splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet \
    --pretrained MCG-NJU/videomae-small-finetuned-kinetics \
    --head-lr-multiplier 5.0 \
    --use-scale-token false \
    --num-epochs 10 \
    --loss-fn focal --focal-gamma 2.0 --reweighting inverse_sqrt \
    --eval-strategy no_cycle --label-mode collapsed13 \
    --collect-grad-norm
```

Recreate the Cell B noise-free re-eval via:

```bash
uv run python scripts/baseline_rerun.py \
    --resume outputs/run_20260510_154227/ckpt/best.pt \
    --use-scale-token false \
    --shard-pattern-fast "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \
    --shard-pattern-slow "data/processed/cicids2017_dt1000ms_v2/*/shards/shard-*.tar" \
    --splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet \
    --output-dir outputs/run_20260510_154227/m5_10_b_videomae_eval/ \
    --task-label "M5.10 round 1 dim 4 cell B — VideoMAE-S K400 head_lr ×5 multi-scale no-token"
```

The `--use-scale-token false` flag is required at re-eval time —
without it, baseline_rerun builds the default `use_scale_token=True`
model and the ckpt load fails with a fail-loud shape mismatch on
`backbone.embeddings.position_embeddings: (1, 257, 384) vs (1, 256, 384)`.

Recreate the Cell C run via:

```bash
uv run python scripts/train.py \
    --shard-pattern "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \
    --splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet \
    --pretrained MCG-NJU/videomae-small-finetuned-kinetics \
    --head-lr-multiplier 5.0 \
    --use-scale-token false \
    --num-epochs 10 \
    --loss-fn focal --focal-gamma 2.0 --reweighting inverse_sqrt \
    --eval-strategy no_cycle --label-mode collapsed13 \
    --collect-grad-norm
```

Note `--shard-pattern` (single-scale legacy flag, M3 codepath) instead
of `--shard-pattern-fast` + `--shard-pattern-slow` (multi-scale).
Trainer's `_validate_args` enforces "multi-scale requires BOTH
--shard-pattern-fast AND --shard-pattern-slow"; passing only one of
the multi-scale flags raises SystemExit. Cell C uses the legacy
single-scale path with the fast-shard glob.

Recreate the Cell C noise-free re-eval (double-stream OOD on slow):

```bash
uv run python scripts/baseline_rerun.py \
    --resume outputs/run_20260510_183624/ckpt/best.pt \
    --use-scale-token false \
    --shard-pattern-fast "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \
    --shard-pattern-slow "data/processed/cicids2017_dt1000ms_v2/*/shards/shard-*.tar" \
    --splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet \
    --output-dir outputs/run_20260510_183624/m5_10_c_videomae_eval/ \
    --task-label "M5.10 round 1 dim 4 cell C — VideoMAE-S K400 head_lr ×5 fast-only single-stream no-token"
```

Recreate the Cell D run via:

```bash
uv run python scripts/train.py \
    --shard-pattern "data/processed/cicids2017_dt1000ms_v2/*/shards/shard-*.tar" \
    --splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet \
    --pretrained MCG-NJU/videomae-small-finetuned-kinetics \
    --head-lr-multiplier 5.0 \
    --use-scale-token false \
    --num-epochs 10 \
    --loss-fn focal --focal-gamma 2.0 --reweighting inverse_sqrt \
    --eval-strategy no_cycle --label-mode collapsed13 \
    --collect-grad-norm
```

Recreate the Cell D noise-free re-eval (double-stream OOD on fast):

```bash
uv run python scripts/baseline_rerun.py \
    --resume outputs/run_20260510_201129/ckpt/best.pt \
    --use-scale-token false \
    --shard-pattern-fast "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \
    --shard-pattern-slow "data/processed/cicids2017_dt1000ms_v2/*/shards/shard-*.tar" \
    --splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet \
    --output-dir outputs/run_20260510_201129/m5_10_d_videomae_eval/ \
    --task-label "M5.10 round 1 dim 4 cell D — VideoMAE-S K400 head_lr ×5 slow-only single-stream no-token"
```

## Findings + Methods draft

Deferred to round 1 closeout (after Cells C and D land + dimensions
1, 2, 4 are aggregated). Findings fact-layer entries + paper Methods
draft will be batch-written once the M5.10 round 1 dimension suite is
complete. Per round 1 spec, `prompts/Findings.md` is NOT edited this
round.

## Round 1 closeout 必修订项 (cross-doc)

The following corrections are deferred to round 1 closeout batch
update (do NOT apply in this revision):

- `docs/m5_10_pretrained_ablation.md`:
  - K400 main P2 ch[0:3] norm citation 3.83 → corrected to 5.24
    (verified deterministic this commit, see Adapter ch[0:3] norm
    forensic correction section).
  - SSv2 ch[0:3] norm citation 5.22 → **already correct** (offline
    verify this commit reproduced 5.219768, matches cited "5.22" to
    three decimal places); no number change needed for this entry.
  - **"SSv2 vs K400 norm discrimination" narrative needs full
    rewrite** (not just number substitution): with K400 = 5.238 and
    SSv2 = 5.220, the |Δ| 0.018 is below the cited two-decimal
    precision and cannot discriminate between the two source ckpts.
    A replacement forensic anchor for "no silent K400 load in the
    SSv2 cell" is needed — candidates: (a) downstream macro_f1
    Δ −0.034 (functional discrimination), (b) per-layer state_dict
    weight signature comparison (e.g., norm of a deeper layer that
    diverges between the two pretraining tasks), (c) explicit
    HuggingFace ckpt id verbatim from `_load_backbone_with_fallback`
    log line.
- `docs/m5_10_motion_channel_ablation.md`:
  - K400-derived ch[0:3] norm citation 3.85 → corrected to 5.24
    (same root cause: prior log was likely from random-init re-eval).

These corrections are cross-doc and should land in a single closeout
commit alongside the broader Findings.md batch update (per
M5_handoff_summary_v4.md §"Round 1 closeout 必修订项").
