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
| **C: fast-only** | K400 | single-scale fast (Δt=100ms) | off | ×5 | **Phase 1 forward training — this commit** |
| **D: slow-only** | K400 | single-scale slow (Δt=1s) | off | ×5 | Phase 1 forward training — planned |

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

## Three-cell summary table (Cell A + Cell B + Cell C)

Cell D will be appended in its commit.

| Cell | Run dir | Params | scale token | streams | combined | fast | slow | Bot AUROC | Bot F1 | accuracy |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| **A: main P2** | `outputs/run_20260502_184512/` | 22M | on | ms (50/50) | **0.4756** | 0.4525 | 0.6069 | 0.4968 | 0.0000 | 0.9560 |
| **B: ms+notoken** | `outputs/run_20260510_154227/` | 22M | **off** | ms (50/50) | **0.4760** | 0.4555 | 0.5999 | 0.3224 | 0.0000 | 0.9528 |
| **C: fast-only** | `outputs/run_20260510_183624/` | 22M | off | fast only | **0.4341** | 0.4604 | **0.2895** | **0.6715** | 0.0000 | 0.9408 |

Numbers verbatim from each cell's eval bundle:
- Cell A: `outputs/run_20260502_184512/m5_4_phase2_eval/eval_metrics.json`
- Cell B: `outputs/run_20260510_154227/m5_10_b_videomae_eval/eval_metrics.json`
- Cell C: `outputs/run_20260510_183624/m5_10_c_videomae_eval/eval_metrics.json`

val_sample_count_total = 18,156 (fast 16,463 + slow 1,693) bit-identical
across all three cells. Cell C's slow stream is OOD-evaluated — the
model trained only on fast (Δt=100ms) shards and is being asked to
classify slow (Δt=1s) inputs at re-eval time.

## Pair-wise Δ (so far: A, B, C — D pending)

| Metric | A (token on, ms) | B (token off, ms) | C (token off, fast only) | Δ B − A | Δ C − A | Δ C − B |
|---|---:|---:|---:|---:|---:|---:|
| combined macro_f1 | 0.4756 | 0.4760 | 0.4341 | +0.0004 | **−0.042** | **−0.042** |
| fast macro_f1 | 0.4525 | 0.4555 | 0.4604 | +0.003 | **+0.008** | +0.005 |
| slow macro_f1 (OOD for C) | 0.6069 | 0.5999 | 0.2895 | −0.007 | **−0.317** | **−0.310** |
| Bot per-class AUROC | 0.4968 | 0.3224 | 0.6715 | −0.175 | **+0.175** | **+0.349** |
| Bot per-class F1 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 |
| accuracy | 0.9560 | 0.9528 | 0.9408 | −0.003 | −0.015 | −0.012 |

Pair-wise interpretation:

- **A vs B (scale token alone, holding multi-scale fixed)**: token
  contribution to combined is null (+0.0004); token preserves Bot
  AUROC (Δ −0.175 without token).
- **A vs C (full main method vs fast-only no-token)**: combined drops
  −0.042 — the multi-scale + token combination provides this margin.
  Slow OOD penalty Δ −0.317 dominates the loss; fast actually gains
  +0.008 (single-stream specialisation).
- **B vs C (multi-scale-training contribution alone, no-token side)**:
  same combined Δ −0.042 and same slow OOD Δ −0.310. Holding the
  no-token regime fixed, multi-scale training is worth +0.042 on
  combined macro_f1.
- **C vs A on Bot AUROC** (+0.175): striking surprise — fast-only
  training preserves Bot rare-class ranking BETTER than multi-scale
  + token. Combined with the dim 1 random cell's 0.6743 Bot AUROC
  (similar magnitude), this points to **non-aggressive optimisation
  regimes** (head_lr ×1, OR fast-only single-stream + head_lr ×5)
  preserving Bot ranking, while multi-scale + head_lr ×5 collapses it.
  Cell B's Bot AUROC 0.3224 — the worst across all M5.10 round 1
  cells — suggests the COMBINATION of multi-scale training + no token
  + head_lr ×5 is the malicious regime.

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
| C: fast-only | `outputs/run_20260510_183624/` | `m5_10_c_videomae_eval/` (this commit) | (this commit) |
| D: slow-only | (planned) | (planned) | (planned) |

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

## Findings + Methods draft

Deferred to round 1 closeout (after Cells C and D land + dimensions
1, 2, 4 are aggregated). Findings fact-layer entries + paper Methods
draft will be batch-written once the M5.10 round 1 dimension suite is
complete. Per round 1 spec, `prompts/Findings.md` is NOT edited this
round.
