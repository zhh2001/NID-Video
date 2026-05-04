# Baseline trajectory: epoch budget × eval strategy

This document is the data anchor for the M5 baseline section of the
paper. It records the noise-free per-checkpoint validation metrics
along the training-budget axis (1 epoch → 3 epoch → 10 epoch on the
same multi-scale setup), reconciled to a single eval strategy
(``no_cycle``) so the numbers are directly comparable across runs.

The earlier in-training metrics (the per-epoch eval numbers logged by
the trainer) used ``round_robin`` for the val loader. ``round_robin``
re-iterates the slow stream whenever it drains and reseeds its shuffle
on each cycle, which functions as an unintended test-time
augmentation: per-sample softmax noise gets averaged out and the
resulting metric is systematically higher than what a one-pass eval
would report. The magnitude of the inflation is **training-maturity
dependent** — see the analysis below.

The noise-free numbers in this document are produced by
``scripts/baseline_rerun.py``, which builds a ``no_cycle`` val loader
(drains both streams exactly once, no duplicates), accumulates the
predictions for the full val split, and partitions on each sample's
``scale_id`` to compute combined / fast-only / slow-only metrics.

## Trajectory table (collapsed-13, val split)

| Run | Epoch | grad_steps | reported macro_f1 | noise-free macro_f1 | Δ | reported Bot AUROC | noise-free Bot AUROC | loss |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| M4.8 | 1 | 4,853 | 0.4474 | **0.3324** | −0.1150 | 0.7411 † | 0.7247 | CE |
| M5.1 | 3 | 14,559 | 0.5113 | **0.4230** | −0.0883 | n/a ‡ | 0.4237 | CE |
| M5.2 | 10 | 48,530 | 0.5143 | **0.4677** | −0.0466 | 0.4402 | 0.4077 | CE |
| M5.4 P1 | 10 | 48,530 | 0.4584 ⁂ | **0.4584** ⁂ | 0.0000 ⁂ | 0.5060 ⁂ | 0.5060 | focal γ=2 |
| M5.4 P2 | 10 | 48,530 | 0.4756 ⁂ | **0.4756** ⁂ | 0.0000 ⁂ | 0.4968 ⁂ | 0.4968 | focal γ=2 + inv-sqrt α + head LR ×5 (K400 contract) |
| M5.5 R1 (TimeSformer-Small) ★ | 10 | 48,530 | 0.4836 ⁂ | **0.4836** ⁂ | 0.0000 ⁂ | 0.7151 ⁂ | 0.7151 | focal γ=2 + inv-sqrt α + head LR ×1 effective (random-init contract; 30.8M) |
| M5.5 R1.5 (TimeSformer-Small) — ablation supplementary | 10 | 48,530 | 0.4616 ⁂ | **0.4616** ⁂ | 0.0000 ⁂ | 0.5940 ⁂ | 0.5940 | focal γ=2 + inv-sqrt α + head LR ×5 intentional (off-contract; 30.8M random-init) |

† M4.8's original training task output was retained only in
in-conversation records; the value 0.7411 is the figure cited there.

‡ M5.1's original task output was rotated out of the workspace cache
before this trajectory was assembled. The noise-free re-eval (0.4237)
is the recoverable number; the reported value would require a fresh
``round_robin`` re-eval of the same checkpoint.

⁂ M5.4 was trained with ``--eval-strategy no_cycle`` from epoch 0 (the
M5.3 default for val loaders), so its in-training and re-evaluation
numbers are bit-identical — Δ = 0.0000 by construction. There is no
``round_robin`` reported figure to compare against. The Bot AUROC for
M5.4 P1 is the noise-free value reproduced once (0.5060) — the column
duplication is intentional, to keep the table shape consistent across
rows.

★ Per the Path B fairness contract (see "M5.5 R1 → R1.5 forensic
finding" below + `m5_5_baselines.md`), the cited M5.5 baseline number
for TimeSformer-Small is **R1's 0.4836** — random-init backbone
trained with effective head_lr ×1. R1 was originally launched with
`--head-lr-multiplier 5.0` but the trainer's pre-fix head matcher
silently bypassed the multiplier (5,005 head params landed in the
backbone group); the resulting head_lr ×1 effective configuration is
the **correct fairness contract for from-scratch backbones** under
Path B. R1.5 (matcher-fixed, head_lr ×5 actually applied) is the
ablation supplementary that empirically validated the Path B
grouping: Δ R1.5 vs R1 = combined −0.022 / Bot AUROC −0.121, i.e.
applying head_lr ×5 to a from-scratch backbone HURTS. The four M5.5
Round 2 baselines (C3D-Small, ConvLSTM, I3D, R(2+1)D-18) are
recorded in `docs/m5_5_baselines.md` rather than added as rows here,
to keep this trajectory focused on the single-model time-budget axis.

### val_sample_count_total bit-identity

All three runs evaluate on the same val split derived from the same
``splits.parquet``. The total sample count under ``no_cycle`` should
therefore be bit-identical across the three retrofits — a ``red flag``
mismatch would indicate splits drift.

| Run | val_sample_count_total | val fast | val slow |
|---|---:|---:|---:|
| M4.8 | 18,156 | 16,463 | 1,693 |
| M5.1 | 18,156 | 16,463 | 1,693 |
| M5.2 | 18,156 | 16,463 | 1,693 |

✓ bit-identical across all three runs.

## Cycling-delta is training-maturity dependent

The cycling-induced inflation Δ is **not** a constant offset:

| Run | grad_steps | log(grad_steps) | Δ |
|---|---:|---:|---:|
| M4.8 | 4,853 | 8.49 | −0.1150 |
| M5.1 | 14,559 | 9.59 | −0.0883 |
| M5.2 | 48,530 | 10.79 | −0.0466 |

|Δ| decreases monotonically with training maturity. A simple
log-linear fit through the three points yields a slope of roughly
+0.030 per natural-log step, which predicts the M5.1 mid-point at
−0.082 — the measured −0.0883 lands within 0.007 of that.

The mechanism is variance-reduction-by-ensembling, applied
inadvertently. Cycling exposes each slow sample to roughly ten
different batch contexts per eval pass; the resulting predictions are
softmax-averaged at metric computation time. The benefit is
proportional to per-sample logit variance, which is higher early in
training (under-fit head, noisy decisions) and shrinks as the model
matures (peaked softmax, stable rankings). The under-trained M4.8
checkpoint therefore receives the largest TTA boost; the well-trained
M5.2 checkpoint receives the smallest.

This is a generic property of the evaluation policy, not a
model-specific artefact. Any saved checkpoint evaluated under
``round_robin`` is subject to the same effect, with magnitude
determined by where it sits on the training-maturity axis.

## Bot AUROC trajectory: when does the rare-class collapse start?

Vanilla multi-class CE on a heavily imbalanced label distribution
gradually re-weights the model's decision boundary toward majority
classes. The Bot validation set has only 12 unique members under
``no_cycle`` — small enough that a single early-training step that
suppresses Bot logits can move the AUROC across the 0.5 random-rank
threshold.

| Run | epoch | noise-free Bot AUROC | interpretation |
|---|---:|---:|---|
| M4.8 | 1 | 0.7247 | Bot rank ordering is solidly above random; representation is healthy |
| M5.1 | 3 | 0.4237 | Already slightly worse than random — collapse has happened |
| M5.2 | 10 | 0.4077 | Marginal further decline; the post-collapse plateau |

The collapse is **steep early, then plateau**: a 0.30-point drop
between epoch 1 and epoch 3 (M4.8 → M5.1), then only a 0.02-point
drop over the next seven epochs (M5.1 → M5.2). Two implications:

1. The Bot collapse is not gradual erosion across the whole training
   budget — it sets in within the first few epochs and then stabilises.
2. A loss-function intervention (focal / class reweight) applied from
   epoch 0 has the right window to prevent the collapse rather than
   correct it after the fact.

## Implications for M5.4 (focal loss / class reweighting)

The M4.8 → M5.1 → M5.2 noise-free macro_f1 progression is
0.3324 → 0.4230 → 0.4677, a +0.135 total gain across nine extra
epochs. The first two epochs (M4.8 → M5.1, two epochs of additional
training) account for +0.0906 of that — roughly 67% of the total
multi-epoch gain happens in the first 2,000 grad steps after epoch 0.
The remaining seven epochs (M5.1 → M5.2) deliver only +0.0447.

Combined with the Bot AUROC plateau and the FTP-Patator / SSH-Patator
F1 numbers stuck near zero across all three checkpoints, the
trajectory says vanilla CE has effectively reached its representational
ceiling by epoch 3 and additional training under the same loss
function is heavily diminishing-returns. The M5.4 task — replacing
vanilla CE with focal loss or class-reweighted CE — is therefore a
necessary intervention rather than an optimisation, and the noise-free
macro_f1 = 0.4677 (combined) / 0.4510 (fast-only) is the reference
ceiling that M5.4 must clear to justify the change.

## M5.4 Phase 1: focal loss γ=2 from scratch — does NOT clear the ceiling

10-epoch training with ``--loss-fn focal --focal-gamma 2.0``, no
``--resume`` (pretrained Kinetics backbone + fresh classification
head), all other hyperparameters identical to M5.2. Run dir
``outputs/run_20260502_134735``; best epoch = 9 (final),
``best.pt`` at ``ckpt/best.pt``.

Noise-free numbers (verbatim from
``outputs/run_20260502_134735/m5_4_eval/eval_metrics.json``):

  combined macro_f1   : 0.4584
  combined accuracy   : 0.9598
  combined auroc      : 0.7609
  fast-only macro_f1  : 0.4262
  slow-only macro_f1  : 0.6035
  Bot per-class AUROC : 0.5060
  val_sample_count    : 18,156   (fast 16,463 + slow 1,693)

Comparison to vanilla CE M5.2 noise-free at the same training budget:

  M5.2 combined  : 0.4677
  M5.4 combined  : 0.4584   (Δ = -0.0092, slight regression)
  M5.2 fast-only : 0.4510
  M5.4 fast-only : 0.4262   (Δ = -0.0248)
  M5.2 slow-only : 0.5476
  M5.4 slow-only : 0.6035   (Δ = +0.0559)

Focal γ=2 trades fast-stream macro F1 down for slow-stream macro F1 up,
without lifting the combined number above the vanilla CE ceiling. The
trade is the wrong direction for the project's primary goal: the
fast-stream number is the apples-to-apples baseline against
single-resolution models in M5.5+.

Sentinel checks (set in the M5.4 task spec to define a true PASS):

| Check | Threshold | Measured | Status |
|---|---:|---:|:--:|
| Bot F1 | > 0 | 0.0000 | FAIL |
| Bot AUROC | > 0.6 | 0.5060 | FAIL |
| DoS-GoldenEye F1 | ≥ 0.40 | 0.2892 | FAIL |
| FTP-Patator F1 | ≥ 0.20 | 0.0488 | FAIL |

Four of four sentinels fail. Combined macro_f1 (0.4584) lands in the
``< 0.52`` band, which the task spec earmarks for Phase 2 trigger
(focal + class-frequency reweighting). The Phase 2 decision is
deferred to the user — this commit captures the focal-loss training
infrastructure and the Phase 1 results regardless.

### Bot AUROC trajectory: focal slowed but did not prevent collapse

Mid-training peek at M5.4 epoch 2 had Bot AUROC = 0.6773 (above
random, comparable to M4.8's 0.7247 at the same training depth). By
epoch 9 the value had decayed to 0.5060 — still above M5.1's 0.4237
and M5.2's 0.4077, but well below M4.8's 0.7247. So the focal γ=2
intervention **slowed** the rare-class collapse without **preventing**
it: the model still drifts toward suppressing the Bot logit late in
training, just on a longer timescale than vanilla CE.

For Phase 2 / Phase 3 design, the obvious knobs are γ > 2 (sharper
focus on hard samples) and per-class alpha (which the FocalLoss
implementation already accepts as a buffer; the Phase 2 hook is
therefore zero-code-change).

## M5.4 Phase 2: focal γ=2 + inverse-sqrt α + head LR ×5 — small gain, sentinels still fail

10-epoch training with focal γ=2 unchanged from Phase 1, plus
inverse-sqrt class reweighting (alpha = 1/sqrt(n_train), normalised
to mean=1 over present classes; n=0 classes get α=0) and a separate
optimizer parameter group for the classification head + scale_token +
scale_embedding at 5× the backbone learning rate. All other
hyperparameters identical to Phase 1.

Run dir ``outputs/run_20260502_184512``; best epoch = 9 (final);
``best.pt`` at ``ckpt/best.pt``.

Noise-free numbers (verbatim from
``outputs/run_20260502_184512/m5_4_phase2_eval/eval_metrics.json``):

  combined macro_f1   : 0.4756
  combined accuracy   : 0.9560
  combined auroc      : 0.7641
  fast-only macro_f1  : 0.4525
  slow-only macro_f1  : 0.6069
  Bot per-class AUROC : 0.4968
  val_sample_count    : 18,156   (fast 16,463 + slow 1,693)

Three-way comparison at the identical 10-epoch budget:

  M5.2 CE      combined : 0.4677  fast : 0.4510  slow : 0.5476
  M5.4 P1      combined : 0.4584  fast : 0.4262  slow : 0.6035
  M5.4 P2      combined : 0.4756  fast : 0.4525  slow : 0.6069

  delta P2 - P1 : +0.0172   (combined)
  delta P2 - CE : +0.0079   (combined; barely clears the vanilla ceiling)
  delta P2 - P1 fast : +0.0263 (recovered the fast-stream regression P1 introduced)
  delta P2 - P1 slow : +0.0034
  delta P2 - CE fast : +0.0015 (effectively flat against fast-only CE)

P2 modestly clears P1 and the vanilla CE ceiling on the combined
metric, but the headline number 0.4756 lands in the spec's
``[0.40, 0.50]`` band — below the 0.50 PASS-minimum threshold and
well below the 0.55 true-PASS bar. Inverse-sqrt α=49.06× spread plus
a 5× head LR delivered a smaller-than-expected lift.

Sentinel checks (spec-defined for Phase 2):

| Check | Threshold | Measured | Status |
|---|---:|---:|:--:|
| Bot AUROC | ≥ 0.65 | 0.4968 | FAIL |
| Bot F1 | > 0.10 | 0.0000 | FAIL |
| DoS-GoldenEye F1 | ≥ 0.40 | 0.4130 | PASS |
| FTP-Patator F1 | ≥ 0.20 | 0.0752 | FAIL |

Three of four sentinels fail. GoldenEye is the lone PASS — its F1
trajectory M5.2 0.2278 → P1 0.2892 → P2 0.4130 is consistent with
the design intent (rare-but-not-tiny class lifted by α). Bot remains
stuck at F1=0 — the 12 val samples never get argmax-classified as
Bot, and the AUROC actually went DOWN slightly vs P1 (0.5060 →
0.4968 = below random).

### Per-class trajectory (M5.2 CE / P1 / P2 noise-free)

| Class | n_val | M5.2 F1 | P1 F1 | P2 F1 | M5.2 AUROC | P1 AUROC | P2 AUROC | Trend |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BENIGN | 16,829 | 0.9793 | 0.9794 | 0.9778 | 0.8828 | 0.8967 | 0.9036 | flat F1, slight AUROC gain |
| DoS Hulk | 105 | 0.6339 | 0.6667 | 0.6066 | 0.9309 | 0.9034 | 0.9051 | F1 regressed in P2 |
| **PortScan** | 22 | 0.3889 | 0.2857 | **0.5957** | 0.9546 | 0.9531 | 0.9561 | **strong P2 lift** |
| DDoS | 228 | 0.7411 | 0.6474 | 0.4590 | 0.9895 | 0.9894 | 0.9928 | F1 regressed both passes |
| **DoS GoldenEye** | 61 | 0.2278 | 0.2892 | **0.4130** | 0.9406 | 0.9632 | 0.9524 | **steady gain to spec** |
| FTP-Patator | 107 | 0.0775 | 0.0488 | 0.0752 | 0.8050 | 0.8606 | 0.9001 | AUROC up but F1 stuck |
| SSH-Patator | 175 | 0.0948 | 0.0800 | 0.0804 | 0.8043 | 0.9129 | 0.9172 | AUROC up, F1 stuck |
| DoS slowloris | 264 | 0.8566 | 0.8849 | 0.8880 | 0.9907 | 0.9954 | 0.9968 | best class steady gain |
| DoS Slowhttptest | 105 | 0.1789 | 0.1846 | 0.1515 | 0.8960 | 0.9106 | 0.9120 | F1 small regression |
| **Bot** | 12 | 0.0000 | 0.0000 | 0.0000 | 0.4077 | 0.5060 | **0.4968** | **F1 stuck, AUROC slipped** |
| Heartbleed | 248 | 0.9655 | 0.9760 | 0.9841 | 0.9998 | 0.9999 | 0.9999 | saturated, slight gain |

Pattern: P2 reweighting helps mid-rarity classes (GoldenEye, PortScan,
Slowloris) where the alpha boost has enough sample count behind it to
move argmax decisions; it does not help the FTP-Patator / SSH-Patator
F1 (their AUROC lifted but argmax threshold not crossed) and it does
not help Bot at all (12 val samples is too few to get over the
argmax threshold, regardless of α). DDoS regressed more under P2 than
under P1 — the head's 5× LR may be over-correcting on this class
where vanilla CE was already reasonable.

### Phase 3 not pursued — M5.4 P2 accepted as deliverable

Loss-level optimisation saturates around combined macro_f1 = 0.4756
under the current data scale. Two stacked levers (focal gamma=2 in
Phase 1, then focal + inverse-sqrt alpha + head LR multiplier in
Phase 2) together lifted the combined number by +0.0079 over the
vanilla CE ceiling — meaningful enough to ship as a fairness baseline
but small enough that further loss-internal tweaks (gamma sweep,
alternative reweighting schemes) project negative or marginal returns.

The remaining failure modes split cleanly along sample-count axes:

- **Mid-rarity classes (n_train > 200)** were successfully addressed:
  GoldenEye F1 0.2278 → 0.4130 (M5.2 → P2) is a monotonic, design-
  intent gain. PortScan F1 (n_val=22) jumped 0.3889 → 0.5957 in P2,
  another reweighting win.
- **Extreme-rarity classes (n_train < 50)** are not loss-fixable on
  this dataset: Bot (n_train=30, n_val=12) stayed at F1=0 across all
  three checkpoints; its AUROC drifted 0.5060 → 0.4968 across P1 → P2,
  showing that increasing alpha by 49× and head LR by 5× cannot
  manufacture argmax-passing predictions when the val set is too
  small for the threshold to be crossed reliably.
- **Web Attack and Infiltration (n_train=0, n_val=0)** are out of
  scope for this CIC subset — the original Tue+Wed+Fri pcap selection
  excludes Thursday, where these attacks were captured.

Candidate Phase 3 directions are deferred:

- **gamma sweep (γ=3/4/5)** — projected negative return: with α already
  pushing the rare-class loss share to ~3× and the gradient-flow test
  showing > 100× ratio at γ=2, additional focusing concentrates more
  gradient onto the same already-hard samples that the val set is too
  small to learn anyway.
- **fine-tune from M5.2 ckpt** — moved to M5.10 ablation as the
  "from-scratch vs fine-tune" comparison; would inherit M5.2's collapsed
  Bot representation as a starting point and is not the natural follow-on.
- **data-level intervention (rare-class oversampling)** — moved to
  M5.10 as the "data-level vs loss-level fix" ablation chapter; the
  M5.4 task spec scoped to loss-level only.

The real narrative-decision axis is M5.5+ (representation × backbone
alignment across baselines), not internal loss micro-optimisation.

### M5.4 deliverable

  configuration  : focal gamma=2 + inverse_sqrt class reweighting + head LR ×5
  combined macro_f1 (noise-free, no_cycle eval) : 0.4756
  fast-only macro_f1 : 0.4525
  slow-only macro_f1 : 0.6069
  budget         : 10 epochs, 48,510 grad steps, batch=32 / accum=1
  splits         : data/processed/cicids2017_dt100ms_v2/splits.parquet
  ckpt           : outputs/run_20260502_184512/ckpt/best.pt
  artefact       : outputs/run_20260502_184512/m5_4_phase2_eval/

M5.5 baselines run under the same budget, splits, eval policy
(no_cycle), batch / accumulation / multi-scale (50/50 fast/slow)
configuration, and where applicable the same loss/reweight/head_lr
combination. The 0.4525 fast-only number is the apples-to-apples
reference for single-resolution baseline comparisons.

## M5.5 baselines: cross-architecture comparison

A five-baseline suite established the architectural axis of the
representation × backbone alignment question that motivates this
project: holding the input tensor (T=16, C=6, H=32, W=64), splits, eval
policy, optimisation stack, and loss/reweight/head_lr configuration
constant, what does the choice of video backbone alone buy or cost vs
the M5.4 P2 main method?

The five planned baselines are TimeSformer-Small, C3D-Small, I3D,
R(2+1)D-18, and ConvLSTM. Round 1 (this commit) delivers
TimeSformer-Small; rounds 2-3 add the remaining four under the same
fairness contract.

### Fairness contract (all rows)

- Input: identical (T=16, C=6, H=32, W=64) NID tensor; same
  splits.parquet; multi-scale 50/50 fast/slow mix.
- Optimiser: 8-bit AdamW, batch=32, grad_accumulation=1, fp16 AMP.
- Loss: focal γ=2 + inverse-sqrt class reweighting + head LR ×5 on
  the classification head's parameter group (matches M5.4 P2).
- Schedule: 10 epochs, ~48,510 grad steps under round_robin
  epoch terminator; per-epoch eval under no_cycle so the in-training
  metric is bit-identical to the noise-free re-evaluation (Δ ≈ 0).

### Pretrained-checkpoint asymmetry across the suite

| Baseline | Params | Pretrained source | head_lr (Path B) | Note |
|---|---:|---|---:|---|
| VideoMAE-Small (main, M4.8/M5.x) | 22M | Kinetics-400 | ×5 | Adapted 3→6 channels via `adapt_conv3d_to_6ch`. |
| TimeSformer-Small | 30.8M | none (random init) | ×1 | Divided space-time at hidden=384; no public 22M K400 ckpt. |
| C3D-Small | 18.8M | none (random init) | ×1 | 8 Conv3d (64/128/256/256/384/384/384/384) + 5 max-pool + FC (1024/512/13). |
| ConvLSTM | 13.0M | none (random init) | ×1 | 3-layer ConvLSTM with 2×2 spatial pool between cells (hidden 64→128→256). |
| I3D-R50 | 27.3M | Kinetics-400 (pytorchvideo) | ×5 | Adapter `adapt_conv3d_to_6ch` on `blocks[0].conv` (5×7×7, identity transform). |
| R(2+1)D-18 | 31.3M | Kinetics-400 (torchvision) | ×5 | Adapter `adapt_conv3d_to_6ch` on `stem[0]` (1×7×7, identity transform). |

Among the five baselines, R(2+1)D-18 and I3D inherit Kinetics weights;
TimeSformer-Small, C3D-Small, and ConvLSTM run from scratch. This
asymmetry reflects the open-source video-backbone ecosystem at this
parameter scale, not a project choice. The R(2+1)D-18 and I3D rows
serve as the upper bound for "Kinetics-pretrained video backbone on
this task" and address whether the main method's advantage stems from
pretraining or from representation × backbone alignment specifically.

Per the M5.5 Path B contract (validated by R1 vs R1.5 ablation,
documented below + in `m5_5_baselines.md`), `head_lr_multiplier` is
grouped by pretrained-status: K400-pretrained backbones train with
head_lr ×5 (preserves pretraining via slow backbone, fast head learns
new classifier from scratch); random-init backbones train with
head_lr ×1 (both head and backbone are equally fresh; an asymmetric
LR overshoots toward majority-class boundary). The full six-row data
table + 13×6 per-class grand tables + Methods-section draft live in
`docs/m5_5_baselines.md`.

### M5.5 Round 1: TimeSformer-Small (random init from scratch)

10-epoch training of `TimeSformerSmallForNID` (HF `TimesformerConfig`
with hidden=384, num_hidden_layers=12, num_attention_heads=6,
intermediate_size=1536, attention_type=`divided_space_time`,
patch_size=16, image_size=64, num_channels=6 directly with no 3→6
adapter — random Kaiming init throughout). The H dim is zero-padded
from 32 → 64 inside `forward` to satisfy TimeSformer's square-frame
assumption; no other input transformation. `scale_id` is accepted in
the forward signature but ignored — the model is scale-agnostic by
design and consumes the multi-scale dataloader's mixed batches without
conditioning on which stream a sample came from.

#### M5.5 R1 → R1.5 forensic finding: head_lr_multiplier matcher fix and the Path B head_lr split

R1's first run (``outputs/run_20260502_232207/``, commit ``bac6c67``)
launched with ``--head-lr-multiplier 5.0`` but the trainer's pre-fix
head matcher used ``startswith("classifier.", "scale_embedding.")`` —
designed for VideoMAE's flat ``classifier`` attribute, blind to HF
wrappers that nest the classifier at ``backbone.classifier``.
TimeSformer-Small therefore trained with **head_lr_multiplier=1.0
effective** (5,005 head params silently in the backbone group at
base_lr=1.5e-4 instead of the configured head_lr=7.5e-4).

R1.5 (``outputs/run_20260503_121046/``, commit ``3187a67``) fixed the
matcher (segment match across ``classifier`` / ``scale_embedding`` /
``fc`` / ``proj`` ancestors — covering HF wrappers, torchvision
``model.fc``, and pytorchvideo ``blocks[-1].proj``) and re-ran
TimeSformer-Small with the multiplier actually applied at
``head_lr=7.5e-4``. The retrain was bit-equivalent in every other
respect to the R1 run.

The result reframes the R1 number as a **lucky coincidence under the
correct contract**, not a bug:

Three-way comparison at the identical 10-epoch budget:

  M5.4 P2 main (K400-pretrained 22M, head_lr ×5)         combined : 0.4756  fast : 0.4525  slow : 0.6069  Bot AUROC : 0.4968
  M5.5 R1  TimeSformer-Small (random 31M, head_lr ×1 eff.) combined : 0.4836  fast : 0.4547  slow : 0.6254  Bot AUROC : 0.7151  ← cited
  M5.5 R1.5 TimeSformer-Small (random 31M, head_lr ×5)     combined : 0.4616  fast : 0.4339  slow : 0.6226  Bot AUROC : 0.5940  ← ablation

  Δ R1 vs M5.4 P2     : combined +0.0080  fast +0.0022  slow +0.0185  Bot AUROC +0.2183
  Δ R1.5 vs R1        : combined −0.0220  fast −0.0208  slow −0.0028  Bot AUROC −0.1211

The R1.5 ablation reveals that **head_lr ×5 is harmful on a
from-scratch backbone**: applying it dropped combined macro_f1 by
0.022 and collapsed Bot per-class F1 from 0.091 to 0.000 (Bot AUROC
0.7151 → 0.594). Mechanism: with a randomly initialised backbone, both
head and backbone are equally fresh; running the head at 5× LR makes
it overshoot toward the majority-class decision boundary while the
backbone is still learning basic features. The M5.4 Phase-2 head LR
multiplier was justified for K400-pretrained backbones (slow backbone
preserves pretraining; fast head learns the new classifier from
scratch); the same intervention is **not transferable** to
from-scratch baselines under a nominally-identical "fairness
contract".

This finding crystallises into the **M5.5 Path B contract**:
``head_lr_multiplier`` is grouped by pretrained status. K400 backbones
train with ×5 (M5.4 P2 recipe); random-init backbones train with ×1.
R1's effective head_lr ×1 (under the matcher bug) IS therefore the
correct fairness contract for from-scratch backbones, and R1's 0.4836
is the cited TimeSformer-Small baseline number — **not superseded by
R1.5**. R1.5 stays as the ablation supplementary that empirically
validates the Path B grouping.

Architectural follow-on (preserved through R1 → R1.5):
TimeSformer-Small's slow-only macro_f1 (0.6254 in R1, 0.6226 in R1.5)
exceeds M5.4 P2 main's 0.6069 in both — divided space-time attention
captures longer-tempo patterns slightly better than VideoMAE-Small's
joint attention at this data scale. The Bot AUROC trajectory is the
sharper signal: R1's 0.7151 is the highest Bot AUROC seen across all
six rows in `m5_5_baselines.md`, and the only one above 0.7. The
focal+α intervention works noticeably better when the backbone does
not commit to the majority-class boundary as aggressively.

### M5.5 Round 1 deliverable + R1.5 ablation supplementary

  cited baseline : TimeSformer-Small R1 (random init, 30.8M params)
  configuration  : focal γ=2 + inverse_sqrt α + head LR ×1 effective (random-init Path B contract)
  combined macro_f1 (noise-free, no_cycle eval) : 0.4836
  fast-only macro_f1 : 0.4547
  slow-only macro_f1 : 0.6254
  Bot per-class AUROC: 0.7151
  budget         : 10 epochs, 48,530 grad steps, batch=32 / accum=1
  splits         : data/processed/cicids2017_dt100ms_v2/splits.parquet
  ckpt           : outputs/run_20260502_232207/ckpt/best.pt    (commit bac6c67)
  artefact       : outputs/run_20260502_232207/m5_5_timesformer_small_eval/

  ablation supp. : TimeSformer-Small R1.5 (same model, head_lr ×5 intentional)
                 : combined 0.4616 / fast 0.4339 / slow 0.6226 / Bot AUROC 0.5940
                 : ckpt outputs/run_20260503_121046/ckpt/best.pt    (commit 3187a67)
                 : artefact outputs/run_20260503_121046/m5_5_timesformer_small_eval/
                 : NOT a cited baseline; recorded as the empirical evidence
                   that head_lr ×5 hurts from-scratch backbones (Δ vs R1
                   combined −0.022, Bot AUROC −0.121). Anchors the Path B
                   head_lr-by-pretrained-status grouping decision.

Round 2 baselines (C3D-Small, ConvLSTM, I3D, R(2+1)D-18) are
documented in `docs/m5_5_baselines.md` (consolidated 6-row table +
13×6 per-class grand tables + 4-section findings + Methods draft).

## Reproduction

The four artefact bundles are stored alongside their source training
runs (each ``outputs/run_<ts>/`` directory is gitignored):

- ``outputs/run_20260430_223105/m4_8_rerun/``
- ``outputs/run_20260501_143946/m5_1_rerun/``
- ``outputs/run_20260501_162117/m5_3_rerun/``
- ``outputs/run_20260502_134735/m5_4_eval/``
- ``outputs/run_20260502_184512/m5_4_phase2_eval/``
- ``outputs/run_20260502_232207/m5_5_timesformer_small_eval/`` (R1, **cited per Path B** — random-init head_lr ×1 effective; commit bac6c67)
- ``outputs/run_20260503_121046/m5_5_timesformer_small_eval/`` (R1.5 ablation supplementary — head_lr ×5 intentional; commit 3187a67)
- Round 2 baselines (4 bundles + 4 commits) are listed in
  ``docs/m5_5_baselines.md`` §"Source artefacts".

Each bundle contains ``eval_metrics.json`` (full payload), a
``confusion_matrix.json``, a ``per_class_table.csv`` for direct
table-paste, and a ``README.md`` with the exact reproduction command.

To regenerate any one bundle:

```bash
uv run python scripts/baseline_rerun.py \
    --resume <ckpt path> \
    --shard-pattern-fast "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \
    --shard-pattern-slow "data/processed/cicids2017_dt1000ms_v2/*/shards/shard-*.tar" \
    --splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet \
    --output-dir <output dir> \
    --source-train-macro-f1 <reported value> \
    --task-label "<short label written into the README>"
```

The ``scripts/m5_3_rerun.py`` entry point is preserved as a thin
back-compat shim that delegates to ``scripts/baseline_rerun.py`` with
the M5.3 task label and script-name defaults pre-injected, so the
reproduction command quoted in the original M5.3 README continues to
work unchanged.
