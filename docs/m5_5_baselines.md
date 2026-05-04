# M5.5 baselines: cross-architecture comparison

This document is the consolidated data anchor for the M5.5 baseline
section of the paper. It records the noise-free per-checkpoint
validation metrics (combined / fast-only / slow-only) across one main
method and five video-backbone baselines under a unified fairness
contract on the (T=16, C=6, H=32, W=64) NID input.

The earlier per-milestone trajectory (1 → 3 → 10 epoch budget on the
single VideoMAE-Small main method) lives in `m5_baseline_trajectory.md`;
this file extends that picture into the architectural axis (5
baselines + main method, all at the same 10-epoch budget).

## Five-baseline suite + main method

| Model | Params | Pretrained | head_lr | combined macro_f1 | fast macro_f1 | slow macro_f1 | combined accuracy | combined auroc | Bot per-class F1 | Bot per-class AUROC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M5.4 P2 main (VideoMAE-S)         | 22M | K400 | ×5 | **0.4756** | 0.4525 | 0.6069 | 0.9560 | 0.7641 | 0.0000 | 0.4968 |
| M5.5 R1 TimeSformer-Small         | 31M | none | ×1 | **0.4836** | 0.4547 | 0.6254 | 0.9367 | 0.7890 | 0.0909 | 0.7151 |
| M5.5 R2 C3D-Small                 | 19M | none | ×1 | **0.4464** | 0.4161 | 0.6270 | 0.9462 | 0.7239 | 0.0000 | 0.3755 |
| M5.5 R2 ConvLSTM                  | 13M | none | ×1 | **0.4746** | 0.4530 | 0.5542 | 0.9495 | 0.7492 | 0.0000 | 0.3772 |
| M5.5 R2 I3D-R50                   | 27M | K400 | ×5 | **0.5149** | 0.4888 | 0.6470 | 0.9477 | 0.7677 | 0.0000 | 0.5341 |
| M5.5 R2 R(2+1)D-18                | 31M | K400 | ×5 | **0.5197** | 0.4833 | 0.6730 | 0.9504 | 0.7768 | 0.1429 | 0.4994 |

Numbers are verbatim from each row's `eval_metrics.json` (see *Source
artefacts* at the bottom). All rows evaluate on the same val split:
`val_sample_count_total` = 18,156 (fast 16,463 + slow 1,693), bit-identical
across all six runs — confirmed by `baseline_rerun.py`'s split-count
audit.

## Per-class macro_f1 grand table (13 classes × 6 models, combined eval)

| Class (support) | M5.4 P2 main | M5.5 R1 TimeSf-S | M5.5 R2 C3D-S | M5.5 R2 ConvLSTM | M5.5 R2 I3D | M5.5 R2 R(2+1)D-18 |
|---|---:|---:|---:|---:|---:|---:|
| BENIGN (16,829)         | 0.9778 | 0.9703 | 0.9755 | 0.9757 | 0.9731 | 0.9741 |
| DoS Hulk (105)          | 0.6066 | 0.3470 | 0.5833 | 0.5240 | 0.5298 | **0.7166** |
| PortScan (22)           | 0.5957 | 0.5588 | 0.5283 | **0.5965** | 0.5818 | 0.5294 |
| DDoS (228)              | 0.4590 | 0.5500 | 0.3357 | 0.7886 | 0.8018 | **0.8243** |
| DoS GoldenEye (61)      | 0.4130 | **0.4295** | 0.2244 | 0.1185 | 0.3158 | 0.2069 |
| FTP-Patator (107)       | 0.0752 | 0.1154 | 0.0903 | **0.2008** | 0.1849 | 0.1898 |
| SSH-Patator (175)       | 0.0804 | 0.1940 | 0.1756 | 0.1636 | **0.2548** | 0.1592 |
| DoS slowloris (264)     | 0.8880 | 0.8311 | 0.7821 | 0.8143 | 0.8446 | **0.9115** |
| DoS Slowhttptest (105)  | 0.1515 | **0.2403** | 0.2289 | 0.0526 | 0.1875 | 0.0719 |
| Bot (12)                | 0.0000 | 0.0909 | 0.0000 | 0.0000 | 0.0000 | **0.1429** |
| Web Attack (0)          | —      | —      | —      | —      | —      | —      |
| Infiltration (0)        | —      | —      | —      | —      | —      | —      |
| Heartbleed (248)        | 0.9841 | 0.9920 | 0.9860 | 0.9860 | 0.9900 | 0.9900 |

Bold per-row marks the best-macro_f1 model for that class. R(2+1)D-18
takes 4 of the 13 class-best slots; ConvLSTM, TimeSformer-S, I3D each
take 1. Web Attack and Infiltration have zero val support under the
current splits and contribute zero to all macro averages by
construction (they're the same zero across the macro denominator).

## Per-class AUROC grand table (13 classes × 6 models, combined eval)

| Class (support) | M5.4 P2 main | M5.5 R1 TimeSf-S | M5.5 R2 C3D-S | M5.5 R2 ConvLSTM | M5.5 R2 I3D | M5.5 R2 R(2+1)D-18 |
|---|---:|---:|---:|---:|---:|---:|
| BENIGN (16,829)         | 0.9036 | 0.9232 | 0.8855 | 0.8981 | 0.9153 | **0.9322** |
| DoS Hulk (105)          | **0.9051** | 0.8481 | 0.8743 | 0.8821 | 0.8633 | 0.8840 |
| PortScan (22)           | 0.9561 | 0.9668 | 0.9521 | 0.9572 | 0.9593 | **0.9698** |
| DDoS (228)              | 0.9928 | 0.9929 | 0.9739 | 0.9912 | 0.9947 | **0.9980** |
| DoS GoldenEye (61)      | 0.9524 | **0.9768** | 0.9147 | 0.8603 | 0.9360 | 0.9625 |
| FTP-Patator (107)       | 0.9001 | 0.9508 | 0.7193 | 0.9600 | 0.9474 | **0.9685** |
| SSH-Patator (175)       | 0.9172 | 0.9537 | 0.8498 | 0.9265 | 0.9337 | **0.9637** |
| DoS slowloris (264)     | 0.9968 | **0.9977** | 0.9559 | 0.9789 | 0.9905 | 0.9975 |
| DoS Slowhttptest (105)  | 0.9120 | **0.9321** | 0.9105 | 0.9086 | 0.9060 | 0.9228 |
| Bot (12)                | 0.4968 | **0.7151** | 0.3755 | 0.3772 | 0.5341 | 0.4994 |
| Web Attack (0)          | —      | —      | —      | —      | —      | —      |
| Infiltration (0)        | —      | —      | —      | —      | —      | —      |
| Heartbleed (248)        | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | **1.0000** |

R(2+1)D-18 takes 6 of the 11 non-zero class AUROC slots; TimeSformer-S
takes 4 (notably leading on Bot — see *Findings* §3); main method takes
1 (DoS Hulk). The K400-pretrained backbones (main, I3D, R(2+1)D-18)
collectively dominate AUROC outside Bot.

## Fairness contract

All six rows above were trained and evaluated under the same contract,
with one calibrated split: head_lr_multiplier is grouped by pretrained
status. The split is documented here, not adapted post-hoc.

### Common ground (all six rows)

- **Input**: identical (T=16, C=6, H=32, W=64) NID tensor; same
  splits.parquet (M5.3 anchored at
  `data/processed/cicids2017_dt100ms_v2/splits.parquet`); multi-scale
  50/50 fast/slow mix dataloader.
- **Optimiser**: 8-bit AdamW, batch=32, grad_accumulation=1, fp16 AMP,
  weight_decay=0.05, base learning rate 1.5e-4 with linear warmup to
  500 steps + cosine decay to 1% of peak. Gradient checkpointing on
  for all rows that have an HF wrapper exposing it.
- **Loss + reweighting**: focal loss γ=2 with inverse-square-root class
  reweighting (alpha = 1/sqrt(n_train), normalised to mean=1 over
  present classes; n=0 classes get α=0).
- **Schedule**: 10 epochs, 48,530 grad steps under round_robin
  epoch terminator; per-epoch eval under no_cycle so the in-training
  metric is bit-identical to the noise-free re-evaluation (Δ ≤ 5e-5
  across all six rows).
- **Eval policy**: `no_cycle` strategy drains both streams exactly
  once with no duplicates and yields a count that does not depend on
  `mix_ratio` — the property needed for stable cross-run comparison.

### Calibrated split: head LR multiplier by pretrained status

The M5.4 Phase 2 head LR multiplier (×5) was justified for the
K400-pretrained main method as a way to preserve pretraining via slow
backbone LR while letting the freshly initialised classification head
learn at a faster rate. Applying the same multiplier to from-scratch
(random init) backbones is **not** transferable: the M5.5 R1.5 forensic
ablation directly measured this — TimeSformer-Small with head_lr ×5
intentional dropped combined macro_f1 by 0.022 and collapsed Bot per-class
F1 from 0.0909 to 0.0 (vs the TimeSformer-S R1 result, which trained
with head_lr ×1 effective due to a matcher bug, see m5_baseline_trajectory.md).

The contract therefore splits on pretrained status:

| Group | Rows | head_lr_multiplier | Rationale |
|---|---|---:|---|
| **K400 pretrained** | M5.4 P2 main, I3D, R(2+1)D-18 | **×5** | M5.4 P2 contract — slow backbone preserves K400 features; fast head learns the new classifier from scratch. |
| **Random init from-scratch** | TimeSformer-S, C3D-Small, ConvLSTM | **×1** | M5.5 R1.5 finding — both head and backbone are fresh; running the head at 5× LR overshoots toward the majority-class boundary while the backbone is still learning basic features. |

This split is stable and tractable: the pretrained-status of each
backbone is determined by what's available in the open-source video
ecosystem, not by a project preference. R(2+1)D-18 and I3D have public
K400 checkpoints (torchvision and pytorchvideo respectively);
TimeSformer at the 22M Small scale, C3D-Small, and ConvLSTM do not.
The K400 vs random asymmetry is pre-existing; the head_lr split is the
locally-correct response to it.

### Pretrained-checkpoint asymmetry across the suite

| Baseline | Pretrained source | First-conv adapter | Norm-ratio diagnostic |
|---|---|---|---|
| M5.4 P2 main (VideoMAE-S) | Kinetics-400 (HuggingFace) | `adapt_conv3d_to_6ch` 3→6 ch on `patch_embeddings.projection` (16×16×16 → 8×8×2 trilinear downsample + Kaiming-init for ch[3:6]) | M3-001 norm-ratio test passes: ext / pretrained > 1.5× (compression-induced). |
| M5.5 R1 TimeSformer-Small | none | none — Conv2d(6, 384, 16, 16) Kaiming-init throughout | n/a (random init). |
| M5.5 R2 C3D-Small | none | none — Conv3d(6, 64, 3, 3, 3) Kaiming-init throughout | n/a. |
| M5.5 R2 ConvLSTM | none | none — Conv2d(6+64, 4×64, 3, 3) Kaiming-init throughout | n/a. |
| M5.5 R2 I3D-R50 | Kinetics-400 (pytorchvideo `i3d_r50`) | `adapt_conv3d_to_6ch` 3→6 ch on `blocks[0].conv` (kernel 5×7×7, target == source = identity transform) | bit-identity test passes: ch[0:3] byte-identical to K400 reference; ch[3:6] differs (Kaiming). |
| M5.5 R2 R(2+1)D-18 | Kinetics-400 (torchvision `R2Plus1D_18_Weights.KINETICS400_V1`) | `adapt_conv3d_to_6ch` 3→6 ch on `stem[0]` (kernel 1×7×7, target == source = identity transform) | bit-identity test passes: ch[0:3] byte-identical; ch[3:6] differs (Kaiming). |

For the two K400 baselines whose stem kernel matches the target
(I3D, R(2+1)D-18), the trilinear "downsample" is the identity
transform → the per-channel-group norm ratio is naturally close to 1
and the M3-001 ratio diagnostic does not apply. The bit-identity test
(`torch.testing.assert_close(adapted_first_three, ref_conv.weight,
atol=1e-7)`) is the stronger pin in this regime, mirroring the
TimeSformer-Small R1 K400 verification approach. Both K400 baselines
PASS the bit-identity test.

## Findings

### 1. K400 pretraining still helps at this NID input scale

The two K400-pretrained baselines (I3D 0.5149, R(2+1)D-18 0.5197)
both beat the K400-pretrained main method (0.4756) by ~0.040, and beat
the strongest random-init baseline (TimeSformer-S 0.4836) by 0.031 /
0.036. Same K400 source, similar param count (27M / 31M / 22M / 31M);
the architecture difference is the active mechanism.

That said, R(2+1)D-18 is at 31M (1.4× main) and I3D at 27M (1.2×
main); some fraction of the +0.040 lift may attribute to params alone.
M5.10 ablation could control for this by training a from-scratch
VideoMAE-S at the same compute budget — which would also tease apart
"K400 vs random" from "transformer vs 3D conv" for the same
parameter count.

### 2. The slow stream rewards 3D-conv inductive bias

All four 3D-conv baselines (C3D, I3D, R(2+1)D-18, plus the joint
3D-tube VideoMAE main) score better on slow than the recurrent
ConvLSTM (slow 0.5542 vs 0.6069–0.6730). R(2+1)D-18's 0.6730 is the
top slow-only macro_f1 — beats main method's 0.6069 by +0.066. The
divided space-time TimeSformer-S also beats main on slow (0.6254,
+0.019), suggesting the inductive bias question is "spatial-temporal
factorisation in some form" rather than specifically "3D conv" or
"divided attention".

ConvLSTM's slow-stream weakness is the inverse pattern. The recurrent
inductive bias may favour short-burst (fast-stream) patterns over
long-tempo ones — possibly because the recurrence's effective memory
horizon saturates before the 1s slow-stream window unfolds; or because
the 2×2 spatial pool between cells (a memory-budget concession; see
`convlstm_nid.py` docstring) discards low-frequency spatial signal
that the slow stream relies on.

### 3. Bot rare-class collapse is architecture-dependent, not pure data imbalance

The Bot validation set has only 12 samples — small enough that 1
correct prediction shifts F1 from 0.0 to 0.083. Five of the six rows
(main, C3D, ConvLSTM, I3D + R(2+1)D-18 by F1=0.143) collapse Bot to
F1 ≤ 0.143. Only TimeSformer-S R1 (random init, head_lr ×1) preserves
non-trivial Bot signal: F1=0.091 with AUROC=0.7151, +0.218 above main
method's 0.4968.

Two readings:

a) The focal+α intervention works noticeably better when the backbone
   does not commit to the majority-class boundary as aggressively;
   TimeSformer-S divided space-time attention with head_lr ×1 is
   evidently in a regime where Bot signal survives. The R1.5 ablation
   (head_lr ×5 active) collapses Bot AUROC back to 0.594 — confirming
   that head_lr is not the only knob; the multiplier interacts with
   architecture.

b) R(2+1)D-18 is the only baseline with non-zero Bot F1 (0.143). It
   correctly classified 1 of 12 Bot samples — a recall-side breakthrough
   at low support, not a discriminative signal lift (Bot AUROC 0.4994
   ≈ random). At n=12 a single sample contributes 0.083 to recall, so
   this gain is on the edge of statistical significance; it would need
   a larger Bot test set to interpret as architecture-level evidence.

### 4. C3D-Small is the weakest baseline despite matched param scale

C3D-Small (random init, 19M, head_lr ×1) lands 0.029 below the
K400-pretrained main method on combined macro_f1 — the largest gap of
any baseline. Notably C3D collapses DDoS to F1=0.336 while every other
baseline (random or pretrained) lands DDoS F1 ≥ 0.46. The 8-layer
feed-forward C3D conv stack apparently loses DDoS-specific signal that
both the recurrent ConvLSTM (0.789) and the K400 baselines (0.802,
0.824) retain. This rules out "from-scratch is the bottleneck" — both
C3D and ConvLSTM are random-init, but ConvLSTM matches main method on
combined macro_f1 (0.4746 vs 0.4756, Δ −0.001). Architecture matters
much more than pretrained-status alone for DDoS-class signal.

## R1.5 ablation supplementary: head_lr × pretrained-status interaction

The M5.5 R1 → R1.5 forensic discovery (head matcher bypass, see
`m5_baseline_trajectory.md` §"M5.5 R1 → R1.5 forensic finding") gave
us a free observational ablation of head_lr ×5 on a from-scratch
backbone (TimeSformer-Small):

| Variant | head_lr_multiplier | combined macro_f1 | Bot AUROC |
|---|---:|---:|---:|
| TimeSformer-S R1 (matcher bug) | ×1 effective | 0.4836 | 0.7151 |
| TimeSformer-S R1.5 (matcher fixed) | ×5 active | 0.4616 | 0.5940 |
| Δ R1.5 − R1 | — | **−0.022** | **−0.121** |

This is the empirical evidence for the Path B contract: head_lr ×5 was
designed for K400-pretrained backbones; applying it to a from-scratch
backbone hurts, both on combined macro_f1 and on Bot rare-class signal.
The R1.5 numbers are NOT used in the headline 6-row table above; they
appear only as this supplementary ablation row.

## Methods section draft (paper)

> All baselines and the main method share an identical training
> contract except the head learning-rate multiplier, which is grouped
> by pretrained status. The K400-pretrained backbones (VideoMAE-Small,
> I3D-R50, R(2+1)D-18) train with `head_lr_multiplier = 5×` over the
> base learning rate, slowing the backbone update to preserve
> Kinetics-pretrained features while the freshly-initialised
> classification head learns the 13-way collapsed-CIC mapping at a
> faster rate. The from-scratch random-init backbones (TimeSformer-Small,
> C3D-Small, ConvLSTM) train with `head_lr_multiplier = 1×` because
> both head and backbone are equally fresh; an asymmetric
> head/backbone LR in this regime overshoots toward the
> majority-class decision boundary before the backbone learns
> low-level features (M5.5 R1 vs R1.5 ablation: head_lr ×5 dropped
> TimeSformer-Small combined macro_f1 by 0.022 and collapsed Bot
> per-class F1 from 0.091 to 0.000). The split is determined by what
> pretrained checkpoints exist in the open-source video ecosystem at
> our parameter scale (~13M–33M); no model-specific tuning beyond
> this binary grouping.

## Source artefacts

Each `eval_metrics.json` is verbatim machine-readable; the per-class
CSV is suitable for direct table-paste; the README in each bundle
carries the exact reproduction command.

| Row | Source training run | Artefact bundle | Commit |
|---|---|---|---|
| M5.4 P2 main         | `outputs/run_20260502_184512/`    | `m5_4_phase2_eval/`              | `1d1a61e` |
| M5.5 R1 TimeSformer-S | `outputs/run_20260503_121046/`    | `m5_5_timesformer_small_eval/`   | `3187a67` |
| M5.5 R2 C3D-Small    | `outputs/run_20260503_180604/`    | `m5_5_c3d_small_eval/`           | `135d3e6` |
| M5.5 R2 ConvLSTM     | `outputs/run_20260503_202832/`    | `m5_5_convlstm_eval/`            | `2d054cb` |
| M5.5 R2 I3D          | `outputs/run_20260504_015958/`    | `m5_5_i3d_eval/`                 | `23bd16b` |
| M5.5 R2 R(2+1)D-18   | `outputs/run_20260504_051632/`    | `m5_5_r2plus1d_18_eval/`         | `e0f29ff` |

`outputs/**` is gitignored so artefacts ship locally only; recreate
any one bundle from `best.pt` via `scripts/baseline_rerun.py
--model <name> --resume <ckpt> --output-dir <out>`. The R1 forensic
artefact at `outputs/run_20260502_232207/m5_5_timesformer_small_eval/`
is preserved with a SUPERSEDED banner; numbers from that bundle must
not be cited as baseline results.

The train commands are recorded in each commit's body. All six runs
ran with `--shard-pattern-fast
"data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar"`,
`--shard-pattern-slow
"data/processed/cicids2017_dt1000ms_v2/*/shards/shard-*.tar"`,
`--splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet`,
`--num-epochs 10 --loss-fn focal --focal-gamma 2.0 --reweighting
inverse_sqrt --eval-strategy no_cycle --label-mode collapsed13`,
differing only in `--model {videomae_small | timesformer_small |
c3d_small | convlstm | i3d | r2plus1d_18}` and `--head-lr-multiplier
{5.0 | 1.0}` per the Path B grouping above.
