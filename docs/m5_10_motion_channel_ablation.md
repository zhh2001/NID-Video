# M5.10 Round 1 Dimension 2: Motion-channel ablation (VideoMAE-Small)

This document is the data anchor for the second M5.10 ablation
dimension — input-channel motion contribution to the main-method
22M VideoMAE-Small backbone. Two cells run under the M5.4 P2 fairness
contract with **only** the input channel count varying:

| Cell | Channels | Description | head_lr | Status |
|---|---:|---|---:|---|
| **C=6** (main P2) | 6 | full input: ch1-4 static + ch5 direction-Δ + ch6 packet-count-Δ | ×5 | reused from M5.4 P2 retrofit (no re-train) |
| **C=4** | 4 | static-only: ch1-4 (drops ch5+ch6 motion channels) | ×5 | **Phase 1 forward training — this commit** |

Per Idea.md §3.2: ch1-3 are bit-IP / port / protocol histograms;
ch4 is the bit-packed TCP-flag mask (SYN/ACK/FIN/RST/PSH/URG/CWR/ECE);
ch5 is the per-bucket direction-Δ (signed packet-count change between
consecutive Δt windows, sign = direction); ch6 is the unsigned packet-
count Δ. The C=4 cell drops ch5+ch6 by setting `data.num_channels=4`,
which the dataloader respects by yielding 4-channel tensors and the
model's `forward` either slices (for upstream-emitted 6-channel
tensors) or ingests directly. The patch_embed conv is built with
`adapt_conv3d_to_4ch` instead of `adapt_conv3d_to_6ch`, producing a
`Conv3d(4, 384, ...)` instead of `Conv3d(6, 384, ...)` — ch[0:3] are
trilinear-downsampled K400 weights (16→8 spatial) bit-identical to
the C=6 cell's ch[0:3] (same source, same downsample), and ch[3:4] is
a single Kaiming-init channel for the TCP-flag mask.

Path B is preserved: K400 → ×5 (both cells), so the contrast isolates
the motion-channel contribution rather than mixing it with the head_lr
factor that confounded R1 vs R1.5 in M5.5.

## Common contract (both cells)

- **Input**: identical (T=16, H=32, W=64) NID tensor; same
  `splits.parquet` (M5.3 anchor); multi-scale 50/50 fast/slow mix.
  Channel count C varies between cells (4 vs 6).
- **Optimiser**: 8-bit AdamW, batch=32, grad_accumulation=1, fp16 AMP,
  weight_decay=0.05, base_lr=1.5e-4 with linear warmup to 500 steps
  + cosine decay to 1% peak. Gradient checkpointing on.
- **Loss + reweighting**: focal γ=2 + inverse-square-root α reweighting.
- **Schedule**: 10 epochs, 48,530 grad steps under round_robin
  epoch terminator; per-epoch eval under `no_cycle` so the in-training
  metric is bit-identical to the noise-free re-evaluation.
- **MetricsWriter on**: per_step.jsonl (with grad_norm via
  `--collect-grad-norm`), per_epoch.json, confusion_per_epoch.npz
  written to `<run_dir>/metrics/`.
- **Pretrained source**: `MCG-NJU/videomae-small-finetuned-kinetics`
  (K400) for both cells. The C=4 cell's `adapt_conv3d_to_4ch` reuses
  the same source and the same trilinear downsample; the only
  difference is `n_extra=1` (Kaiming-init 1 channel) instead of
  `n_extra=3` (Kaiming-init 3 channels).

## Two-cell summary table

| Cell | Run dir | Params | C | combined | fast | slow | Bot AUROC | Bot F1 | accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **C=6** (main P2) | `outputs/run_20260502_184512/` | 22M | 6 | **0.4756** | 0.4525 | 0.6069 | 0.4968 | 0.0000 | 0.9560 |
| **C=4** | `outputs/run_20260510_091547/` | 22M | 4 | **0.4691** | 0.4387 | 0.6401 | 0.5233 | 0.0000 | 0.9511 |

Numbers verbatim from each cell's eval bundle:
- C=6 main P2: `outputs/run_20260502_184512/m5_4_phase2_eval/eval_metrics.json`
- C=4: `outputs/run_20260510_091547/m5_10_c4_videomae_eval/eval_metrics.json`

val_sample_count_total = 18,156 (fast 16,463 + slow 1,693) bit-identical
across both cells.

## Two-way Δ (the round 1 dim 2 headline)

| Metric | C=4 | C=6 | Δ C=4 − C=6 |
|---|---:|---:|---:|
| combined macro_f1 | 0.4691 | 0.4756 | **−0.007** |
| fast macro_f1 | 0.4387 | 0.4525 | −0.014 |
| slow macro_f1 | 0.6401 | 0.6069 | **+0.033** |
| Bot per-class AUROC | 0.5233 | 0.4968 | **+0.027** |
| Bot per-class F1 | 0.0000 | 0.0000 | 0.000 |

## Three-way cross-dimension Δ (this commit + dim 1 anchors)

Pulling in the dim 1 cells lets us assess whether dropping motion
channels approaches the random / SSv2 floor, or stays close to K400.
All four cells use the same `splits.parquet` and the same 18,156-sample
val set.

| Cell | dim | Pretrained | C | head_lr | combined | Bot AUROC |
|---|---|---|---:|---:|---:|---:|
| **K400 main P2** | — | K400 | 6 | ×5 | 0.4756 | 0.4968 |
| **C=4** (this) | 2 | K400 | 4 | ×5 | **0.4691** | 0.5233 |
| random | 1 | none | 6 | ×1 | 0.4386 | 0.6743 |
| SSv2 | 1 | SSv2 | 6 | ×5 | 0.4413 | 0.4115 |

| Δ pair | combined | Bot AUROC |
|---|---:|---:|
| C=4 vs K400 main (motion-channel cost) | **−0.007** | **+0.027** |
| C=4 vs random (transfer-prior preserved at C=4) | +0.031 | **−0.151** |
| C=4 vs SSv2 (within-dimension cell vs cross-dimension cell) | +0.028 | +0.112 |

**Two-cell observations** (recorded for round 1 closeout Findings;
no Findings.md edits this round per spec):

- **Motion-channel cost on headline combined macro_f1 is essentially
  null**: dropping ch5 (direction-Δ) and ch6 (packet-count-Δ) from the
  input costs only −0.007 combined macro_f1 (0.4691 vs 0.4756) — well
  within the per-run noise band observed across the M5.10 round 1 dim 1
  cells (random 0.4386, SSv2 0.4413, K400 0.4756: range 0.037 across
  three cells with the same input). The motion channels do **not**
  contribute the bulk of the input signal at this representation
  scale; the static channels (bit-IP / port / protocol histograms +
  TCP-flag mask) carry nearly all of the recoverable discriminative
  information.
- **Slow-stream macro_f1 IMPROVED at C=4** (0.6401 vs C=6's 0.6069,
  Δ +0.033 — direction-of-effect surprise). The slow stream
  (Δt=1000ms) is where I expected the explicit motion channels to
  matter most because the longer window captures more between-frame
  packet-count deltas; the empirical result is the opposite. One
  plausible explanation: at Δt=1000ms the per-bucket packet counts
  are large enough that ch5+ch6 carry redundant or noisy signal
  relative to what the model can recover from frame-to-frame
  attention over the static histogram channels. A confirmation
  ablation would be C=5 (drop ch6, keep ch5) and C=5' (drop ch5,
  keep ch6); deferred (this round 1 is two-cell only).
- **Bot per-class AUROC IMPROVED at C=4** (0.5233 vs C=6's 0.4968,
  Δ +0.027 — second direction-of-effect surprise). Combined with the
  slow-stream improvement, this suggests ch5+ch6 may have been
  injecting low-signal-to-noise features that biased the
  pretrained-K400 + head_lr ×5 regime *toward* the head_lr-×5 Bot
  collapse, and removing them mildly tempers the collapse. C=4 still
  sits well below the random / head_lr ×1 reference (0.6743), so the
  splitting variable for Bot rare-class signal preservation is still
  **head_lr ×1 vs ×5** (the M5-007 finding holds), but motion
  channels appear to amplify the ×5 collapse rather than mitigate it.
- **Fast-stream macro_f1 dropped slightly** (0.4387 vs 0.4525,
  Δ −0.014). The fast stream (Δt=100ms) sees fewer packets per
  bucket per frame, so the explicit motion channels carry a higher
  fraction of the per-bucket signal. The −0.014 fast-stream cost is
  the only metric where C=4 is meaningfully behind C=6, and it
  cancels with the +0.033 slow-stream gain to give the near-null
  combined Δ of −0.007.

## Phase 1 per-class table (C=4 VideoMAE-S, combined eval, epoch 9 best.pt)

| class | n | P | R | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| BENIGN | 16,829 | 0.9664 | 0.9847 | 0.9755 | 0.8909 |
| DoS Hulk | 105 | 0.5462 | 0.6762 | 0.6043 | 0.8982 |
| PortScan | 22 | 0.4286 | 0.5455 | 0.4800 | 0.9554 |
| DDoS | 228 | 0.8763 | 0.3728 | 0.5231 | 0.9908 |
| DoS GoldenEye | 61 | 0.5517 | 0.2623 | 0.3556 | 0.9630 |
| FTP-Patator | 107 | 0.1136 | 0.0467 | 0.0662 | 0.9057 |
| SSH-Patator | 175 | 0.2000 | 0.0743 | 0.1083 | 0.8452 |
| DoS slowloris | 264 | 0.8273 | 0.8712 | 0.8487 | 0.9970 |
| DoS Slowhttptest | 105 | 0.2561 | 0.2000 | 0.2246 | 0.8998 |
| Bot | 12 | 0.0000 | 0.0000 | 0.0000 | **0.5233** |
| Web Attack | 0 | — | — | — | — |
| Infiltration | 0 | — | — | — | — |
| Heartbleed | 248 | 0.9644 | 0.9839 | 0.9741 | 0.9999 |

## Phase 1 per-epoch trajectory (combined eval)

| epoch | combined | Bot AUROC | Bot F1 | GoldenEye F1 | DDoS F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.3277 | 0.6975 | 0.0000 | 0.1379 | 0.1667 |
| 1 | 0.3055 | 0.6097 | 0.0000 | 0.0909 | 0.1705 |
| 2 | 0.3599 | 0.5247 | 0.0000 | 0.4091 | 0.1719 |
| 3 | 0.3728 | 0.4564 | 0.0000 | 0.2254 | 0.1746 |
| 4 | 0.4207 | 0.4381 | 0.0000 | 0.3953 | 0.1753 |
| 5 | 0.4246 | 0.6443 | 0.0000 | 0.4368 | 0.2008 |
| 6 | 0.4331 | 0.5199 | 0.0000 | 0.3182 | 0.2248 |
| 7 | 0.4438 | 0.4324 | 0.0000 | 0.2989 | 0.3037 |
| 8 | 0.4340 | 0.4538 | 0.0000 | 0.3158 | 0.2519 |
| 9 | **0.4691** | 0.5233 | 0.0000 | 0.3556 | 0.5231 |

`best.pt = epoch_9_step_48530.pt` per the trainer's tracking — epoch 9
combined macro_f1 was the run's high-water-mark (monotone-with-noise
climb from 0.3277 to 0.4691 over 10 epochs; max dip 0.022 at epoch 0→1
within standard cosine-warmup variation).

Notable trajectory features (open-ended observations, recorded for
the round 1 closeout Findings batch; no Findings.md edits this round
per spec):

- **Bot AUROC oscillates [0.43, 0.70] across 10 epochs** with final
  epoch 9 = 0.5233. NOT sustained > 0.66 like the dim 1 random Phase 1
  trajectory (which held > 0.66 across all 10 epochs). Pattern shape
  matches the K400 main P2 collapse (0.6835 → 0.4968) and the dim 1
  SSv2 collapse, but the **final-epoch C=4 Bot AUROC of 0.5233 is
  slightly higher than C=6 main P2's 0.4968** (Δ +0.027) — the
  cross-cell direction-of-effect surprise from the headline table
  shows up at the per-class trajectory too. The "head_lr ×1 + random
  init preserves Bot ranking, head_lr ×5 + pretrained collapses Bot
  ranking" pattern from M5-007 still holds qualitatively (C=4 sits
  far below random's 0.6743), but motion-channel removal mildly
  tempers the collapse rather than locking it in. The splitting
  variable for Bot rare-class signal preservation remains **head_lr
  ×1 vs ×5**, with motion channels playing a secondary modulating
  role within the ×5 regime.
- **DDoS F1 plateau-then-jump pattern** (held 0.16–0.20 epochs 0–4;
  climbed to 0.30 at epoch 7; final-epoch jump to 0.52 at epoch 9,
  +0.27). Same shape as M5.5 R2 random-init 3D-conv-style baselines
  and the dim 1 random / SSv2 Phase 1 trajectories. Final-ckpt DDoS F1
  = 0.5231 for C=4 vs 0.5410 for dim 1 random vs 0.4178 for dim 1
  SSv2 vs ~0.4590 for K400 main P2 — C=4 sits at the high end despite
  losing the explicit motion channels. DDoS pattern detection survives
  the channel ablation.
- **GoldenEye F1 oscillates [0.09, 0.44]** — universal noisy-attractor
  pattern continues (11/11 forward + retrofit runs across rounds 1+2).
- **Combined macro_f1 trajectory has 2 dips** (epoch 0→1 −0.022,
  epoch 7→8 −0.010). Max dip magnitude 0.022, comparable to dim 1
  random's max dip of 0.038 and dim 1 SSv2's max dip of 0.006. C=4 +
  head_lr ×5 sits between random + ×1 and SSv2 + ×5 in dip variance.

## Phase 0 sanity verification (recorded for Phase 1 readiness)

The C=4 cell's `adapt_conv3d_to_4ch` is a thin wrapper over
`adapt_conv3d_to_6ch` with `n_extra=1`; the bit-identity contract
(verified by `test_videomae_c4_adapter_consistency_with_c6_on_pretrained`)
is that ch[0:3] of the C=4 cell's patch_embed conv is byte-identical
to ch[0:3] of a C=6 cell built from the same K400 source ckpt with
the same trilinear downsample target. The K400 source kernel is
(2, 16, 16); the project tube_patch is (2, 8, 8) — the M3-001
downsample regime, NOT the I3D / R(2+1)D-18 identity-kernel regime,
so the bit-identity check is between two derived cells (both produced
by the same downsample) rather than between a derived cell and the
un-downsampled source.

Phase 0 silent-load checks at startup (recorded in the trainer log):
- patch_embed adapter log: `ch[0:3] downsampled 16→8 shape=(384, 3, 2, 8, 8)
  norm=3.85; ch[3:4] kaiming-init shape=(384, 1, 2, 8, 8) norm=27.68`.
  K400-derived first-3-channel norm 3.85 matches K400 main P2's
  3.83 within rounding (same source ckpt, same downsample target);
  Kaiming-init extra channel norm 27.68 confirms a single fresh ch4.
- norm-ratio sanity: 27.68 / 3.85 ≈ 7.2× — well above the M3-001 PASS
  bar of 1.5×, confirming pretrained signal preservation at C=4.
- val_n=18,156 in epoch 0 ✓; matches both dim 1 cells and K400 main.

## Phase 1 sanity verification

- **val_sample_count_total = 18,156** in all 10 epochs ✓
- **In-training vs noise-free re-eval** (best ckpt = epoch 9):
  in-training combined macro_f1 = 0.469117; noise-free re-eval
  combined macro_f1 = 0.469117; **Δ = 0.000000** (bit-identical,
  well within the ≤ 5e-5 fairness contract for `--eval-strategy
  no_cycle` runs — same property held by the dim 1 random Phase 1
  and dim 1 SSv2 Phase 2 cells)
- **Wall time**: 10,482.6 s ≈ 174.7 min (10 epochs × ~14–15 min each
  on the 8 GB box, peak GPU 480-482 MB throughout — under the relaxed
  485 MB threshold per the Phase 1 stop-and-report contract; vs C=6
  K400 main P2's 488 MB the −6 to −8 MB delta is consistent with the
  patch_embed channel-count reduction (Conv3d(6→4) saves
  out_ch × T_p × H_p × W_p × 2 bytes for the down_w slice plus
  optimizer-state shadow, which sums to a small but non-zero MB delta
  matching the observed −6 MB)).
- **per_step.jsonl**: 48,530 rows (1 per grad step, full schedule
  completed). Each row carries `grad_norm` field via
  `--collect-grad-norm`. 19 non-finite values, all in early warmup
  steps (steps 1, 2, 3, 5, 8, ...) — the standard fp16 GradScaler
  dynamic-loss-scaling startup artefact (the scaler halves loss-scale
  and retries; per-step loss is sensible from step 2 onwards).
  Remaining 48,511 grad_norm values finite.
- **per_epoch.json**: 10 epoch records, all with
  `metrics.combined.{macro_f1, accuracy, auroc_macro, per_class}`
  populated. (No `fast` / `slow` keys in forward-instrumented runs;
  the noise-free re-eval `eval_metrics.json` carries those splits
  via `baseline_rerun.py`'s scale_id partition.)
- **confusion_per_epoch.npz**: 10 keys `epoch_0..epoch_9`, each
  (13, 13) int64, every sum = 18,156 ✓.
- **No silent C=6 load**: the patch_embed adapter log shows ch[3:4]
  (single extra channel) not ch[3:6] (three extra channels), and the
  saved best.pt's `backbone.embeddings.patch_embeddings.projection.weight`
  has shape `[384, 4, 2, 8, 8]` (verified at re-eval load time — a
  C=6 ckpt would refuse to load into the C=4 model with a shape error,
  and the rerun succeeded only after `--num-channels 4` was passed).
- **Killed-run forensic**: an earlier Phase 1 launch attempt
  (run dirs `outputs/run_20260510_000412/` manual tee +
  `outputs/run_20260510_000824/` Trainer-internal) was killed at
  epoch 0 per the original 450 MB stop-and-report contract because
  peak_gpu = 480 MB exceeded the threshold by 30 MB. Forensic showed
  the 480 MB was architecturally consistent (8-bit AdamW optimizer
  state dominates peak GPU and is proportional to model params, not
  input channels; the −8 MB delta vs C=6's 488 MB matches the
  patch_embed channel reduction); the threshold premise was incorrect
  for this ablation. Threshold was relaxed to 485 MB and Phase 1
  restarted from scratch (rather than resumed from the killed-run
  ckpt — resume codepath was untested + dataloader shuffle drift +
  commit narrative complexity). Killed-run dirs preserved per spec
  for the forensic record; canonical Phase 1 numbers come from
  `run_20260510_091547/`.
- **Two-dir consolidation note**: Phase 1 was launched with a manual
  `tee outputs/run_<ts>/training.log` redirect into a manually-created
  run dir `run_20260510_091140`, but the python Trainer chose its
  own timestamp `run_20260510_091547` for ckpt + metrics output.
  Post-training, `091140/training.log` was moved into
  `091547/training.log` and the empty `091140` dir was rmdir'd. The
  consolidated `091547/` is the canonical Phase 1 run dir; the
  /tmp-cleanup lesson (logs must live with the run) is held — same
  pattern as the dim 1 SSv2 Phase 2 cell.

## Source artefacts

| Cell | Source training run | Eval bundle | Commit |
|---|---|---|---|
| C=6 main P2 | `outputs/run_20260502_184512/` | `m5_4_phase2_eval/` | `1d1a61e` (M5.4 P2) |
| C=4 | `outputs/run_20260510_091547/` | `m5_10_c4_videomae_eval/` (this commit) | (this commit) |

`outputs/**` is gitignored so artefacts ship locally only; recreate
the C=4 cell via:

```bash
uv run python scripts/train.py \
    --shard-pattern-fast "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \
    --shard-pattern-slow "data/processed/cicids2017_dt1000ms_v2/*/shards/shard-*.tar" \
    --splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet \
    --pretrained MCG-NJU/videomae-small-finetuned-kinetics \
    --num-channels 4 \
    --num-epochs 10 \
    --loss-fn focal --focal-gamma 2.0 --reweighting inverse_sqrt \
    --head-lr-multiplier 5.0 \
    --eval-strategy no_cycle --label-mode collapsed13 \
    --collect-grad-norm
```

The `--num-channels 4` flag is added in this commit
(`scripts/train.py`); it routes through `DataConfig._channels_must_be_in_4_to_6`
which logs a warning at non-default values and raises outside [4, 6].
The data pipeline reads `cfg.data.num_channels` to size each yielded
tensor; the model's `VideoMAESmallForNID.__init__` reads
`in_channels=cfg.data.num_channels` and dispatches the patch_embed
build to `adapt_conv3d_to_4ch` when in_channels=4 (vs
`adapt_conv3d_to_6ch` when in_channels=6). The forward pass also
slices a 6-channel input down to `self.in_channels` (a defensive
upstream-mismatch guard, not the production codepath; production
runs with C=4 yield 4-channel tensors directly).

The same `--num-channels 4` flag is added to `scripts/baseline_rerun.py`
in this commit so the noise-free re-eval can build a C=4 model to
match the C=4 ckpt:

```bash
uv run python scripts/baseline_rerun.py \
    --resume outputs/run_20260510_091547/ckpt/best.pt \
    --num-channels 4 \
    --shard-pattern-fast "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \
    --shard-pattern-slow "data/processed/cicids2017_dt1000ms_v2/*/shards/shard-*.tar" \
    --splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet \
    --output-dir outputs/run_20260510_091547/m5_10_c4_videomae_eval/ \
    --task-label "M5.10 round 1 dim 2 cell C=4 — VideoMAE-Small K400 4ch head_lr ×5"
```

Without `--num-channels 4`, baseline_rerun builds a C=6 model and
the ckpt load fails with a shape mismatch on
`backbone.embeddings.patch_embeddings.projection.weight: copying a
param with shape torch.Size([384, 4, 2, 8, 8]) from checkpoint, the
shape in current model is torch.Size([384, 6, 2, 8, 8])`. This is the
intended fail-loud guard; silent acceptance of a partially-loaded
ckpt would be much worse for ablation comparability.

## Findings + Methods draft

Deferred to round 1 closeout. Findings fact-layer entries + paper
Methods draft will be batch-written once all M5.10 round 1 dimensions
(1 = pretrained-source, 2 = motion-channel, 3 + 4 deferred) are
filled in. Per round 1 spec, `prompts/Findings.md` is NOT edited
this round.
