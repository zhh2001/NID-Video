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

| Run | Epoch | grad_steps | reported macro_f1 (round_robin) | noise-free macro_f1 (no_cycle) | Δ | reported Bot AUROC | noise-free Bot AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| M4.8 | 1 | 4,853 | 0.4474 | **0.3324** | −0.1150 | 0.7411 † | 0.7247 |
| M5.1 | 3 | 14,559 | 0.5113 | **0.4230** | −0.0883 | n/a ‡ | 0.4237 |
| M5.2 | 10 | 48,530 | 0.5143 | **0.4677** | −0.0466 | 0.4402 | 0.4077 |

† M4.8's original training task output was retained only in
in-conversation records; the value 0.7411 is the figure cited there.

‡ M5.1's original task output was rotated out of the workspace cache
before this trajectory was assembled. The noise-free re-eval (0.4237)
is the recoverable number; the reported value would require a fresh
``round_robin`` re-eval of the same checkpoint.

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

## Reproduction

The four artefact bundles are stored alongside their source training
runs (each ``outputs/run_<ts>/`` directory is gitignored):

- ``outputs/run_20260430_223105/m4_8_rerun/``
- ``outputs/run_20260501_143946/m5_1_rerun/``
- ``outputs/run_20260501_162117/m5_3_rerun/``

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
