# M3 — Memory & throughput benchmark

Empirical numbers from the dev box (RTX 4060 Mobile / 8 GB / WSL2 Ubuntu).
All measurements use `pretrained=None` (random VideoMAE-S init) so model
weights are deterministic across runs; the input volume on the GPU is the same
either way.

> **Re-scoped after the M3 task 3.4 demo measured 0.20 GB peak.** The 8 GB ceiling
> is comfortably distant — the relevant question shifted from "do we fit?" to
> "what configuration maximizes throughput given memory is abundant?".

> ⚠️ **HARD LIMIT: B ≤ 512 on 8 GB cards.** Empirically, B=1024 hung the WSL2
> NVIDIA driver mid-test, requiring `kill -9` and ~1 minute of recovery time
> (`nvidia-smi` reported integer-overflow garbage values until the kernel
> released the phantom CUDA context). **Do not exceed B=512 even if the
> peak-memory report suggests room** — the OOM transition is not graceful in
> WSL2 and a hung driver cancels your run rather than raising a recoverable
> Python exception.

## TL;DR

| | |
|---|---|
| OOM ceiling (full FP16+8-bit AdamW+GC stack) | **between 768 and 1024** (B=768 → 6.6 GB peak; B=1024 hung GPU on dev box) |
| Recommended max-safe batch | **B=512** (4.5 GB peak; ~3.5 GB headroom for misc allocator + activations growth) |
| Throughput plateau | **~220 samples/sec at B=128–256** (GPU-only); **~280 samples/sec at B=128 with `num_workers=4`** through the real DataLoader |
| Single biggest memory saver | **gradient checkpointing** (4× at B=32: 1735 → 426 MB) |
| `base.yaml` ergonomic verdict | **B=2/accum=16 is 7-12× slower than B=32–128/accum=1** because batch=2 is well below the GPU's compute saturation point |

## Phase 1 — batch sweep (FP16 + 8-bit AdamW + grad checkpointing)

Each measurement is the peak `torch.cuda.max_memory_allocated()` after 3
forward+backward+step iterations on synthetic `(B, 16, 6, 32, 64)` input.

| batch | peak MB | peak GB | note |
|------:|--------:|--------:|------|
| 1   | 231    | 0.23  | most of this is the model + optimizer state, not activations |
| 2   | 241    | 0.24  | (matches the 3.4 demo's 290 MB closely; small allocator variance) |
| 4   | 265    | 0.26  |  |
| 8   | 290    | 0.28  |  |
| 16  | 335    | 0.33  |  |
| 32  | 426    | 0.42  | (used as the ablation reference batch below) |
| 64  | 688    | 0.67  |  |
| 96  | 959    | 0.94  |  |
| 128 | 1230   | 1.20  |  |
| 192 | 1771   | 1.73  |  |
| 256 | 2312   | 2.26  |  |
| 384 | 3394   | 3.31  |  |
| 512 | 4477   | 4.37  | **recommended max-safe** with margin |
| 768 | 6642   | 6.49  | last clean measurement before driver state went bad |
| 1024 | — | — | hung the WSL2 GPU driver mid-test; treat as the practical OOM line |

Memory grows almost linearly past B=64: marginal cost ≈ **10–11 MB/sample**
(activations dominate as parameters/optim state amortize). The crossover where
activations equal parameter-side cost is around B=64.

## Phase 2 — ablations at B=32 (peak MB)

Same input shape, varying one optimization at a time. All five configurations
fit in well under 1 GB.

| config | peak MB | Δ vs baseline |
|---|---:|---:|
| **FP16 + 8-bit AdamW + GC** (baseline) | 426 | — |
| FP16 + 8-bit AdamW + **GC OFF** | 1735 | **+1309 (×4.07)** |
| FP16 + **32-bit AdamW** + GC | 553 | +127 |
| **BF16** + 8-bit AdamW + GC | 426 | 0 |
| **FP32** + 8-bit AdamW + GC | 554 | +128 |

Reading:
* **GC is the dominant memory tool.** Disabling it 4×s the budget. Keep it on.
* **8-bit vs 32-bit AdamW** is worth ~130 MB at 22 M params (8-bit state ≈ 2 B/param,
  32-bit ≈ 8 B/param → diff ≈ 22 M × 6 B ≈ 132 MB). Cheap and accumulates with
  bigger batches.
* **FP16 vs BF16 vs FP32** barely matters here because GC keeps activation
  memory small. With GC disabled the FP16/BF16 gap would widen.
* **BF16 has identical footprint to FP16** (no GradScaler buffers, but same
  half-precision weights/activations).

If GC ever needs to come off (e.g. for debugging gradient flow), expect peak
to roughly 4× — at B=32 that's 1.7 GB, still fine; at B=512 that would push past
the GPU.

## Phase 3 — GPU-only throughput (synthetic data, no DataLoader)

30-iter timing of forward+backward+step under FP16+8-bit+GC. Synthetic input is
generated on-GPU so the DataLoader is bypassed; this isolates the model-compute
ceiling.

| batch | steps/sec | samples/sec |
|------:|----------:|------------:|
|   2 | 11.94 |  23.9 |
|   8 | 11.09 |  88.7 |
|  32 |  6.35 | 203.2 |
| 128 |  1.69 | 216.5 |
| 256 |  0.87 | 222.1 |
| 512 |  0.41 | 210.5 |

* **Throughput saturates around B=128–256 at ~220 samples/sec.** Beyond that
  the GPU is compute-bound and bigger batches give nothing.
* **B=2 is 9× slower than B=32** in samples-per-second despite costing less per
  step. The Python/launch overhead dominates at small B.
* **B=512 is fractionally slower than B=256** — possibly allocator overhead at
  the OOM edge. Don't push past B=256 for throughput.

## Phase 4 — Real DataLoader + num_workers (B=32, B=128)

Synthetic shards (256 samples / 16 shards) read through the actual `build_dataloader`
pipeline. 15-grad-step warmup + 15 measured steps.

| batch | num_workers | samples/sec | note |
|------:|------------:|------------:|------|
|  32 | 0 | 168 | DataLoader overhead vs GPU-only (203) ≈ 17 % |
|  32 | 2 | 181 |  |
|  32 | 4 | 173 |  |
| 128 | 0 | 166 | DataLoader heavily IO-bound at this batch (217 GPU-only → 166) |
| 128 | 2 | 130 | multiprocessing overhead overwhelms small per-shard work |
| 128 | 4 | **283** | best measured throughput; workers prefetch while GPU computes |

* **num_workers benefit appears at higher batch sizes**, because per-step GPU
  work is fast enough that DataLoader latency becomes the limiter.
* At B=128, going from `num_workers=0` to `num_workers=4` gives a **70 % speedup**.
* **`num_workers=2` is sometimes worse than `=0`** — multiprocessing overhead
  beats the parallelism benefit when workload is small. Pick 0 or 4, skip 2.

## Projection: 500 K-sample epoch

Assuming the training loop is fed by the real DataLoader (Phase 4 numbers, not
the GPU-only Phase 3 ceiling).

| config | samples/sec | wall time / 500 K-sample epoch |
|---|---:|---:|
| Current `base.yaml` (B=2 / accum=16, single worker) | ≈24 | **~5.8 hours** |
| Recommended (B=32 / accum=1, workers=2) | 181 | **~46 min** |
| Aggressive (B=128 / accum=1, workers=4) | 283 | **~29 min** |

A 10–12× speedup is on the table by abandoning the small-batch / high-accumulation
strategy that the original `base.yaml` was set up for.

## Conclusions

### Two configs ship: `base.yaml` for tests, `training_perf.yaml` for production

The `batch_size=2 / grad_accumulation=16` configuration was designed under a
"survive 8 GB VRAM at any cost" assumption. **That assumption no longer holds**:

* The model + activations at B=128 fit in 1.2 GB.
* batch=2 sits 9× below the GPU's compute saturation point.
* Eight-step accumulation already destroys the wall-clock advantage of a small per-step batch.

To capture this without breaking the M3 test baseline, M3 ships **two** configs:

| file | purpose | training stack |
|---|---|---|
| `configs/base.yaml` | CI / smoke tests / small-scale sanity checks (baseline stability) | B=2 / accum=16 / num_workers=0 |
| `configs/training_perf.yaml` (extends base) | Real M4+ training (throughput-optimized) | B=32 / accum=1 / num_workers=2 |

`training_perf.yaml` uses our `extends: base.yaml` config-inheritance pattern
(implemented in `utils/config.py`), so any future change to `base.yaml`
propagates automatically — the perf override only touches the three training
fields that change.

Selection rule:
- **Tests / CI / first-time sanity runs** → `--config configs/base.yaml`
- **Real training on CIC-IDS-2017 shards** → `--config configs/training_perf.yaml`

If the user wants a still-more-aggressive `B=128 / num_workers=4` tier, that
changes the *effective* batch (32 → 128) and should land alongside an LR-vs-batch
sanity check in M4.

### Re-evaluating Idea.md "memory-constrained" framing

**Defer to user decision** (per M3 task 3.5 instructions): the relevant Idea.md
section says the small-batch + accumulation choice is part of the cost model
arguing why this work fits a single 8 GB GPU. The numbers above don't undermine
that argument — they say the cost model has more *headroom* than originally
quoted. Whether to revise the framing in Idea.md (e.g., "B=2 was chosen to fit
8 GB at training time" → "B=32 is comfortable and B=128 is feasible") is an
editorial decision worth making once empirical numbers from real CIC-IDS data
are in (M4-M5).

### M4 considerations (decisions deferred from M3)

1. **8-bit vs 32-bit AdamW: re-evaluate against `safetensors` resume.**
   At our 22 M-param scale, 8-bit AdamW saves only **127 MB** (vs 32-bit) — well
   below the textbook ~75 % savings, because 8-bit only quantizes the optimizer
   state (momentum + variance), not the parameters themselves. Smaller models
   amortize less.

   M4 will need full resume (model + optimizer state + step count).
   `bitsandbytes` quantized state isn't `safetensors`-friendly without
   custom dequantization-and-reserialization code. Trade:
   - **32-bit AdamW** (`optimizer: "adamw"`): standard `safetensors` resume,
     +127 MB. Free at our scale.
   - **8-bit AdamW** (`optimizer: "adamw_8bit"`, status quo): need to write a
     custom resume codec. Saves 127 MB.

   *Recommendation:* M4 should default to 32-bit AdamW for resume simplicity;
   keep 8-bit on a CLI flag for users who actually need the savings (large
   models, or when memory headroom shrinks at multi-scale / longer T).

2. **Re-measure on real CIC-IDS shards.** All numbers in this doc are
   synthetic-input upper bounds (random tensors generated on-GPU or written
   directly to webdataset shards). M4 should re-run Phase 3 and Phase 4 on
   actual CIC-IDS shards: tar decoding + JSON parse will add per-batch IO
   overhead, and the relative benefit of `num_workers > 0` may grow.

3. **Idea.md "memory-constrained" framing.** That section currently justifies
   the small-batch / accumulation choice as a hardware-fit argument. The
   numbers above show that headroom is much larger than the framing implies.
   Editorial revision deferred to M4–M5 once real-data numbers are in.

### Open caveats

1. **All measurements use synthetic data**. Real CIC-IDS shards (compressed
   ~150 KB/sample) will be slower in IO; the ratio of GPU-only-vs-DataLoader
   throughput in Phase 4 is the synthetic-input upper bound.
2. **Phase 4 timing has high variance** (only 15 measured grad steps each).
   The ~280 sps at B=128/workers=4 is a single trial; treat as ±20 %.
3. **The B=1024 driver hang** is documented as the HARD LIMIT at the top of
   this doc. Repeating: stay at B=512 or below.
