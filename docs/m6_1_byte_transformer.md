# M6.1 — 1D Byte Transformer (Phase 0 design)

Cross-paradigm baseline cell 6.1. The fast-only val partition is the
shared evaluation surface with all other cells (M5.4 main P2 + M5.10
dim 1/2/4 + M6.2 RF/XGB + M6.3 IN/RN); the 1D byte-sequence paradigm
provides the third paradigm row alongside video + 2D snapshot + flow,
filling out the 12-cell cross-paradigm table.

This document is the Phase 0 design source-of-truth. It commits the
non-⚠ default choices in advance of code (step 2). Two ⚠ items (A
byte-extraction source + D closed-world hygiene mask) require
explicit design-layer approval before step 2 begins; recommendations
are documented inline but not yet locked.

The contract recap (locked from handoff §7, NOT re-negotiated):

  * Input: K=16 packets × N=128 bytes per window
  * Model: 6-layer 1D Transformer encoder, d_model=256, nhead=8
  * Stream: fast-only (no slow analogue; the 1D paradigm consumes
    raw bytes from packets in time order — there is no natural
    "1 Hz aggregated" version that maps to slow Δt=1s windows)
  * Init: from-scratch random
  * Path B head_lr ×1 (random-init group, M5-005 v3)
  * Loss: focal γ=2.0 + inverse_sqrt α
  * Epochs: 10 (round_robin counted against the fast stream)
  * Batch: B=32, grad_accumulation=1
  * Eval: noise-free no_cycle
  * Labels: 13-class collapsed

The `splits.parquet` `(pcap_source, start_time)` key set is the
invariant that ties M6.1 to every other cell — the 1D paradigm's
training samples must align 1-to-1 with the fast (Δt=100ms)
webdataset windows. val_n_fast = 16,463 ✓.

## A — Byte extraction source [DECIDED: A-i (design layer N+1 session)]

Locked: **A-i offline extraction → new parallel shards** at
`data/processed/cicids2017_dt100ms_v2_bytes/`. Rationale: minimal
disk (~230 MB on 795 GB free), forensic-preserves existing v2
shards, deterministic + testable, A-ii's 10-epoch re-parse cost
(~10h wall) was a fatal ceiling.


### Trade-off space

| option | what it does | disk Δ | wall (one-shot) | training IO | implementation complexity | failure modes |
|---|---|---:|---:|---|---|---|
| **A-i** offline extraction → parquet/webdataset | scan pcaps once, emit per-window `(K=16, N=128) uint8` + attention_mask + `pcap_source` + `start_time`. New webdataset shards under `data/processed/cicids2017_dt100ms_v2_bytes/`. | ~230 MB (Tues+Wed+Fri val+train+test) | 1-2 h (pcap_parser-class speed, ~33 GB scan) | cheap: webdataset shard reads, no random-seek into pcaps at train time | medium: new ETL script mirrors `run_etl.py` shape but emits byte tensor not 6-channel histogram tensor | (1) misaligned window-key set vs existing v2 shards → val_n drift; (2) malformed-packet handling drift vs v2 ETL |
| **A-ii** on-the-fly extraction in DataLoader | DataLoader hook re-walks the pcap stream per epoch, indexed by `splits.parquet` start_time → packet stream → first 16 packets → bytes | 0 | 0 (no precompute) | expensive: every epoch re-parses 33 GB pcaps; either ~1 h/epoch via streaming or random-seek into pcap which dpkt does not support natively | high: needs custom dataset that bridges splits.parquet windows ↔ pcap_parser streams; epoch ordering / shuffle complicates window-level random sampling | (1) random-seek not native in pcap; (2) 10 epochs × ~1 h re-parse = 10 h wall; (3) hard to make deterministic for tests |
| **A-iii** retrofit v2 shards with extra byte fields | extend ETL to emit `bytes.npy` (K=16, N=128) alongside `tensor.npy` in each existing shard | ~230 MB + reshard cost | full v2 ETL re-run ~6-8 h | cheap (one shard read serves both video tensor + bytes) | high: violates forensic preservation principle ("现有 v2 shards 不动"); requires re-ETL for all 11 cells implicitly | (1) breaks reproducibility anchor for already-trained M5.10 + M5.5 cells; (2) ETL re-run cost dominates Phase 1 budget |

### Recommendation: **A-i** offline extraction

Rationale:
- **Forensic preservation honoured**: existing v2 shards untouched
  (per Idea.md / handoff doc rule "现有 v2 shards 不动"). New shards
  land at `data/processed/cicids2017_dt100ms_v2_bytes/` parallel to
  v2 — additive, not modifying.
- **Disk cost minimal**: ~230 MB total ≈ 0.7% of existing
  cicids2017_dt100ms_v2 (82 GB), 0.03% of disk-free (795 GB).
- **Training IO clean**: webdataset shard reads at ~hundreds of
  MB/s match Phase 1 budget; no random-seek into 33 GB pcaps every
  step.
- **Implementation complexity tractable**: mirrors existing
  `etl_pipeline.py` shape (windower → per-window emit → ShardWriter).
  Reuses `SlidingWindow` to align byte-window keys 1-to-1 with v2
  fast windows.
- **Deterministic + testable**: same input pcaps + same seed →
  bit-identical output shards. The bit-identity contract
  (`splits.parquet` keys exactly match) can be verified once at
  ETL end and again at training epoch 0.

A-ii has a fatal cost ceiling (10 h training wall × 10 epochs is
worse than the entire Phase 1 budget). A-iii violates the forensic
preservation rule and the cost is dominated by ETL re-run rather
than Phase 1 training, which is wrong incentive.

**STOP-AND-ASK**: design-layer approval needed before proceeding to
step 2 with A-i. Alternatives are documented but not implemented.

## B — Packet selection K=16 [self-decided]

**Locked: B-i first 16 packets in temporal order**.

Right-pad with attention_mask=0 when the window has fewer than 16
packets. M5.4 P2 retrofit logs show median ~30 packets per 1.6s
fast window with long tail; the lower bound is rarely <5 packets
(empty / quiet windows are dropped before encoding). Right-pad
fraction will be reported per the smoke run.

Rationale: temporal order is the natural reading direction for the
1D Transformer; first-16 is causal (no peek ahead) and matches video
cells' T=16 frame budget — fairness contract for "16 timesteps per
window".

## C — Byte truncate/pad N=128 [self-decided]

**Locked: C-i first 128 bytes of L2 (Ethernet) frame**.

The L2 frame layout is: `Ethernet header (14 bytes) | IP header
(20 bytes for IPv4) | TCP/UDP header (8-20 bytes) | payload (0-N
bytes) | trailer`. First 128 bytes of L2 covers the full L2/L3/L4
headers plus the first ~88 bytes of payload for short packets, or
the full headers + first ~94 bytes of payload for typical TCP.

Right-pad with 0x00 + attention_mask=0 for packets <128 bytes;
truncate beyond 128 (record truncation rate in smoke log).

Rationale: keeping headers gives the model access to TCP flags,
options, port numbers (all in the 14-44 byte range) — fingerprintable
signals that the M5/M5.10 channel encoder already extracts in
aggregated form. C-ii (skip headers, payload-only) would discard
the parts the channel encoder explicitly leverages and is worse for
a "1D paradigm vs spatial encoder" comparison.

## D — Closed-world hygiene mask [DECIDED: D-ii (design layer N+1 session)]

Locked: **D-ii** — zero L2 bytes 0-11 (MAC dst+src) + 26-29 (IPv4
src) + 30-33 (IPv4 dst). Rationale: both link + network identifiers
masked (CIC-IDS-2017 attacker MAC is fixed across the day; D-i
alone leaks via MAC). Ports retained (D-iii would destroy
legitimate-service-port signal for SSH/FTP-Patator). 20 of 128
bytes masked → 84.375% remaining.


CIC-IDS-2017 has trivially label-correlated network metadata. The
attacker workstation has a fixed MAC + IP across the entire test
day; raw bytes containing that MAC/IP would let the model memorize
"this MAC = attack day Friday afternoon DDoS" rather than learn
generalizable byte signatures. The existing 6-channel encoder
avoids this via the IP hash-clustering step (Idea.md §3.2) that
clusters IPs by behavior, not identity.

For raw bytes:

| option | what to zero out | bytes masked | rem fraction | residual label leak risk |
|---|---|---:|---:|---|
| **D-i** mask IPv4 src + dst | L2 bytes 26-29 + 30-33 (IPv4 src + dst inside IP header) | 8 of 128 | 120/128 = 93.75% | medium: Ethernet MAC still present (12 bytes) carrying same attacker identity |
| **D-ii** D-i + Ethernet MAC src + dst | L2 bytes 0-11 (MAC dst[0:6] + MAC src[6:12]) + 26-29 + 30-33 | 20 of 128 | 108/128 = 84.375% | low: both link-layer and network-layer identifiers anonymized |
| **D-iii** D-i + D-ii + port mask | + L2 bytes 34-35 + 36-37 (TCP/UDP sport + dport) | 24 of 128 | 104/128 = 81.25% | very low but **destroys** legitimate-port-vs-attacker-port discrimination signal (SSH-Patator 22 / FTP-Patator 21 land on legitimate service ports anyway; port mask would zero out *both* attack and benign port signals, eliminating a real learnable feature) |

### Recommendation: **D-ii** IP + MAC, NOT port

Rationale:
- D-i alone is insufficient: CIC-IDS-2017 attacker MAC is fixed and
  hash-equivalent to attacker IP within the day. A model seeing
  MAC `00:..attacker..` always paired with attack labels learns
  the MAC as label proxy.
- D-iii (port mask) is a false friend: legitimate attacks like
  SSH-Patator and FTP-Patator use the legitimate service ports
  (22, 21). Masking the port destroys exactly the
  `legitimate-service-port-receiving-attack-traffic` signal that
  the model should learn. Idea.md's whole port-bucket-encoder
  design depends on port being available as a feature; the byte
  paradigm should mirror that.
- D-ii leaves 108/128 (~84%) of each packet's bytes intact,
  including TCP flags / sequence number / window size / urgent
  pointer / options + payload first bytes. These are the
  fingerprintable signals.

The masked bytes are *zeroed* (set to byte 0x00 = 0) rather than
removed; attention_mask still attends to them (they're real
positions, just anonymized). Keeping the position with 0x00 byte
preserves K × N = 2048 token grid invariant.

**STOP-AND-ASK**: design-layer approval needed for D-ii. Alternatives
are documented but not implemented.

## E — Token vocabulary [self-decided]

**Locked: vocab_size = 257 (256 raw bytes + 1 [PAD])**.

  * Byte values 0-255 map directly to token ids 0-255.
  * [PAD] = token id 256, used for both packet-level pad (when
    window has <16 packets) and byte-level pad (when packet is
    <128 bytes).
  * No [CLS] token; classification head pools over attention-masked
    non-pad positions (mean-pool over the K×N sequence positions
    with mask=1).

Rationale: smallest sensible vocab; no extra special tokens needed
because the attention_mask carries the pad information. Mean-pool
is simpler than CLS and avoids a fresh special-token-init concern
(CLS would need its own init scheme + ablation).

## F — Positional encoding [self-decided]

**Locked: F-ii 2D factorized learnable**.

  * Packet-axis (K=16): `pos_k = nn.Parameter(torch.zeros(16, 256))`,
    `pos_k.uniform_(-0.02, 0.02)` init.
  * Byte-axis (N=128): `pos_n = nn.Parameter(torch.zeros(128, 256))`,
    same init.
  * Token at `(k, n)` gets embed[byte] + pos_k[k] + pos_n[n].

Rationale: factored PE lets the token know it's "packet i, byte j"
explicitly. F-i (flat 1D PE over 2048 positions) loses the packet
boundary; F-iii (sinusoidal) is fixed and may underfit a small
specialized corpus where learnable PEs have shown advantages on
similar 1D-token-stream tasks.

Parameter cost: (16 + 128) × 256 = 36,864 floats ≈ negligible.

## G — 1D Transformer module details [self-decided + SDPA refinement]

Locked from handoff:
  * 6 encoder layers
  * d_model = 256
  * nhead = 8 (head_dim = 32)

Self-decided:
  * FFN dim = 1024 (4× d_model, standard ratio)
  * **Pre-norm** (LayerNorm before each sub-layer) — modern default,
    stabler training for from-scratch random init
  * Dropout = 0.1 (attention + FFN), 0.0 on embedding
  * Activation = GELU (matches VideoMAE / standard 2025 default)
  * Final layernorm before classifier head
  * Classifier: `nn.Linear(256, 13)`, fresh init, head segment
    name = "classifier" (matches trainer head matcher)

### Attention kernel — SDPA (Phase 0 step 3 stop-and-report follow-up)

Attention kernel = `F.scaled_dot_product_attention` (PyTorch SDPA /
Flash Attention path). Default `nn.TransformerEncoder` + pre-norm
(`norm_first=True`) disables the nested-tensor fast-path, sending
attention through the O(seqlen²) slow path that materializes the
full attention matrix. At (B=32, heads=8, seqlen=2048) this is
~4.3 GB fp16 — over the 8 GB box budget after activations + grads.
SDPA is mathematically equivalent (same scaled dot-product
attention formula); memory drops to O(seqlen) via the mem-efficient
backend (Flash Attention itself does not accept arbitrary padding
masks, so SDPA falls to mem-efficient — both are O(seqlen)).

This is a framework implementation choice, **non-design**: pre-norm /
d_model / nhead / FFN dim / GELU / dropout / 6 layers / fresh
classifier — all locked decisions intact. The custom
`ByteEncoderLayer` mirrors `nn.TransformerEncoderLayer(norm_first=
True)` semantics exactly:

  ```
  pre-norm flow:
    x = x + dropout(out_proj(SDPA(Q,K,V, key_padding_mask)))   # attn block
    x = x + dropout(ffn(norm2(x)))                              # ffn block
  ```

Decision recorded as N+2 design-layer approval after Phase 0 step 3
smoke surfaced 4.5 GB peak GPU > 4 GB stop-and-report threshold.
Re-smoke after SDPA refinement expected ~1 GB peak.

Parameter count breakdown (approximate):

| component | params |
|---|---:|
| token embedding (257 × 256) | 65,792 |
| positional embedding ((16+128) × 256) | 36,864 |
| 6 encoder layers (each ~787 K) | ~4,720,000 |
| final layernorm (2 × 256) | 512 |
| classifier (256 × 13 + 13) | 3,341 |
| **total** | **~4.83 M params** |

Smaller than every video baseline (smallest is ConvLSTM 13 M) and
smaller than ResNet-18 (11.2 M); fairness implication is M6.1 has
fewer parameters than every other cell, **biasing against the 1D
paradigm**. This is honest framing for the paper (don't compensate
artificially; the paradigm's natural model size on this input scale
is small).

## H — Stream alignment [locked from contract]

  * `splits.parquet`: `data/processed/cicids2017_dt100ms_v2/splits.parquet`
  * fast slice val_n = **16,463** (bit-identical contract)
  * 13-class collapsed labeling (same as all other cells)
  * noise-free no_cycle eval policy

The byte-extraction ETL emits one byte-sample per fast window key
in `splits.parquet`, in `(pcap_source, start_time)`-keyed order.
val_n alignment is verified at smoke time (50-window subset) and at
Phase 1 epoch 0 (full val).

## I — Training contract [locked from contract; gradient checkpointing revised]

  * Loss: focal γ=2.0
  * α reweighting: inverse_sqrt
  * head_lr_multiplier: ×1 (random init, Path B)
  * Epochs: 10
  * Batch: 32 (grad_accumulation=1)
  * Optimizer: 8-bit AdamW (bitsandbytes)
  * Precision: fp16 AMP with GradScaler
  * **Gradient checkpointing: ON by default** — revised after Phase 0
    step 3 stop-and-report. The B=32 × seqlen=2048 backward pass
    saves ~1.5 GB of FFN intermediate activations + Q/K/V cache
    across the 6 layers, with peak GPU ~4.7 GB without
    checkpointing (over the 4 GB smoke threshold + close to the
    8 GB box ceiling). Checkpointing trades ~30% extra compute for
    memory; re-smoke peak GPU drops to ~1.4 GB. Non-design
    implementation choice (the Phase 0 design contract permits
    "gradient checkpointing on if memory tight" — this is exactly
    that case).
  * Scheduler: linear warmup 500 + cosine to 1% peak (same as M5
    cells)

## Sanity tests (planned for step 2)

  1. `test_model_forward_shape`: input `(B=2, K=16, N=128)` (int8 or
     int64 ids) + attention_mask `(B=2, K*N=2048)` → logits
     `(B=2, 13)`.
  2. `test_byte_extraction_determinism`: same `(pcap_source,
     start_time)` window key extracted twice → byte-identical
     output.
  3. `test_val_n_alignment`: byte-extraction restricted to
     `splits.parquet split=='val'` rows produces exactly **16,463
     samples** (no drift).
  4. `test_mask_correctness_D_ii`: extracted bytes from any sample
     have:
       - L2 bytes 0-11 (MAC src/dst) all 0x00 ✓
       - L2 bytes 26-29, 30-33 (IPv4 src/dst) all 0x00 ✓
       - L2 byte 34-35, 36-37 (sport/dport) NOT all 0x00 (i.e., not
         port-masked — this catches accidental D-iii application)
  5. `test_param_count_in_band`: model parameter count ∈ [4.5 M, 5.2 M].

## Smoke run (planned for step 3)

50 windows from val split (the first 50 by `(pcap_source,
start_time)` order). Smoke ETL extracts those 50 to
`outputs/run_<ts>_m6_1_smoke/bytes_smoke.parquet` (or webdataset
shard); model does 1 epoch forward + backward + step on that subset.

Reported metrics:
  * val_n alignment: 50 in → 50 out
  * per-step grad_norm finiteness (allow first few fp16 warmup
    artefact, then must be all finite)
  * peak GPU MB (expect < 1 GB on B=32, model ~5 M params, K×N=2048
    tokens with d_model=256)
  * byte-extraction wall (end-to-end 50 windows)
  * forward + backward + step wall per micro-batch
  * Phase 1 full epoch wall projection: extrapolate from 50-window
    smoke to (16,463 train fast + per-epoch eval) and ×10 epochs

## Phase 1 wall budget projection (heuristic, pre-smoke)

Pre-smoke estimate:
  * Training: 16,463 train windows × 10 epochs / B=32 = 5,144 grad
    steps. At ~0.5-1.0 s/step with B=32 + small Transformer →
    ~45-90 min per training pass + ~5 min eval per epoch → **~10 h
    total wall**. May fit in a single overnight slot.
  * ETL (one-shot, A-i): ~1-2 h scan over 33 GB pcaps.

Total Phase 1: ~12-14 h wall (ETL + train + eval).

If smoke reveals significantly higher numbers (e.g., 3× slower per
step than expected) the budget extrapolates to ~30 h, which would
trigger a stop-and-report at the smoke completion step.

## Phase 0 step 2-5 status

All ⚠ items resolved in design-layer N+1 session:
  * **A** = A-i offline extraction (LOCKED)
  * **D** = D-ii IP + MAC masking (LOCKED)

Step 2 (code), step 3 (smoke), step 4 (commit), step 5 (final report)
proceed in this Phase 0 turn.

## Reproduction

The canonical M6.1 eval artefact bundle lives at
`outputs/run_20260516_090240/m6_1_byte_transformer_eval/`. The bundle
contains:

- `eval_metrics.json` — full metric payload (combined + per-stream),
  schema_version=1, machine-readable.
- `confusion_matrix.json` — combined confusion matrix; row=true,
  col=pred; raw counts.
- `per_class_table.csv` — combined per-class table (label_id /
  label_name / support / precision / recall / f1 / auroc).
- `README.md` — task label, source checkpoint, configuration,
  per-stream table, and reproduction commands.

To reproduce the full bundle (per-stream breakdown included), re-run
`scripts/baseline_rerun.py` against the saved checkpoint:

```bash
uv run python scripts/baseline_rerun.py \
    --resume outputs/run_20260516_090240/ckpt/best.pt \
    --shard-pattern-fast "data/processed/cicids2017_dt100ms_v2_bytes/shards/shard-*.tar" \
    --shard-pattern-slow "data/processed/cicids2017_dt100ms_v2_bytes/shards/shard-*.tar" \
    --splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet \
    --output-dir outputs/run_20260516_090240/m6_1_byte_transformer_eval \
    --task-label "M6.1 — 1D byte Transformer, 6-layer SDPA, K=16 N=128, fast-only, head_lr ×1"
```

### Dual-stream artefact note

`baseline_rerun.py` enforces a dual-stream contract: both
`--shard-pattern-fast` and `--shard-pattern-slow` are mandatory. M6.1
is a single-stream fast-only paradigm (the byte Transformer trains on
the fast stream only), so the workaround above passes the same fast
shards to both flags.

Consequences:

- `val_sample_count_total = 32,926` (= 16,463 × 2) — the total is
  doubled because the fast windows are loaded twice.
- The combined / fast / slow rows of the eval bundle report identical
  numbers (macro_f1 = 0.2444, accuracy = 0.9066, auroc_macro = 0.6185
  in all three rows) because the predictions are identical: the same
  data passes through the same forward pass twice.
- Per-class supports `n` are doubled, but F1 / AUROC ratios are
  invariant under doubling and remain the true per-class scores.

### Canonical M6.1 metric

The canonical M6.1 cross-paradigm metric is:

- **macro_f1 = 0.2444 at val_n_fast = 16,463**

This uses the same fast-only protocol as the dim 4 Cell C/D ablation
arm, giving an apples-to-apples comparison against the video
fast-only counterpart (dim 4 Cell C, macro_f1 = 0.4341 at 24,260
grad steps + head_lr ×1).

Caveat: do NOT read the doubled `combined` or `slow` figure from the
eval bundle as a separate measurement; it is the same fast-slice
number reported under a different label due to the dual-pass
workaround.

### Future infrastructure work (deferred)

Extending `baseline_rerun.py` to natively accept a single-stream
fast-only paradigm (e.g. allowing an empty `--shard-pattern-slow`)
is deferred future infrastructure work. The current paper's results
do not depend on it: the dual-pass workaround is forensic-preserved
in the artefact bundle and operationally adequate for producing the
canonical M6.1 metric.
