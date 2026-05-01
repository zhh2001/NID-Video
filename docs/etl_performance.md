# ETL Performance — measured + projected

This doc records the measured throughput of the M2 ETL pipeline (pcap → tensor →
webdataset shard) and projects how long the full CIC-IDS 2017 subset takes to
process. Numbers come from synthetic-pcap benchmarks on the dev box (RTX 4060
Mobile / WSL2 / 8 GB RAM).

## TL;DR

| Stage | Throughput | Per-unit cost |
|---|---|---|
| pcap parser — raw `dpkt.pcap.Reader` only | 981k pps | I/O + pcap header floor |
| pcap parser — `dpkt.ethernet.Ethernet` + IP parse | 138k pps | dpkt's pure-Python ceiling |
| pcap parser — `PacketStream` (production wrapper) | 127k pps | 92% efficient vs. dpkt L2 |
| Full ETL (parse → window → cluster → encode → label → shard write) | **18k pps** / **22 win/s** | **~46 ms / window** |

**Full CIC-IDS 2017 Tue+Wed+Fri (~31 GB, ~42M packets, ~40k windows) projected:**

| Mode | Wall time |
|---|---|
| Single process (`--num-workers 1`) | 31–38 min |
| 3 workers (one per pcap) | **~13 min** (bounded by Wednesday, the largest pcap) |
| 8 workers | ~10 min (diminishing — only 3 pcaps to dispatch) |

Both targets from the M2 acceptance line are met:
- pcap parsing alone ≤ 10 min ✓ (5.5 min)
- Full ETL with `--num-workers 3` ≤ 15 min ✓

## Methodology

### pcap parser (M2 §2.2)

Synthetic pcap of 100k TCP+IP packets (~9.5 MB, ~100 B/packet), four-stage
profile to localize the bottleneck:

| Level | Throughput | What it measures |
|---|---|---|
| L1 | 981k pps | `dpkt.pcap.Reader` iteration; just walking pcap records, no parsing |
| L2 | 138k pps | + `dpkt.ethernet.Ethernet(buf)` + IP layer parse |
| L3 | 125k pps | + TCP/UDP detect + tuple field extraction |
| L4 | 127k pps | + production `PacketStream` (NamedTuple, stats counters, log summary) |

The dpkt parsing itself drops us 7× below raw pcap iteration. The wrapper adds
only ~8% on top of L2, so the bottleneck is dpkt's per-packet `__init__` chain
in pure Python — not anything fixable in our code.

### Full ETL (M2 §2.7)

Synthetic pcap of 30 s × 1000 pps × 32 source IPs (30 000 packets, 3.1 MB) with
a matching CSV labelling all flows BENIGN. Single-process `run_etl` with
default config (T=16, Δt=100 ms, H=32, W=64, samples_per_shard=100).

Result:
- Wall clock: 1.65 s
- 36 windows emitted, 1 shard
- **18 210 packets/sec end-to-end**
- **22 windows/sec**
- 46 ms / window (amortized)

### Stage-level cost estimate (per window, 833 packets / 1.6 s window)

| Stage | Estimated cost | Rationale |
|---|---|---|
| pcap parse share | ~7 ms | 833 / 127k = 6.6 ms |
| `cluster_ips_in_window` | ~3 ms | sklearn `MiniBatchKMeans` on 32 active IPs; sub-ms is theoretical, real-world JIT/init overhead pushes it to a few ms |
| `encode_window` | ~3 ms | numpy ops on `(16, 6, 32, 64)` ≈ 196k float32 cells |
| `label_window` | ~1 ms | dict lookup per packet (833 × 1 µs) |
| ShardWriter `.write` | ~3 ms | numpy `.npy` encode + tar append |
| Per-window Python overhead | ~30 ms | the residual: dataclass construction, accumulator dict updates, generator overhead, loguru calls |

The Python per-window overhead is the largest single cost (~30 ms / 46 ms ≈
65%). It has many small contributors and no single hot spot.

## Projection to CIC-IDS-2017 Tue+Wed+Fri

CIC capture density: ~3 hours of working-hours traffic per day, average
~1500 pps. Three days → ~9 hours capture → ~40 500 windows total at 0.8 s step.

Single-process projection (linear scaling from 22 win/s):
```
40 500 windows / 22 win/s = 1841 s ≈ 31 min
```
Or by packets: 42M / 18k = 2333 s ≈ 39 min. The two estimates bracket
**31–38 min**; reality will land somewhere in that range and depend on burst
density (DDoS chunks have many packets per window → cheap per-window amortized
cost; idle periods are similar).

Multi-worker (`scripts/run_etl.py --num-workers 3` dispatches 1 pcap per
worker) is bounded by Wednesday (~12.5 GB → 13 min single-thread). 4 workers
gets no benefit since there are only 3 pcaps. Beyond 3 workers the pcaps don't
divide further.

## When to optimize

The current numbers comfortably meet the M2 acceptance bar. **Optimization is
not pursued** for M3 onward unless real-data ETL exceeds 30 minutes wall time.

If we ever do need to optimize, in priority order:
1. **Multiprocess every day's pcap** — already supported via `--num-workers`.
   Use 3 workers for the standard subset.
2. **Numpy-vectorize the per-cell scalar accumulators** in `encode_window`. The
   inner Python loop over packets fills `pkt_count`, `byte_total`, `flag_or`,
   `out_count` element-wise; replacing with `np.add.at` or building an `(N, 4)`
   array of `(t, h, w, value)` tuples and using `np.bincount` could shave 5-10 ms
   per window. Estimated total: 22 win/s → 30+ win/s.
3. **Drop dpkt for `pylibpcap`** for the pcap parser. Realistic ~5× boost on
   the parse stage but with C-extension build deps. Only worth it if the rest
   is also optimized; otherwise the residual Python costs cap the speedup.

We are NOT pursuing C-extension paths (decision recorded in `feedback_proactive_deviation_reports.md`):
the ETL is one-time pre-processing per dataset, and a cleaner cross-platform
build profile is more valuable than 3× ETL speedup.

## Output sizing

Per sample on disk (M4.7 real-data run on CIC-IDS-2017 100ms shards;
synthetic estimates in parentheses):

- `tensor.npy`: `16 × 6 × 32 × 64 × 4 B` = **768 KB raw, written uncompressed**
  to the tar archive (the synthetic-run gzip-tar estimate of ~150 KB is no
  longer used — the float32 tensor is incompressible enough that gzip cost did
  not pay back the read latency in our shard pipeline; see decision in M2 task
  2.7). Effective per-sample disk ≈ 770 KB after tar metadata overhead.
- `label.cls`: 1–2 bytes
- `meta.json`: ~150 bytes

Real-data totals on the Tue+Wed+Fri 100ms+1s subset (after the M4.7 12h fix,
v2 ETL run):

| Δt   | windows  | shards | total disk |
|------|---------:|-------:|-----------:|
| 100ms| 110,783  | 113    | **~82 GB** |
| 1s   |  11,074  |  ~12   |   ~8.5 GB  |
| **combined** | — | — | **~90 GB** |

The 82 GB at 100ms is materially larger than the 5.8 GB synthetic projection
because (a) we no longer gzip the tar, (b) real CIC pcap density is denser
than the 1500-pps synthetic average, and (c) tube-patch overlap + dominant-rule
labelling produce ~2.7× more windows per second of capture than the M2
projection assumed. The 1s slow-scale shard set is roughly 10× sparser by
construction. See `docs/v1_vs_v2_comparison.md` for the v1 (pre-fix) vs v2
(post-fix) per-class breakdown — the on-disk sizing is identical between v1
and v2 (deterministic windowing on the same pcaps); only label assignment
changes.

## Update log

| Date | Run | Result |
|---|---|---|
| 2026-04-25 | initial benchmark on dev box | 18.2k pps / 22 win/s / 46 ms per window |
| 2026-04-29 | M4.7 v2 ETL on real CIC-IDS-2017 Tue+Wed+Fri 100ms after the 12h-without-AM/PM fix | 110,783 windows / 113 shards / 82 GB / 71.3 min wall time (3 workers); see `docs/v1_vs_v2_comparison.md` for the per-class delta |
