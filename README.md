# nid-video

> Network intrusion detection by representing traffic as video.
> A 5-tuple flow window becomes a `(T=16, C=6, H=32, W=64)` tensor and is fed
> to a VideoMAE-Small backbone, so the model sees how communication patterns
> co-evolve in space and time — not just static snapshots.

The full design rationale lives in [docs/Idea.md](docs/Idea.md). This README
only covers how to set the project up and run things.

---

## Requirements

| | |
|---|---|
| OS | Windows 11 + WSL2 (Ubuntu 24.04) |
| Python | 3.10 |
| Package manager | [`uv`](https://docs.astral.sh/uv/) ≥ 0.11 |
| NVIDIA driver | Modern driver supporting CUDA 13 (tested on 595.97) |
| GPU | 1 × NVIDIA GPU with ≥ 8 GB VRAM (tested on RTX 4060 Mobile) |
| Disk | ≥ 30 GB free for the recommended CIC-IDS 2017 subset |

CUDA toolkit on the host is not required — the `torch==2.11.0+cu130` wheel
ships its own CUDA libraries. The driver alone is enough.

> If you only have CUDA 12.x driver/runtime, swap the index URL in
> `pyproject.toml` to `https://download.pytorch.org/whl/cu121` (or `cu124`)
> and rerun `uv sync`. The rest of the project is CUDA-version-agnostic.

---

## Quick start

```bash
# 1. Get the code
git clone <repo-url> nid-video
cd nid-video

# 2. Install dependencies (creates .venv automatically)
uv sync

# 3. Verify the toolchain (config + logger + CUDA matmul + ETL pipeline)
uv run pytest tests/ -v

# 4. (Optional, ~30 GB) Download the CIC-IDS 2017 Tue/Wed/Fri subset.
#    The CIC mirror gates downloads behind a free registration. Register at
#    https://www.unb.ca/cic/datasets/ids-2017.html, log into cicresearch.ca,
#    and copy the `Token` cookie value, then:
uv run python scripts/download_cicids2017.py --dry-run
uv run python scripts/download_cicids2017.py --cookie-token "$CIC_TOKEN" --yes

# 5. (After step 4) Run ETL: pcap -> (T,C,H,W) webdataset shards.
#    The CSV ZIPs from step 4 must be unpacked first (TrafficLabelling/*.csv).
#    Use --num-workers 3 to dispatch one worker per pcap.
uv run python scripts/run_etl.py \
    --pcap-dir data/raw/cicids2017/PCAPs \
    --label-dir data/raw/cicids2017/TrafficLabelling \
    --output-dir data/processed/cicids2017 \
    --csv-dayfirst \
    --num-workers 3

# 6. Train. Three scenarios — pick one.

# 6a. Real M4 training (multi-scale + splits + per-epoch eval, default config):
#     ETL twice first (Δt=100ms uses configs/base.yaml; Δt=1s uses
#     configs/data_dt1000ms.yaml — see step 5 with --output-dir cicids2017_dt100ms_v2
#     and a second pass with --config configs/data_dt1000ms.yaml --output-dir
#     cicids2017_dt1000ms_v2). Then build splits from the 100ms shards:
uv run python scripts/run_split.py \
    --shard-pattern "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \
    --label-dir data/raw/cicids2017/TrafficLabelling \
    --output data/processed/cicids2017_dt100ms_v2/splits.parquet \
    --csv-dayfirst

uv run python scripts/train.py \
    --shard-pattern-fast "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \
    --shard-pattern-slow "data/processed/cicids2017_dt1000ms_v2/*/shards/shard-*.tar" \
    --splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet \
    --num-epochs 20
# Defaults: configs/training_perf.yaml (B=32 / accum=1 / workers=2),
#           track-best=macro_f1, mix-ratio=0.5, warmup-steps=500.
# Outputs: outputs/run_<ts>/ckpt/epoch_<N>_step_<M>.pt + best.pt (highest val macro-F1).

# 6b. Smoke test (single-scale, FP32, no eval — CI / sanity baseline):
uv run python scripts/train.py \
    --config configs/base.yaml \
    --shard-pattern "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \
    --debug --max-steps 20

# 6c. Eval-only on a saved checkpoint (defaults to val split; --keep-split test for held-out):
uv run python scripts/train.py \
    --shard-pattern-fast "data/processed/cicids2017_dt100ms_v2/*/shards/shard-*.tar" \
    --shard-pattern-slow "data/processed/cicids2017_dt1000ms_v2/*/shards/shard-*.tar" \
    --splits-path data/processed/cicids2017_dt100ms_v2/splits.parquet \
    --eval-only --resume outputs/run_<ts>/ckpt/best.pt \
    [--keep-split test]
```

After step 3 you should see `191 passed` on the fast tier. After step 4 you'll have:

```
data/raw/cicids2017/
├── PCAPs/{Tuesday,Wednesday,Friday}-*.pcap
└── CSVs/{GeneratedLabelledFlows,MachineLearningCSV}.zip
```

After step 5 (~80 min wall clock for 100ms with 3 workers on full Tue+Wed+Fri pcaps;
much smaller for 1s):

```
data/processed/cicids2017_dt100ms_v2/
├── <pcap_stem>/shards/shard-NNNNNN.tar    # ~770 KB/sample (raw float32, no tar gzip), ~1000/shard
└── <pcap_stem>/manifest.parquet            # per-shard label distribution
```

After step 6 the M4 trainer writes a full-state ``.pt`` checkpoint per epoch
(model + optimizer + scheduler + scaler + RNG, supports ``--resume``) plus a
``best.pt`` copy whenever val ``macro_f1`` strictly improves:

```
outputs/run_<timestamp>/ckpt/
├── epoch_{N}_step_{M}.pt                   # full state, ~few hundred MB / epoch
└── best.pt                                  # copy of best-macro-F1 epoch
```

---

## Project layout

```
nid-video/
├── configs/             # OmegaConf YAML configs (validated by pydantic; supports `extends:`)
│   ├── base.yaml              # CI / smoke / sanity baseline (B=2, accum=16)
│   └── training_perf.yaml     # production throughput (B=32, accum=1, workers=2)
├── data/
│   ├── raw/             # pcaps land here (gitignored)
│   └── processed/       # ETL output webdataset shards (gitignored)
├── outputs/             # training logs + checkpoints (gitignored)
├── docs/
│   ├── etl_performance.md     # M2 ETL throughput
│   └── m3_perf.md             # M3 memory + throughput sweep
├── scripts/
│   ├── download_cicids2017.py
│   ├── run_etl.py                          # pcap → shards
│   ├── run_split.py                        # shards + LabelIndex → splits.parquet (M4)
│   ├── diagnose_unparseable_timestamps.py  # CIC CSV forensics tool
│   └── train.py                            # M4: multi-scale / resume / eval-only
├── src/nid_video/
│   ├── data/            # ETL stages + dataset adapter
│   │   ├── pcap_parser.py     # dpkt-backed PacketStream
│   │   ├── windowing.py       # SlidingWindow → Window/Frame
│   │   ├── channels.py        # encode_window → (T,C,H,W) tensor
│   │   ├── ip_clustering.py   # per-window k-means on source IPs
│   │   ├── labeling.py        # CIC TrafficLabelling alignment
│   │   ├── etl_pipeline.py    # run_etl + manifest
│   │   ├── dataset.py         # NidShardDataset + MultiScaleNidDataset    (M3, M4)
│   │   └── split.py           # train/val/test by time-based partition    (M4)
│   ├── models/
│   │   └── videomae_nid.py    # VideoMAE-S + scale-token (3-ch→6-ch, 16→8 patch) (M3, M4)
│   ├── trainer/
│   │   ├── trainer.py         # FP16 + AMP + AdamW + grad-accum + resume  (M3, M4)
│   │   ├── scheduler.py       # cosine LR with warmup                      (M4)
│   │   └── evaluator.py       # macro-F1 / per-class / AUROC / CM          (M4)
│   └── utils/
│       ├── config.py    # OmegaConf load + extends + pydantic validate
│       └── logger.py    # loguru wrapper
└── tests/               # 191 tests in two tiers: -m "not slow" (fast, ~7m) / full (~13m)
```

---

## Configuration

All hyperparameters live in `configs/base.yaml` and are validated by the
pydantic schema in `src/nid_video/utils/config.py`. Adding a key the schema
doesn't know about will fail loudly — this is intentional, the
`(T, C, H, W) = (16, 6, 32, 64)` tensor shape and the 6-channel semantics
are project contracts, not knobs.

Override at runtime with the standard OmegaConf pattern:

```python
from nid_video.utils import load_config
cfg = load_config("configs/base.yaml")
```

---

## Milestone status

| | | |
|---|---|---|
| M1 | Project skeleton + dependencies + config + smoke tests + CIC download script | ✅ done |
| M2 | ETL: pcap → `(T,C,H,W)` shards (per-window k-means, motion channels, webdataset) | ✅ done |
| M3 | VideoMAE-S adaptation + dataset/dataloader + minimal trainer (FP16+AMP+8bit AdamW), 1-epoch smoke | ✅ done |
| M4 | Multi-scale + scale token + LR scheduler + resume + eval + time-based split + 12h-CIC fix | ⏳ in progress |
| M5 | Baselines (C3D / I3D / R(2+1)D / TimeSformer / 1D-Transformer) + cross-dataset transfer | – |
| M6 | Core ablations + paper figures + deployment efficiency | – |

---

## Development

```bash
# Tests come in two tiers:
uv run pytest -m "not slow"      # fast (≈7m, dev-iteration loop; 191 tests)
uv run pytest                    # full (+~13 slow tests for real-HF / ETL / trainer-CUDA; pre-commit / CI)

uv run ruff check .              # lint
uv run ruff format .             # format
uv run mypy src/                 # type-check
```

The `slow` marker tags end-to-end ETL tests that build synthetic pcaps via
scapy and run the full pipeline; see `pyproject.toml` for the registration.

For perf characteristics and projections to the full CIC-IDS subset, see
[`docs/etl_performance.md`](docs/etl_performance.md).
