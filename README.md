# nid-video

> Network intrusion detection by representing traffic as video.
> A 5-tuple flow window becomes a `(T=16, C=6, H=32, W=64)` tensor and is fed
> to a VideoMAE-Small backbone, so the model sees how communication patterns
> co-evolve in space and time — not just static snapshots.

The full design rationale lives in `prompts/Idea.md` (untracked). This README
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
```

After step 3 you should see `72 passed`. After step 4 you'll have:

```
data/raw/cicids2017/
├── PCAPs/{Tuesday,Wednesday,Friday}-*.pcap
└── CSVs/{GeneratedLabelledFlows,MachineLearningCSV}.zip
```

After step 5 (~13 min wall clock with 3 workers; see [docs/etl_performance.md](docs/etl_performance.md)):

```
data/processed/cicids2017/
├── <pcap_stem>/shards/shard-NNNNNN.tar    # ~150 KB/sample, ~1000/shard
└── <pcap_stem>/manifest.parquet            # per-shard label distribution
```

---

## Project layout

```
nid-video/
├── configs/             # OmegaConf YAML configs (validated by pydantic)
│   └── base.yaml
├── data/
│   ├── raw/             # pcaps land here (gitignored)
│   └── processed/       # ETL output webdataset shards (gitignored)
├── outputs/             # training logs + checkpoints (gitignored)
├── docs/
│   └── etl_performance.md
├── scripts/
│   ├── download_cicids2017.py
│   └── run_etl.py
├── src/nid_video/
│   ├── data/            # ETL stages (M2)
│   │   ├── pcap_parser.py     # dpkt-backed PacketStream
│   │   ├── windowing.py       # SlidingWindow → Window/Frame
│   │   ├── channels.py        # encode_window → (T,C,H,W) tensor
│   │   ├── ip_clustering.py   # per-window k-means on source IPs
│   │   ├── labeling.py        # CIC TrafficLabelling alignment
│   │   └── etl_pipeline.py    # run_etl + manifest
│   ├── models/          # VideoMAE backbone + heads      (M3)
│   ├── trainer/         # training loop + eval            (M4)
│   └── utils/
│       ├── config.py    # OmegaConf load + pydantic validate
│       └── logger.py    # loguru wrapper
└── tests/               # 72 tests; mark `slow` for end-to-end ETL (~10s)
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
| M3 | VideoMAE-Small backbone integration + tube-patch tokenizer | ⏳ next |
| M4 | Training loop (FP16 + 8-bit AdamW + grad checkpointing) on 8 GB VRAM | – |
| M5 | Evaluation: in-domain + cross-dataset transfer | – |
| M6 | Ablations + paper figures | – |

---

## Development

```bash
# Tests come in two tiers:
uv run pytest -m "not slow"      # fast (≈3s, dev-iteration loop)
uv run pytest                    # full (≈8s, includes end-to-end ETL — pre-commit / CI)

uv run ruff check .              # lint
uv run ruff format .             # format
uv run mypy src/                 # type-check
```

The `slow` marker tags end-to-end ETL tests that build synthetic pcaps via
scapy and run the full pipeline; see `pyproject.toml` for the registration.

For perf characteristics and projections to the full CIC-IDS subset, see
[`docs/etl_performance.md`](docs/etl_performance.md).
