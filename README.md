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

# 3. Verify the toolchain (config + logger + CUDA matmul)
uv run pytest tests/ -v

# 4. (Optional, ~30 GB) Download the CIC-IDS 2017 Tue/Wed/Fri subset.
#    The CIC mirror gates downloads behind a free registration. Register at
#    https://www.unb.ca/cic/datasets/ids-2017.html, log into cicresearch.ca,
#    and copy the `Token` cookie value, then:
uv run python scripts/download_cicids2017.py --dry-run
uv run python scripts/download_cicids2017.py --cookie-token "$CIC_TOKEN" --yes
```

After step 3 you should see `4 passed`. After step 4 you'll have:

```
data/raw/cicids2017/
├── PCAPs/{Tuesday,Wednesday,Friday}-*.pcap
└── CSVs/{GeneratedLabelledFlows,MachineLearningCSV}.zip
```

---

## Project layout

```
nid-video/
├── configs/             # OmegaConf YAML configs (validated by pydantic)
│   └── base.yaml
├── data/
│   ├── raw/             # pcaps land here (gitignored)
│   └── processed/       # ETL output tensors (gitignored)
├── outputs/             # training logs + checkpoints (gitignored)
├── scripts/
│   └── download_cicids2017.py
├── src/nid_video/
│   ├── data/            # ETL: pcap → (T,C,H,W) tensor   (M2)
│   ├── models/          # VideoMAE backbone + heads      (M3)
│   ├── trainer/         # training loop + eval            (M4)
│   └── utils/
│       ├── config.py    # OmegaConf load + pydantic validate
│       └── logger.py    # loguru wrapper
└── tests/
    └── test_smoke.py    # config / logger / CUDA matmul
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
| M2 | ETL: pcap → `(T, C, H, W)` tensor (k-means IP buckets, port columns, motion channels) | ⏳ next |
| M3 | VideoMAE-Small backbone integration + tube-patch tokenizer | – |
| M4 | Training loop (FP16 + 8-bit AdamW + grad checkpointing) on 8 GB VRAM | – |
| M5 | Evaluation: in-domain + cross-dataset transfer | – |
| M6 | Ablations + paper figures | – |

---

## Development

```bash
uv run pytest tests/ -v          # run all tests
uv run ruff check .              # lint
uv run ruff format .             # format
uv run mypy src/                 # type-check
```
