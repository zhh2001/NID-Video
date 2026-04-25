"""Trainer tests — all marked slow (each spins up the real backbone)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import webdataset as wds
from safetensors.torch import load_file

from nid_video.data.dataset import build_dataloader
from nid_video.models.videomae_nid import VideoMAESmallForNID
from nid_video.trainer.trainer import Trainer, TrainResult
from nid_video.utils.config import TrainingConfig


# ---------------------------------------------------------------------------
# Fixtures: synthetic shards with a learnable signal
# ---------------------------------------------------------------------------


def _build_synthetic_shards(out_dir: Path, n_samples: int = 64,
                            label_id: int = 0, maxcount: int = 16) -> str:
    """Write n_samples shards. Every sample carries the same label so the
    model can converge fast (the loss-down test only needs a learnable target,
    not class-balance)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "shard-%06d.tar")
    rng = np.random.default_rng(0)
    with wds.ShardWriter(pattern, maxcount=maxcount) as w:
        for i in range(n_samples):
            tensor = rng.standard_normal((16, 6, 32, 64), dtype=np.float32)
            w.write({
                "__key__": f"{i:010d}",
                "tensor.npy": tensor,
                "label.cls": label_id,
                "meta.json": {
                    "start_time": 0.0, "pcap_source": "x.pcap",
                    "label": "BENIGN", "label_id": label_id,
                    "dominant_attack_ratio": 1.0, "n_unmatched": 0,
                },
            })
    return str(out_dir / "shard-*.tar")


@pytest.fixture
def synthetic_shards(tmp_path: Path) -> str:
    return _build_synthetic_shards(tmp_path / "shards", n_samples=32, maxcount=8)


def _base_training_config(**override) -> TrainingConfig:
    base = dict(
        batch_size=2,
        grad_accumulation=2,
        num_epochs=1,
        lr=1.5e-4,
        weight_decay=0.05,
        optimizer="adamw_8bit",
        precision="fp16",
        gradient_checkpointing=True,
    )
    base.update(override)
    return TrainingConfig(**base)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_trainer_runs_5_steps_no_nan_inf(synthetic_shards, tmp_path: Path) -> None:
    """With 5 micro-steps, the loss must stay finite (not NaN/Inf)."""
    if not torch.cuda.is_available():
        pytest.skip("trainer tests require CUDA (8-bit AdamW + AMP)")
    cfg = _base_training_config(num_epochs=10)   # plenty for max_steps cap
    model = VideoMAESmallForNID(num_classes=13, pretrained=None,
                                gradient_checkpointing=False)
    loader = build_dataloader(synthetic_shards, batch_size=cfg.batch_size,
                              num_workers=0, shuffle_buffer=0,
                              pin_memory=False)
    trainer = Trainer(model=model, train_loader=loader, config=cfg,
                      run_dir=tmp_path / "run")
    res = trainer.train(max_steps=5, log_every=1)
    assert res.micro_steps >= 1
    assert torch.isfinite(torch.tensor(res.final_avg_loss)), \
        f"loss not finite: {res.final_avg_loss}"


@pytest.mark.slow
def test_trainer_checkpoint_save_and_load(synthetic_shards, tmp_path: Path) -> None:
    """After save_checkpoint, load_file should round-trip the state_dict."""
    if not torch.cuda.is_available():
        pytest.skip("trainer tests require CUDA")
    cfg = _base_training_config(num_epochs=1)
    model = VideoMAESmallForNID(num_classes=13, pretrained=None,
                                gradient_checkpointing=False)
    loader = build_dataloader(synthetic_shards, batch_size=2, num_workers=0,
                              shuffle_buffer=0, pin_memory=False)
    trainer = Trainer(model=model, train_loader=loader, config=cfg,
                      run_dir=tmp_path / "run")
    res = trainer.train(max_steps=4, log_every=1)
    assert len(res.checkpoint_paths) >= 1

    ckpt = res.checkpoint_paths[0]
    assert ckpt.is_file()
    loaded = load_file(str(ckpt))
    # Same key set as the model state_dict
    model_state = model.state_dict()
    assert set(loaded.keys()) == set(model_state.keys())
    # And the largest tensor matches in shape
    for k, v in model_state.items():
        if v.numel() > 1000:
            assert loaded[k].shape == v.shape, f"shape mismatch {k}"
            break


@pytest.mark.slow
def test_trainer_debug_fp32_path_works(synthetic_shards, tmp_path: Path) -> None:
    """`--debug` overrides: FP32 + vanilla AdamW + no grad checkpointing."""
    if not torch.cuda.is_available():
        pytest.skip("trainer tests require CUDA")
    cfg = _base_training_config(precision="fp32", optimizer="adamw",
                                 gradient_checkpointing=False, num_epochs=1)
    model = VideoMAESmallForNID(num_classes=13, pretrained=None,
                                gradient_checkpointing=False)
    loader = build_dataloader(synthetic_shards, batch_size=2, num_workers=0,
                              shuffle_buffer=0, pin_memory=False)
    trainer = Trainer(model=model, train_loader=loader, config=cfg,
                      run_dir=tmp_path / "run")
    assert trainer.amp_enabled is False
    assert trainer.scaler is None
    res = trainer.train(max_steps=4, log_every=1)
    assert torch.isfinite(torch.tensor(res.final_avg_loss))


@pytest.mark.slow
def test_grad_accumulation_math_skipped() -> None:
    """TODO: Verify that batch=2/accum=4 vs batch=8/accum=1 produces the same
    parameter update direction. Deferred — needs deterministic sample ordering
    + identical optimizer states across runs, which interacts subtly with
    bitsandbytes' 8-bit quantized state. Revisit in M4 alongside resume."""
    pytest.skip("TODO M4: rigorous accumulation-equivalence check")
