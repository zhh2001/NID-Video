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
def test_trainer_checkpoint_save_writes_full_pt_dict(synthetic_shards, tmp_path: Path) -> None:
    """After save_checkpoint the .pt file contains the full M4 schema:
    model + optimizer + scheduler + scaler + rng + counters + configs."""
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

    ckpt_path = res.checkpoint_paths[0]
    assert ckpt_path.is_file()
    assert ckpt_path.suffix == ".pt"
    assert "_step_" in ckpt_path.name

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    for key in ("schema_version", "model", "optimizer", "scheduler", "scaler",
                "epoch", "global_grad_step", "global_micro_step",
                "scheduler_config", "training_config", "rng"):
        assert key in ckpt, f"missing checkpoint field: {key}"
    assert set(ckpt["model"].keys()) == set(model.state_dict().keys())
    for sk in ("warmup_steps", "total_steps", "min_lr_ratio"):
        assert sk in ckpt["scheduler_config"]
    for rk in ("python", "numpy", "torch", "torch_cuda"):
        assert rk in ckpt["rng"]
    assert ckpt["schema_version"] == 1


@pytest.mark.slow
def test_trainer_export_model_safetensors_writes_weights_only(synthetic_shards, tmp_path: Path) -> None:
    """export_model_safetensors writes a deploy-only file with model weights
    and no optimizer/scheduler/etc state."""
    if not torch.cuda.is_available():
        pytest.skip("trainer tests require CUDA")
    cfg = _base_training_config(num_epochs=1)
    model = VideoMAESmallForNID(num_classes=13, pretrained=None,
                                gradient_checkpointing=False)
    loader = build_dataloader(synthetic_shards, batch_size=2, num_workers=0,
                              shuffle_buffer=0, pin_memory=False)
    trainer = Trainer(model=model, train_loader=loader, config=cfg,
                      run_dir=tmp_path / "run")
    out = tmp_path / "deploy.safetensors"
    trainer.export_model_safetensors(out)
    assert out.is_file()
    loaded = load_file(str(out))
    assert set(loaded.keys()) == set(model.state_dict().keys())


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


# ---------------------------------------------------------------------------
# M4 task 4.4: resume capability
# ---------------------------------------------------------------------------


class _FixedSamples(torch.utils.data.Dataset):
    """Map-style dataset of N deterministic (tensor, label) pairs.

    We use a Dataset (not IterableDataset/webdataset) for the resume tests
    because resume determinism requires the dataloader to produce the SAME
    samples in the SAME order on each .train() call. Webdataset shard order
    + worker splitting is harder to pin to that contract; a tiny map-style
    dataset is deterministic by construction with shuffle=False.
    """

    def __init__(self, n: int = 5, seed: int = 123) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.tensors = [torch.randn(16, 6, 32, 64, generator=gen) for _ in range(n)]
        self.labels = [int(i % 13) for i in range(n)]

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> dict:
        return {
            "tensor": self.tensors[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "meta": {},
        }


def _build_trainer_for_resume_test(tmp_path: Path, *, run_subdir: str = "run",
                                    n_samples: int = 5) -> Trainer:
    """Construct a CPU-friendly Trainer for the resume test. Forces vanilla
    AdamW + FP32 + no GC so determinism doesn't depend on GradScaler/8-bit
    optimizer state."""
    from nid_video.data.dataset import _collate
    cfg = _base_training_config(
        precision="fp32",
        optimizer="adamw",
        gradient_checkpointing=False,
        batch_size=1,
        grad_accumulation=1,
        num_epochs=2,
    )
    model = VideoMAESmallForNID(num_classes=13, pretrained=None,
                                gradient_checkpointing=False)
    ds = _FixedSamples(n=n_samples)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=_collate,
    )
    return Trainer(
        model=model, train_loader=loader, config=cfg,
        device="cpu", run_dir=tmp_path / run_subdir,
        warmup_steps=2, total_steps=20, min_lr_ratio=0.1,
    )


def _seed_all(seed: int = 42) -> None:
    import random as _r
    import numpy as _np
    _r.seed(seed); _np.random.seed(seed); torch.manual_seed(seed)


def test_save_load_state_dict_roundtrip(tmp_path: Path) -> None:
    """Single ckpt round-trip: build → save → fresh build → load. After load,
    optimizer/scheduler/counter state must match the saver's."""
    _seed_all(42)
    t = _build_trainer_for_resume_test(tmp_path, run_subdir="r1")
    t.train(num_epochs=1, log_every=10)             # 5 steps
    saved_ckpt = t.run_dir / "ckpt"
    files = sorted(saved_ckpt.glob("epoch_*_step_*.pt"))
    assert len(files) == 1, f"expected 1 ckpt, got {files}"

    saved_grad_step = t.global_grad_step
    saved_lr = t.scheduler.get_last_lr()

    _seed_all(0)            # different seed; load must override
    t2 = _build_trainer_for_resume_test(tmp_path, run_subdir="r2")
    t2.load_checkpoint(files[0])

    assert t2.global_grad_step == saved_grad_step
    assert t2.scheduler.get_last_lr() == saved_lr
    # AdamW exp_avg / exp_avg_sq tensors restored
    for sg in t2.optimizer.state_dict()["state"].values():
        for k in ("exp_avg", "exp_avg_sq"):
            assert k in sg


def test_resume_continues_at_correct_global_step(tmp_path: Path) -> None:
    _seed_all(42)
    t = _build_trainer_for_resume_test(tmp_path, run_subdir="r1")
    t.train(num_epochs=1, log_every=10)             # 5 grad steps
    ckpt = sorted((t.run_dir / "ckpt").glob("*.pt"))[-1]

    t2 = _build_trainer_for_resume_test(tmp_path, run_subdir="r2")
    assert t2.global_grad_step == 0
    t2.load_checkpoint(ckpt)
    assert t2.global_grad_step == 5     # not 0 — we resumed at step 5


def test_resume_continues_at_correct_lr(tmp_path: Path) -> None:
    _seed_all(42)
    t = _build_trainer_for_resume_test(tmp_path, run_subdir="r1")
    t.train(num_epochs=1, log_every=10)
    saved_lr = t.scheduler.get_last_lr()
    ckpt = sorted((t.run_dir / "ckpt").glob("*.pt"))[-1]

    t2 = _build_trainer_for_resume_test(tmp_path, run_subdir="r2")
    t2.load_checkpoint(ckpt)
    assert t2.scheduler.get_last_lr() == saved_lr


def test_resume_rng_state_produces_same_random_draw(tmp_path: Path) -> None:
    """After load_checkpoint, draws from torch / numpy / random should
    proceed from the same point in the RNG stream as if the saver had
    continued — specifically, the post-load draws must match the saver's
    immediately-after-save draws."""
    import random as _r
    import numpy as _np

    _seed_all(42)
    t = _build_trainer_for_resume_test(tmp_path, run_subdir="r1")
    t.train(num_epochs=1, log_every=10)
    ckpt = sorted((t.run_dir / "ckpt").glob("*.pt"))[-1]

    # After save, draw three different RNG streams.
    py_draw_ref = _r.random()
    np_draw_ref = _np.random.rand()
    torch_draw_ref = torch.rand(1).item()

    # Resume: load_checkpoint restores the snapshot taken at save time.
    t2 = _build_trainer_for_resume_test(tmp_path, run_subdir="r2")
    t2.load_checkpoint(ckpt)
    py_draw_resumed = _r.random()
    np_draw_resumed = _np.random.rand()
    torch_draw_resumed = torch.rand(1).item()

    assert py_draw_resumed == pytest.approx(py_draw_ref)
    assert np_draw_resumed == pytest.approx(np_draw_ref)
    assert torch_draw_resumed == pytest.approx(torch_draw_ref)


def test_resume_produces_identical_loss_curve(tmp_path: Path) -> None:
    """The fingerprint test: a continuous 10-step run vs (5-step + ckpt
    round-trip + 5-step) must produce element-wise identical loss values
    to within 1e-5. M4 task 4.4 hard acceptance criterion."""
    # --- Run A: continuous 10 steps (2 epochs of 5 samples each) ---
    _seed_all(42)
    a = _build_trainer_for_resume_test(tmp_path, run_subdir="run_a")
    a.train(num_epochs=2, log_every=10)
    losses_a = list(a.step_losses)

    # --- Run B: 5 steps → save → load into fresh trainer → 5 more steps ---
    _seed_all(42)
    b1 = _build_trainer_for_resume_test(tmp_path, run_subdir="run_b1")
    b1.train(num_epochs=1, log_every=10)
    losses_b_first = list(b1.step_losses)
    ckpt = sorted((b1.run_dir / "ckpt").glob("*.pt"))[-1]

    # New trainer (different seed; load must override).
    _seed_all(0)
    b2 = _build_trainer_for_resume_test(tmp_path, run_subdir="run_b2")
    b2.load_checkpoint(ckpt)
    b2.train(num_epochs=2, log_every=10)        # already done epoch 0; runs epoch 1 only
    losses_b_second = list(b2.step_losses)

    losses_b = losses_b_first + losses_b_second
    assert len(losses_a) == len(losses_b) == 10, (
        f"len mismatch — a={len(losses_a)} b={len(losses_b)}"
    )
    for i, (la, lb) in enumerate(zip(losses_a, losses_b)):
        assert abs(la - lb) < 1e-5, (
            f"step {i}: continuous loss={la:.7f} vs resumed loss={lb:.7f} "
            f"(|diff|={abs(la - lb):.2e}, threshold=1e-5)"
        )


def test_load_checkpoint_unknown_field_does_not_raise(tmp_path: Path) -> None:
    """Forward-compat: an unknown extra field in the ckpt dict must NOT
    raise. Future schema additions can land in old code without breaking it."""
    _seed_all(42)
    t = _build_trainer_for_resume_test(tmp_path, run_subdir="r1")
    t.train(num_epochs=1, log_every=10)
    ckpt_path = sorted((t.run_dir / "ckpt").glob("*.pt"))[-1]
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt["fancy_new_field_from_the_future"] = {"extra": [1, 2, 3]}
    torch.save(ckpt, str(ckpt_path))

    t2 = _build_trainer_for_resume_test(tmp_path, run_subdir="r2")
    t2.load_checkpoint(ckpt_path)              # must not raise
    assert t2.global_grad_step == 5


def test_load_checkpoint_newer_schema_version_raises(tmp_path: Path) -> None:
    """Forward-compat boundary: a ckpt from a future schema_version we don't
    understand must raise explicitly. Silent best-effort would corrupt state."""
    _seed_all(42)
    t = _build_trainer_for_resume_test(tmp_path, run_subdir="r1")
    t.train(num_epochs=1, log_every=10)
    ckpt_path = sorted((t.run_dir / "ckpt").glob("*.pt"))[-1]
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt["schema_version"] = 999
    torch.save(ckpt, str(ckpt_path))

    t2 = _build_trainer_for_resume_test(tmp_path, run_subdir="r2")
    with pytest.raises(ValueError, match="schema_version"):
        t2.load_checkpoint(ckpt_path)


@pytest.mark.slow
def test_grad_accumulation_math_skipped() -> None:
    """TODO: Verify that batch=2/accum=4 vs batch=8/accum=1 produces the same
    parameter update direction. Deferred — needs deterministic sample ordering
    + identical optimizer states across runs, which interacts subtly with
    bitsandbytes' 8-bit quantized state. Revisit in M4 alongside resume."""
    pytest.skip("TODO M4: rigorous accumulation-equivalence check")
