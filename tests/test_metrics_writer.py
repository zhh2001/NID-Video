"""Tests for src/nid_video/trainer/metrics_writer.py (M5.10 prep, Part 1).

Six unit tests pin the file-format contracts under both default and
``collect_grad_norm=True`` modes; one smoke test wires the writer to
the real Trainer + Evaluator with a tiny synthetic shard, runs 2
epochs at a few micro-steps each, and verifies the three artefacts
materialise on disk with the expected schema.

The smoke test is the only test that touches the real training loop;
the unit tests exercise the writer in isolation against mock metric
dicts. Both tiers are pinned because (a) future refactors of the
writer can break the JSONL schema independently of the trainer, and
(b) future refactors of the trainer hooks can break the call sequence
independently of the writer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nid_video.trainer.metrics_writer import MetricsWriter


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _mock_combined_metrics(macro_f1: float = 0.5) -> dict:
    """Build a minimal ``combined`` block matching the per_epoch.json
    schema. 13 classes (collapsed CIC) so future refactors that hardcode
    a different num_classes get a shape-mismatch test failure here."""
    names = ["BENIGN", "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye",
             "FTP-Patator", "SSH-Patator", "DoS slowloris",
             "DoS Slowhttptest", "Bot", "Web Attack", "Infiltration",
             "Heartbleed"]
    per_class = {
        n: {"f1": float(0.1 + i * 0.05), "p": 0.5, "r": 0.5,
            "auroc": 0.7, "n": 100 + i}
        for i, n in enumerate(names)
    }
    return {
        "n_samples": 18156,
        "accuracy": 0.95,
        "macro_f1": float(macro_f1),
        "auroc_macro": 0.78,
        "per_class": per_class,
    }


def _mock_confusion(num_classes: int = 13, fill: int = 0) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    if fill > 0:
        # Diagonal-heavy fake confusion to make the test data plausible.
        for i in range(num_classes):
            cm[i, i] = fill
    return cm


# ---------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------


def test_writer_creates_metrics_dir(tmp_path: Path) -> None:
    """Constructor must create ``<run_dir>/metrics/`` even when
    ``run_dir`` does not pre-exist (the trainer creates ckpt/ before
    instantiating the writer, but the writer must be self-sufficient
    for retrofit / standalone use)."""
    run_dir = tmp_path / "run_test"
    assert not run_dir.exists()
    w = MetricsWriter(run_dir)
    assert (run_dir / "metrics").is_dir()
    assert w.metrics_dir == run_dir / "metrics"
    # per_step.jsonl is touched (empty) on construction so re-runs
    # don't append to a stale file.
    assert (run_dir / "metrics" / "per_step.jsonl").is_file()
    w.finalize()


def test_writer_per_step_jsonl_schema_default(tmp_path: Path) -> None:
    """Default mode (collect_grad_norm=False): five log_step calls with
    grad_norm passed-but-ignored produce five rows that lack the
    grad_norm field. The default schema is exactly
    {step, epoch, loss, lr, wall_time_s} — no extras."""
    w = MetricsWriter(tmp_path / "run_default")
    for i in range(5):
        # Pass grad_norm even though collect_grad_norm=False — writer
        # must still drop the field. This is the critical pin.
        w.log_step(step=i + 1, epoch=0, loss=2.0 - 0.1 * i, lr=1e-4,
                   wall_time_s=0.31, grad_norm=12.0 + i)
    w.finalize()

    rows = [json.loads(line)
            for line in (tmp_path / "run_default" / "metrics" / "per_step.jsonl").read_text().splitlines()
            if line.strip()]
    assert len(rows) == 5
    expected_keys = {"step", "epoch", "loss", "lr", "wall_time_s"}
    for row in rows:
        assert set(row.keys()) == expected_keys, (
            f"default schema regressed: row keys = {set(row.keys())}; "
            f"expected exactly {expected_keys} (no grad_norm field)"
        )
        assert "grad_norm" not in row, (
            "grad_norm leaked into default-mode per_step.jsonl"
        )


def test_writer_per_step_jsonl_schema_with_grad_norm(tmp_path: Path) -> None:
    """``collect_grad_norm=True``: rows include the grad_norm field
    when a non-None value is supplied. Pass None on one of the five
    calls to verify the field is omitted on that row only (per-call
    optionality, not all-or-nothing)."""
    w = MetricsWriter(tmp_path / "run_grad", collect_grad_norm=True)
    for i in range(5):
        gn = 12.0 + i if i != 2 else None     # row 3 has no grad_norm
        w.log_step(step=i + 1, epoch=0, loss=2.0, lr=1e-4,
                   wall_time_s=0.3, grad_norm=gn)
    w.finalize()

    rows = [json.loads(line)
            for line in (tmp_path / "run_grad" / "metrics" / "per_step.jsonl").read_text().splitlines()
            if line.strip()]
    assert len(rows) == 5
    for i, row in enumerate(rows):
        if i == 2:
            assert "grad_norm" not in row, (
                f"row {i}: grad_norm should be omitted when caller "
                f"passes None even with collect_grad_norm=True"
            )
        else:
            assert "grad_norm" in row, (
                f"row {i}: collect_grad_norm=True + non-None gn → "
                f"field expected, got keys {set(row.keys())}"
            )
            assert row["grad_norm"] == pytest.approx(12.0 + i)


def test_writer_per_epoch_json_schema(tmp_path: Path) -> None:
    """log_epoch flushes a per_epoch.json with run_id + config + epochs
    list. Each epoch record carries epoch / grad_steps / wall_time_s /
    metrics; the metrics dict has at minimum a ``combined`` key whose
    value carries macro_f1 + per_class."""
    w = MetricsWriter(tmp_path / "run_epoch",
                      config={"model": "videomae_small", "lr": 1.5e-4})
    cm = _mock_confusion(fill=10)
    for ep in range(2):
        w.log_epoch(epoch=ep, grad_steps=4853 * (ep + 1),
                    wall_time_s=1237.0,
                    metrics={"combined": _mock_combined_metrics(0.5 + 0.1 * ep)},
                    confusion=cm)
    w.finalize()

    payload = json.loads(
        (tmp_path / "run_epoch" / "metrics" / "per_epoch.json").read_text()
    )
    assert payload["run_id"] == "run_epoch"
    assert payload["config"]["model"] == "videomae_small"
    assert isinstance(payload["epochs"], list)
    assert len(payload["epochs"]) == 2
    e0 = payload["epochs"][0]
    assert e0["epoch"] == 0
    assert e0["grad_steps"] == 4853
    assert "wall_time_s" in e0
    assert "combined" in e0["metrics"]
    assert e0["metrics"]["combined"]["macro_f1"] == pytest.approx(0.5)
    assert "per_class" in e0["metrics"]["combined"]
    assert "Bot" in e0["metrics"]["combined"]["per_class"]
    bot = e0["metrics"]["combined"]["per_class"]["Bot"]
    assert {"f1", "p", "r", "auroc", "n"} <= set(bot.keys())


def test_writer_confusion_npz_keys(tmp_path: Path) -> None:
    """Three log_epoch calls → npz with keys ``epoch_0``, ``epoch_1``,
    ``epoch_2``; each loaded value is a (13, 13) int64 matrix."""
    w = MetricsWriter(tmp_path / "run_cm")
    for ep in range(3):
        cm = _mock_confusion(fill=ep + 1)
        w.log_epoch(epoch=ep, grad_steps=100 * (ep + 1), wall_time_s=10.0,
                    metrics={"combined": _mock_combined_metrics()},
                    confusion=cm)
    w.finalize()

    loaded = np.load(tmp_path / "run_cm" / "metrics" / "confusion_per_epoch.npz")
    keys = sorted(loaded.files)
    assert keys == ["epoch_0", "epoch_1", "epoch_2"]
    for ep in range(3):
        m = loaded[f"epoch_{ep}"]
        assert m.shape == (13, 13)
        assert m.dtype == np.int64
        # Diagonal entries reflect the fill we passed.
        assert m[0, 0] == ep + 1


def test_writer_retrofit_mode_skips_per_step(tmp_path: Path) -> None:
    """``purpose="retrofit"`` (used by scripts/baseline_rerun.py
    --ckpt-glob) does not touch per_step.jsonl. A prior
    forward-instrumentation per_step.jsonl in the same metrics/ dir
    must be preserved verbatim, and log_step calls must raise."""
    run_dir = tmp_path / "run_retro"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    # Plant a fake forward-instrumented per_step.jsonl from a prior run.
    prior = '{"step":1,"epoch":0,"loss":2.5,"lr":1e-4,"wall_time_s":0.3}\n'
    (metrics_dir / "per_step.jsonl").write_text(prior)

    w = MetricsWriter(run_dir, purpose="retrofit")
    # File untouched after construction.
    assert (metrics_dir / "per_step.jsonl").read_text() == prior

    # log_step in retrofit mode raises (BEFORE finalize, so the "closed"
    # check doesn't shadow the "retrofit" check).
    with pytest.raises(RuntimeError, match="retrofit"):
        w.log_step(step=2, epoch=0, loss=1.0, lr=1e-4, wall_time_s=0.1)

    # log_epoch + finalize work normally.
    w.log_epoch(epoch=0, grad_steps=4853, wall_time_s=1237.0,
                metrics={"combined": _mock_combined_metrics()},
                confusion=_mock_confusion(fill=10))
    w.finalize()

    # Per-step file still untouched after the full retrofit cycle.
    assert (metrics_dir / "per_step.jsonl").read_text() == prior


def test_writer_finalize_idempotent(tmp_path: Path) -> None:
    """Repeated finalize() must not raise. log_step / log_epoch after
    finalize() must raise (closed-state contract)."""
    w = MetricsWriter(tmp_path / "run_idem")
    w.log_step(step=1, epoch=0, loss=1.0, lr=1e-4, wall_time_s=0.1)
    w.log_epoch(epoch=0, grad_steps=1, wall_time_s=1.0,
                metrics={"combined": _mock_combined_metrics()},
                confusion=_mock_confusion())
    w.finalize()
    w.finalize()        # should be no-op
    w.finalize()        # ditto

    with pytest.raises(RuntimeError, match="closed"):
        w.log_step(step=2, epoch=0, loss=1.0, lr=1e-4, wall_time_s=0.1)
    with pytest.raises(RuntimeError, match="closed"):
        w.log_epoch(epoch=1, grad_steps=2, wall_time_s=1.0,
                    metrics={"combined": _mock_combined_metrics()},
                    confusion=_mock_confusion())


# ---------------------------------------------------------------------
# Smoke test (real trainer + evaluator)
# ---------------------------------------------------------------------


def _build_synthetic_shards(out_dir: Path, n_samples: int = 32,
                            label_id: int = 0, maxcount: int = 8) -> str:
    """Inlined from tests/test_trainer.py with the same sample schema
    (tensor.npy + label.cls + meta.json) so the smoke does not depend
    on that test module's fixtures."""
    import webdataset as wds
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "shard-%06d.tar")
    rng = np.random.default_rng(0)
    with wds.ShardWriter(pattern, maxcount=maxcount) as sink:
        for i in range(n_samples):
            tensor = rng.standard_normal((16, 6, 32, 64), dtype=np.float32)
            sink.write({
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


@pytest.mark.slow
def test_trainer_with_metrics_writer_smoke(tmp_path: Path) -> None:
    """End-to-end smoke: synthetic 1-class shards → real Trainer with
    MetricsWriter wired (default settings, grad_norm collection OFF) →
    2 epochs × handful of steps → verify all three artefact files
    materialise + schema is correct + per_step rows do NOT carry the
    grad_norm field by default."""
    import torch
    from nid_video.data.dataset import build_dataloader
    from nid_video.models.videomae_nid import VideoMAESmallForNID
    from nid_video.trainer.trainer import Trainer
    from nid_video.utils.config import TrainingConfig

    if not torch.cuda.is_available():
        pytest.skip("smoke test requires CUDA")

    pattern = _build_synthetic_shards(tmp_path / "shards", n_samples=32)
    train_loader = build_dataloader(
        pattern, batch_size=2, num_workers=0,
        label_mode="collapsed13", shuffle_buffer=0,
    )
    val_loader = build_dataloader(
        pattern, batch_size=2, num_workers=0,
        label_mode="collapsed13", shuffle_buffer=0,
    )

    cfg = TrainingConfig(
        batch_size=2, grad_accumulation=1, num_epochs=2,
        lr=1e-4, weight_decay=0.01,
        precision="fp16", optimizer="adamw",
        gradient_checkpointing=False,
        head_lr_multiplier=1.0,
    )
    model = VideoMAESmallForNID(num_classes=13, pretrained=None,
                                 gradient_checkpointing=False)

    run_dir = tmp_path / "run_smoke"
    trainer = Trainer(
        model=model, train_loader=train_loader, config=cfg,
        device="cuda", run_dir=run_dir,
        warmup_steps=2, total_steps=20,
        val_loader=val_loader, num_classes=13,
        # write_metrics defaults True; collect_grad_norm defaults False —
        # the smoke verifies the project-wide default behaviour.
    )
    trainer.train(num_epochs=2, max_steps=8)

    metrics_dir = run_dir / "metrics"
    assert metrics_dir.is_dir()
    assert (metrics_dir / "per_step.jsonl").is_file()
    assert (metrics_dir / "per_epoch.json").is_file()
    assert (metrics_dir / "confusion_per_epoch.npz").is_file()

    # per_step.jsonl: at least 1 row, default schema, no grad_norm field.
    rows = [json.loads(line)
            for line in (metrics_dir / "per_step.jsonl").read_text().splitlines()
            if line.strip()]
    assert len(rows) >= 1, "smoke produced no per-step rows"
    expected_keys = {"step", "epoch", "loss", "lr", "wall_time_s"}
    for row in rows:
        assert set(row.keys()) == expected_keys, (
            f"smoke per-step schema drift: row keys = {set(row.keys())}; "
            f"expected exactly {expected_keys} (default = no grad_norm)"
        )

    # per_epoch.json: at most 2 epoch records (max_steps may have cut us
    # short on epoch 2 — be permissive but require ≥1).
    payload = json.loads((metrics_dir / "per_epoch.json").read_text())
    assert payload["run_id"] == run_dir.name
    assert "epochs" in payload
    assert len(payload["epochs"]) >= 1
    e0 = payload["epochs"][0]
    assert e0["epoch"] == 0
    assert "combined" in e0["metrics"]
    assert "macro_f1" in e0["metrics"]["combined"]
    assert "per_class" in e0["metrics"]["combined"]

    # confusion_per_epoch.npz: ≥1 epoch_X key, each (13, 13) int64.
    loaded = np.load(metrics_dir / "confusion_per_epoch.npz")
    assert len(loaded.files) >= 1
    for k in loaded.files:
        assert k.startswith("epoch_")
        assert loaded[k].shape == (13, 13)
        assert loaded[k].dtype == np.int64
