"""Tests for NidShardDataset and build_dataloader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import webdataset as wds

from nid_video.data.dataset import (
    NUM_CLASSES_COLLAPSED,
    NUM_CLASSES_RAW,
    MultiScaleNidDataset,
    NidShardDataset,
    _collate,
    _resolve_shard_urls,
    _to_torch_sample,
    build_dataloader,
    build_multi_scale_dataloader,
    num_classes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_meta(label_id: int) -> dict:
    return {
        "start_time": float(label_id),
        "pcap_source": f"src_{label_id}.pcap",
        "label": f"label_{label_id}",
        "label_id": label_id,
        "dominant_attack_ratio": 1.0,
        "n_unmatched": 0,
    }


def _write_shards(out_dir: Path, label_ids: list[int], maxcount: int = 4) -> str:
    """Write deterministic synthetic shards covering the given raw label IDs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "shard-%06d.tar")
    rng = np.random.default_rng(7)
    with wds.ShardWriter(pattern, maxcount=maxcount) as w:
        for i, lid in enumerate(label_ids):
            tensor = rng.standard_normal((16, 6, 32, 64), dtype=np.float32)
            w.write({
                "__key__": f"{i:010d}",
                "tensor.npy": tensor,
                "label.cls": lid,
                "meta.json": _fake_meta(lid),
            })
    return str(out_dir / "shard-*.tar")


@pytest.fixture
def shards_with_known_labels(tmp_path: Path) -> tuple[str, list[int]]:
    # Cover BENIGN, all 3 Web Attack subtypes, Infiltration, Heartbleed, plus repeats
    label_ids = [0, 10, 11, 12, 13, 14, 0, 3, 9, 1]
    pattern = _write_shards(tmp_path / "shards", label_ids, maxcount=4)
    return pattern, label_ids


# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------


def test_resolve_shard_urls_expands_glob(tmp_path: Path) -> None:
    a = tmp_path / "shard-000000.tar"
    b = tmp_path / "shard-000001.tar"
    a.write_bytes(b"")
    b.write_bytes(b"")
    urls = _resolve_shard_urls(str(tmp_path / "shard-*.tar"))
    assert urls == [str(a), str(b)]


def test_resolve_shard_urls_passes_lists_through() -> None:
    urls = _resolve_shard_urls(["a.tar", "b.tar"])
    assert urls == ["a.tar", "b.tar"]


def test_resolve_shard_urls_missing_glob_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _resolve_shard_urls(str(tmp_path / "no-such-*.tar"))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_dataset_yields_correct_schema(shards_with_known_labels) -> None:
    pattern, _ = shards_with_known_labels
    ds = NidShardDataset(pattern, shuffle_buffer=0)
    item = next(iter(ds))
    assert isinstance(item, dict)
    t = item["tensor"]
    assert isinstance(t, torch.Tensor)
    assert t.shape == (16, 6, 32, 64)
    assert t.dtype == torch.float32
    assert t.is_contiguous()
    label = item["label"]
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long
    assert label.dim() == 0
    meta = item["meta"]
    assert isinstance(meta, dict)
    for k in ("start_time", "pcap_source", "label", "label_id",
              "dominant_attack_ratio", "n_unmatched"):
        assert k in meta


# ---------------------------------------------------------------------------
# Label modes
# ---------------------------------------------------------------------------


def test_collapsed13_label_in_range(shards_with_known_labels) -> None:
    pattern, _ = shards_with_known_labels
    ds = NidShardDataset(pattern, label_mode="collapsed13", shuffle_buffer=0)
    for item in ds:
        v = item["label"].item()
        assert 0 <= v < NUM_CLASSES_COLLAPSED
        assert v < 13


def test_raw15_label_in_range(shards_with_known_labels) -> None:
    pattern, _ = shards_with_known_labels
    ds = NidShardDataset(pattern, label_mode="raw15", shuffle_buffer=0)
    for item in ds:
        v = item["label"].item()
        assert 0 <= v < NUM_CLASSES_RAW
        assert v < 15


def test_collapsed13_merges_web_attack_subtypes(shards_with_known_labels) -> None:
    """raw 11 (Web Attack XSS) and raw 12 (Sql Injection) both → collapsed 10."""
    pattern, _ = shards_with_known_labels
    ds = NidShardDataset(pattern, label_mode="collapsed13", shuffle_buffer=0)
    raw_to_collapsed: dict[int, int] = {}
    for item in ds:
        raw_to_collapsed[int(item["meta"]["label_id"])] = int(item["label"].item())
    assert raw_to_collapsed[10] == 10
    assert raw_to_collapsed[11] == 10
    assert raw_to_collapsed[12] == 10
    assert raw_to_collapsed[13] == 11
    assert raw_to_collapsed[14] == 12


def test_unknown_label_mode_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        NidShardDataset(tmp_path / "x.tar", label_mode="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_shuffle_buffer_zero_preserves_shard_order(shards_with_known_labels) -> None:
    pattern, label_ids = shards_with_known_labels
    ds = NidShardDataset(pattern, label_mode="raw15", shuffle_buffer=0)
    seen = [int(item["meta"]["label_id"]) for item in ds]
    assert seen == label_ids


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


def test_dataloader_collates_into_batches(shards_with_known_labels) -> None:
    pattern, label_ids = shards_with_known_labels
    loader = build_dataloader(
        pattern,
        batch_size=2,
        num_workers=0,
        label_mode="collapsed13",
        shuffle_buffer=0,
        pin_memory=False,
    )
    batch = next(iter(loader))
    assert batch["tensor"].shape == (2, 16, 6, 32, 64)
    assert batch["tensor"].dtype == torch.float32
    assert batch["label"].shape == (2,)
    assert batch["label"].dtype == torch.long
    assert isinstance(batch["meta"], list) and len(batch["meta"]) == 2


def test_dataloader_consumes_all_samples(shards_with_known_labels) -> None:
    pattern, label_ids = shards_with_known_labels
    loader = build_dataloader(
        pattern,
        batch_size=2,
        num_workers=0,
        label_mode="raw15",
        shuffle_buffer=0,
        pin_memory=False,
    )
    seen: list[int] = []
    for batch in loader:
        seen.extend(int(x) for x in batch["label"].tolist())
    assert sorted(seen) == sorted(label_ids)


def test_collate_fn_directly() -> None:
    """The collate fn should preserve the meta-as-list structure."""
    a = {"tensor": torch.zeros(1), "label": torch.tensor(0, dtype=torch.long),
         "meta": {"k": "a"}}
    b = {"tensor": torch.ones(1), "label": torch.tensor(1, dtype=torch.long),
         "meta": {"k": "b"}}
    out = _collate([a, b])
    assert out["tensor"].shape == (2, 1)
    assert out["label"].tolist() == [0, 1]
    assert out["meta"] == [{"k": "a"}, {"k": "b"}]


def test_num_classes_helper() -> None:
    assert num_classes("raw15") == 15
    assert num_classes("collapsed13") == 13


# ---------------------------------------------------------------------------
# M4 task 4.2: MultiScaleNidDataset
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_and_slow_shards(tmp_path: Path) -> tuple[str, str]:
    """Two physically independent shard sets (fast 100ms-style, slow 1s-style).

    Fast has 10× more samples than slow, mimicking the real Δt=100ms /
    Δt=1s ratio. Label sets differ to make per-stream provenance verifiable
    in tests.
    """
    fast_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    slow_labels = [3, 9]                # different labels → "fast vs slow" provenance check
    fast_pattern = _write_shards(tmp_path / "fast", fast_labels, maxcount=10)
    slow_pattern = _write_shards(tmp_path / "slow", slow_labels, maxcount=10)
    return fast_pattern, slow_pattern


def test_multi_scale_attaches_scale_id_to_each_sample(fast_and_slow_shards) -> None:
    """Yielded samples carry scale_id (0 or 1) and provenance is consistent:
    fast labels {0,1} only with scale_id=0; slow labels {3,9} only with scale_id=1."""
    fast, slow = fast_and_slow_shards
    ds = MultiScaleNidDataset(fast, slow, mix_ratio=0.5,
                              shuffle_buffer=0, label_mode="raw15", seed=42)
    seen_fast_labels: set[int] = set()
    seen_slow_labels: set[int] = set()
    for s in ds:
        scale_id = int(s["scale_id"].item())
        label = int(s["label"].item())
        assert scale_id in (0, 1)
        if scale_id == 0:
            seen_fast_labels.add(label)
        else:
            seen_slow_labels.add(label)
    assert seen_fast_labels.issubset({0, 1}), seen_fast_labels
    assert seen_slow_labels.issubset({3, 9}), seen_slow_labels
    # Both streams contributed (mix worked, not stuck on one stream)
    assert len(seen_fast_labels) > 0
    assert len(seen_slow_labels) > 0


def test_multi_scale_deterministic_under_same_seed(fast_and_slow_shards) -> None:
    """Same seed → same scale_id sequence (modulo stream-internal order)."""
    fast, slow = fast_and_slow_shards
    a = MultiScaleNidDataset(fast, slow, mix_ratio=0.5, shuffle_buffer=0,
                             label_mode="raw15", seed=123)
    b = MultiScaleNidDataset(fast, slow, mix_ratio=0.5, shuffle_buffer=0,
                             label_mode="raw15", seed=123)
    seq_a = [int(s["scale_id"].item()) for s in a]
    seq_b = [int(s["scale_id"].item()) for s in b]
    assert seq_a == seq_b


def test_multi_scale_slow_exhausted_strategy_stops_at_slow_end(
    fast_and_slow_shards,
) -> None:
    """``epoch_end_strategy='slow_exhausted'`` (the M4.2 default): iteration
    ends as soon as the slow stream (2 samples) is empty. Fast has 20 samples
    but most are not seen — that's the documented intended behaviour."""
    fast, slow = fast_and_slow_shards
    ds = MultiScaleNidDataset(
        fast, slow, mix_ratio=0.5, shuffle_buffer=0,
        label_mode="raw15", seed=42,
        epoch_end_strategy="slow_exhausted",
    )
    samples = list(ds)
    n_slow = sum(1 for s in samples if int(s["scale_id"].item()) == 1)
    n_fast = sum(1 for s in samples if int(s["scale_id"].item()) == 0)
    # Slow contributed at most 2 (its full size). After that, the next slow
    # draw triggers StopIteration and ends the epoch.
    assert n_slow <= 2, f"slow over-drawn: {n_slow}"
    # We did NOT drain all 20 fast samples — slow ran out first.
    assert n_fast < 20, f"fast was drained ({n_fast}); slow_exhausted didn't trigger"


def test_multi_scale_dataloader_collates_scale_id_into_batch(fast_and_slow_shards) -> None:
    """build_multi_scale_dataloader produces batches with a stacked scale_id
    (B,) tensor of dtype long, alongside tensor / label / meta."""
    fast, slow = fast_and_slow_shards
    loader = build_multi_scale_dataloader(
        fast, slow, batch_size=2, num_workers=0,
        label_mode="raw15", shuffle_buffer=0, mix_ratio=0.5, seed=7,
        pin_memory=False,
    )
    batch = next(iter(loader))
    assert "scale_id" in batch
    assert batch["scale_id"].dtype == torch.long
    assert batch["scale_id"].shape == (2,)
    assert set(batch["scale_id"].tolist()) <= {0, 1}
    # Tensor / label still correct
    assert batch["tensor"].shape == (2, 16, 6, 32, 64)
    assert batch["label"].shape == (2,)


def test_multi_scale_invalid_mix_ratio_rejected(fast_and_slow_shards) -> None:
    fast, slow = fast_and_slow_shards
    with pytest.raises(ValueError, match="mix_ratio"):
        MultiScaleNidDataset(fast, slow, mix_ratio=1.5)
    with pytest.raises(ValueError, match="mix_ratio"):
        MultiScaleNidDataset(fast, slow, mix_ratio=-0.1)


def test_collate_handles_scale_id_when_present() -> None:
    """_collate auto-stacks scale_id only when samples carry it; single-scale
    samples (no scale_id) round-trip unchanged."""
    a = {"tensor": torch.zeros(1), "label": torch.tensor(0, dtype=torch.long),
         "meta": {"k": "a"}, "scale_id": torch.tensor(0, dtype=torch.long)}
    b = {"tensor": torch.ones(1), "label": torch.tensor(1, dtype=torch.long),
         "meta": {"k": "b"}, "scale_id": torch.tensor(1, dtype=torch.long)}
    out = _collate([a, b])
    assert "scale_id" in out
    assert out["scale_id"].tolist() == [0, 1]
