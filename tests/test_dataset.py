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
    NidShardDataset,
    _collate,
    _resolve_shard_urls,
    _to_torch_sample,
    build_dataloader,
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
