"""IterableDataset over the M2 ETL output (webdataset shards).

M2 shards store RAW 15-class label IDs in `label.cls`. This dataset converts
them to whichever scheme the trainer needs at load time (`raw15` or
`collapsed13`), so the same shards serve both ablation paths without re-ETL.

M3 does NOT z-score normalize. The (T,C,H,W) tensor was already log-scaled in
the channel encoder; manifest-driven normalization is deferred to M4+.
"""

from __future__ import annotations

import glob
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import torch
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset

from nid_video.data.labeling import collapse_to_13
from nid_video.data.split import SplitName, WindowKey, load_splits
from nid_video.utils import logger

EpochEndStrategy = Literal["slow_exhausted", "max_len"]

LabelMode = Literal["raw15", "collapsed13"]
NUM_CLASSES_RAW = 15
NUM_CLASSES_COLLAPSED = 13


def num_classes(mode: LabelMode) -> int:
    return NUM_CLASSES_RAW if mode == "raw15" else NUM_CLASSES_COLLAPSED


def _to_torch_sample(sample: dict, label_mode: LabelMode) -> dict:
    """Convert a webdataset-decoded sample dict into our training format."""
    arr = sample["tensor.npy"]
    raw_label = int(sample["label.cls"])
    if label_mode == "collapsed13":
        label = collapse_to_13(raw_label)
    elif label_mode == "raw15":
        label = raw_label
    else:
        raise ValueError(f"unknown label_mode: {label_mode}")

    tensor = torch.from_numpy(arr).contiguous()
    return {
        "tensor": tensor,
        "label": torch.tensor(label, dtype=torch.long),
        "meta": sample["meta.json"],
    }


def _resolve_shard_urls(shard_pattern: str | Path | list[str]) -> list[str] | str:
    """Expand glob wildcards in a shard pattern; pass other forms straight through.

    webdataset's WebDataset accepts a list of URLs or a brace-expandable string,
    but does not natively expand `*` globs. We expand here so callers can pass
    natural patterns like `data/processed/*/shards/shard-*.tar`.
    """
    if isinstance(shard_pattern, list):
        return [str(p) for p in shard_pattern]
    pattern = str(shard_pattern)
    if any(c in pattern for c in "*?"):
        urls = sorted(glob.glob(pattern, recursive=True))
        if not urls:
            raise FileNotFoundError(f"no shards matched pattern: {pattern}")
        return urls
    return pattern


class NidShardDataset(IterableDataset):
    """webdataset IterableDataset adapter for M2 ETL shards.

    Yielded items::

        {
          "tensor": torch.float32, shape (T=16, C=6, H=32, W=64), contiguous,
          "label":  torch.long, scalar, in [0, num_classes(label_mode)),
          "meta":   dict (start_time, pcap_source, label, label_id,
                          dominant_attack_ratio, n_unmatched),
        }

    Args:
      shard_pattern: glob (e.g. ``"data/processed/*/shards/shard-*.tar"``),
        a list of shard URLs, or a single .tar path.
      label_mode: ``"raw15"`` returns label IDs verbatim; ``"collapsed13"``
        applies :func:`nid_video.data.labeling.collapse_to_13` so the three
        Web-Attack subtypes share one ID.
      shuffle_buffer: 0 disables; else N samples in the in-memory shuffle
        reservoir. Default 1000.
    """

    def __init__(
        self,
        shard_pattern: str | Path | list[str],
        *,
        label_mode: LabelMode = "collapsed13",
        shuffle_buffer: int = 1000,
        splits_path: Path | str | None = None,
        keep_split: SplitName | None = None,
    ) -> None:
        if label_mode not in ("raw15", "collapsed13"):
            raise ValueError(f"unknown label_mode: {label_mode}")
        if (splits_path is None) != (keep_split is None):
            raise ValueError(
                "splits_path and keep_split must be set together "
                "(both None to disable filtering)"
            )
        self._urls = _resolve_shard_urls(shard_pattern)
        self.label_mode: LabelMode = label_mode
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self._splits: dict[WindowKey, SplitName] | None = (
            load_splits(Path(splits_path)) if splits_path is not None else None
        )
        self.keep_split: SplitName | None = keep_split
        n = len(self._urls) if isinstance(self._urls, list) else 1
        logger.info(
            f"NidShardDataset: {n} url(s), label_mode={label_mode}, "
            f"shuffle_buffer={self.shuffle_buffer}, keep_split={keep_split}"
        )

    def _split_predicate(self, sample: dict) -> bool:
        """Return True iff this sample's (pcap_source, start_time) maps to the
        target split. Called only when ``keep_split`` is set."""
        meta = sample["meta.json"]
        key = WindowKey(
            pcap_source=str(meta["pcap_source"]),
            start_time=float(meta["start_time"]),
        )
        return self._splits.get(key) == self.keep_split   # type: ignore[union-attr]

    def _build_pipeline(self) -> wds.WebDataset:
        # shardshuffle=False: shard order is deterministic; sample-level shuffle
        # below via .shuffle(buffer). Keeping shardshuffle off makes the
        # shuffle_buffer=0 case strictly reproducible.
        pipeline = wds.WebDataset(self._urls, shardshuffle=False).decode()
        if self.keep_split is not None:
            # Filter happens before shuffle so the shuffle buffer holds only
            # samples for the target split — avoids wasting buffer slots on
            # samples we'd then drop.
            pipeline = pipeline.select(self._split_predicate)
        if self.shuffle_buffer > 0:
            pipeline = pipeline.shuffle(self.shuffle_buffer)
        return pipeline.map(lambda s: _to_torch_sample(s, self.label_mode))

    def __iter__(self) -> Iterator[dict]:
        return iter(self._build_pipeline())


def _collate(batch: list[dict]) -> dict:
    """Stack tensors / labels; keep meta as a list of per-sample dicts.

    If samples carry ``scale_id`` (multi-scale path), it is stacked into a
    (B,) long tensor for the model's scale-token forward.
    """
    out: dict = {
        "tensor": torch.stack([b["tensor"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "meta": [b["meta"] for b in batch],
    }
    if "scale_id" in batch[0]:
        out["scale_id"] = torch.stack([b["scale_id"] for b in batch], dim=0)
    return out


class MultiScaleNidDataset(IterableDataset):
    """50/50 (configurable) mix of fast (Δt=100ms) and slow (Δt=1s) shards.

    Each yielded sample carries ``scale_id`` ∈ {0=fast, 1=slow}. The model's
    scale-token forward (``VideoMAESmallForNID.forward``) reads it.

    Idea.md M4 task 4.2 decision: option (a) — two physically independent
    shard sets, no aggregation. Per-sample mixing is done in Python via a
    worker-seeded ``random.Random``; webdataset's own shard splitting handles
    multi-worker non-overlap on each underlying NidShardDataset.

    Args:
      fast_pattern / slow_pattern: shard globs for the two scales.
      mix_ratio: P(fast). 0.5 by default.
      epoch_end_strategy: how to terminate when the two streams have
        different lengths.

        * ``"slow_exhausted"`` (default): epoch ends at the FIRST
          ``StopIteration`` from either stream. In practice the slow
          stream goes first since it's ~10× sparser. Simple and
          deterministic; the cost is that fast samples beyond the slow
          stream's length are not seen this epoch.
        * ``"max_len"``: reserved for future revisits, raises
          NotImplementedError.

      seed: base RNG seed; per-worker offset added by ``__iter__``.
    """

    def __init__(
        self,
        fast_pattern: str | Path | list[str],
        slow_pattern: str | Path | list[str],
        *,
        mix_ratio: float = 0.5,
        label_mode: LabelMode = "collapsed13",
        shuffle_buffer: int = 1000,
        splits_path: Path | str | None = None,
        keep_split: SplitName | None = None,
        seed: int = 42,
        epoch_end_strategy: EpochEndStrategy = "slow_exhausted",
    ) -> None:
        if not 0.0 <= mix_ratio <= 1.0:
            raise ValueError(f"mix_ratio must be in [0, 1], got {mix_ratio}")
        if epoch_end_strategy != "slow_exhausted":
            raise NotImplementedError(
                f"epoch_end_strategy={epoch_end_strategy!r} not implemented. "
                f"Only 'slow_exhausted' is supported; 'max_len' is reserved."
            )
        self.fast = NidShardDataset(
            fast_pattern, label_mode=label_mode, shuffle_buffer=shuffle_buffer,
            splits_path=splits_path, keep_split=keep_split,
        )
        self.slow = NidShardDataset(
            slow_pattern, label_mode=label_mode, shuffle_buffer=shuffle_buffer,
            splits_path=splits_path, keep_split=keep_split,
        )
        self.mix_ratio = float(mix_ratio)
        self.seed = int(seed)
        self.epoch_end_strategy: EpochEndStrategy = epoch_end_strategy
        logger.info(
            f"MultiScaleNidDataset: mix_ratio={mix_ratio}, seed={seed}, "
            f"epoch_end_strategy={epoch_end_strategy}"
        )

    def __iter__(self) -> Iterator[dict]:
        winfo = torch.utils.data.get_worker_info()
        worker_seed = self.seed + (winfo.id if winfo is not None else 0)
        rng = random.Random(worker_seed)
        fast_iter = iter(self.fast)
        slow_iter = iter(self.slow)

        while True:
            if rng.random() < self.mix_ratio:
                try:
                    sample = next(fast_iter)
                except StopIteration:
                    return
                sample["scale_id"] = torch.tensor(0, dtype=torch.long)
            else:
                try:
                    sample = next(slow_iter)
                except StopIteration:
                    return
                sample["scale_id"] = torch.tensor(1, dtype=torch.long)
            yield sample


def build_dataloader(
    shard_pattern: str | Path | list[str],
    *,
    batch_size: int = 2,
    num_workers: int = 0,
    label_mode: LabelMode = "collapsed13",
    shuffle_buffer: int = 1000,
    splits_path: Path | str | None = None,
    keep_split: SplitName | None = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Standard DataLoader over :class:`NidShardDataset` (single-scale).

    With ``num_workers > 0`` webdataset's default node/worker shard splitting
    ensures each shard is read by exactly one worker (no duplicate samples).
    Pass ``splits_path`` + ``keep_split`` to restrict to one split.

    For multi-scale training (M4) use ``build_multi_scale_dataloader``.
    """
    ds = NidShardDataset(
        shard_pattern=shard_pattern,
        label_mode=label_mode,
        shuffle_buffer=shuffle_buffer,
        splits_path=splits_path,
        keep_split=keep_split,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )


def build_multi_scale_dataloader(
    fast_pattern: str | Path | list[str],
    slow_pattern: str | Path | list[str],
    *,
    batch_size: int = 2,
    num_workers: int = 0,
    label_mode: LabelMode = "collapsed13",
    shuffle_buffer: int = 1000,
    mix_ratio: float = 0.5,
    splits_path: Path | str | None = None,
    keep_split: SplitName | None = None,
    seed: int = 42,
    epoch_end_strategy: EpochEndStrategy = "slow_exhausted",
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """DataLoader over :class:`MultiScaleNidDataset` (M4).

    Yields batches with ``scale_id`` (B,) alongside ``tensor`` / ``label`` /
    ``meta``. ``scale_id`` is consumed by ``VideoMAESmallForNID.forward``.
    """
    ds = MultiScaleNidDataset(
        fast_pattern=fast_pattern,
        slow_pattern=slow_pattern,
        mix_ratio=mix_ratio,
        label_mode=label_mode,
        shuffle_buffer=shuffle_buffer,
        splits_path=splits_path,
        keep_split=keep_split,
        seed=seed,
        epoch_end_strategy=epoch_end_strategy,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
