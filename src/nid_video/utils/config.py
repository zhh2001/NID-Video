"""Configuration loading + validation.

Loads YAML via OmegaConf (so we get interpolation / merge semantics later if needed)
and validates the resulting tree with pydantic v2 for strict typing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DataConfig(BaseModel):
    """Frame-construction parameters. Idea.md §3.2."""

    model_config = ConfigDict(extra="forbid")

    raw_pcap_dir: str
    processed_dir: str
    delta_t_ms: int = Field(default=100, gt=0)
    num_frames: int = Field(default=16, gt=0)
    num_ip_buckets: int = Field(default=32, gt=0)        # H
    num_port_buckets: int = Field(default=64, gt=0)      # W
    num_channels: int = Field(default=6, gt=0)           # C
    window_overlap: float = Field(default=0.5, ge=0.0, lt=1.0)

    @field_validator("num_channels")
    @classmethod
    def _channels_must_be_six(cls, v: int) -> int:
        # Idea.md §3.2 fixes the 6-channel design. Surface a clear error
        # if the YAML drifts from that contract.
        if v != 6:
            raise ValueError(
                "num_channels must stay at 6 — see Idea.md §3.2 for channel semantics"
            )
        return v


class ModelConfig(BaseModel):
    """Backbone selection. Idea.md §3.4."""

    model_config = ConfigDict(extra="forbid")

    backbone: Literal["videomae_small", "videomae_base"] = "videomae_small"
    pretrained: Literal["kinetics400", "ssv2", "none"] = "kinetics400"
    tube_patch: tuple[int, int, int] = (2, 8, 8)


class TrainingConfig(BaseModel):
    """Training loop hyperparameters + memory tactics. Idea.md §4.1."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=2, gt=0)
    grad_accumulation: int = Field(default=16, gt=0)
    num_epochs: int = Field(default=20, gt=0)
    lr: float = Field(default=1.5e-4, gt=0)
    weight_decay: float = Field(default=0.05, ge=0)
    optimizer: Literal["adamw_8bit", "adamw"] = "adamw_8bit"
    precision: Literal["fp16", "bf16", "fp32"] = "fp16"
    gradient_checkpointing: bool = True
    num_workers: int = Field(default=0, ge=0)
    # Loss function (M5.4). ``"ce"`` keeps the historical default; ``"focal"``
    # selects multi-class focal loss with focusing parameter ``focal_gamma``.
    loss_fn: Literal["ce", "focal"] = "ce"
    focal_gamma: float = Field(default=2.0, gt=0)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = "outputs"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class NIDVideoConfig(BaseModel):
    """Top-level config tree."""

    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig


def load_config(path: str | Path) -> NIDVideoConfig:
    """Load a YAML config file and validate it against the pydantic schema.

    Supports a top-level ``extends: <relative-or-absolute-path>`` field for
    composition: the parent file is loaded first, then the child overrides
    are merged on top. ``extends`` may itself extend another file (recursive).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = _load_with_extends(p)
    plain = OmegaConf.to_container(raw, resolve=True)
    return NIDVideoConfig.model_validate(plain)


def _load_with_extends(path: Path):
    """Load a YAML, recursively resolving any top-level ``extends`` chain."""
    raw = OmegaConf.load(path)
    if "extends" not in raw:
        return raw
    parent_path = Path(str(raw.extends))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    if not parent_path.is_file():
        raise FileNotFoundError(
            f"{path}: 'extends: {raw.extends}' resolves to {parent_path} which does not exist"
        )
    parent = _load_with_extends(parent_path)
    del raw["extends"]
    return OmegaConf.merge(parent, raw)


def project_root() -> Path:
    """Walk up from this file to find the project root (the dir containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "pyproject.toml").is_file():
            return ancestor
    raise RuntimeError("pyproject.toml not found above utils/config.py")
