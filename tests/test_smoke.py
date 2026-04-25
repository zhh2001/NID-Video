"""Smoke tests for M1: config loads, logger emits, CUDA reachable + usable."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from nid_video.utils import (
    NIDVideoConfig,
    load_config,
    logger,
    project_root,
    setup_logger,
)


@pytest.fixture(autouse=True)
def _reset_logger() -> Iterator[None]:
    """Strip any logger sinks installed by a test so the next test starts clean."""
    yield
    logger.remove()


# --- config -----------------------------------------------------------------


def test_config_loads_with_canonical_shape() -> None:
    """Lock the (T, C, H, W) contract from Idea.md §3.2."""
    cfg = load_config(project_root() / "configs" / "base.yaml")

    assert isinstance(cfg, NIDVideoConfig)
    assert cfg.data.num_frames == 16            # T
    assert cfg.data.num_channels == 6           # C
    assert cfg.data.num_ip_buckets == 32        # H
    assert cfg.data.num_port_buckets == 64      # W
    assert cfg.data.delta_t_ms == 100
    assert 0.0 <= cfg.data.window_overlap < 1.0

    assert cfg.model.backbone == "videomae_small"
    assert cfg.model.tube_patch == (2, 8, 8)

    assert cfg.training.batch_size == 2
    assert cfg.training.grad_accumulation == 16
    assert cfg.training.optimizer == "adamw_8bit"
    assert cfg.training.precision == "fp16"
    assert cfg.training.gradient_checkpointing is True


# --- logger -----------------------------------------------------------------


def test_logger_writes_to_file(tmp_path: Path) -> None:
    """Configure a file sink, emit a message, flush, and read it back."""
    logfile = tmp_path / "smoke.log"
    setup_logger(level="INFO", log_file=logfile)
    logger.info("smoke-test-marker-{}", 12345)
    logger.remove()  # synchronously flushes the enqueue=True sink
    content = logfile.read_text()
    assert "smoke-test-marker-12345" in content


# --- CUDA -------------------------------------------------------------------


def test_cuda_is_available() -> None:
    """Project requires a real GPU — CPU-only env should fail loudly here."""
    assert torch.cuda.is_available(), "CUDA must be available (cu130 wheel installed in M1)"
    assert torch.cuda.get_device_name(0)


def test_cuda_matmul_matches_cpu() -> None:
    """A small matmul on the GPU should agree with the CPU reference."""
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda")
    b = torch.randn(128, 64, device="cuda")
    c = a @ b
    assert c.shape == (64, 64)
    assert torch.allclose(c.cpu(), a.cpu() @ b.cpu(), atol=1e-3)
