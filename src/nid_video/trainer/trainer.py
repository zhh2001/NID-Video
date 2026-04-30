"""Trainer: FP16 + AdamW + grad-accumulation loop with cosine LR scheduling.

Scope (M3 baseline + M4 task 4.3 LR scheduler):
  * FP16 AMP via `torch.autocast` + `GradScaler`
  * Gradient accumulation
  * 8-bit AdamW (bitsandbytes) with a `--debug` switch back to vanilla AdamW
  * **M4: cosine LR with linear warmup**, stepped per grad step
  * Per-step loguru logging (every 10 micro-batches)
  * Per-epoch checkpoint to `outputs/run_<ts>/ckpt/epoch_{N}.safetensors`

Out of scope (deferred to M4 later tasks):
  * Resume from checkpoint, optimizer state save/load (task 4.4)
  * Eval / val split / metrics (task 4.5)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader

from nid_video.trainer.scheduler import make_cosine_scheduler
from nid_video.utils import logger
from nid_video.utils.config import TrainingConfig


@dataclass
class TrainResult:
    final_avg_loss: float
    epochs_completed: int
    micro_steps: int
    grad_steps: int
    peak_gpu_memory_mb: float
    elapsed_seconds: float
    checkpoint_paths: list[Path] = field(default_factory=list)


def _build_optimizer(
    model: nn.Module,
    cfg: TrainingConfig,
    optimizer_class: Callable | None,
) -> torch.optim.Optimizer:
    """Return AdamW8bit / vanilla AdamW / caller-supplied class. lr & wd from config."""
    if optimizer_class is not None:
        return optimizer_class(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw_8bit":
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer in TrainingConfig: {cfg.optimizer!r}")


class Trainer:
    """Minimal training loop for the M3 milestone — see module docstring."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: TrainingConfig,
        device: str = "cuda",
        run_dir: Path | None = None,
        criterion: nn.Module | None = None,
        optimizer_class: Callable | None = None,
        warmup_steps: int = 500,
        total_steps: int | None = None,
        min_lr_ratio: float = 0.01,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = device

        if run_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path("outputs") / f"run_{ts}"
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "ckpt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = _build_optimizer(self.model, config, optimizer_class)

        # M4 task 4.3: cosine LR with linear warmup. ``total_steps`` is the
        # full grad-step budget (NOT micro-step). When ``None`` we fall back
        # to a num_epochs-based estimate; task 4.6 will compute it precisely
        # from the manifest sample count once the dataset is final. The
        # placeholder is a heuristic, not a contract — log a WARNING so
        # debug runs notice they're on the fallback path.
        if total_steps is None:
            total_steps = max(warmup_steps + 1, config.num_epochs * 1000)
            logger.warning(
                f"total_steps was None; using fallback estimate {total_steps} "
                f"({config.num_epochs} epochs × 1000 grad steps/epoch). Pass "
                f"total_steps explicitly when the dataset size is known."
            )
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.scheduler = make_cosine_scheduler(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            min_ratio=self.min_lr_ratio,
        )

        # Mixed-precision setup
        if device == "cuda" and config.precision == "fp16":
            self.amp_enabled = True
            self.amp_dtype = torch.float16
            self.scaler: torch.amp.GradScaler | None = torch.amp.GradScaler("cuda")
        elif device == "cuda" and config.precision == "bf16":
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
            self.scaler = None      # bf16 doesn't need a scaler
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32
            self.scaler = None

        self.grad_accum = max(1, int(config.grad_accumulation))
        self.global_micro_step = 0
        self.global_grad_step = 0

        logger.info(
            f"Trainer ready: device={device}, precision={config.precision}, "
            f"optimizer={config.optimizer}, lr={config.lr}, wd={config.weight_decay}, "
            f"batch_size={config.batch_size}, grad_accumulation={self.grad_accum}, "
            f"warmup_steps={self.warmup_steps}, total_steps={self.total_steps}, "
            f"min_lr_ratio={self.min_lr_ratio}, run_dir={self.run_dir}"
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        num_epochs: int | None = None,
        max_steps: int | None = None,
        log_every: int = 10,
    ) -> TrainResult:
        """Run training. Returns TrainResult.

        Args:
          num_epochs: override `config.num_epochs` if provided.
          max_steps: optional cap on micro-batch count (for `--max-steps` debug).
          log_every: emit a loguru INFO every N micro-batches. Default 10.
        """
        n_epochs = num_epochs if num_epochs is not None else self.config.num_epochs

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t_run_start = time.perf_counter()
        ckpts: list[Path] = []
        epoch_avg_losses: list[float] = []

        for epoch in range(n_epochs):
            avg_loss, t_epoch = self._train_one_epoch(epoch, max_steps, log_every)
            epoch_avg_losses.append(avg_loss)
            peak_mb = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if self.device == "cuda" else 0.0
            )
            logger.info(
                f"=== epoch {epoch} done: avg_loss={avg_loss:.4f} "
                f"elapsed={t_epoch:.1f}s peak_gpu={peak_mb:.0f}MB "
                f"micro_steps={self.global_micro_step} grad_steps={self.global_grad_step} ==="
            )

            ckpt_path = self.ckpt_dir / f"epoch_{epoch}.safetensors"
            self.save_checkpoint(ckpt_path, epoch)
            ckpts.append(ckpt_path)

            if max_steps is not None and self.global_micro_step >= max_steps:
                logger.info(f"--max-steps {max_steps} reached; stopping early")
                break

        peak_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if self.device == "cuda" else 0.0
        )
        return TrainResult(
            final_avg_loss=epoch_avg_losses[-1] if epoch_avg_losses else float("nan"),
            epochs_completed=len(epoch_avg_losses),
            micro_steps=self.global_micro_step,
            grad_steps=self.global_grad_step,
            peak_gpu_memory_mb=peak_mb,
            elapsed_seconds=time.perf_counter() - t_run_start,
            checkpoint_paths=ckpts,
        )

    def _train_one_epoch(
        self,
        epoch: int,
        max_steps: int | None,
        log_every: int,
    ) -> tuple[float, float]:
        self.model.train()
        loss_sum = 0.0
        loss_n = 0
        t_epoch_start = time.perf_counter()

        for micro_idx, batch in enumerate(self.train_loader):
            x = batch["tensor"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)
            # Multi-scale path attaches scale_id (B,) per sample. Single-scale
            # legacy path doesn't — default to fast (0).
            if "scale_id" in batch:
                scale_id = batch["scale_id"].to(self.device, non_blocking=True)
            else:
                scale_id = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

            with torch.autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.amp_enabled):
                out = self.model(x, scale_id=scale_id)
                raw_loss = self.criterion(out["logits"], y)

            scaled_loss = raw_loss / self.grad_accum
            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            loss_sum += float(raw_loss.detach().item())
            loss_n += 1

            grad_norm: float | None = None
            do_step = (micro_idx + 1) % self.grad_accum == 0
            if do_step:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0,
                ).item()
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                # Advance the cosine scheduler once per grad step (NOT micro
                # step). PyTorch convention: scheduler.step() AFTER optimizer
                # step. NB: when GradScaler skipped due to inf/nan grads the
                # optimizer didn't actually step, but we advance the scheduler
                # anyway — bounded drift, acceptable in M4 baseline.
                self.scheduler.step()
                self.global_grad_step += 1

            self.global_micro_step += 1

            if micro_idx % log_every == 0 or do_step:
                gpu_mb = (
                    torch.cuda.memory_allocated() / (1024 * 1024)
                    if self.device == "cuda" else 0.0
                )
                gn_str = f" grad_norm={grad_norm:.3f}" if grad_norm is not None else ""
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"epoch={epoch} micro={micro_idx} grad={self.global_grad_step} "
                    f"loss={raw_loss.item():.4f} lr={lr:.2e}{gn_str} "
                    f"gpu_mem={gpu_mb:.0f}MB"
                )

            if max_steps is not None and self.global_micro_step >= max_steps:
                break

        return loss_sum / max(loss_n, 1), time.perf_counter() - t_epoch_start

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        """Save model state_dict to safetensors.

        M3 saves model weights only; optimizer state is deferred to M4 (resume)
        because bitsandbytes 8-bit optimizer state isn't safetensors-friendly
        without dequantization, and M3's prompt explicitly says resume is M4.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # safetensors requires contiguous tensors and disallows shared storage.
        state = {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()}
        save_file(
            state,
            str(path),
            metadata={
                "epoch": str(epoch),
                "precision": self.config.precision,
                "optimizer": self.config.optimizer,
                "batch_size": str(self.config.batch_size),
                "grad_accumulation": str(self.config.grad_accumulation),
                "lr": str(self.config.lr),
            },
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"checkpoint saved: {path} ({size_mb:.1f} MB)")
