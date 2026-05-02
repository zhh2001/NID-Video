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

import math
import random
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader

from nid_video.trainer.evaluator import Evaluator
from nid_video.trainer.loss import build_criterion
from nid_video.trainer.scheduler import make_cosine_scheduler
from nid_video.utils import logger
from nid_video.utils.config import TrainingConfig

# Schema version for the .pt checkpoint dict. Bumped when the on-disk layout
# changes in a way that requires a code-side migration (not just adding a new
# optional field — those are forward-compat).
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class TrainResult:
    final_avg_loss: float
    epochs_completed: int
    micro_steps: int
    grad_steps: int
    peak_gpu_memory_mb: float
    elapsed_seconds: float
    checkpoint_paths: list[Path] = field(default_factory=list)


def _build_param_groups(
    model: nn.Module,
    base_lr: float,
    head_lr_multiplier: float,
) -> list[dict]:
    """Split model parameters into two groups: backbone (Kinetics-pretrained)
    at ``base_lr`` and head (fresh-init components) at
    ``base_lr * head_lr_multiplier``.

    Head = classifier + scale_token + scale_embedding. All three are
    initialised at construction (Phase-2-of-M5.4 design point: fresh
    components benefit from a higher LR than the slowly-fine-tuned
    Kinetics backbone). Backbone = everything else, including the
    patch_embed channels that mix Kinetics-init and Kaiming-init weights
    (those Kaiming channels co-train with the rest of the encoder
    pipeline; isolating them would over-engineer a third group).

    With ``head_lr_multiplier=1.0`` (default), both groups have the same
    effective LR — bit-equivalent to the single-group path used in
    M5.4 Phase 1 and earlier.
    """
    head_prefixes = ("classifier.", "scale_embedding.")
    head_params: list[torch.Tensor] = []
    backbone_params: list[torch.Tensor] = []
    for name, p in model.named_parameters():
        is_head = (name == "scale_token") or any(name.startswith(pre) for pre in head_prefixes)
        (head_params if is_head else backbone_params).append(p)
    return [
        {"params": backbone_params, "lr": float(base_lr)},
        {"params": head_params, "lr": float(base_lr) * float(head_lr_multiplier)},
    ]


def _build_optimizer(
    model: nn.Module,
    cfg: TrainingConfig,
    optimizer_class: Callable | None,
) -> torch.optim.Optimizer:
    """Return AdamW8bit / vanilla AdamW / caller-supplied class with
    backbone+head param groups. wd from config; per-group lr set by
    ``_build_param_groups`` based on ``cfg.head_lr_multiplier``.
    """
    param_groups = _build_param_groups(model, cfg.lr, cfg.head_lr_multiplier)
    if optimizer_class is not None:
        return optimizer_class(param_groups, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw_8bit":
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit(param_groups, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
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
        val_loader: DataLoader | None = None,
        num_classes: int = 13,
        eval_every: int = 1,
        track_best_metric: str = "macro_f1",
        class_names: Sequence[str] | None = None,
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

        self.criterion = criterion if criterion is not None else build_criterion(config)
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
        # If a checkpoint was loaded, this records the last fully-completed
        # epoch so train() skips epochs that already finished. -1 means "no
        # epoch completed yet" (fresh trainer).
        self._resumed_from_epoch: int = -1
        # Per-micro-step raw-loss trace. Populated by _train_one_epoch.
        # Used by determinism tests (M4 task 4.4 acceptance) — the resume
        # path must produce element-wise identical losses to a continuous run.
        # Memory cost: 8 bytes/step → ~400 KB for 50k-step real training.
        self.step_losses: list[float] = []

        # M4 task 4.5: optional eval + best-model selection.
        self.eval_every = max(1, int(eval_every))
        self.track_best_metric = track_best_metric
        self.class_names = list(class_names) if class_names is not None else None
        self._best_metric_value: float = -math.inf
        self._best_ckpt_path: Path | None = None
        self.eval_history: list[dict] = []   # appended each eval
        if val_loader is not None:
            self.evaluator: Evaluator | None = Evaluator(
                model=self.model, val_loader=val_loader,
                num_classes=int(num_classes), device=device,
            )
            logger.info(
                f"Evaluator wired: track_best_metric={self.track_best_metric}, "
                f"eval_every={self.eval_every}, num_classes={num_classes}"
            )
        else:
            self.evaluator = None

        # Report the *configured* per-group LRs (i.e. the values used as
        # ``initial_lr`` by LambdaLR), not the live param_group["lr"] —
        # LambdaLR sets the live value to ``initial_lr × cosine_factor(0)``
        # = 0.0 at construction (the warmup-step-0 factor), which would
        # be misleading in the startup log.
        configured_backbone_lr = float(config.lr)
        configured_head_lr = float(config.lr) * float(config.head_lr_multiplier)
        logger.info(
            f"Trainer ready: device={device}, precision={config.precision}, "
            f"optimizer={config.optimizer}, lr={config.lr}, wd={config.weight_decay}, "
            f"head_lr_multiplier={config.head_lr_multiplier}, "
            f"configured_backbone_lr={configured_backbone_lr:.2e}, "
            f"configured_head_lr={configured_head_lr:.2e}, "
            f"batch_size={config.batch_size}, grad_accumulation={self.grad_accum}, "
            f"warmup_steps={self.warmup_steps}, total_steps={self.total_steps}, "
            f"min_lr_ratio={self.min_lr_ratio}, criterion={type(self.criterion).__name__}, "
            f"run_dir={self.run_dir}"
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

        # If we resumed from epoch K, train() should run epochs K+1..n_epochs-1.
        # Fresh trainers have _resumed_from_epoch=-1 → start at 0.
        start_epoch = self._resumed_from_epoch + 1
        if start_epoch > 0:
            logger.info(
                f"resuming from epoch {self._resumed_from_epoch}; "
                f"training epochs {start_epoch}..{n_epochs - 1}"
            )

        for epoch in range(start_epoch, n_epochs):
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

            # M4 task 4.5: eval + best-model selection BEFORE checkpoint save,
            # so the checkpoint can record the eval-determined best status.
            is_best = False
            if self.evaluator is not None and (epoch + 1) % self.eval_every == 0:
                metrics = self.evaluator.evaluate()
                self.evaluator.pretty_print(metrics, class_names=self.class_names)
                self.eval_history.append({"epoch": epoch, **metrics})
                if self.track_best_metric not in metrics:
                    raise KeyError(
                        f"track_best_metric={self.track_best_metric!r} not in "
                        f"evaluator output keys {sorted(metrics.keys())}"
                    )
                metric_val = float(metrics[self.track_best_metric])
                # Strict greater-than tiebreak: don't re-copy when metric
                # plateaus (avoids late-epoch I/O thrash with small lr).
                is_best = metric_val > self._best_metric_value
                if is_best:
                    self._best_metric_value = metric_val
                    logger.info(
                        f"new best {self.track_best_metric}={metric_val:.4f} "
                        f"at epoch {epoch}"
                    )

            ckpt_path = self.ckpt_dir / f"epoch_{epoch}_step_{self.global_grad_step}.pt"
            self.save_checkpoint(ckpt_path, epoch)
            ckpts.append(ckpt_path)
            self._resumed_from_epoch = epoch   # record completion for re-resume

            if is_best:
                # Plain copy (not symlink) for Windows/WSL2 portability.
                self._best_ckpt_path = self.ckpt_dir / "best.pt"
                shutil.copyfile(ckpt_path, self._best_ckpt_path)
                logger.info(f"best.pt updated → {self._best_ckpt_path}")

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

            raw_loss_val = float(raw_loss.detach().item())
            loss_sum += raw_loss_val
            loss_n += 1
            self.step_losses.append(raw_loss_val)

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

    def save_checkpoint(self, path: Path, epoch: int, *, step: int | None = None) -> None:
        """Persist full training state for resume (M4 task 4.4).

        Single ``.pt`` file. Contents (schema v1):

          - ``model``           : ``model.state_dict()``
          - ``optimizer``       : ``optimizer.state_dict()`` (8-bit AdamW
                                  state is serializable but only restorable
                                  back into the same optimizer class)
          - ``scheduler``       : ``LambdaLR.state_dict()`` (the lambda
                                  closure itself is not persisted; rebuild
                                  uses ``scheduler_config`` below)
          - ``scaler``          : ``GradScaler.state_dict()`` or ``None``
          - ``epoch``, ``global_grad_step``, ``global_micro_step``
          - ``rng``             : torch / cuda / numpy / python RNG states
          - ``training_config`` : pydantic ``model_dump()`` of
                                  ``TrainingConfig`` for sanity check on load
          - ``scheduler_config``: warmup_steps / total_steps / min_lr_ratio
                                  needed to rebuild LambdaLR with the
                                  matching lambda
          - ``schema_version``  : int, currently 1

        We use ``torch.save`` (pickle), not safetensors, because optimizer
        and RNG state are non-tensor Python objects. For deployment-only
        exports (model weights only) use :meth:`export_model_safetensors`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if step is None:
            step = self.global_grad_step
        ckpt = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "epoch": int(epoch),
            "global_grad_step": int(self.global_grad_step),
            "global_micro_step": int(self.global_micro_step),
            "scheduler_config": {
                "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps,
                "min_lr_ratio": self.min_lr_ratio,
            },
            "training_config": self.config.model_dump(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
            },
        }
        torch.save(ckpt, str(path))
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(
            f"checkpoint saved: {path} ({size_mb:.1f} MB) — "
            f"epoch={epoch}, grad_step={step}"
        )

    def load_checkpoint(self, path: Path) -> None:
        """Restore full training state from a ``.pt`` checkpoint.

        Forward-compat policy:
          - Unknown extra fields are ignored (do not raise).
          - Missing optional fields fall back to safe defaults.
          - ``schema_version`` newer than this code's
            ``CHECKPOINT_SCHEMA_VERSION`` raises explicitly — refusing to
            misread a future format is safer than silent best-effort.
          - Model state loads with ``strict=False`` so M3 ckpts (no
            scale_token / scale_embedding) still resume cleanly.
        """
        path = Path(path)
        # weights_only=False because the dict contains non-tensor python
        # objects (rng state, optimizer state). The file is trusted (we wrote
        # it ourselves on the same machine).
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

        version = ckpt.get("schema_version", 0)
        if version > CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema_version={version} newer than this code's "
                f"CHECKPOINT_SCHEMA_VERSION={CHECKPOINT_SCHEMA_VERSION}; "
                f"refusing to read"
            )

        # Model — strict=False for M3-from-M4 (missing scale_token etc).
        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            logger.info(f"load_checkpoint: model missing keys: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            logger.warning(f"load_checkpoint: model unexpected keys: {list(unexpected)[:5]}")

        # Optimizer
        self.optimizer.load_state_dict(ckpt["optimizer"])

        # Scheduler — uses the lambda from THIS Trainer's make_cosine_scheduler;
        # only last_epoch and counter state come from the ckpt.
        self.scheduler.load_state_dict(ckpt["scheduler"])

        # Scaler — only restore if both sides have one
        scaler_state = ckpt.get("scaler")
        if scaler_state is not None and self.scaler is not None:
            self.scaler.load_state_dict(scaler_state)
        elif scaler_state is not None and self.scaler is None:
            logger.warning(
                "load_checkpoint: ckpt has GradScaler state but current trainer has none "
                "(precision changed?); skipping"
            )
        elif scaler_state is None and self.scaler is not None:
            logger.warning(
                "load_checkpoint: ckpt has no GradScaler state but current trainer has one; "
                "scaler will keep its fresh-init state"
            )

        # RNG — restore all four. Use .get() with defaults for forward-compat
        # against checkpoints that didn't capture cuda state.
        rng = ckpt.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "torch_cuda" in rng and torch.cuda.is_available() and rng["torch_cuda"]:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

        # Counters + resume marker
        self.global_grad_step = int(ckpt.get("global_grad_step", 0))
        self.global_micro_step = int(ckpt.get("global_micro_step", 0))
        self._resumed_from_epoch = int(ckpt.get("epoch", -1))

        # Sanity check: scheduler_config must match what we built with.
        sc = ckpt.get("scheduler_config", {})
        for k, expected in (("warmup_steps", self.warmup_steps),
                            ("total_steps", self.total_steps),
                            ("min_lr_ratio", self.min_lr_ratio)):
            if k in sc and sc[k] != expected:
                logger.warning(
                    f"load_checkpoint: scheduler_config.{k} mismatch — "
                    f"ckpt={sc[k]} vs current trainer={expected}. "
                    f"This will cause lr-curve drift; rebuild trainer with "
                    f"matching scheduler args before resume."
                )

        logger.info(
            f"checkpoint loaded: {path} — epoch={self._resumed_from_epoch}, "
            f"grad_step={self.global_grad_step}, micro_step={self.global_micro_step}"
        )

    def export_model_safetensors(self, path: Path) -> None:
        """Export model weights only to safetensors for deployment.

        Distinct from :meth:`save_checkpoint`: that saves training state
        (optimizer/scheduler/RNG/...); this writes only the model
        ``state_dict`` in a safetensors file suitable for inference-only
        environments that don't ship pickle/torch.save trust.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()}
        save_file(
            state, str(path),
            metadata={
                "format": "safetensors-deploy",
                "precision": self.config.precision,
                "optimizer": self.config.optimizer,
                "lr": str(self.config.lr),
            },
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"model weights exported (safetensors): {path} ({size_mb:.1f} MB)")
