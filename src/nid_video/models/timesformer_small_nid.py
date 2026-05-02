"""TimeSformer-Small backbone, random-init from scratch, for NID (M5.5 baseline).

This is the matched-param baseline — ~22M parameters, mirroring the main
method's VideoMAE-Small. The architecture follows the TimeSformer paper
(divided space-time attention) at Small dims:

  * hidden_size=384, num_hidden_layers=12, num_attention_heads=6
    (head_dim=64), intermediate_size=1536, attention_type="divided_space_time"
  * num_frames=16, image_size=64, patch_size=16  →  spatial grid 4×4 = 16
    patches per frame × 16 frames = 256 spatial-temporal patches + 1 CLS
  * num_channels=6: Conv2d's first layer is built directly with 6 input
    channels and Kaiming-initialised. No 3→6 adapter is needed because
    no pretrained weights to preserve.
  * 13-class linear head over the CLS-pooled token output.

Pretrained-checkpoint asymmetry vs the M5.5 baseline suite:
  * R(2+1)D-18 + I3D inherit Kinetics-400 weights (publicly available
    at the project's target scale).
  * TimeSformer at 22M-Small + C3D-Small + ConvLSTM have no
    publicly-released checkpoints at this scale and run from scratch.
  * The asymmetry reflects the open-source video-backbone ecosystem,
    not a project choice. The trajectory.md fairness table records
    pretrained-status per row so reviewers can read the contract
    directly.

Forward signature mirrors VideoMAESmallForNID: ``forward(x, *, scale_id)
→ {"logits", "features"}``. ``scale_id`` is ignored — TimeSformer-Small
is scale-agnostic by design and consumes the multi-scale dataloader's
mixed batches without conditioning on which stream each sample came
from. This asymmetry is the comparison's purpose: the multi-scale
conditioning is a contribution of the main method, and the comparison
reveals whether scale_token contributes beyond exposure to multi-scale
data alone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import TimesformerConfig, TimesformerForVideoClassification

from nid_video.utils import logger


def _timesformer_small_config(
    num_frames: int, image_size: int, num_labels: int, num_channels: int,
) -> TimesformerConfig:
    """TimeSformer-Small at 22M params — matched to the main method's
    VideoMAE-Small. Divided space-time attention is the published
    TimeSformer default and the comparison-relevant choice for this
    project (vs joint-attention / temporal-only variants).
    """
    return TimesformerConfig(
        num_frames=num_frames,
        image_size=image_size,
        patch_size=16,
        num_channels=num_channels,
        num_labels=num_labels,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,        # head_dim=64
        intermediate_size=1536,
        attention_type="divided_space_time",
    )


class TimeSformerSmallForNID(nn.Module):
    """TimeSformer-Small (~22M, random-init from-scratch) adapted for the
    (T=16, C=6, H=32, W=64) NID tensor.

    Args:
      num_classes: classification head output dim. Default 13.
      in_channels: input channel count (default 6). Used directly as
        the patch_embed Conv2d's input dim — no 3→6 adapter needed
        because there are no pretrained weights to preserve.
      target_image_size: square image size the model is constructed
        for. Default 64 — the smallest multiple-of-16 that contains
        our 32×64 input after H-padding. Must be a multiple of
        patch_size=16.
      num_frames: temporal extent. Default 16 (matches our T=16
        window).
      gradient_checkpointing: default ``True`` to match the
        training_perf.yaml convention used by the main method.
    """

    def __init__(
        self,
        num_classes: int = 13,
        in_channels: int = 6,
        target_image_size: int = 64,
        num_frames: int = 16,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if target_image_size % 16 != 0:
            raise ValueError(
                f"target_image_size must be a multiple of patch_size=16; "
                f"got {target_image_size}"
            )
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.target_image_size = target_image_size
        self.num_frames = num_frames

        cfg = _timesformer_small_config(
            num_frames=num_frames,
            image_size=target_image_size,
            num_labels=num_classes,
            num_channels=in_channels,
        )
        self.backbone = TimesformerForVideoClassification(cfg)

        # Diagnostics: surface the random-init patch_embed norm so a
        # downstream regression that accidentally re-weights the conv
        # (e.g. a future refactor that imports an adapter) is visible.
        proj = self.backbone.timesformer.embeddings.patch_embeddings.projection
        logger.info(
            f"timesformer-small built (random init): "
            f"patch_embed Conv2d weight shape={tuple(proj.weight.shape)} "
            f"norm={proj.weight.data.norm().item():.2f}, "
            f"hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
            f"heads={cfg.num_attention_heads}, attn={cfg.attention_type}"
        )

        if gradient_checkpointing:
            self.enable_gradient_checkpointing()

    # ----- gradient checkpointing -----

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    # ----- forward -----

    def forward(
        self,
        x: torch.Tensor,
        *,
        scale_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, C, H, W) float tensor with our project's
                T=16, C=6, H=32, W=64. The H dim is zero-padded to
                ``target_image_size`` (default 64) before being fed
                to TimeSformer, which expects square frames.
            scale_id: (B,) long tensor — IGNORED. TimeSformer-Small
                is scale-agnostic; it consumes the multi-scale
                dataloader's mixed batches but does not condition
                representations on which stream a sample came from.
                Accepted in the signature so the trainer's uniform
                call site works.

        Returns:
            dict with
              ``logits``:   (B, num_classes)
              ``features``: (B, hidden_size) — mean-pooled
                pre-classifier representation across token outputs.
        """
        del scale_id   # intentionally ignored

        B, T, C, H, W = x.shape
        if H > self.target_image_size or W > self.target_image_size:
            raise ValueError(
                f"input H={H}, W={W} exceeds target_image_size="
                f"{self.target_image_size}"
            )
        pad_h = self.target_image_size - H
        pad_w = self.target_image_size - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        out = self.backbone(pixel_values=x)
        logits = out.logits

        out_full = self.backbone.timesformer(
            pixel_values=x, output_hidden_states=False, return_dict=True,
        )
        feat = out_full.last_hidden_state.mean(dim=1)
        return {"logits": logits, "features": feat}
