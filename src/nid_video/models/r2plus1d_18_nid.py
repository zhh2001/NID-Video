"""R(2+1)D-18 backbone, K400-pretrained, for NID (M5.5 R2 baseline 4).

Wraps torchvision's K400-pretrained R(2+1)D-18 (Tran et al. 2018,
~31.5M params) with two adaptations for the project's (T=16, C=6,
H=32, W=64) NID input:

  1. patch-stem ``stem[0]`` Conv3d adapted from 3 → 6 channels via
     ``adapt_conv3d_to_6ch``. The source kernel is (1, 7, 7) and we
     target (1, 7, 7) too — i.e. the trilinear "downsample" is the
     identity transform; ch[0:3] is BYTE-IDENTICAL to the K400
     pretrained stem weights, ch[3:6] is Kaiming-initialised. Pinned
     by ``test_r2plus1d_18_pretrained_first_three_channels_preserved``.

  2. ``fc`` replaced from K400 ``Linear(512, 400)`` to fresh
     ``Linear(512, 13)`` for our 13-class collapsed CIC labels.

Unlike I3D, R(2+1)D-18's head pool is already adaptive
(``AdaptiveAvgPool3d(1)``) so no pool replacement is needed; the model
forwards cleanly at our (T=16, 32, 64) input.

Head LR group (M5.5 Path B): R(2+1)D-18 inherits K400 pretraining and
trains with head_lr_multiplier=5.0 (M5.4 P2 contract). The head matcher
segment-matches ``fc`` → head group; the renamed K400 ``fc.weight`` /
``fc.bias`` (now Linear(512, 13)) lands in the head group at 5× LR.

Forward signature mirrors the M5.5 baselines convention:
``forward(x, *, scale_id) → {"logits", "features"}``. ``scale_id`` is
ignored — R(2+1)D-18 is scale-agnostic.
"""

from __future__ import annotations

import torch
from torch import nn

from nid_video.models._adapters import adapt_conv3d_to_6ch
from nid_video.utils import logger


def _load_backbone_with_fallback(pretrained: bool) -> nn.Module:
    """Load torchvision r2plus1d_18; fall back to random init on failure."""
    from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
    try:
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        model = r2plus1d_18(weights=weights)
        if pretrained:
            logger.info("loaded pretrained backbone: torchvision r2plus1d_18 (K400)")
        else:
            logger.warning("r2plus1d_18 built with weights=None (random init)")
        return model
    except (OSError, ConnectionError, ValueError) as exc:
        logger.warning(
            f"failed to load pretrained r2plus1d_18 K400 weights: "
            f"{type(exc).__name__}: {exc}"
        )
        return r2plus1d_18(weights=None)


class R2Plus1D18ForNID(nn.Module):
    """R(2+1)D-18 (~31.5M, K400-pretrained) adapted for the
    (T=16, C=6, H=32, W=64) NID tensor.

    Args:
      num_classes: classification head output dim. Default 13.
      pretrained: load torchvision K400 weights (default True). Pass
        False for fast random-init tests.
      in_channels: input channel count (default 6). Adapted from the
        K400 3-channel stem via ``adapt_conv3d_to_6ch``.
      gradient_checkpointing: default ``True`` to match the
        training_perf.yaml convention; this model is small enough at
        batch=32 that checkpointing has marginal effect, but the API
        is here for trainer-uniformity.
    """

    def __init__(
        self,
        num_classes: int = 13,
        pretrained: bool = True,
        in_channels: int = 6,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if in_channels < 3:
            raise ValueError(
                f"in_channels must be ≥ 3 to receive the 3-channel pretraining; "
                f"got {in_channels}"
            )
        self.num_classes = num_classes
        self.in_channels = in_channels
        self._grad_ckpt_enabled = bool(gradient_checkpointing)

        self.backbone = _load_backbone_with_fallback(pretrained)

        self._adapt_patch_stem()
        self._replace_classifier(num_classes)

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(
            f"r2plus1d_18 built: pretrained={pretrained}, in_ch={in_channels}, "
            f"num_classes={num_classes}, params={n_params:.2f}M"
        )

    # ----- gradient checkpointing -----

    def enable_gradient_checkpointing(self) -> None:
        self._grad_ckpt_enabled = True

    def disable_gradient_checkpointing(self) -> None:
        self._grad_ckpt_enabled = False

    # ----- patch stem adaptation (3 → 6 channels) -----

    def _adapt_patch_stem(self) -> None:
        """Adapt stem[0] from Conv3d(3, 45, 1, 7, 7) to Conv3d(in,
        45, 1, 7, 7). Source kernel == target → ch[0:3] bit-identical
        to K400.
        """
        stem = self.backbone.stem
        old_conv: nn.Conv3d = stem[0]
        if old_conv.weight.shape[1] != 3:
            raise RuntimeError(
                f"expected K400 stem[0] with in_ch=3, got {old_conv.weight.shape[1]}"
            )
        T_orig, H_orig, W_orig = old_conv.weight.shape[2:]
        n_extra = self.in_channels - 3

        new_conv = adapt_conv3d_to_6ch(
            old_conv,
            target_kernel=(T_orig, H_orig, W_orig),
            target_stride=tuple(old_conv.stride),
            n_extra=n_extra,
        )
        stem[0] = new_conv

        ch3_norm = new_conv.weight.data[:, :3].norm().item()
        ext_norm = new_conv.weight.data[:, 3:].norm().item() if n_extra > 0 else 0.0
        logger.info(
            f"r2plus1d_18 patch_stem adapted: ch[0:3] bit-identical to K400 "
            f"shape={tuple(new_conv.weight.data[:, :3].shape)} norm={ch3_norm:.2f}; "
            f"ch[3:{self.in_channels}] kaiming-init shape="
            f"{tuple(new_conv.weight.data[:, 3:].shape) if n_extra > 0 else 'n/a'} "
            f"norm={ext_norm:.2f}"
        )

    # ----- classifier replacement -----

    def _replace_classifier(self, num_classes: int) -> None:
        """Replace fc from K400 Linear(512, 400) with a fresh
        Linear(512, 13).
        """
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    # ----- forward -----

    def forward(
        self,
        x: torch.Tensor,
        *,
        scale_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T=16, C=6, H=32, W=64) float tensor. Permuted
                internally to (B, C, T, H, W) for torchvision's
                expected layout.
            scale_id: (B,) long tensor — IGNORED.

        Returns:
            dict with
              ``logits``:   (B, num_classes)
              ``features``: (B, 512)  — pre-fc representation captured
                via forward hook on ``backbone.fc`` input.
        """
        del scale_id

        # (B, T, C, H, W) → (B, C, T, H, W) for torchvision.
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        feat_holder: list[torch.Tensor] = []

        def hook(_module, inputs, _output):
            feat_holder.append(inputs[0].detach().clone())

        h = self.backbone.fc.register_forward_hook(hook)
        try:
            logits = self.backbone(x)             # (B, num_classes)
        finally:
            h.remove()

        if feat_holder:
            feat = feat_holder[0]
            while feat.dim() > 2:
                feat = feat.squeeze(1)
        else:
            feat = logits.new_zeros(logits.size(0), 512)
        return {"logits": logits, "features": feat}
