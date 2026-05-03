"""I3D-R50 backbone, K400-pretrained, for NID (M5.5 R2 baseline 3).

Wraps pytorchvideo's K400-pretrained I3D ResNet-50 (Carreira &
Zisserman 2017, ~28M params) with three adaptations for the project's
(T=16, C=6, H=32, W=64) NID input:

  1. patch-stem Conv3d adapted from 3 → 6 channels via
     ``adapt_conv3d_to_6ch``. The source kernel is (5, 7, 7) and we
     target (5, 7, 7) too — i.e. the trilinear "downsample" is the
     identity transform; ch[0:3] is BYTE-IDENTICAL to the K400
     pretrained patch_embed weights, ch[3:6] is Kaiming-initialised.
     Pinned by ``test_i3d_pretrained_first_three_channels_preserved``.

  2. ``blocks[6].pool`` replaced from a fixed ``AvgPool3d(4, 7, 7)``
     to ``AdaptiveAvgPool3d(1)``: at our (T=16, H=32, W=64) input the
     post-blocks feature map is (T=8, H=1, W=2), too narrow for a
     7×7 spatial pool. Adaptive collapses (T', H', W') → (1, 1, 1)
     regardless of input geometry.

  3. ``blocks[6].proj`` replaced with a fresh ``nn.Linear(2048, 13)``
     classifier for our 13-class collapsed CIC labels.

The K400 transformer-block weights (~27.5M of the 28M total) all
transfer cleanly. Only the patch_embed extra channels (ch[3:6]) and
the classifier are fresh.

Head LR group (M5.5 Path B): I3D inherits K400 pretraining, so it
trains with head_lr_multiplier=5.0 (M5.4 P2 contract — slow backbone
preserves pretraining, fast head learns from scratch). The head
matcher segment-matches ``proj`` (the renamed
``blocks.6.proj.classifier`` path) and ``classifier`` if any
ancestor has that name.

Forward signature mirrors the M5.5 baselines convention:
``forward(x, *, scale_id) → {"logits", "features"}``. ``scale_id`` is
ignored — I3D is scale-agnostic.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nid_video.models._adapters import adapt_conv3d_to_6ch
from nid_video.utils import logger


_K400_HUB = "i3d_r50"   # pytorchvideo.models.hub.i3d_r50


def _load_backbone_with_fallback(pretrained: bool) -> nn.Module:
    """Load pytorchvideo's I3D-R50; fall back to random init on failure.

    pytorchvideo's hub returns a ``pytorchvideo.models.net.Net`` whose
    ``blocks`` ModuleList is the standard ResNet-50 I3D layout (stem +
    4 res-stages + head).
    """
    from pytorchvideo.models.hub import i3d_r50
    try:
        model = i3d_r50(pretrained=bool(pretrained))
        if pretrained:
            logger.info(f"loaded pretrained backbone: pytorchvideo {_K400_HUB} (K400)")
        else:
            logger.warning(
                f"i3d_r50 built with pretrained=False (random init)"
            )
        return model
    except (OSError, ConnectionError, ValueError) as exc:
        logger.warning(
            f"failed to load pretrained {_K400_HUB!r}: "
            f"{type(exc).__name__}: {exc}"
        )
        return i3d_r50(pretrained=False)


class I3DForNID(nn.Module):
    """I3D-R50 (28M, K400-pretrained) adapted for the (T=16, C=6,
    H=32, W=64) NID tensor.

    Args:
      num_classes: classification head output dim. Default 13.
      pretrained: load pytorchvideo K400 weights (default True).
        Pass False for fast random-init tests.
      in_channels: input channel count (default 6). Adapted from the
        K400 3-channel patch_embed via ``adapt_conv3d_to_6ch``.
      gradient_checkpointing: default ``True`` to match the
        training_perf.yaml convention.
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
        self._replace_head_pool()
        self._replace_classifier(num_classes)

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(
            f"i3d-r50 built: pretrained={pretrained}, in_ch={in_channels}, "
            f"num_classes={num_classes}, params={n_params:.2f}M"
        )

    # ----- gradient checkpointing -----

    def enable_gradient_checkpointing(self) -> None:
        self._grad_ckpt_enabled = True

    def disable_gradient_checkpointing(self) -> None:
        self._grad_ckpt_enabled = False

    # ----- patch stem adaptation (3 → 6 channels) -----

    def _adapt_patch_stem(self) -> None:
        """Adapt blocks[0].conv from Conv3d(3, 64, 5, 7, 7) to
        Conv3d(in_channels, 64, 5, 7, 7) via the shared 3D adapter.
        Source kernel == target kernel, so ch[0:3] is bit-identical to
        K400 (the trilinear "downsample" is the identity transform).
        """
        stem = self.backbone.blocks[0]
        old_conv: nn.Conv3d = stem.conv
        if old_conv.weight.shape[1] != 3:
            raise RuntimeError(
                f"expected K400 stem with in_ch=3, got {old_conv.weight.shape[1]}"
            )
        out_ch = old_conv.weight.shape[0]
        T_orig, H_orig, W_orig = old_conv.weight.shape[2:]
        n_extra = self.in_channels - 3

        new_conv = adapt_conv3d_to_6ch(
            old_conv,
            target_kernel=(T_orig, H_orig, W_orig),
            target_stride=tuple(old_conv.stride),
            n_extra=n_extra,
        )
        stem.conv = new_conv

        ch3_norm = new_conv.weight.data[:, :3].norm().item()
        ext_norm = new_conv.weight.data[:, 3:].norm().item() if n_extra > 0 else 0.0
        logger.info(
            f"i3d patch_stem adapted: ch[0:3] bit-identical to K400 "
            f"shape={tuple(new_conv.weight.data[:, :3].shape)} norm={ch3_norm:.2f}; "
            f"ch[3:{self.in_channels}] kaiming-init shape="
            f"{tuple(new_conv.weight.data[:, 3:].shape) if n_extra > 0 else 'n/a'} "
            f"norm={ext_norm:.2f}"
        )

    # ----- head pool replacement (fixed → adaptive) -----

    def _replace_head_pool(self) -> None:
        """The K400 head's ``AvgPool3d(4, 7, 7)`` assumes feature map
        ≥ (4, 7, 7). At our (T=16, 32, 64) input the post-blocks map is
        (T=8, H=1, W=2) — too narrow. ``AdaptiveAvgPool3d(1)`` collapses
        to (1, 1, 1) regardless of input shape.

        ``output_pool`` (an AdaptiveAvgPool3d in the original head) is
        replaced with ``Identity`` since adaptive_pool above already
        produces the (1, 1, 1) collapse.
        """
        head = self.backbone.blocks[6]
        head.pool = nn.AdaptiveAvgPool3d(1)
        head.output_pool = nn.Identity()

    # ----- classifier replacement -----

    def _replace_classifier(self, num_classes: int) -> None:
        """Replace blocks[6].proj from K400 Linear(2048, 400) with a
        fresh Linear(2048, 13) for our collapsed CIC labels.
        """
        head = self.backbone.blocks[6]
        in_features = head.proj.in_features
        head.proj = nn.Linear(in_features, num_classes)

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
                internally to (B, C, T, H, W) for I3D's expected
                layout.
            scale_id: (B,) long tensor — IGNORED. I3D is scale-agnostic.

        Returns:
            dict with
              ``logits``:   (B, num_classes)
              ``features``: (B, 2048)  — pre-classifier representation
                read by hooking the input to ``blocks[6].proj`` so we
                avoid a second forward.
        """
        del scale_id

        # (B, T, C, H, W) → (B, C, T, H, W) for I3D.
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # Capture the pre-classifier representation via a forward-hook
        # registered on blocks[6].proj — the input to proj is the
        # adaptive-pooled, dropout'd 2048-d feature vector.
        feat_holder: list[torch.Tensor] = []

        def hook(_module, inputs, _output):
            feat_holder.append(inputs[0].detach().clone())

        head_proj = self.backbone.blocks[6].proj
        h = head_proj.register_forward_hook(hook)
        try:
            logits = self.backbone(x)             # (B, num_classes)
        finally:
            h.remove()

        # The hook captures the proj input post-pool / post-dropout
        # post-permute. pytorchvideo's ResNetBasicHead permutes the
        # (B, C, T', H', W') pool output to (B, T', H', W', C) before
        # the Linear, so the captured shape is (B, 1, 1, 1, 2048) when
        # adaptive_pool collapses the spatio-temporal dims to 1. Flatten
        # the singleton dims to give the conventional (B, 2048).
        if feat_holder:
            feat = feat_holder[0]
            # Squeeze any singleton dims between batch and the final
            # feature dim.
            while feat.dim() > 2:
                feat = feat.squeeze(1)
        else:
            feat = logits.new_zeros(logits.size(0), 2048)
        return {"logits": logits, "features": feat}
