"""M6.3 — 2D snapshot paradigm baseline (ResNet-18 on middle frame t=8).

Two sub-cells under one architecture:

  * **M6.3.IN** — ResNet-18 with torchvision's ImageNet-pretrained
    ``conv1`` (and downstream block weights) adapted from 3-channel to
    6-channel via ``adapt_conv2d_to_6ch(..., n_extra=3)``. The
    downstream layers (layer1..layer4 + fc) inherit ImageNet weights
    unchanged; the ``fc`` is replaced with a fresh 13-class head.

  * **M6.3.RN** — ResNet-18 random-init, no ImageNet weights anywhere.
    ``conv1`` is freshly constructed with 6 input channels; all other
    layers Kaiming-init via torchvision's default constructor with
    ``weights=None``.

Both sub-cells consume the same data contract as the video cells:
input ``x`` shape ``(B, T=16, C=6, H=32, W=64)`` with the t=8 middle
frame extracted *inside* ``forward()`` (the dataloader is unmodified,
preserving splits.parquet identity / val_n bit-identity with the
video cells).

The model's ``forward`` signature matches the project's other
backbones (``forward(x, *, scale_id) → {"logits", "features"}``) so the
trainer's call site at ``Trainer._train_one_epoch`` doesn't need to
branch on model type. ``scale_id`` is accepted and ignored.

Path B head_lr contract (M5.5 R1 result):
  * M6.3.IN → ``--head-lr-multiplier 5.0`` (pretrained group)
  * M6.3.RN → ``--head-lr-multiplier 1.0`` (random group)

The trainer's ``_build_param_groups`` matcher already treats ``.fc``
as a head segment (torchvision ResNet's classifier is named ``fc``).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tv_models

from nid_video.models._adapters import adapt_conv2d_to_6ch
from nid_video.utils import logger


class ResNet18SnapshotForNID(nn.Module):
    """torchvision ResNet-18 adapted to 6-channel 2D snapshot input.

    Constructor selects the init regime via ``pretrained``:

      * ``True``  / ``"imagenet"`` → torchvision DEFAULT ImageNet-pretrained
        weights, then 6-channel conv1 adapter + fresh 13-class fc.
      * ``False`` / ``None`` / ``""`` / ``"none"`` / ``"false"`` / ``"0"`` →
        random Kaiming-init for the full network (torchvision
        ``weights=None``), 6-channel conv1, fresh 13-class fc.

    The 6-channel conv1 is built via ``adapt_conv2d_to_6ch`` with
    ``target_kernel=(7, 7)`` to match ResNet-18's native conv1 spatial
    kernel. When pretrained=True the adapter bilinear-interpolates the
    3-channel ImageNet kernel to (7, 7) (degenerate identity at matched
    sizes) and Kaiming-inits 3 extra channels. When pretrained=False
    the adapter is bypassed and a fresh 6-channel Conv2d is built
    directly via the standard Kaiming pattern.

    The ``forward`` extracts the middle frame ``x[:, 8]`` from the
    ``(B, T=16, C=6, H, W)`` input and runs ResNet-18 on the resulting
    ``(B, 6, H, W)`` 2D snapshot. The `t=8 middle frame` choice is
    documented in Phase 0 design — it's the most-information-rich
    snapshot index for a 16-frame window under the fairness contract.
    """

    MIDDLE_FRAME_INDEX = 8

    def __init__(
        self,
        num_classes: int = 13,
        pretrained: bool | str | None = True,
        in_channels: int = 6,
        gradient_checkpointing: bool = False,  # ResNet-18 is small; not needed
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be ≥ 1; got {in_channels}")

        self.in_channels = in_channels
        self.num_classes = num_classes

        load_pretrained = self._coerce_pretrained_flag(pretrained)
        self._init_regime = "imagenet" if load_pretrained else "random"

        if load_pretrained:
            self.backbone = tv_models.resnet18(
                weights=tv_models.ResNet18_Weights.DEFAULT,
            )
            logger.info(
                "ResNet-18 backbone loaded: torchvision DEFAULT ImageNet-pretrained"
            )
            # Replace conv1 with 6-channel adapter — preserves ImageNet
            # ch[0:3] kernels; Kaiming-inits the extra 3 channels.
            old_conv1 = self.backbone.conv1
            new_conv1 = adapt_conv2d_to_6ch(
                old_conv1,
                target_kernel=tuple(old_conv1.kernel_size),    # (7, 7)
                target_stride=tuple(old_conv1.stride),         # (2, 2)
                n_extra=in_channels - 3,
            )
            # Preserve padding (Conv2d.padding) from the source conv1.
            # ``adapt_conv2d_to_6ch`` copies it verbatim — verify here so the
            # 2D-snapshot stem matches ResNet-18's standard receptive field.
            assert tuple(new_conv1.padding) == tuple(old_conv1.padding), (
                f"padding drift: new={new_conv1.padding} vs old={old_conv1.padding}"
            )
            self.backbone.conv1 = new_conv1
        else:
            self.backbone = tv_models.resnet18(weights=None)
            logger.warning(
                "ResNet-18 random-init: torchvision weights=None; Kaiming throughout"
            )
            # Replace conv1 with a fresh 6-channel Conv2d (same hp as ResNet's
            # default — kernel=7 stride=2 padding=3 bias=False).
            self.backbone.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            nn.init.kaiming_normal_(
                self.backbone.conv1.weight, mode="fan_out", nonlinearity="relu",
            )

        # Diagnostics — record norms of ch[0:3] and ch[3:6] for the
        # forensic record (mirrors the video-cell adapter log).
        proj_w = self.backbone.conv1.weight.data
        ch3_norm = float(proj_w[:, :3].norm().item())
        ch_extra_norm = (
            float(proj_w[:, 3:].norm().item()) if in_channels > 3 else 0.0
        )
        logger.info(
            f"conv1 adapted ({self._init_regime}): "
            f"ch[0:3] norm={ch3_norm:.4f}; "
            f"ch[3:{in_channels}] norm={ch_extra_norm:.4f}; "
            f"shape={tuple(proj_w.shape)}"
        )

        # Replace fc with a 13-class head (fresh init). Trainer's param-group
        # matcher already treats ``fc`` as a head segment — no rename needed.
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        nn.init.normal_(self.backbone.fc.weight, std=0.02)
        nn.init.zeros_(self.backbone.fc.bias)

    @staticmethod
    def _coerce_pretrained_flag(p: bool | str | None) -> bool:
        if isinstance(p, bool):
            return p
        if p is None:
            return False
        if isinstance(p, str):
            return p.lower() not in ("", "none", "false", "0")
        return bool(p)

    # ----- forward -----

    def forward(
        self,
        x: torch.Tensor,
        *,
        scale_id: torch.Tensor | None = None,    # accepted + ignored
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: ``(B, T=16, C=in_channels, H, W)`` float tensor — the project's
              standard video input. The middle frame ``x[:, 8]`` is extracted
              inside this method.
            scale_id: ignored. Accepted for compatibility with the trainer's
              uniform model call site.

        Returns:
            ``{"logits": (B, num_classes), "features": (B, 512)}`` — the 512
            features come from ResNet-18's avgpool output (the input to fc).
        """
        if x.dim() != 5:
            raise ValueError(
                f"expected (B, T, C, H, W); got {tuple(x.shape)}"
            )
        T = x.size(1)
        if T <= self.MIDDLE_FRAME_INDEX:
            raise ValueError(
                f"need T > {self.MIDDLE_FRAME_INDEX}; got T={T}"
            )
        # Middle-frame snapshot extraction. Indexing is purely a view — no
        # data copy. Resulting shape: (B, C=in_channels, H, W).
        snapshot = x[:, self.MIDDLE_FRAME_INDEX]

        # Tolerate dataloader > model channel count (mirrors the M5.10 dim-2
        # forward guard in videomae_nid). Project default produces C=6; if a
        # future dataloader emits more channels we slice; if fewer we fail
        # loud.
        assert snapshot.size(1) >= self.in_channels, (
            f"input has {snapshot.size(1)} channels at t={self.MIDDLE_FRAME_INDEX}; "
            f"model expects ≥ {self.in_channels}"
        )
        if snapshot.size(1) > self.in_channels:
            snapshot = snapshot[:, :self.in_channels]

        # Replicate torchvision ResNet's forward up to but not including
        # fc, so we can expose ``features`` separately for downstream use.
        b = self.backbone
        h = b.conv1(snapshot)
        h = b.bn1(h)
        h = b.relu(h)
        h = b.maxpool(h)

        h = b.layer1(h)
        h = b.layer2(h)
        h = b.layer3(h)
        h = b.layer4(h)

        h = b.avgpool(h)
        feat = torch.flatten(h, 1)
        logits = b.fc(feat)
        return {"logits": logits, "features": feat}
