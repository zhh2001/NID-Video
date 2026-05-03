"""C3D-Small backbone, random-init from scratch, for NID (M5.5 R2 baseline).

Slim variant of the published C3D (Tran et al. 2014, Sports-1M),
trimmed to ~19M params and the project's (T=16, C=6, H=32, W=64) NID
input. The depth (8 conv3d + 5 max-pool + 3 FC) and stride pattern
follow the original C3D recipe; the channel widths follow the
published 64/128/256/256/512 ladder but cap conv5 at 384 to land in
the matched-baseline band (~20M) — the published C3D's bulk lived in
its 4096-dim FC layers (~70M), which we replace with 1024/512 because
our (T=16, 32×64) input post-pool5 yields a much smaller flat dim
(384 × 1 × 1 × 2 = 768) than the original 8 × 7 × 7 × 512 = 25,088.

Pool plan for input (T=16, H=32, W=64):
  pool1 = (1, 2, 2)  →  T=16  H=16  W=32   (preserve temporal extent)
  pool2 = (2, 2, 2)  →  T= 8  H= 8  W=16
  pool3 = (2, 2, 2)  →  T= 4  H= 4  W= 8
  pool4 = (2, 2, 2)  →  T= 2  H= 2  W= 4
  pool5 = (2, 2, 2)  →  T= 1  H= 1  W= 2
After pool5 the conv volume is (256, 1, 1, 2) — flat dim 512 → FC6.

Pretrained source: NONE. The original C3D ckpt is Sports-1M and at the
original 8×112×112 spatial shape; remapping to (T=16, 32×64) and then
to 6 channels would require both spatial-temporal and channel
adaptation, with little signal preservation. No public small-variant
K400 ckpt exists at this scale. Random init is the honest contract.

Head LR group (M5.5 Path B): the M5.5 R1.5 forensic finding showed
head_lr ×5 hurts from-scratch backbones (random init dropped combined
macro_f1 by 0.022 vs head_lr ×1). C3D-Small belongs to the random-init
fairness group and trains with head_lr_multiplier=1.0; only the K400
pretrained baselines (I3D / R(2+1)D-18, M5.5 R2 rows 3–4) inherit
the M5.4 P2 head_lr ×5 recipe.

Forward signature mirrors the M5.5 baselines convention:
``forward(x, *, scale_id) → {"logits", "features"}``. ``scale_id`` is
ignored — C3D is scale-agnostic and consumes the multi-scale
dataloader's mixed batches without per-stream conditioning.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nid_video.utils import logger


class C3DSmallForNID(nn.Module):
    """C3D-Small (~20M random-init) for the NID (T=16, C=6, H=32, W=64)
    tensor.

    Args:
      num_classes: classification head output dim. Default 13.
      in_channels: input channel count (default 6). Used directly as
        the first Conv3d's input dim — no 3→6 adapter (no pretrained
        weights to preserve).
      gradient_checkpointing: default ``True`` to match the
        training_perf.yaml convention; the model is small enough to
        run without checkpointing in fp16, but keep the API uniform
        across baselines.
    """

    # Channel widths for the 8 conv3d layers (follows the published
    # C3D 64/128/256/256/512 ladder, capped at 384 for conv5 to land
    # at the matched-baseline ~19M total).
    _CHANNELS: tuple[int, int, int, int, int, int, int, int] = (
        64, 128, 256, 256, 384, 384, 384, 384,
    )

    def __init__(
        self,
        num_classes: int = 13,
        in_channels: int = 6,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self._grad_ckpt_enabled = bool(gradient_checkpointing)

        c1, c2, c3a, c3b, c4a, c4b, c5a, c5b = self._CHANNELS

        # Conv block 1 (T preserved by pool1).
        self.conv1 = nn.Conv3d(in_channels, c1, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Conv block 2.
        self.conv2 = nn.Conv3d(c1, c2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Conv block 3 (two 3d convs before pool, per published C3D).
        self.conv3a = nn.Conv3d(c2, c3a, kernel_size=3, padding=1)
        self.conv3b = nn.Conv3d(c3a, c3b, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Conv block 4.
        self.conv4a = nn.Conv3d(c3b, c4a, kernel_size=3, padding=1)
        self.conv4b = nn.Conv3d(c4a, c4b, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Conv block 5.
        self.conv5a = nn.Conv3d(c4b, c5a, kernel_size=3, padding=1)
        self.conv5b = nn.Conv3d(c5a, c5b, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        # FC head: pool5 yields (c5b, 1, 1, 2) → flat 768 with c5b=384.
        self._flat_dim = c5b * 1 * 1 * 2
        self.fc6 = nn.Linear(self._flat_dim, 1024)
        self.fc7 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(p=0.5)

        # ``classifier`` is the contract attribute the M5.5 R1.5
        # matcher segment-matches against to put head params in the
        # head_lr group (when head_lr_multiplier > 1.0). C3D-Small uses
        # head_lr_multiplier=1.0 per Path B so the grouping is a no-op
        # in this run, but the convention is kept for parity.
        self.classifier = nn.Linear(512, num_classes)

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(
            f"c3d-small built (random init): in_ch={in_channels}, "
            f"channels={self._CHANNELS}, fc_dims=(1024, 512, {num_classes}), "
            f"params={n_params:.2f}M"
        )

    # ----- gradient checkpointing -----
    #
    # C3D activation memory at batch=32 is small (the largest tensor
    # is post-conv1 at 32 × 32 × 16 × 32 × 64 ≈ 64 MB fp16); the API
    # exists for trainer-uniformity and records the flag for parity
    # diagnostics.

    def enable_gradient_checkpointing(self) -> None:
        self._grad_ckpt_enabled = True

    def disable_gradient_checkpointing(self) -> None:
        self._grad_ckpt_enabled = False

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
                internally to (B, C, T, H, W) for the Conv3d stack.
            scale_id: (B,) long tensor — IGNORED. C3D-Small is
                scale-agnostic; it consumes the multi-scale dataloader's
                mixed batches but does not condition on which stream a
                sample came from.

        Returns:
            dict with
              ``logits``:   (B, num_classes)
              ``features``: (B, 512)  — the FC7 activation, used as
                the representation slot for parity with the other
                M5.5 baselines.
        """
        del scale_id

        # (B, T, C, H, W) → (B, C, T, H, W) for Conv3d.
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)

        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        x = self.pool4(x)

        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = F.relu(self.fc7(x))
        feat = x
        logits = self.classifier(self.dropout(x))
        return {"logits": logits, "features": feat}
