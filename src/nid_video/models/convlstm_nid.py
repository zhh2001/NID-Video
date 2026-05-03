"""ConvLSTM backbone, random-init from scratch, for NID (M5.5 R2 baseline).

Stacked ConvLSTM (Shi et al. 2015 — originally precipitation nowcasting;
3D recurrence at input spatial resolution). For our (T=16, C=6, H=32,
W=64) NID input the original "no spatial reduction" design produces
prohibitive activation memory under autograd over T=16 timesteps
(measured: 9.8 GB peak at hidden=(64,128,192)). We adopt the standard
classification-task adaptation: spatial 2×2 max-pool between cells, so
hidden state shrinks from (32, 64) → (16, 32) → (8, 16) over the three
stacked cells. Hidden widths grow inversely (64 → 128 → 256) to keep
representation capacity roughly constant per cell.

Per timestep flow:
  x_t        : (B, 6,  32, 64)
  cell1 → h  : (B, 64, 32, 64)
  pool12     : (B, 64, 16, 32)
  cell2 → h  : (B, 128,16, 32)
  pool23     : (B, 128, 8, 16)
  cell3 → h  : (B, 256, 8, 16)

After T=16 timesteps the final cell's hidden state is pooled 2×2 to
(256, 4, 8), flattened to 8192, projected through a 1024-dim
representation layer (named ``feature_proj`` to avoid the M5.5 R1.5
segment matcher mistaking it for the classifier head), then classified
to 13 logits via ``self.classifier``. Total params ~13.1M lands at
the matched-baseline scale.

Pretrained source: NONE. ConvLSTM has no widely-released video-action
pretrained ckpt at this hidden size and 3-channel input would not
transfer to our 6-channel NID input cleanly. Random init is the
honest contract.

Head LR group (M5.5 Path B): the M5.5 R1.5 forensic finding showed
head_lr ×5 hurts from-scratch backbones. ConvLSTM trains with
head_lr_multiplier=1.0; only K400 pretrained baselines (I3D /
R(2+1)D-18) inherit the M5.4 P2 head_lr ×5 recipe.

Forward signature mirrors the M5.5 baselines convention:
``forward(x, *, scale_id) → {"logits", "features"}``. ``scale_id`` is
ignored.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nid_video.utils import logger


class _ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell (Shi et al. 2015).

    The 4-gate Conv2d projects [input ; hidden] of shape
    (B, in_ch + hidden_ch, H, W) to (B, 4 × hidden_ch, H, W), then
    splits into input / forget / cell / output gates with sigmoid +
    tanh activations. GroupNorm(num_groups=1) on the new hidden state
    is the LayerNorm-on-feature-map analogue (PyTorch has no clean
    LayerNorm-2d module).
    """

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=hidden_channels)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)
        gate_logits = self.gates(combined)
        i, f, g, o = torch.chunk(gate_logits, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        h_new = self.norm(h_new)
        return h_new, c_new

    def init_state(
        self, batch_size: int, spatial: tuple[int, int],
        device: torch.device, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = spatial
        h = torch.zeros(batch_size, self.hidden_channels, H, W,
                        device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_channels, H, W,
                        device=device, dtype=dtype)
        return h, c


class ConvLSTMForNID(nn.Module):
    """3-layer ConvLSTM (~13.1M random-init) for the NID
    (T=16, C=6, H=32, W=64) tensor.

    Args:
      num_classes: classification head output dim. Default 13.
      in_channels: input channel count (default 6).
      hidden_channels: per-layer hidden state widths. Default
        (64, 128, 256). Each cell's input spatial extent is halved
        vs the previous cell's (after a 2×2 max-pool between cells).
      gradient_checkpointing: default ``True`` for trainer-uniformity.
        ConvLSTM's recurrent unroll is the tight memory regime; the
        spatial pooling strategy above brings it within 8 GB at
        batch=32 fp16 without per-timestep checkpointing, so this
        flag is metadata only.
    """

    def __init__(
        self,
        num_classes: int = 13,
        in_channels: int = 6,
        hidden_channels: tuple[int, int, int] = (64, 128, 256),
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self._grad_ckpt_enabled = bool(gradient_checkpointing)

        h1, h2, h3 = hidden_channels
        self.cell1 = _ConvLSTMCell(in_channels, h1)
        self.cell2 = _ConvLSTMCell(h1, h2)
        self.cell3 = _ConvLSTMCell(h2, h3)

        # Spatial pools between cells: (32,64) → (16,32) → (8,16).
        self.pool12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool23 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Final pool after T=16 timesteps: (8,16) → (4,8). Flat = h3 × 32.
        self.pool_out = nn.MaxPool2d(kernel_size=2, stride=2)
        self._flat_dim = h3 * 4 * 8

        # Pre-classifier feature projection — named ``feature_proj`` to
        # avoid the M5.5 R1.5 segment matcher (which discovers
        # ``classifier`` / ``fc`` / ``proj`` ancestors as head params)
        # treating this 8M-param Linear as classification head. The
        # ``classifier`` Linear below is the actual head.
        self.feature_proj = nn.Linear(self._flat_dim, 1024)
        self.dropout = nn.Dropout(p=0.5)

        # ``classifier`` is the contract attribute the M5.5 R1.5
        # matcher segment-matches against. ConvLSTM uses
        # head_lr_multiplier=1.0 per Path B so the grouping is a no-op
        # in this run, but the convention is kept for parity.
        self.classifier = nn.Linear(1024, num_classes)

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(
            f"convlstm built (random init): in_ch={in_channels}, "
            f"hidden={hidden_channels}, "
            f"spatial_per_cell=((32,64),(16,32),(8,16)), "
            f"feature_proj=1024, classes={num_classes}, "
            f"params={n_params:.2f}M"
        )

    def enable_gradient_checkpointing(self) -> None:
        self._grad_ckpt_enabled = True

    def disable_gradient_checkpointing(self) -> None:
        self._grad_ckpt_enabled = False

    def forward(
        self,
        x: torch.Tensor,
        *,
        scale_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T=16, C=6, H=32, W=64) float tensor. Iterated along
                the T axis; each timestep updates (h, c) state of each
                cell. Spatial 2×2 max-pool between cells reduces
                activation memory across the recurrent unroll.
            scale_id: (B,) long tensor — IGNORED.

        Returns:
            dict with
              ``logits``:   (B, num_classes)
              ``features``: (B, 1024) — post-feature_proj
                representation, used as the feature slot for parity
                with the other M5.5 baselines.
        """
        del scale_id

        B, T, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        h1, c1 = self.cell1.init_state(B, (32, 64), device, dtype)
        h2, c2 = self.cell2.init_state(B, (16, 32), device, dtype)
        h3, c3 = self.cell3.init_state(B, (8, 16), device, dtype)

        # Iterate timesteps; the recurrent dependency precludes
        # vectorisation. Each cell receives the previous cell's hidden
        # state at the same timestep (deep stacked ConvLSTM, Shi 2015),
        # spatially pooled between cells.
        for t in range(T):
            xt = x[:, t]                                   # (B, C, 32, 64)
            h1, c1 = self.cell1(xt, h1, c1)                # h1: (B, h1, 32, 64)
            in2 = self.pool12(h1)                          #     (B, h1, 16, 32)
            h2, c2 = self.cell2(in2, h2, c2)               # h2: (B, h2, 16, 32)
            in3 = self.pool23(h2)                          #     (B, h2,  8, 16)
            h3, c3 = self.cell3(in3, h3, c3)               # h3: (B, h3,  8, 16)

        pooled = self.pool_out(h3)                          # (B, h3, 4, 8)
        flat = pooled.flatten(start_dim=1)                  # (B, h3 * 32)
        feat = F.relu(self.feature_proj(flat))              # (B, 1024)
        logits = self.classifier(self.dropout(feat))
        return {"logits": logits, "features": feat}
