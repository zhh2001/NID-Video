"""VideoMAE-Small backbone adapted for network intrusion detection.

Adaptation strategy (decided after the M3 task 3.3 transformers-5.x exploration):

  1. Load the original 3-channel checkpoint with `from_pretrained` (no
     `ignore_mismatched_sizes`) so all encoder weights flow in clean.
  2. Surgically rebuild patch_embed:
       * trilinear down-sample 16×16 → 8×8 spatially for ch[0:3], preserving
         the Kinetics-pretrained signal in the RGB analogues.
       * Kaiming-initialize ch[3:6] (the motion channels) — they have no
         analogue in the 3-channel video pretraining.
  3. Recompute the sinusoidal position embedding for the new (8, 4, 8) tube
     grid via the upstream `get_sinusoid_encoding_table` helper. VideoMAE's
     position embedding is a fixed sinusoidal table, not a learnable Parameter,
     so we just reassign the attribute.
  4. Add a fresh classification head over the mean-pooled token features.

Note: VideoMAE's pretrained checkpoint stores attention biases as q_bias/v_bias
(no k_bias) per the original VideoMAE-pytorch implementation. transformers 5.x
VideoMAEModel uses standard query.bias / key.bias / value.bias and does NOT
auto-map. These ~14k bias parameters are zero-initialized rather than loaded.
If pretrained transfer underperforms expectation in M5 baseline comparison,
this is the first place to check. See exploration report from M3 task 3.3.

Idea.md §3.4.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from transformers import VideoMAEConfig, VideoMAEModel

from nid_video.utils import logger

ScaleInit = Literal["zero", "trunc_normal"]
NUM_SCALES = 2     # 0 = fast (Δt=100ms), 1 = slow (Δt=1s)


# --------------------------------------------------------------------------
# Bridge transformers' stdlib `logging` (which emits the LOAD REPORT at
# WARNING level) into the project loguru sink, so the `Key | Status |` table
# from from_pretrained shows up in our regular logs.
# --------------------------------------------------------------------------


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # `transformers` spawns background threads (e.g. safetensors auto-conversion)
        # that may emit log records after pytest has closed its stderr-captured
        # stream, causing loguru's stderr sink to raise. Standard logging-handler
        # contract is to never propagate handler errors: swallow via handleError.
        try:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            frame, depth = logging.currentframe(), 2
            while frame is not None and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
        except Exception:
            self.handleError(record)


_LOG_BRIDGE_INSTALLED = False


def _install_transformers_log_bridge() -> None:
    """Forward `logging.getLogger('transformers')` records to loguru. Idempotent."""
    global _LOG_BRIDGE_INSTALLED
    if _LOG_BRIDGE_INSTALLED:
        return
    hf_logger = logging.getLogger("transformers")
    hf_logger.addHandler(_InterceptHandler())
    hf_logger.setLevel(logging.INFO)
    _LOG_BRIDGE_INSTALLED = True


# --------------------------------------------------------------------------
# Backbone loading
# --------------------------------------------------------------------------


def _videomae_small_config() -> VideoMAEConfig:
    """VideoMAE-S architecture as published by MCG-NJU.

    `VideoMAEConfig()` with no args defaults to VideoMAE-**Base** (hidden=768).
    For the offline / fallback random-init path we want Small dims to match
    what the real `from_pretrained("MCG-NJU/...")` ckpt would deliver.
    """
    return VideoMAEConfig(
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=16,    # MCG-NJU ckpt uses 16 (head_dim=24); match it
        intermediate_size=1536,
    )


def _load_backbone_with_fallback(pretrained: str | None) -> VideoMAEModel:
    """Load a HF VideoMAEModel; fall back to random VideoMAE-S on offline / missing.

    M3 fallback is "no pretraining": VideoMAEModel built with the explicit
    VideoMAE-S config (`_videomae_small_config()`) and a clear WARNING. A
    timm vit_small_patch16_224 fallback was considered but deferred — its
    2D patch embed and 14×14 spatial layout don't align with VideoMAE's 3D
    tube embedding without significant remapping work, and HF cache
    reliability has been good in practice.
    """
    _install_transformers_log_bridge()
    if not pretrained:
        logger.warning(
            "pretrained is None/empty; building VideoMAEModel with VideoMAE-S config — RANDOM init"
        )
        return VideoMAEModel(_videomae_small_config())
    try:
        model = VideoMAEModel.from_pretrained(pretrained)
        logger.info(f"loaded pretrained backbone: {pretrained}")
        return model
    except (OSError, ConnectionError, ValueError) as exc:
        logger.warning(f"failed to load pretrained {pretrained!r}: {type(exc).__name__}: {exc}")
        logger.warning("falling back to VideoMAEModel(VideoMAE-S config) — RANDOM init")
        return VideoMAEModel(_videomae_small_config())


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


class VideoMAESmallForNID(nn.Module):
    """VideoMAE-Small backbone adapted for the (T=16, C=6, H=32, W=64) NID tensor.

    Args:
        num_classes: classification head output dim. Default 13 (collapsed CIC).
        pretrained: HF model ID, ``None``/``""`` to skip loading and start random.
        in_channels: input channel count (default 6 — packets/bytes/avg/flags + 2 motion).
        tube_patch: (T_p, H_p, W_p) — kernel/stride of patch_embed; default (2, 8, 8).
        spatial_grid: (H, W) of the input tensor; default (32, 64).
        gradient_checkpointing: default True for the 8 GB VRAM target.
    """

    def __init__(
        self,
        num_classes: int = 13,
        pretrained: str | None = "MCG-NJU/videomae-small-finetuned-kinetics",
        in_channels: int = 6,
        tube_patch: tuple[int, int, int] = (2, 8, 8),
        spatial_grid: tuple[int, int] = (32, 64),
        gradient_checkpointing: bool = True,
        scale_init: ScaleInit = "zero",
    ) -> None:
        super().__init__()
        if in_channels < 3:
            raise ValueError(
                f"in_channels must be ≥ 3 to receive the 3-channel pretraining; got {in_channels}"
            )
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.tube_patch = tube_patch
        self.spatial_grid = spatial_grid

        self.backbone = _load_backbone_with_fallback(pretrained)

        self._adapt_patch_embedding()
        self._adapt_position_embedding()
        self._build_scale_components(scale_init)

        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden, num_classes)

        if gradient_checkpointing:
            self.enable_gradient_checkpointing()

    # ----- gradient checkpointing -----

    def enable_gradient_checkpointing(self) -> None:
        self.backbone.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self) -> None:
        self.backbone.gradient_checkpointing_disable()

    # ----- patch embedding adaptation -----

    def _adapt_patch_embedding(self) -> None:
        T_p, H_p, W_p = self.tube_patch
        H_g, W_g = self.spatial_grid

        pe = self.backbone.embeddings.patch_embeddings
        old_proj = pe.projection
        old_w = old_proj.weight.data            # (out_ch, 3, T_orig, H_orig, W_orig)
        old_b = old_proj.bias.data              # (out_ch,)
        out_ch, _, T_orig, H_orig, W_orig = old_w.shape

        # Trilinear interpolate. F.interpolate(mode='trilinear') needs (N, C, D, H, W)
        # and only knows how to spatially-volumetrically resample. We treat each
        # (out_ch, in_ch_subset) pair as an independent tube of values and resample
        # the (D=T, H, W) of each tube. Reshape to (out_ch * 3, 1, T, H, W).
        flat = old_w.reshape(out_ch * 3, 1, T_orig, H_orig, W_orig)
        downed = F.interpolate(
            flat, size=(T_p, H_p, W_p),
            mode="trilinear", align_corners=False,
        )                                       # (out_ch * 3, 1, T_p, H_p, W_p)
        down_w = downed.reshape(out_ch, 3, T_p, H_p, W_p)

        n_extra = self.in_channels - 3
        if n_extra > 0:
            fresh = torch.empty(out_ch, n_extra, T_p, H_p, W_p)
            nn.init.kaiming_normal_(fresh, nonlinearity="relu")
            new_w = torch.cat([down_w, fresh], dim=1)
        else:
            new_w = down_w

        new_proj = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=out_ch,
            kernel_size=(T_p, H_p, W_p),
            stride=(T_p, H_p, W_p),
        )
        new_proj.weight.data.copy_(new_w)
        new_proj.bias.data.copy_(old_b)

        # Replace the projection module + sync the patch_embeddings module's metadata.
        pe.projection = new_proj
        pe.num_channels = self.in_channels
        pe.image_size = (H_g, W_g)
        pe.patch_size = (H_p, W_p)
        pe.tubelet_size = T_p
        pe.num_patches = (H_g // H_p) * (W_g // W_p) * (self.backbone.config.num_frames // T_p)
        self.backbone.embeddings.num_patches = pe.num_patches

        # And the model.config (downstream code reads it).
        cfg = self.backbone.config
        cfg.num_channels = self.in_channels
        cfg.image_size = (H_g, W_g)
        cfg.patch_size = H_p
        cfg.tubelet_size = T_p

        # Diagnostics — the central validation that the pretraining survived
        # rather than being silently overwritten by fresh init.
        ch3_norm = down_w.norm().item()
        ext_norm = float(fresh.norm().item()) if n_extra > 0 else 0.0
        logger.info(
            f"patch_embed adapted: ch[0:3] downsampled {16}→{H_p} "
            f"shape={tuple(down_w.shape)} norm={ch3_norm:.2f}; "
            f"ch[3:{self.in_channels}] kaiming-init shape="
            f"{(out_ch, n_extra, T_p, H_p, W_p) if n_extra > 0 else 'n/a'} "
            f"norm={ext_norm:.2f}"
        )

    # ----- scale token (M4) -----

    def _build_scale_components(self, scale_init: ScaleInit) -> None:
        """Create the scale token + scale embedding for multi-scale training.

        The architecture has two pieces (Idea.md M4 task 4.2):
          * ``scale_token``: a CLS-like learnable parameter, shared across
            scales, prepended to the patch sequence at every forward pass.
            Init: zero (HF VideoMAE has no CLS by default; an explicit zero
            start lets the model decide whether to use this slot).
          * ``scale_embedding``: an nn.Embedding(2, hidden) producing a
            *conditional offset* added to ``scale_token`` based on
            ``scale_id`` ∈ {0=fast, 1=slow}. Init governed by ``scale_init``:
            - ``"zero"``: scale token at step 0 is identical for both
              scales — model starts as if single-scale, scale conditioning
              learned from gradient.
            - ``"trunc_normal"``: small random offset (std=0.02), fast/slow
              tokens differ from step 0.
        Default ``"zero"`` keeps ablations clean (zero initial bias).
        """
        h = self.backbone.config.hidden_size
        self.scale_token = nn.Parameter(torch.zeros(1, 1, h))
        self.scale_embedding = nn.Embedding(NUM_SCALES, h)
        if scale_init == "zero":
            nn.init.zeros_(self.scale_embedding.weight)
        elif scale_init == "trunc_normal":
            nn.init.trunc_normal_(self.scale_embedding.weight, std=0.02)
        else:
            raise ValueError(f"unknown scale_init: {scale_init!r}")
        logger.info(
            f"scale components built: NUM_SCALES={NUM_SCALES}, hidden={h}, "
            f"scale_init={scale_init}"
        )
        self.scale_init: ScaleInit = scale_init

    # ----- position embedding adaptation -----

    def _adapt_position_embedding(self) -> None:
        # Recompute the sinusoidal position table at length (num_patches + 1)
        # to leave room for the prepended scale token at index 0. The
        # sinusoidal formula is position-dependent (PE_table(N)[i] depends on
        # absolute index ``i``), so naively prepending a zero row would
        # silently shift all patch tokens to the wrong positions vs the
        # 256-table the M3 model used. Recomputing at N=257 keeps the math
        # consistent: scale_token sees pos=0, patch i sees pos=i+1.
        #
        # Re-use upstream's exact sinusoidal formula rather than rolling our
        # own — keeps the math identical to pretraining. Flatten order for
        # patches is time-major (verified by
        # `test_patch_token_flatten_order_is_time_major`).
        #
        # Domain-shift caveat: the pretrained grid (8, 14, 14) had H-neighbour
        # distance = 14 steps, T-neighbour = 196. Ours (8, 4, 8) has 8 and 32.
        # The attention's learned relative-distance preferences will not
        # transfer cleanly. If M5 cross-architecture comparisons find weak
        # fine-grained spatial signal, factorized 3D position embeddings would
        # be the next thing to try.
        from transformers.models.videomae.modeling_videomae import get_sinusoid_encoding_table

        n_pos = self.backbone.embeddings.num_patches + 1   # +1 for scale token
        hidden = self.backbone.config.hidden_size
        new_pe = get_sinusoid_encoding_table(n_pos, hidden)
        # In transformers VideoMAEEmbeddings, position_embeddings is a plain Tensor
        # attribute (not a Parameter, not a buffer). Direct assignment is correct
        # and matches upstream's __init__ convention.
        self.backbone.embeddings.position_embeddings = new_pe
        logger.info(f"position_embedding rebuilt: shape={tuple(new_pe.shape)} (scale token + 256 patches)")

    # ----- forward -----

    def forward(
        self,
        x: torch.Tensor,
        *,
        scale_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T=16, C=in_channels, H=32, W=64) float tensor.
            scale_id: (B,) long tensor with values in [0, NUM_SCALES). 0=fast,
              1=slow. The dataset (``MultiScaleNidDataset``) attaches this per
              sample; single-scale callers can pass a fixed-zero tensor.

        Returns:
            dict with
              ``logits``:   (B, num_classes)
              ``features``: (B, hidden_size)  -- mean-pooled across all 257
                            tokens (256 patches + 1 scale token).

        We bypass ``backbone(x)`` and inline the embedding-stack so we can
        prepend the scale token before the position embedding addition.
        Equivalent to ``backbone(x)`` minus the scale-token concat.
        """
        B = x.size(0)
        pe_layer = self.backbone.embeddings.patch_embeddings
        embeddings = pe_layer(x)                                 # (B, 256, h)

        scale_tok = self.scale_token.expand(B, -1, -1) + \
            self.scale_embedding(scale_id).unsqueeze(1)          # (B, 1, h)
        embeddings = torch.cat([scale_tok, embeddings], dim=1)   # (B, 257, h)

        # Position table is a plain tensor (not parameter/buffer); make sure
        # it lands on the embeddings' device/dtype before broadcasting.
        pos_emb = self.backbone.embeddings.position_embeddings
        pos_emb = pos_emb.to(device=embeddings.device, dtype=embeddings.dtype)
        embeddings = embeddings + pos_emb

        encoder_outputs = self.backbone.encoder(embeddings)
        sequence_output = encoder_outputs[0]
        # HF's VideoMAEModel.forward applies layernorm conditionally —
        # ``self.backbone.layernorm`` can be ``None`` when the config disables
        # final normalization. Mirror that None-check rather than assuming it
        # is always present.
        if getattr(self.backbone, "layernorm", None) is not None:
            sequence_output = self.backbone.layernorm(sequence_output)

        feat = sequence_output.mean(dim=1)
        logits = self.classifier(feat)
        return {"logits": logits, "features": feat}
