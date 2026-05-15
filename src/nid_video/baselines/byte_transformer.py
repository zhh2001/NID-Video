"""M6.1 — 1D byte Transformer (Phase 0 model module).

Cross-paradigm baseline cell 6.1. Consumes raw packet bytes as a
2D (K=16 packets × N=128 bytes) sequence, flattens to K×N=2048 tokens,
and runs a 6-layer Transformer encoder. Pools attention-masked
non-pad positions and feeds a 13-class linear classifier.

Design source-of-truth: ``docs/m6_1_byte_transformer.md`` (Phase 0
design report). Locked decisions at write-time:

  * vocab_size = 257 (bytes 0-255 + [PAD]=256)
  * d_model = 256, nhead = 8, n_layers = 6, ffn_dim = 1024
  * pre-norm, GELU, dropout = 0.1
  * 2D factorized learnable positional encoding (K-axis 16 + N-axis 128)
  * mean-pool over non-pad positions
  * classifier head named ``classifier`` (matches trainer
    ``_build_param_groups`` head matcher → Path B head_lr ×1)

Forward signature mirrors the project's other backbones —
``forward(x, *, scale_id) → {logits, features}`` — so the trainer's
call site at ``Trainer._train_one_epoch`` doesn't branch on model
type. ``scale_id`` is accepted and ignored (M6.1 is fast-only by
design — no slow analogue for raw byte sequences).

The input ``x`` shape contract is the byte-shard convention:
``(B, K=16, N=128)`` int64 token ids in [0, 257). The attention_mask
is derived inside ``forward`` from the [PAD] token id (256); callers
need not provide it explicitly. This matches the existing trainer's
``model(x, scale_id=...)`` call signature unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nid_video.utils import logger

PAD_TOKEN_ID = 256
VOCAB_SIZE = 257  # 256 raw bytes + 1 [PAD]


class ByteEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer using F.scaled_dot_product_attention.

    Replaces ``nn.TransformerEncoderLayer(norm_first=True)`` to bypass the
    O(seqlen²) slow path (PyTorch disables the nested-tensor fast-path
    under pre-norm, materializing the full attention matrix). SDPA is
    mathematically equivalent to the standard scaled dot-product
    attention; the mem-efficient backend keeps memory O(seqlen).

    The pre-norm flow mirrors nn.TransformerEncoderLayer(norm_first=True)
    semantics exactly:
      ``x = x + dropout(out_proj(SDPA(Q,K,V, key_padding_mask)))``
      ``x = x + dropout(ffn(norm2(x)))``

    Decision recorded in ``docs/m6_1_byte_transformer.md`` §G.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, f"d_model={d_model} not divisible by nhead={nhead}"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Pre-norm + multi-head Q/K/V projection.
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Pre-norm FFN: Linear → GELU → Dropout → Linear.
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)

        self.dropout_p = float(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, L, D)`` — pre-residual input.
            key_padding_mask: ``(B, L)`` bool, True = pad position to ignore.

        Returns:
            ``(B, L, D)`` post-block residual sum.
        """
        B, L, _ = x.shape
        H = self.nhead
        Dh = self.head_dim

        # --- attention sub-layer -------------------------------------------
        h = self.norm1(x)
        q = self.q_proj(h).view(B, L, H, Dh).transpose(1, 2)   # (B, H, L, Dh)
        k = self.k_proj(h).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(h).view(B, L, H, Dh).transpose(1, 2)

        if key_padding_mask is not None:
            # SDPA bool attn_mask convention: True = attend, False = mask out.
            # key_padding_mask convention: True = pad. Convert + broadcast to
            # (B, 1, 1, L). Shape (B, 1, 1, L) broadcasts to (B, H, L, L) — no
            # explicit .expand() needed; SDPA accepts broadcastable masks.
            attn_mask = (~key_padding_mask)[:, None, None, :]
        else:
            attn_mask = None

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )                                                       # (B, H, L, Dh)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        attn_out = self.out_proj(attn_out)
        x = x + self.dropout(attn_out)

        # --- FFN sub-layer -------------------------------------------------
        h = self.norm2(x)
        ffn_out = self.fc2(self.dropout(F.gelu(self.fc1(h))))
        x = x + self.dropout(ffn_out)
        return x


class ByteTransformerForNID(nn.Module):
    """Small 1D byte-stream Transformer encoder for NID classification."""

    def __init__(
        self,
        num_classes: int = 13,
        d_model: int = 256,
        n_layers: int = 6,
        nhead: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        K: int = 16,
        N: int = 128,
        gradient_checkpointing: bool = True,   # on by default — see Phase 0 §I revision
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.K = K
        self.N = N
        self.pad_id = PAD_TOKEN_ID
        self.vocab_size = VOCAB_SIZE

        # Token embedding (vocab_size 257; ids 0-255 = raw byte values, 256 = [PAD]).
        self.token_embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_TOKEN_ID)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embed.weight[PAD_TOKEN_ID].zero_()

        # 2D factorized positional encoding: K-axis (packet idx) + N-axis (byte idx)
        # additive. Each token (k, n) gets embed[byte] + pos_k[k] + pos_n[n].
        self.pos_packet = nn.Parameter(torch.zeros(K, d_model))
        self.pos_byte = nn.Parameter(torch.zeros(N, d_model))
        nn.init.uniform_(self.pos_packet, -0.02, 0.02)
        nn.init.uniform_(self.pos_byte, -0.02, 0.02)

        # Stack of pre-norm SDPA-based encoder layers. See ByteEncoderLayer
        # docstring for the kernel-choice rationale (docs/m6_1_byte_transformer.md
        # §G).
        self.layers = nn.ModuleList([
            ByteEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ffn_dim,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        self._gradient_checkpointing = bool(gradient_checkpointing)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"ByteTransformerForNID built: d_model={d_model} n_layers={n_layers} "
            f"nhead={nhead} ffn={ffn_dim} K={K} N={N} vocab={VOCAB_SIZE} "
            f"params={n_params:,} ({n_params/1e6:.2f}M)"
        )

    def _build_position_table(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return (K*N, d_model) flattened position table.

        pos_table[k*N + n] = pos_packet[k] + pos_byte[n]. Computed once
        per forward call (not cached as buffer — keeps it differentiable
        across pos_packet / pos_byte updates).
        """
        pos_k = self.pos_packet.to(device=device, dtype=dtype)      # (K, d)
        pos_n = self.pos_byte.to(device=device, dtype=dtype)        # (N, d)
        return (pos_k.unsqueeze(1) + pos_n.unsqueeze(0)).reshape(self.K * self.N, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        scale_id: torch.Tensor | None = None,    # accepted + ignored
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: ``(B, K=16, N=128)`` long tensor of byte-token ids in
              [0, 257). [PAD]=256 marks both packet-pad (full packet
              padded out) and byte-pad (packet shorter than 128 bytes).
            scale_id: ignored. Accepted for trainer-call-site parity.

        Returns:
            ``{"logits": (B, num_classes), "features": (B, d_model)}``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"ByteTransformerForNID expects (B, K, N); got {tuple(x.shape)}"
            )
        if x.size(1) != self.K or x.size(2) != self.N:
            raise ValueError(
                f"ByteTransformerForNID expects K={self.K} N={self.N}; "
                f"got K={x.size(1)} N={x.size(2)}"
            )
        if x.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"ByteTransformerForNID expects integer token ids; got dtype={x.dtype}"
            )

        B = x.size(0)
        flat = x.reshape(B, self.K * self.N)              # (B, K*N)

        # Token embedding + 2D-factored position. embed handles padding_idx
        # internally (PAD row already zero via padding_idx kwarg).
        emb = self.token_embed(flat)                      # (B, K*N, d)
        pos = self._build_position_table(emb.device, emb.dtype)  # (K*N, d)
        emb = emb + pos.unsqueeze(0)                      # (B, K*N, d)

        # Padding mask: True = pad position, ignored in attention.
        pad_mask = flat.eq(self.pad_id)                   # (B, K*N) bool

        h = emb
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                # Re-compute layer outputs during backward instead of storing
                # activations — trade compute for memory. use_reentrant=False
                # is required for proper module-attribute capture under
                # PyTorch ≥ 2.1.
                from torch.utils.checkpoint import checkpoint
                h = checkpoint(layer, h, pad_mask, use_reentrant=False)
            else:
                h = layer(h, pad_mask)
        encoded = self.final_norm(h)

        # Mean-pool over non-pad positions. Avoid division by zero with
        # clamp(min=1) — a window with all-pad rows should never happen
        # (every window has ≥ 1 real packet by construction), but the
        # clamp is a safety guard.
        keep = (~pad_mask).unsqueeze(-1).to(encoded.dtype)        # (B, K*N, 1)
        summed = (encoded * keep).sum(dim=1)              # (B, d)
        denom = keep.sum(dim=1).clamp(min=1.0)            # (B, 1)
        feat = summed / denom                             # (B, d)

        logits = self.classifier(feat)
        return {"logits": logits, "features": feat}
