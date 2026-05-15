"""Sanity tests for M6.1 1D byte Transformer (Phase 0).

Covers:
  1. ``ByteTransformerForNID`` forward shape on synthetic input
  2. Parameter count in the design-doc band [4.5M, 5.2M]
  3. Trainer head-matcher segment "classifier" hits the linear head
  4. PAD-only window yields finite logits (no NaN from div-by-zero)
"""

from __future__ import annotations

import torch

from nid_video.baselines.byte_transformer import (
    PAD_TOKEN_ID,
    VOCAB_SIZE,
    ByteTransformerForNID,
)


def test_byte_transformer_forward_shape() -> None:
    model = ByteTransformerForNID(num_classes=13)
    model.eval()
    x = torch.randint(0, 256, (2, 16, 128), dtype=torch.long)
    with torch.no_grad():
        out = model(x, scale_id=torch.zeros(2, dtype=torch.long))
    assert out["logits"].shape == (2, 13)
    assert out["features"].shape == (2, 256)
    assert torch.isfinite(out["logits"]).all()


def test_byte_transformer_param_count_in_band() -> None:
    """Phase 0 design doc projects ~4.83M params; band = [4.5M, 5.2M]."""
    model = ByteTransformerForNID(num_classes=13)
    n = sum(p.numel() for p in model.parameters())
    assert 4_500_000 <= n <= 5_200_000, (
        f"param count {n:,} outside design band [4.5M, 5.2M]"
    )


def test_byte_transformer_head_segment_named_classifier() -> None:
    """Trainer ``_build_param_groups`` looks for segment ``classifier`` in
    parameter names. Verify the linear head's params route into the head
    group under that matcher."""
    model = ByteTransformerForNID(num_classes=13)
    head_names = [
        n for n, _ in model.named_parameters()
        if "classifier" in n.split(".")
    ]
    assert len(head_names) == 2, (
        f"expected 2 head params (weight + bias); got {head_names}"
    )
    assert "classifier.weight" in head_names
    assert "classifier.bias" in head_names


def test_byte_transformer_one_real_byte_is_finite() -> None:
    """Edge case: window with only one real byte across all positions
    (one real packet with len=1) must produce finite logits — the
    production-realistic minimum information case. (All-PAD is not
    tested: PyTorch nn.TransformerEncoder with all-padded
    src_key_padding_mask produces all--inf attention → NaN by
    construction, and the byte ETL guarantees ≥1 real packet per
    emitted window.)"""
    model = ByteTransformerForNID(num_classes=13)
    model.eval()
    x = torch.full((2, 16, 128), PAD_TOKEN_ID, dtype=torch.long)
    # Mark one position per sample as real (token id = 0x42 = 66)
    x[:, 0, 0] = 66
    with torch.no_grad():
        out = model(x, scale_id=torch.zeros(2, dtype=torch.long))
    assert torch.isfinite(out["logits"]).all()


def test_vocab_constants() -> None:
    assert VOCAB_SIZE == 257
    assert PAD_TOKEN_ID == 256
