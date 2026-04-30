"""Tests for VideoMAESmallForNID. Offline path uses pretrained=None; the
slow tests exercise the real HF download/load."""

from __future__ import annotations

import pytest
import torch

from nid_video.models.videomae_nid import (
    VideoMAESmallForNID,
    _load_backbone_with_fallback,
)


# ---------------------------------------------------------------------------
# Offline (random-init) tests — fast, no network
# ---------------------------------------------------------------------------


def test_instantiation_offline() -> None:
    m = VideoMAESmallForNID(num_classes=13, pretrained=None)
    assert isinstance(m, torch.nn.Module)


def test_param_count_in_expected_range() -> None:
    m = VideoMAESmallForNID(num_classes=13, pretrained=None)
    n = sum(p.numel() for p in m.parameters())
    # M3 prompt: ~22M ± 5M (17M..27M). Our adapter adds patch_embed delta + classifier.
    assert 17e6 < n < 27e6, f"got {n / 1e6:.2f}M params"


def test_patch_embed_input_channels_is_six() -> None:
    m = VideoMAESmallForNID(num_classes=13, pretrained=None, in_channels=6)
    pe = m.backbone.embeddings.patch_embeddings
    assert pe.projection.in_channels == 6
    assert pe.projection.kernel_size == (2, 8, 8)
    assert pe.projection.stride == (2, 8, 8)
    assert pe.projection.weight.shape == (384, 6, 2, 8, 8)


def test_patch_embed_metadata_synced_with_new_grid() -> None:
    m = VideoMAESmallForNID(num_classes=13, pretrained=None, in_channels=6)
    pe = m.backbone.embeddings.patch_embeddings
    assert pe.num_channels == 6
    assert pe.image_size == (32, 64)
    assert pe.patch_size == (8, 8)
    assert pe.tubelet_size == 2
    assert pe.num_patches == 256


def test_position_embedding_shape_matches_token_count() -> None:
    """For (T=16, H=32, W=64) with tube (2,8,8): patches = 8*4*8 = 256;
    plus 1 scale token at index 0 → seq length 257 (M4 task 4.2)."""
    m = VideoMAESmallForNID(num_classes=13, pretrained=None)
    pe = m.backbone.embeddings.position_embeddings
    assert pe.shape == (1, 257, 384), f"got {tuple(pe.shape)}"


def test_forward_output_shapes_no_grad() -> None:
    m = VideoMAESmallForNID(num_classes=13, pretrained=None,
                            gradient_checkpointing=False)
    m.eval()
    x = torch.randn(2, 16, 6, 32, 64)
    scale_id = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        out = m(x, scale_id=scale_id)
    assert out["logits"].shape == (2, 13)
    assert out["features"].shape == (2, 384)


def test_forward_backward_runs() -> None:
    m = VideoMAESmallForNID(num_classes=13, pretrained=None,
                            gradient_checkpointing=False)
    m.train()
    x = torch.randn(2, 16, 6, 32, 64)
    scale_id = torch.zeros(2, dtype=torch.long)
    out = m(x, scale_id=scale_id)
    loss = out["logits"].sum()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.parameters()
    )
    assert has_grad, "no parameter received gradient — backward failed silently"


def test_in_channels_below_three_rejected() -> None:
    with pytest.raises(ValueError):
        VideoMAESmallForNID(num_classes=13, pretrained=None, in_channels=2)


def test_disable_gradient_checkpointing_does_not_raise() -> None:
    m = VideoMAESmallForNID(num_classes=13, pretrained=None,
                            gradient_checkpointing=True)
    m.disable_gradient_checkpointing()
    m.enable_gradient_checkpointing()


def test_classifier_head_output_dim_follows_num_classes() -> None:
    m = VideoMAESmallForNID(num_classes=15, pretrained=None)
    assert isinstance(m.classifier, torch.nn.Linear)
    assert m.classifier.out_features == 15
    assert m.classifier.in_features == 384


def test_patch_token_flatten_order_is_time_major() -> None:
    """The position embedding alignment depends on the patch_embed flatten order.

    transformers' VideoMAEPatchEmbeddings does:
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
    Conv3d output is (B, hidden, T_tubes, H_patches, W_patches); flatten(2) is
    row-major over those three axes, so token index i corresponds to:
        t = i // (H_patches * W_patches)
        h = (i % (H_patches * W_patches)) // W_patches
        w = i % W_patches

    Our recomputed sinusoidal position_embeddings is a 1D table of length 256;
    pos[0, i] is added to token i. The two orderings line up by construction
    (both are 1D 0..255). This test pins that contract empirically: a future
    transformers refactor that flips the flatten direction would break the
    pretrained spatial relationships silently — this test catches that.
    """
    m = VideoMAESmallForNID(num_classes=13, pretrained=None,
                            gradient_checkpointing=False)
    pe = m.backbone.embeddings.patch_embeddings

    # Surgical projection: only weight[c_out=0, c_in=0, 0, 0, 0] = 1, rest = 0.
    # That makes out[c_out=0, t, h, w] = input[c_in=0, t*tubelet, h*patch, w*patch].
    pe.projection.weight.data.zero_()
    pe.projection.weight.data[0, 0, 0, 0, 0] = 1.0
    pe.projection.bias.data.zero_()

    # Encode each tube-corner with a unique tag = t*H*W + h*W + w (the t-major index).
    T_tubes, H_pat, W_pat = 8, 4, 8
    x = torch.zeros(1, 16, 6, 32, 64)
    for t_idx in range(T_tubes):
        for h_idx in range(H_pat):
            for w_idx in range(W_pat):
                tag = t_idx * H_pat * W_pat + h_idx * W_pat + w_idx
                x[0, t_idx * 2, 0, h_idx * 8, w_idx * 8] = float(tag)

    out = pe(x)                         # (1, 256, 384)
    # Token i's first hidden dim should carry tag i if and only if flatten is t-major.
    expected = torch.arange(256, dtype=out.dtype)
    torch.testing.assert_close(out[0, :, 0], expected, rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------
# Slow (real HF) tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_pretrained_load_keeps_param_count_close_to_22M() -> None:
    """Real HF download (cached after first run)."""
    m = VideoMAESmallForNID(num_classes=13)
    n = sum(p.numel() for p in m.parameters())
    assert 21e6 < n < 23e6, f"got {n / 1e6:.2f}M params"


@pytest.mark.slow
def test_real_pretrained_ch_0_3_norm_smaller_than_ch_3_6() -> None:
    """KEY VALIDATION (M3 task 3.3 user requirement).

    The trilinear-down-sampled pretraining for ch[0:3] is initialized from a
    weight tensor whose elements have std ≈ 0.02 (ViT default trunc-normal).
    The fresh Kaiming init for ch[3:6] uses std ≈ sqrt(2 / fan_in) ≈ 0.07,
    so its norm should be visibly larger. Equality of norms across the two
    halves would mean we silently dropped the pretraining (the failure mode
    we caught in the transformers 5.x exploration).
    """
    m = VideoMAESmallForNID(
        num_classes=13,
        pretrained="MCG-NJU/videomae-small-finetuned-kinetics",
    )
    w = m.backbone.embeddings.patch_embeddings.projection.weight.data
    n_pre = w[:, :3].norm().item()
    n_fresh = w[:, 3:].norm().item()
    assert n_fresh > 1.5 * n_pre, (
        f"pretrained ch[0:3] norm={n_pre:.2f} should be << fresh ch[3:6] norm={n_fresh:.2f}; "
        "if they're close, pretraining was silently dropped"
    )


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------


def test_fallback_returns_random_videomae_when_pretrained_is_none() -> None:
    bb = _load_backbone_with_fallback(None)
    from transformers import VideoMAEModel
    assert isinstance(bb, VideoMAEModel)


def test_fallback_returns_random_videomae_when_pretrained_is_empty_string() -> None:
    bb = _load_backbone_with_fallback("")
    from transformers import VideoMAEModel
    assert isinstance(bb, VideoMAEModel)


# ---------------------------------------------------------------------------
# M4 task 4.2: scale token, scale embedding, 257-PE
# ---------------------------------------------------------------------------


def test_position_embedding_for_257_tokens_is_sinusoidal_extension_of_256() -> None:
    """257-PE is a fresh sinusoidal table at length 257, NOT a 256-PE with a
    zero row prepended. Because the formula is position-dependent
    (PE[i] depends on absolute index i), 257-PE[1:] (positions 1..256) is
    not equal to 256-PE (positions 0..255) — they are shifted by one.

    Pinning this explicitly: a future regression to ``cat([zeros, old_pe])``
    would silently put patches at the wrong sinusoidal positions and break
    pretraining transfer.
    """
    from transformers.models.videomae.modeling_videomae import (
        get_sinusoid_encoding_table,
    )

    m = VideoMAESmallForNID(num_classes=13, pretrained=None)
    table_257 = m.backbone.embeddings.position_embeddings        # (1, 257, 384)
    assert table_257.shape == (1, 257, 384)

    table_256 = get_sinusoid_encoding_table(256, 384)            # (1, 256, 384)
    # shape match, content does NOT (257[1..256] uses positions 1..256 vs
    # 256[0..255] uses positions 0..255).
    assert table_257[0, 1:].shape == table_256[0].shape
    assert not torch.allclose(table_257[0, 1:], table_256[0])

    # And: positions 1..255 are SHARED (PE depends only on absolute index).
    table_at_n2 = get_sinusoid_encoding_table(2, 384)
    torch.testing.assert_close(table_257[0, 1, :], table_at_n2[0, 1, :], rtol=0, atol=1e-6)


def test_scale_token_and_embedding_are_learnable_parameters() -> None:
    m = VideoMAESmallForNID(num_classes=13, pretrained=None)
    names = {n for n, _ in m.named_parameters()}
    assert "scale_token" in names
    assert "scale_embedding.weight" in names
    assert m.scale_token.requires_grad
    assert m.scale_embedding.weight.requires_grad


def test_scale_init_zero_makes_fast_slow_initially_equivalent() -> None:
    """With scale_init='zero', scale_embedding(0) == scale_embedding(1) == 0,
    so the full scale_token (= shared scale_token + zero offset) is identical
    for both scales. forward(x, scale_id=0) and forward(x, scale_id=1) must
    therefore produce identical logits at step 0."""
    m = VideoMAESmallForNID(num_classes=13, pretrained=None,
                            gradient_checkpointing=False, scale_init="zero")
    m.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 16, 6, 32, 64)
    with torch.no_grad():
        out_fast = m(x, scale_id=torch.tensor([0, 0], dtype=torch.long))
        out_slow = m(x, scale_id=torch.tensor([1, 1], dtype=torch.long))
    torch.testing.assert_close(out_fast["logits"], out_slow["logits"])


def test_scale_init_trunc_normal_breaks_fast_slow_equivalence() -> None:
    """Random-init scale_embedding makes fast/slow logits diverge from step 0."""
    m = VideoMAESmallForNID(num_classes=13, pretrained=None,
                            gradient_checkpointing=False,
                            scale_init="trunc_normal")
    m.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 16, 6, 32, 64)
    with torch.no_grad():
        out_fast = m(x, scale_id=torch.tensor([0, 0], dtype=torch.long))
        out_slow = m(x, scale_id=torch.tensor([1, 1], dtype=torch.long))
    assert not torch.allclose(out_fast["logits"], out_slow["logits"])


def test_scale_init_unknown_value_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scale_init"):
        VideoMAESmallForNID(num_classes=13, pretrained=None,
                            scale_init="bogus")  # type: ignore[arg-type]


def test_forward_with_mixed_scale_ids_in_a_batch() -> None:
    """A single batch can contain both scales — gradient flows for both rows."""
    m = VideoMAESmallForNID(num_classes=13, pretrained=None,
                            gradient_checkpointing=False,
                            scale_init="trunc_normal")
    m.train()
    x = torch.randn(4, 16, 6, 32, 64)
    scale_id = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    out = m(x, scale_id=scale_id)
    out["logits"].sum().backward()
    assert m.scale_token.grad is not None
    assert m.scale_embedding.weight.grad is not None
    # Both rows of the embedding got gradient (mixed batch).
    assert m.scale_embedding.weight.grad.abs().sum(dim=1).gt(0).all().item()


def test_m3_state_dict_loads_with_strict_false_for_scale_params() -> None:
    """An M3 ckpt has no scale_token / scale_embedding entries. load_state_dict
    must accept that with ``strict=False`` and report them as missing keys —
    this is the M4 backward-compat hook."""
    m_new = VideoMAESmallForNID(num_classes=13, pretrained=None)
    # Simulate an M3 state dict by stripping the new params.
    state = {k: v for k, v in m_new.state_dict().items()
             if not k.startswith(("scale_token", "scale_embedding"))}
    fresh = VideoMAESmallForNID(num_classes=13, pretrained=None)
    missing, unexpected = fresh.load_state_dict(state, strict=False)
    assert "scale_token" in missing
    assert "scale_embedding.weight" in missing
    assert unexpected == []
