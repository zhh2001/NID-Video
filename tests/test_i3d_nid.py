"""Tests for src/nid_video/models/i3d_nid.py (M5.5 R2 baseline 3).

Pin the architectural contract: param count band (~28M K400-pretrained),
forward shape (B=2 → logits (2,13) features (2,2048)), scale_id ignore
semantics, native 6-channel patch_stem after adapter, K400-pretrained
signal preservation through ``adapt_conv3d_to_6ch`` (bit-identity
because source kernel == target kernel), finite gradient flow.
"""

from __future__ import annotations

import pytest
import torch

from nid_video.models.i3d_nid import I3DForNID


def test_i3d_param_count_within_target() -> None:
    """I3D-R50 K400-pretrained is ~28M params at the standard
    ResNet-50 layout. Pin within ±5M to catch accidental config drift
    (e.g. a future refactor swapping to a different ResNet depth)."""
    m = I3DForNID(num_classes=13, pretrained=False, gradient_checkpointing=False)
    n_params = sum(p.numel() for p in m.parameters())
    n_params_M = n_params / 1e6
    assert 23 < n_params_M < 33, (
        f"I3D-R50 param count out of band: {n_params_M:.1f}M "
        f"(expected ~28M ±5M)"
    )


def test_i3d_forward_shape() -> None:
    """input (B=2, T=16, C=6, H=32, W=64), scale_id (2,) → logits (2,
    13) and features (2, 2048). The features dim is the I3D-R50
    pre-classifier hidden (2048 = 4 × 512 ResNet-50 stage 5 width)."""
    m = I3DForNID(num_classes=13, pretrained=False, gradient_checkpointing=False)
    m.eval()
    x = torch.randn(2, 16, 6, 32, 64)
    scale_id = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        out = m(x, scale_id=scale_id)
    assert out["logits"].shape == (2, 13), out["logits"].shape
    assert out["features"].shape == (2, 2048), out["features"].shape


def test_i3d_ignores_scale_id() -> None:
    """I3D does not condition on scale_id. Forwarding the same input
    with scale_id={all 0} vs {all 1} must yield identical logits."""
    m = I3DForNID(num_classes=13, pretrained=False, gradient_checkpointing=False)
    m.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 16, 6, 32, 64)
    with torch.no_grad():
        out_fast = m(x, scale_id=torch.tensor([0, 0], dtype=torch.long))
        out_slow = m(x, scale_id=torch.tensor([1, 1], dtype=torch.long))
    torch.testing.assert_close(out_fast["logits"], out_slow["logits"])


def test_i3d_six_channel_input_works() -> None:
    """``in_channels=6`` flows through to the patch stem after
    ``adapt_conv3d_to_6ch`` runs. Backwards-facing pin against a
    future refactor that hardcodes 3 input channels."""
    m = I3DForNID(num_classes=13, pretrained=False, in_channels=6,
                  gradient_checkpointing=False)
    stem_conv = m.backbone.blocks[0].conv
    assert stem_conv.weight.shape[1] == 6, (
        f"i3d stem expected 6 input channels, got {stem_conv.weight.shape[1]}"
    )
    m.eval()
    x = torch.randn(1, 16, 6, 32, 64)
    with torch.no_grad():
        out = m(x, scale_id=torch.zeros(1, dtype=torch.long))
    assert out["logits"].shape == (1, 13)


def test_i3d_loss_finite_and_grad_flows() -> None:
    """Forward + focal-loss + backward yields finite gradients on most
    optimizable parameters."""
    from nid_video.trainer.loss import FocalLoss

    m = I3DForNID(num_classes=13, pretrained=False, gradient_checkpointing=False)
    m.train()
    torch.manual_seed(0)
    x = torch.randn(2, 16, 6, 32, 64)
    targets = torch.randint(0, 13, (2,))
    out = m(x, scale_id=torch.zeros(2, dtype=torch.long))
    loss = FocalLoss(gamma=2.0)(out["logits"], targets)
    assert torch.isfinite(loss), f"loss not finite: {loss.item()}"
    loss.backward()
    n_grad = sum(1 for p in m.parameters() if p.grad is not None)
    n_total = sum(1 for _ in m.parameters())
    assert n_grad >= 0.95 * n_total, (
        f"gradient did not reach most parameters: "
        f"{n_grad}/{n_total} have .grad set"
    )


@pytest.mark.slow
def test_i3d_pretrained_first_three_channels_preserved() -> None:
    """Pin that K400 pretraining survives the 6-channel adapter on the
    first 3 input channels. Because I3D's native patch stem kernel is
    (5, 7, 7) — the same as our target — the trilinear "downsample" is
    an identity transform; ``adapt_conv3d_to_6ch`` therefore produces
    ch[0:3] BYTE-IDENTICAL to the original K400 patch_stem weights.
    Asserting bit-identity is stronger than a norm-ratio pin (which
    fails here precisely BECAUSE there's no compression).

    Secondary check: ch[3:6] is Kaiming-initialised, so its weights
    should NOT match ch[0:3]. Catches a regression where a future
    refactor accidentally broadcasts ch[0:3] into ch[3:6].

    Slow-marked because it pulls the real 28M ckpt from the FAIR hub.
    """
    from pytorchvideo.models.hub import i3d_r50

    m = I3DForNID(num_classes=13, in_channels=6, gradient_checkpointing=False)
    stem_conv = m.backbone.blocks[0].conv
    assert stem_conv.weight.shape[1] == 6

    # Reference: load fresh K400 ckpt to compare ch[0:3] against.
    ref = i3d_r50(pretrained=True)
    ref_conv = ref.blocks[0].conv
    assert ref_conv.weight.shape[1] == 3

    adapted_first_three = stem_conv.weight.data[:, :3]
    torch.testing.assert_close(
        adapted_first_three, ref_conv.weight.data,
        atol=1e-7, rtol=0.0,
    )

    # Secondary: ch[3:6] is NOT a copy of ch[0:3].
    adapted_extra = stem_conv.weight.data[:, 3:]
    assert not torch.allclose(adapted_first_three, adapted_extra), (
        "ch[3:6] looks identical to ch[0:3] — adapter may have broadcast "
        "pretrained weights into the extra channels instead of "
        "Kaiming-initialising them"
    )
