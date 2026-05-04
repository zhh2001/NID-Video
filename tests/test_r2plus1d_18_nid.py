"""Tests for src/nid_video/models/r2plus1d_18_nid.py (M5.5 R2 baseline 4).

Pin the architectural contract: param count band (~31.5M K400-pretrained),
forward shape (B=2 → logits (2,13) features (2,512)), scale_id ignore
semantics, native 6-channel patch_stem after adapter, K400-pretrained
signal preservation through ``adapt_conv3d_to_6ch`` (bit-identity
because source kernel == target kernel), finite gradient flow.
"""

from __future__ import annotations

import pytest
import torch

from nid_video.models.r2plus1d_18_nid import R2Plus1D18ForNID


def test_r2plus1d_18_param_count_within_target() -> None:
    """R(2+1)D-18 K400-pretrained is ~31.5M params at the standard
    torchvision layout. Pin within ±5M to catch accidental config drift
    (e.g. a future refactor swapping to a different ResNet depth)."""
    m = R2Plus1D18ForNID(num_classes=13, pretrained=False,
                         gradient_checkpointing=False)
    n_params = sum(p.numel() for p in m.parameters())
    n_params_M = n_params / 1e6
    assert 27 < n_params_M < 37, (
        f"R(2+1)D-18 param count out of band: {n_params_M:.1f}M "
        f"(expected ~31.5M ±5M)"
    )


def test_r2plus1d_18_forward_shape() -> None:
    """input (B=2, T=16, C=6, H=32, W=64), scale_id (2,) → logits (2,
    13) and features (2, 512). The features dim is the R(2+1)D-18
    pre-classifier hidden (512 = ResNet-18 stage 5 width)."""
    m = R2Plus1D18ForNID(num_classes=13, pretrained=False,
                         gradient_checkpointing=False)
    m.eval()
    x = torch.randn(2, 16, 6, 32, 64)
    scale_id = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        out = m(x, scale_id=scale_id)
    assert out["logits"].shape == (2, 13), out["logits"].shape
    assert out["features"].shape == (2, 512), out["features"].shape


def test_r2plus1d_18_ignores_scale_id() -> None:
    """R(2+1)D-18 does not condition on scale_id."""
    m = R2Plus1D18ForNID(num_classes=13, pretrained=False,
                         gradient_checkpointing=False)
    m.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 16, 6, 32, 64)
    with torch.no_grad():
        out_fast = m(x, scale_id=torch.tensor([0, 0], dtype=torch.long))
        out_slow = m(x, scale_id=torch.tensor([1, 1], dtype=torch.long))
    torch.testing.assert_close(out_fast["logits"], out_slow["logits"])


def test_r2plus1d_18_six_channel_input_works() -> None:
    """``in_channels=6`` flows through to stem[0] after
    ``adapt_conv3d_to_6ch`` runs."""
    m = R2Plus1D18ForNID(num_classes=13, pretrained=False, in_channels=6,
                         gradient_checkpointing=False)
    stem_conv = m.backbone.stem[0]
    assert stem_conv.weight.shape[1] == 6, (
        f"r2plus1d_18 stem[0] expected 6 input channels, got {stem_conv.weight.shape[1]}"
    )
    m.eval()
    x = torch.randn(1, 16, 6, 32, 64)
    with torch.no_grad():
        out = m(x, scale_id=torch.zeros(1, dtype=torch.long))
    assert out["logits"].shape == (1, 13)


def test_r2plus1d_18_loss_finite_and_grad_flows() -> None:
    """Forward + focal-loss + backward yields finite gradients on most
    optimizable parameters."""
    from nid_video.trainer.loss import FocalLoss

    m = R2Plus1D18ForNID(num_classes=13, pretrained=False,
                         gradient_checkpointing=False)
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
def test_r2plus1d_18_pretrained_first_three_channels_preserved() -> None:
    """Pin that K400 pretraining survives the 6-channel adapter on the
    first 3 input channels. R(2+1)D-18's stem[0] kernel is (1, 7, 7) —
    same as our target — so the trilinear "downsample" is an identity
    transform; ``adapt_conv3d_to_6ch`` therefore produces ch[0:3]
    BYTE-IDENTICAL to the original K400 stem weights.

    Secondary check: ch[3:6] is Kaiming-initialised, so its weights
    should NOT match ch[0:3].

    Slow-marked because it pulls the real ckpt from the torchvision hub.
    """
    from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

    m = R2Plus1D18ForNID(num_classes=13, in_channels=6,
                         gradient_checkpointing=False)
    stem_conv = m.backbone.stem[0]
    assert stem_conv.weight.shape[1] == 6

    ref = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    ref_conv = ref.stem[0]
    assert ref_conv.weight.shape[1] == 3

    adapted_first_three = stem_conv.weight.data[:, :3]
    torch.testing.assert_close(
        adapted_first_three, ref_conv.weight.data,
        atol=1e-7, rtol=0.0,
    )

    adapted_extra = stem_conv.weight.data[:, 3:]
    assert not torch.allclose(adapted_first_three, adapted_extra), (
        "ch[3:6] looks identical to ch[0:3] — adapter may have broadcast "
        "pretrained weights into the extra channels instead of "
        "Kaiming-initialising them"
    )
