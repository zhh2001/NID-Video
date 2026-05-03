"""Tests for src/nid_video/models/c3d_small_nid.py (M5.5 R2 baseline 1).

Pin the architectural contract: param count band (~20M random init,
matched-baseline scale), forward shape, scale_id ignore semantics,
native 6-channel first conv, finite gradient flow with focal loss.

No pretrained-related test — C3D-Small runs random-init from-scratch
(no public small-variant K400 ckpt at the project's input scale).
"""

from __future__ import annotations

import torch

from nid_video.models.c3d_small_nid import C3DSmallForNID


def test_c3d_small_param_count_within_target() -> None:
    """C3D-Small at channels (32, 64, 128, 128, 192, 192, 256, 256) and
    FC dims (512, 512, 13) lands at ~20M params — slimmed from the
    published C3D 64/128/256/512 (~78M). Pin within ±5M to catch
    accidental width drift (e.g. a future refactor swapping back to the
    full C3D widths or trimming further to a pure CIFAR-style stem)."""
    m = C3DSmallForNID(num_classes=13, gradient_checkpointing=False)
    n_params = sum(p.numel() for p in m.parameters())
    n_params_M = n_params / 1e6
    assert 15 < n_params_M < 25, (
        f"C3D-Small param count out of band: {n_params_M:.1f}M "
        f"(expected ~20M ±5M for the slimmed widths above)"
    )


def test_c3d_small_forward_shape() -> None:
    """input (B=2, T=16, C=6, H=32, W=64), scale_id (2,) → logits (2,
    13) and features (2, 512). The features dim equals the FC7 width
    (the representation slot returned for parity with other baselines)."""
    m = C3DSmallForNID(num_classes=13, gradient_checkpointing=False)
    m.eval()
    x = torch.randn(2, 16, 6, 32, 64)
    scale_id = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        out = m(x, scale_id=scale_id)
    assert out["logits"].shape == (2, 13), out["logits"].shape
    assert out["features"].shape == (2, 512), out["features"].shape


def test_c3d_small_ignores_scale_id() -> None:
    """C3D-Small does not condition on scale_id (no scale_token /
    scale_embedding mechanism). Forwarding the same input with
    scale_id={all 0} vs {all 1} must yield identical logits."""
    m = C3DSmallForNID(num_classes=13, gradient_checkpointing=False)
    m.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 16, 6, 32, 64)
    with torch.no_grad():
        out_fast = m(x, scale_id=torch.tensor([0, 0], dtype=torch.long))
        out_slow = m(x, scale_id=torch.tensor([1, 1], dtype=torch.long))
    torch.testing.assert_close(out_fast["logits"], out_slow["logits"])


def test_c3d_small_six_channel_input_works() -> None:
    """``in_channels=6`` flows through to a Conv3d(6, 32, 3, 3, 3)
    without a 3→6 adapter. Backwards-facing pin against a future
    refactor that hardcodes 3 input channels."""
    m = C3DSmallForNID(num_classes=13, in_channels=6,
                       gradient_checkpointing=False)
    assert m.conv1.weight.shape[1] == 6, (
        f"conv1 expected 6 input channels, got {m.conv1.weight.shape[1]}"
    )
    m.eval()
    x = torch.randn(1, 16, 6, 32, 64)
    with torch.no_grad():
        out = m(x, scale_id=torch.zeros(1, dtype=torch.long))
    assert out["logits"].shape == (1, 13)


def test_c3d_small_loss_finite_and_grad_flows() -> None:
    """Forward + focal-loss + backward yields finite gradients on most
    optimizable parameters. Catches accidental ``detach()`` /
    ``requires_grad=False`` regressions."""
    from nid_video.trainer.loss import FocalLoss

    m = C3DSmallForNID(num_classes=13, gradient_checkpointing=False)
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
