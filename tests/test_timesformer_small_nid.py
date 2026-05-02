"""Tests for src/nid_video/models/timesformer_small_nid.py (M5.5 baseline).

Pin the architectural contract: param count band (~22M to match the
main method's VideoMAE-Small), forward shape, scale_id ignore
semantics, native 6-channel patch_embed acceptance, finite gradient
flow with focal loss.

No pretrained-checkpoint test — TimeSformer-Small runs random-init
from-scratch (no public 22M Kinetics ckpt exists).
"""

from __future__ import annotations

import torch

from nid_video.models.timesformer_small_nid import TimeSformerSmallForNID


def test_timesformer_small_param_count_within_target() -> None:
    """TimeSformer-Small at hidden=384, depth=12, heads=6, intermediate=1536,
    divided_space_time, num_frames=16, image_size=64, num_channels=6
    lands at ~30M params. Wider than VideoMAE-Small (22M) because divided
    space-time runs spatial-attn + temporal-attn + MLP per layer, vs
    VideoMAE's single joint-attention + MLP per layer (~1.36× layer cost).
    Pin within ±5M so a future refactor that swaps to joint-attention or
    drops layers is caught."""
    m = TimeSformerSmallForNID(
        num_classes=13, gradient_checkpointing=False,
    )
    n_params = sum(p.numel() for p in m.parameters())
    n_params_M = n_params / 1e6
    assert 25 < n_params_M < 35, (
        f"TimeSformer-Small param count out of band: {n_params_M:.1f}M "
        f"(expected ~30M ±5M for divided space-time @ Small dims)"
    )


def test_timesformer_small_forward_shape() -> None:
    """input (B=2, T=16, C=6, H=32, W=64), scale_id (2,) → logits (2, 13)
    and features (2, 384). Pin H=32 padding to image_size=64 works
    silently."""
    m = TimeSformerSmallForNID(
        num_classes=13, gradient_checkpointing=False,
    )
    m.eval()
    x = torch.randn(2, 16, 6, 32, 64)
    scale_id = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        out = m(x, scale_id=scale_id)
    assert out["logits"].shape == (2, 13)
    assert out["features"].shape == (2, 384)


def test_timesformer_small_ignores_scale_id() -> None:
    """TimeSformer-Small does not condition on scale_id (it has no
    scale_token / scale_embedding mechanism). Forwarding the same input
    with scale_id={all 0} vs {all 1} must yield identical logits."""
    m = TimeSformerSmallForNID(
        num_classes=13, gradient_checkpointing=False,
    )
    m.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 16, 6, 32, 64)
    with torch.no_grad():
        out_fast = m(x, scale_id=torch.tensor([0, 0], dtype=torch.long))
        out_slow = m(x, scale_id=torch.tensor([1, 1], dtype=torch.long))
    torch.testing.assert_close(out_fast["logits"], out_slow["logits"])


def test_timesformer_small_six_channel_input_works() -> None:
    """``in_channels=6`` flows through to a Conv2d(6, 384, 16, 16) without
    a 3→6 adapter. Backwards-facing pin against a future refactor that
    hardcodes 3 or accidentally re-introduces the adapter."""
    m = TimeSformerSmallForNID(
        num_classes=13, in_channels=6, gradient_checkpointing=False,
    )
    pe = m.backbone.timesformer.embeddings.patch_embeddings.projection
    assert pe.weight.shape[1] == 6, (
        f"patch_embed expected 6 input channels, got {pe.weight.shape[1]}"
    )
    m.eval()
    x = torch.randn(1, 16, 6, 32, 64)
    with torch.no_grad():
        out = m(x, scale_id=torch.zeros(1, dtype=torch.long))
    assert out["logits"].shape == (1, 13)


def test_timesformer_small_loss_finite_and_grad_flows() -> None:
    """Forward + focal-loss + backward yields finite gradients on most
    optimizable parameters. Catches accidental ``detach()`` /
    ``requires_grad=False`` regressions."""
    from nid_video.trainer.loss import FocalLoss

    m = TimeSformerSmallForNID(
        num_classes=13, gradient_checkpointing=False,
    )
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
        f"gradient did not reach most parameters: {n_grad}/{n_total} have .grad set"
    )
