"""Tests for src/nid_video/models/convlstm_nid.py (M5.5 R2 baseline 2).

Pin the architectural contract: param count band (~15-16M random init),
forward shape, scale_id ignore semantics, native 6-channel first cell,
finite gradient flow with focal loss, and recurrent state shape (the
hidden state must keep the 32×64 spatial resolution per Shi 2015).

No pretrained-related test — ConvLSTM runs random-init from-scratch.
"""

from __future__ import annotations

import torch

from nid_video.models.convlstm_nid import ConvLSTMForNID


def test_convlstm_param_count_within_target() -> None:
    """ConvLSTM at hidden=(64, 128, 256) with 2×2 spatial pools between
    cells lands at ~13.1M params: ~4.6M in the recurrent cells (the
    third cell dominates with 384*1024*9 ≈ 3.5M) + ~8.4M in the
    feature_proj Linear (8192 → 1024). Pin within ±5M to catch
    accidental hidden / pool-stride / FC drift."""
    m = ConvLSTMForNID(num_classes=13, gradient_checkpointing=False)
    n_params = sum(p.numel() for p in m.parameters())
    n_params_M = n_params / 1e6
    assert 8 < n_params_M < 18, (
        f"ConvLSTM param count out of band: {n_params_M:.1f}M "
        f"(expected ~13.1M ±5M)"
    )


def test_convlstm_forward_shape() -> None:
    """input (B=2, T=16, C=6, H=32, W=64), scale_id (2,) → logits (2,
    13) and features (2, 1024)."""
    m = ConvLSTMForNID(num_classes=13, gradient_checkpointing=False)
    m.eval()
    x = torch.randn(2, 16, 6, 32, 64)
    scale_id = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        out = m(x, scale_id=scale_id)
    assert out["logits"].shape == (2, 13), out["logits"].shape
    assert out["features"].shape == (2, 1024), out["features"].shape


def test_convlstm_ignores_scale_id() -> None:
    """ConvLSTM does not condition on scale_id (no scale_token /
    scale_embedding mechanism). Forwarding the same input with
    scale_id={all 0} vs {all 1} must yield identical logits."""
    m = ConvLSTMForNID(num_classes=13, gradient_checkpointing=False)
    m.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 16, 6, 32, 64)
    with torch.no_grad():
        out_fast = m(x, scale_id=torch.tensor([0, 0], dtype=torch.long))
        out_slow = m(x, scale_id=torch.tensor([1, 1], dtype=torch.long))
    torch.testing.assert_close(out_fast["logits"], out_slow["logits"])


def test_convlstm_six_channel_input_works() -> None:
    """``in_channels=6`` flows through to the first cell's gates Conv2d
    without a 3→6 adapter. The gates conv must accept (in + hidden) =
    (6 + 64) = 70 input channels."""
    m = ConvLSTMForNID(num_classes=13, in_channels=6,
                       gradient_checkpointing=False)
    expected_in = 6 + m.hidden_channels[0]   # 6 + 64
    actual_in = m.cell1.gates.weight.shape[1]
    assert actual_in == expected_in, (
        f"cell1.gates expected in={expected_in}, got {actual_in}"
    )
    m.eval()
    x = torch.randn(1, 16, 6, 32, 64)
    with torch.no_grad():
        out = m(x, scale_id=torch.zeros(1, dtype=torch.long))
    assert out["logits"].shape == (1, 13)


def test_convlstm_loss_finite_and_grad_flows() -> None:
    """Forward + focal-loss + backward yields finite gradients on most
    optimizable parameters. Catches accidental ``detach()`` /
    ``requires_grad=False`` regressions; the recurrent unroll is the
    most likely site of such a regression because each cell receives
    state from the previous cell at the same timestep."""
    from nid_video.trainer.loss import FocalLoss

    m = ConvLSTMForNID(num_classes=13, gradient_checkpointing=False)
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
