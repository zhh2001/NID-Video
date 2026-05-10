"""Shared input-conv adapters for going from a pretrained 3-channel
backbone to the project's 6-channel NID input. Two families:

  * ``adapt_conv3d_to_6ch`` — for 3D-conv first-stage backbones
    (VideoMAE patch_embed, I3D stem, C3D first conv, R(2+1)D stem).
    Trilinear-downsamples the existing 3-channel kernel to the target
    spatial-temporal kernel size, then concatenates ``n_extra``
    Kaiming-initialised channels.

  * ``adapt_conv2d_to_6ch`` — for 2D-conv first-stage backbones
    (TimeSformer patch_embed, ConvLSTM front-end).
    Bilinear-downsamples the existing 3-channel kernel to the target
    spatial kernel size, then concatenates ``n_extra`` Kaiming
    channels.

Both helpers mirror the pattern established in M3-001 (forensic
finding: the trilinear downsample produces a ~5× lower kernel norm
than the Kaiming-init extra channels — pinning that the norm-ratio
test passes with ratio kaiming/pretrained > 1.5× confirms the
pretrained signal survived the adaptation rather than being
overwritten by fresh init).

The helpers return a NEW conv module with the adapted weights;
callers replace the original module by attribute assignment.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def adapt_conv3d_to_6ch(
    conv3d: nn.Conv3d,
    target_kernel: tuple[int, int, int],
    target_stride: tuple[int, int, int] | None = None,
    n_extra: int = 3,
) -> nn.Conv3d:
    """Build a new ``Conv3d(3 + n_extra, out_ch, ...)`` initialised by
    trilinear-downsampling the source ``conv3d``'s kernel from its
    native ``(T, H, W)`` to ``target_kernel`` for the first 3 input
    channels, and Kaiming-init for the next ``n_extra`` channels.

    Bias is copied from the source verbatim (it doesn't depend on
    input channels).

    Args:
      conv3d: pretrained source conv. ``conv3d.weight`` shape
        ``(out_ch, 3, T_orig, H_orig, W_orig)``.
      target_kernel: ``(T_p, H_p, W_p)`` of the new conv.
      target_stride: same as kernel by default (non-overlapping
        patch projection); pass explicitly for stride != kernel.
      n_extra: number of additional input channels to Kaiming-init.
        Default 3 (3 → 6 channel adaptation, the project default).
    """
    if conv3d.weight.dim() != 5:
        raise ValueError(
            f"expected a Conv3d weight (5-D), got {conv3d.weight.shape}"
        )
    out_ch, in_ch, T_orig, H_orig, W_orig = conv3d.weight.shape
    if in_ch != 3:
        raise ValueError(
            f"adapt_conv3d_to_6ch requires source in_channels=3, got {in_ch}"
        )
    T_p, H_p, W_p = target_kernel
    stride = target_stride if target_stride is not None else target_kernel

    flat = conv3d.weight.data.reshape(out_ch * 3, 1, T_orig, H_orig, W_orig)
    downed = F.interpolate(
        flat, size=(T_p, H_p, W_p), mode="trilinear", align_corners=False,
    )
    down_w = downed.reshape(out_ch, 3, T_p, H_p, W_p)

    if n_extra > 0:
        fresh = torch.empty(out_ch, n_extra, T_p, H_p, W_p)
        nn.init.kaiming_normal_(fresh, nonlinearity="relu")
        new_w = torch.cat([down_w, fresh], dim=1)
    else:
        new_w = down_w

    new_in_ch = 3 + n_extra
    new_conv = nn.Conv3d(
        in_channels=new_in_ch,
        out_channels=out_ch,
        kernel_size=target_kernel,
        stride=stride,
        padding=conv3d.padding,
        dilation=conv3d.dilation,
        bias=conv3d.bias is not None,
    )
    new_conv.weight.data.copy_(new_w)
    if conv3d.bias is not None:
        new_conv.bias.data.copy_(conv3d.bias.data)
    return new_conv


def adapt_conv3d_to_4ch(
    conv3d: nn.Conv3d,
    target_kernel: tuple[int, int, int],
    target_stride: tuple[int, int, int] | None = None,
) -> nn.Conv3d:
    """Build a new ``Conv3d(4, out_ch, ...)`` for the M5.10 dim-2 motion-
    channel ablation: trilinear-downsample the source 3-channel kernel
    from its native ``(T, H, W)`` to ``target_kernel`` for ch[0:3] and
    Kaiming-init a single extra channel for ch4 (the project's
    bit-packed TCP-flag mask, ch1-4 static-only — see Idea.md §3.2).

    This is a thin convenience wrapper over ``adapt_conv3d_to_6ch``
    with ``n_extra=1``; named separately so call sites read as
    "C=4 adapter" rather than "to_6ch adapter passed n_extra=1".

    Regime note: in the project's VideoMAE-S setup the K400 source
    kernel is (2, 16, 16) and the project's target tube_patch is
    (2, 8, 8) — i.e. this is the M3-001 norm-ratio downsample regime,
    not the I3D / R(2+1)D-18 identity-kernel regime. The bit-identity
    sanity in tests therefore compares C=4 vs C=6 cells produced from
    the *same* source ckpt + the *same* downsample (ch[0:3] of both
    cells must match byte-for-byte), rather than asserting the
    returned conv equals the un-downsampled source weights.

    Args:
      conv3d: pretrained source. ``conv3d.weight`` shape
        ``(out_ch, 3, T_orig, H_orig, W_orig)``.
      target_kernel: ``(T_p, H_p, W_p)`` of the new conv.
      target_stride: same as kernel by default.
    """
    return adapt_conv3d_to_6ch(
        conv3d, target_kernel, target_stride=target_stride, n_extra=1,
    )


def adapt_conv2d_to_6ch(
    conv2d: nn.Conv2d,
    target_kernel: tuple[int, int],
    target_stride: tuple[int, int] | None = None,
    n_extra: int = 3,
) -> nn.Conv2d:
    """2D analogue of :func:`adapt_conv3d_to_6ch`. Bilinear-downsamples
    the source 3-ch kernel to ``target_kernel`` and Kaiming-inits
    ``n_extra`` extra channels. Bias copied verbatim.

    Args:
      conv2d: pretrained source conv. ``conv2d.weight`` shape
        ``(out_ch, 3, H_orig, W_orig)``.
      target_kernel: ``(H_p, W_p)`` of the new conv.
      target_stride: same as kernel by default.
      n_extra: number of additional input channels to Kaiming-init.
    """
    if conv2d.weight.dim() != 4:
        raise ValueError(
            f"expected a Conv2d weight (4-D), got {conv2d.weight.shape}"
        )
    out_ch, in_ch, H_orig, W_orig = conv2d.weight.shape
    if in_ch != 3:
        raise ValueError(
            f"adapt_conv2d_to_6ch requires source in_channels=3, got {in_ch}"
        )
    H_p, W_p = target_kernel
    stride = target_stride if target_stride is not None else target_kernel

    flat = conv2d.weight.data.reshape(out_ch * 3, 1, H_orig, W_orig)
    downed = F.interpolate(
        flat, size=(H_p, W_p), mode="bilinear", align_corners=False,
    )
    down_w = downed.reshape(out_ch, 3, H_p, W_p)

    if n_extra > 0:
        fresh = torch.empty(out_ch, n_extra, H_p, W_p)
        nn.init.kaiming_normal_(fresh, nonlinearity="relu")
        new_w = torch.cat([down_w, fresh], dim=1)
    else:
        new_w = down_w

    new_in_ch = 3 + n_extra
    new_conv = nn.Conv2d(
        in_channels=new_in_ch,
        out_channels=out_ch,
        kernel_size=target_kernel,
        stride=stride,
        padding=conv2d.padding,
        dilation=conv2d.dilation,
        bias=conv2d.bias is not None,
    )
    new_conv.weight.data.copy_(new_w)
    if conv2d.bias is not None:
        new_conv.bias.data.copy_(conv2d.bias.data)
    return new_conv
