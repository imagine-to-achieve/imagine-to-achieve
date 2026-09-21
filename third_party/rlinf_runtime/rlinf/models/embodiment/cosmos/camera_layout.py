# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared camera-layout transforms for Cosmos three-view policies."""

from __future__ import annotations

import torch
import torch.nn.functional as F

COSMOS_CAMERA_LAYOUTS = {"main_top", "droid"}


def _resize_chw_images(
    images: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    """Resize ``[..., C, H, W]`` images while preserving leading axes."""
    leading_shape = images.shape[:-3]
    channels = images.shape[-3]
    flattened = images.reshape(-1, channels, *images.shape[-2:])
    resized = F.interpolate(
        flattened,
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(*leading_shape, channels, *size)


def stitch_cosmos_three_view_chw(
    main_view: torch.Tensor,
    side_view_a: torch.Tensor,
    side_view_b: torch.Tensor,
    *,
    layout: str = "main_top",
    real_camera_aspect_ratio: float | None = None,
    upscale_to_main_height: int | None = None,
) -> torch.Tensor:
    """Build the Cosmos three-camera composite from ``[..., C, H, W]`` views.

    ``main_top`` places ``main_view`` on top and the two side views below it.
    ``droid`` places ``side_view_a`` on top, with ``main_view`` and
    ``side_view_b`` on the bottom. Side tiles are half the top tile's height
    and width, matching the action-policy training transform.

    Leading dimensions are preserved, so the same implementation handles a
    single policy input ``[C,H,W]`` and reward videos ``[B,T,C,H,W]``.
    """
    if layout not in COSMOS_CAMERA_LAYOUTS:
        raise ValueError(
            f"Unsupported Cosmos camera layout {layout!r}; "
            f"expected one of {sorted(COSMOS_CAMERA_LAYOUTS)}."
        )
    for name, view in (
        ("main_view", main_view),
        ("side_view_a", side_view_a),
        ("side_view_b", side_view_b),
    ):
        if view.ndim < 3:
            raise ValueError(
                f"{name} must have shape [...,C,H,W], got {tuple(view.shape)}."
            )
    expected_prefix = main_view.shape[:-3]
    expected_channels = main_view.shape[-3]
    for name, view in (
        ("side_view_a", side_view_a),
        ("side_view_b", side_view_b),
    ):
        if view.shape[:-3] != expected_prefix or view.shape[-3] != expected_channels:
            raise ValueError(
                "Three-view leading dimensions and channels must match: "
                f"main_view={tuple(main_view.shape)}, {name}={tuple(view.shape)}."
            )

    if real_camera_aspect_ratio is not None:
        if real_camera_aspect_ratio <= 0:
            raise ValueError("real_camera_aspect_ratio must be positive.")
        current_height, current_width = main_view.shape[-2:]
        current_ratio = current_height / current_width
        if abs(current_ratio - real_camera_aspect_ratio) > 1.0e-3:
            corrected_width = max(
                1, round(current_height / real_camera_aspect_ratio)
            )
            corrected_size = (current_height, corrected_width)
            main_view = _resize_chw_images(main_view, corrected_size)
            side_view_a = _resize_chw_images(side_view_a, corrected_size)
            side_view_b = _resize_chw_images(side_view_b, corrected_size)

    if layout == "droid":
        main_view, side_view_a = side_view_a, main_view

    main_height, main_width = main_view.shape[-2:]
    side_size = (main_height // 2, main_width // 2)
    if min(side_size) < 1:
        raise ValueError(
            "Cosmos three-view layout requires a top view of at least 2x2, "
            f"got {(main_height, main_width)}."
        )
    side_view_a = _resize_chw_images(side_view_a, side_size)
    side_view_b = _resize_chw_images(side_view_b, side_size)
    bottom_row = torch.cat([side_view_a, side_view_b], dim=-1)
    composite = torch.cat([main_view, bottom_row], dim=-2)

    if (
        upscale_to_main_height is not None
        and main_height != upscale_to_main_height
    ):
        if upscale_to_main_height <= 0:
            raise ValueError("upscale_to_main_height must be positive.")
        scale = upscale_to_main_height / main_height
        composite = _resize_chw_images(
            composite,
            (
                round(composite.shape[-2] * scale),
                round(composite.shape[-1] * scale),
            ),
        )

    return composite


def split_cosmos_three_view_chw(
    composite: torch.Tensor,
    *,
    layout: str = "main_top",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invert :func:`stitch_cosmos_three_view_chw` into semantic views."""
    if layout not in COSMOS_CAMERA_LAYOUTS:
        raise ValueError(
            f"Unsupported Cosmos camera layout {layout!r}; "
            f"expected one of {sorted(COSMOS_CAMERA_LAYOUTS)}."
        )
    if composite.ndim < 3:
        raise ValueError(
            "Cosmos composite must have shape [...,C,H,W], "
            f"got {tuple(composite.shape)}."
        )
    height, width = composite.shape[-2:]
    if height % 3 != 0 or width % 2 != 0:
        raise ValueError(
            "Cosmos three-view composite must have H divisible by 3 and W "
            f"divisible by 2, got {(height, width)}."
        )

    top_height = height * 2 // 3
    top = composite[..., :top_height, :]
    bottom = composite[..., top_height:, :]
    bottom_a, bottom_b = torch.chunk(bottom, chunks=2, dim=-1)
    output_size = (top_height, width)
    bottom_a = _resize_chw_images(bottom_a, output_size)
    bottom_b = _resize_chw_images(bottom_b, output_size)

    if layout == "droid":
        return bottom_a, top, bottom_b
    return top, bottom_a, bottom_b
