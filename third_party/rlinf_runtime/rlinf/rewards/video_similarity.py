# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.models.embodiment.cosmos.camera_layout import (
    stitch_cosmos_three_view_chw,
)

VIDEO_SIMILARITY_NEG_MSE = "neg_mse"
VIDEO_RANGE_POLICIES = {"error", "clamp"}
VIDEO_ALIGNMENT_MODES = {"legacy", "aligned"}
EDGE_FULL_FOV_LAYOUT = "edge_full_fov"


def _video_to_bcthw(video: torch.Tensor) -> torch.Tensor:
    if video.dim() != 5:
        raise ValueError(
            "Expected video tensor [B,T,H,W,C] or [B,C,T,H,W], "
            f"got {video.shape}"
        )
    if video.shape[-1] in (1, 3):
        video = video.permute(0, 4, 1, 2, 3)
    elif video.shape[1] in (1, 3):
        video = video
    else:
        raise ValueError(
            f"Cannot infer channel dimension for video tensor {video.shape}"
        )
    return video.contiguous()


def _normalize_video(
    video: torch.Tensor,
    *,
    name: str,
    range_policy: str,
) -> torch.Tensor:
    if range_policy not in VIDEO_RANGE_POLICIES:
        raise ValueError(
            f"Unsupported video range policy '{range_policy}'. "
            "Supported values: 'error', 'clamp'."
        )

    video = _video_to_bcthw(video)
    if video.numel() == 0:
        return video.to(torch.float32)

    if not torch.is_floating_point(video):
        return video.to(torch.float32) / 255.0

    video = video.to(torch.float32)
    video_min = float(video.min().item())
    video_max = float(video.max().item())
    if 0.0 <= video_min and video_max <= 1.0:
        return video
    if -1.0 <= video_min and video_max <= 1.0:
        return (video + 1.0) / 2.0
    if range_policy == "clamp":
        return torch.clamp(video, 0.0, 1.0)
    raise ValueError(
        f"{name} has unsupported float range [{video_min}, {video_max}]. "
        "Expected [0,1] or [-1,1]."
    )


def _resize_video(video: torch.Tensor, size: tuple[int, int] | list[int]) -> torch.Tensor:
    height, width = int(size[0]), int(size[1])
    bsz, channels, timesteps, _, _ = video.shape
    video = video.permute(0, 2, 1, 3, 4).reshape(
        bsz * timesteps, channels, *video.shape[-2:]
    )
    video = F.interpolate(
        video, size=(height, width), mode="bilinear", align_corners=False
    )
    return video.reshape(bsz, timesteps, channels, height, width).permute(
        0, 2, 1, 3, 4
    )


def _resize_video_letterbox(
    video: torch.Tensor, size: tuple[int, int] | list[int]
) -> torch.Tensor:
    """Resize preserving aspect ratio, padding with black to fill `size`.

    Unlike `_resize_video`, this never visibly stretches/distorts the image
    content. Needed for human-facing comparison video where the source's
    native aspect ratio can differ a lot from the target `size` (e.g.
    Cosmos's native 3-view concat input is much taller-than-wide than the
    reward model's own video_similarity size) -- a plain resize would squash
    or stretch the picture.
    """
    target_h, target_w = int(size[0]), int(size[1])
    bsz, channels, timesteps, cur_h, cur_w = video.shape
    video = video.permute(0, 2, 1, 3, 4).reshape(
        bsz * timesteps, channels, cur_h, cur_w
    )
    ratio = max(cur_w / target_w, cur_h / target_h)
    resized_h = max(1, int(round(cur_h / ratio)))
    resized_w = max(1, int(round(cur_w / ratio)))
    video = F.interpolate(
        video, size=(resized_h, resized_w), mode="bilinear", align_corners=False
    )
    pad_h0, remainder_h = divmod(target_h - resized_h, 2)
    pad_w0, remainder_w = divmod(target_w - resized_w, 2)
    video = F.pad(
        video, (pad_w0, pad_w0 + remainder_w, pad_h0, pad_h0 + remainder_h), value=0.0
    )
    return video.reshape(bsz, timesteps, channels, target_h, target_w).permute(
        0, 2, 1, 3, 4
    )


def _normalize_mask(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    timesteps: int,
    device: torch.device,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.shape == (batch_size, timesteps, 1):
        mask = mask.squeeze(-1)
    if mask.shape != (batch_size, timesteps):
        raise ValueError(
            f"Expected video similarity mask shape [B,T], got {mask.shape}."
        )
    return mask.to(device=device, dtype=torch.bool)


def _safe_stat(values: torch.Tensor, op: str) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=values.device)
    if op == "mean":
        return values.mean()
    if op == "std":
        return values.std(unbiased=False)
    if op == "min":
        return values.min()
    if op == "max":
        return values.max()
    raise ValueError(f"Unsupported stat op: {op}")


def _video_reward_stats(
    *,
    imagined: torch.Tensor,
    actual: torch.Tensor,
    mse: torch.Tensor,
    reward: torch.Tensor,
    mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    valid_mse = mse[mask] if mask is not None else mse.reshape(-1)
    valid_reward = reward[mask] if mask is not None else reward.reshape(-1)
    valid_count = (
        mask.sum().to(dtype=torch.float32)
        if mask is not None
        else torch.tensor(float(reward.numel()), device=reward.device)
    )
    return {
        "per_frame_mse": mse,
        "valid_frame_count": valid_count,
        "imagined_min": imagined.min(),
        "imagined_max": imagined.max(),
        "ctrl_world_min": actual.min(),
        "ctrl_world_max": actual.max(),
        "reference_min": actual.min(),
        "reference_max": actual.max(),
        "mse_mean": _safe_stat(valid_mse, "mean"),
        "mse_std": _safe_stat(valid_mse, "std"),
        "mse_min": _safe_stat(valid_mse, "min"),
        "mse_max": _safe_stat(valid_mse, "max"),
        "reward_mean": _safe_stat(valid_reward, "mean"),
        "reward_std": _safe_stat(valid_reward, "std"),
        "reward_min": _safe_stat(valid_reward, "min"),
        "reward_max": _safe_stat(valid_reward, "max"),
    }


def _build_aligned_ctrl_world_composite(
    *,
    imagined: torch.Tensor,
    ctrl_main: torch.Tensor,
    ctrl_wrist_video: torch.Tensor | None,
    ctrl_extra_video: torch.Tensor | None,
    camera_layout: str,
    range_policy: str,
) -> torch.Tensor:
    """Stitch Ctrl-World views into the semantic layout imagined by Cosmos."""
    missing = [
        name
        for name, video in (
            ("ctrl_world_video_chunk_wrist", ctrl_wrist_video),
            ("ctrl_world_video_chunk_extra", ctrl_extra_video),
        )
        if video is None
    ]
    if missing:
        raise ValueError(
            "Aligned video similarity requires all three Ctrl-World views; "
            f"missing {missing}."
        )

    ctrl_wrist = _normalize_video(
        ctrl_wrist_video,
        name="ctrl_world_video_chunk_wrist",
        range_policy=range_policy,
    ).to(imagined.device)
    ctrl_extra = _normalize_video(
        ctrl_extra_video,
        name="ctrl_world_video_chunk_extra",
        range_policy=range_policy,
    ).to(imagined.device)

    for name, view in (
        ("ctrl_world_video_chunk", ctrl_main),
        ("ctrl_world_video_chunk_wrist", ctrl_wrist),
        ("ctrl_world_video_chunk_extra", ctrl_extra),
    ):
        if view.shape[:3] != imagined.shape[:3]:
            raise ValueError(
                "Aligned view batch, channel, and time dimensions must match: "
                f"imagined={tuple(imagined.shape)}, {name}={tuple(view.shape)}."
            )

    imagined_height, imagined_width = imagined.shape[-2:]
    if camera_layout == EDGE_FULL_FOV_LAYOUT:
        # Edge training uses wrist at full size on top, front/right at half size below.
        # This path resizes full views and never crops source pixels.
        ctrl_wrist = _resize_video(ctrl_wrist, (480, 640))
        ctrl_main = _resize_video(ctrl_main, (240, 320))
        ctrl_extra = _resize_video(ctrl_extra, (240, 320))
        ctrl_composite = torch.cat(
            [ctrl_wrist, torch.cat([ctrl_main, ctrl_extra], dim=-1)], dim=-2
        )
        return _resize_video(ctrl_composite, (imagined_height, imagined_width))

    if imagined_height % 3 != 0 or imagined_width % 2 != 0:
        raise ValueError(
            "Aligned Cosmos composite must have height divisible by 3 and "
            f"width divisible by 2, got {(imagined_height, imagined_width)}."
        )

    target_main_height = 2 * (imagined_height // 3)
    target_main_aspect_ratio = target_main_height / imagined_width

    # BCTHW -> BTCHW lets the shared layout helper preserve B and T as
    # leading axes while applying the exact policy-input camera transform.
    ctrl_composite = stitch_cosmos_three_view_chw(
        ctrl_main.permute(0, 2, 1, 3, 4),
        ctrl_wrist.permute(0, 2, 1, 3, 4),
        ctrl_extra.permute(0, 2, 1, 3, 4),
        layout=camera_layout,
        real_camera_aspect_ratio=target_main_aspect_ratio,
        upscale_to_main_height=target_main_height,
    ).permute(0, 2, 1, 3, 4)

    if ctrl_composite.shape != imagined.shape:
        raise ValueError(
            "Aligned Ctrl-World composite did not match imagined video: "
            f"imagined={tuple(imagined.shape)}, "
            f"ctrl_composite={tuple(ctrl_composite.shape)}."
        )
    return ctrl_composite


def _aligned_view_mse_stats(
    *,
    imagined: torch.Tensor,
    actual: torch.Tensor,
    camera_layout: str,
    mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Return per-view MSE diagnostics from aligned three-view composites."""
    _, _, _, height, width = imagined.shape
    top_end = round(2 * height / 3)
    left_end = width // 2

    top = (slice(0, top_end), slice(0, width))
    bottom_left = (slice(top_end, height), slice(0, left_end))
    bottom_right = (slice(top_end, height), slice(left_end, width))
    view_slices = (
        {"main": top, "wrist": bottom_left, "extra": bottom_right}
        if camera_layout == "main_top"
        else {"wrist": top, "main": bottom_left, "extra": bottom_right}
    )

    pixel_mse = (imagined - actual).square().mean(dim=1)
    stats: dict[str, torch.Tensor] = {}
    for view_name, (height_slice, width_slice) in view_slices.items():
        per_frame = pixel_mse[..., height_slice, width_slice].mean(dim=(-2, -1))
        valid = per_frame[mask] if mask is not None else per_frame.reshape(-1)
        stats[f"{view_name}_per_frame_mse"] = per_frame
        stats[f"{view_name}_mse_mean"] = _safe_stat(valid, "mean")
    return stats


def compute_video_similarity_reward(
    imagined_video_chunk: torch.Tensor,
    ctrl_world_video_chunk: torch.Tensor,
    size: tuple[int, int] | list[int] | None = None,
    *,
    ctrl_world_video_chunk_wrist: torch.Tensor | None = None,
    ctrl_world_video_chunk_extra: torch.Tensor | None = None,
    alignment_mode: str = "legacy",
    camera_layout: str = "main_top",
    metric: str = VIDEO_SIMILARITY_NEG_MSE,
    mask: torch.Tensor | None = None,
    return_stats: bool = False,
    range_policy: str = "error",
    composite_view_stats: bool = False,
    view_weights: Mapping[str, float] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return per-frame negative-MSE reward with shape `[B, T]`.

    Weights name physical main/wrist/extra cameras, independent of canvas area.
    None preserves the original pixel-average reward.
    """

    if metric != VIDEO_SIMILARITY_NEG_MSE:
        raise ValueError(
            f"Unsupported video similarity metric '{metric}'. "
            f"Supported metrics: ['{VIDEO_SIMILARITY_NEG_MSE}']."
        )
    if alignment_mode not in VIDEO_ALIGNMENT_MODES:
        raise ValueError(
            f"Unsupported video alignment mode {alignment_mode!r}; "
            f"expected one of {sorted(VIDEO_ALIGNMENT_MODES)}."
        )

    weights = None
    if view_weights is not None:
        if camera_layout not in {"droid", "main_top"} or not (
            alignment_mode == "aligned" or composite_view_stats
        ):
            raise ValueError("View weights require a droid/main_top three-view composite.")
        if set(view_weights) != {"main", "wrist", "extra"}:
            raise ValueError("View weights must name exactly main, wrist, and extra.")
        weights = {name: float(value) for name, value in view_weights.items()}
        if any(not np.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("View weights must be finite and nonnegative.")
        if not np.isclose(sum(weights.values()), 1.0, rtol=0, atol=1e-6):
            raise ValueError("View weights must sum to 1.")

    imagined = _normalize_video(
        imagined_video_chunk, name="imagined_video_chunk", range_policy=range_policy
    )
    actual = _normalize_video(
        ctrl_world_video_chunk,
        name="ctrl_world_video_chunk",
        range_policy=range_policy,
    ).to(imagined.device)
    if alignment_mode == "aligned":
        actual = _build_aligned_ctrl_world_composite(
            imagined=imagined,
            ctrl_main=actual,
            ctrl_wrist_video=ctrl_world_video_chunk_wrist,
            ctrl_extra_video=ctrl_world_video_chunk_extra,
            camera_layout=camera_layout,
            range_policy=range_policy,
        )
    if imagined.shape[:3] != actual.shape[:3]:
        raise ValueError(
            "Video batch, channel, and time dimensions must match: "
            f"{imagined.shape=} {actual.shape=}"
        )
    if size is None and imagined.shape[-2:] != actual.shape[-2:]:
        raise ValueError(
            "Video spatial dimensions must match when resize size is not set: "
            f"{imagined.shape=} {actual.shape=}"
        )

    if size is not None:
        imagined = _resize_video(imagined, size)
        actual = _resize_video(actual, size)

    bsz, _, timesteps, _, _ = imagined.shape
    valid_mask = _normalize_mask(
        mask, batch_size=bsz, timesteps=timesteps, device=imagined.device
    )
    pixel_mse = ((imagined - actual) ** 2).mean(dim=(1, 3, 4))
    view_stats = {}
    if weights is not None or (return_stats and (
        alignment_mode == "aligned" or composite_view_stats
    )):
        view_stats = _aligned_view_mse_stats(
            imagined=imagined, actual=actual, camera_layout=camera_layout,
            mask=valid_mask,
        )
    mse = pixel_mse if weights is None else sum(
        weights[name] * view_stats[f"{name}_per_frame_mse"]
        for name in ("main", "wrist", "extra")
    )
    reward = -mse
    if valid_mask is not None:
        reward = reward.masked_fill(~valid_mask, 0.0)

    if return_stats:
        stats = _video_reward_stats(
            imagined=imagined,
            actual=actual,
            mse=mse,
            reward=reward,
            mask=valid_mask,
        )
        stats.update(view_stats)
        stats["pixel_per_frame_mse"] = pixel_mse
        stats["pixel_mse_mean"] = _safe_stat(
            pixel_mse[valid_mask] if valid_mask is not None else pixel_mse.reshape(-1),
            "mean",
        )
        return reward, stats
    return reward


def compute_pixel_video_similarity_reward(
    predicted_video_chunk: torch.Tensor,
    reference_video_chunk: torch.Tensor,
    size: tuple[int, int] | list[int] | None = None,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Neutral-name wrapper for pixel negative-MSE video reward.

    The original function remains the backward-compatible Ctrl-World API.
    """
    return compute_video_similarity_reward(
        predicted_video_chunk, reference_video_chunk, size=size, **kwargs
    )


_CLIP_MODEL_CACHE: dict[str, tuple] = {}


def _load_clip_model(clip_model_path: str, device: torch.device):
    """Lazily load and cache a frozen CLIP vision encoder for one process."""
    cache_key = f"{clip_model_path}:{device}"
    if cache_key in _CLIP_MODEL_CACHE:
        return _CLIP_MODEL_CACHE[cache_key]
    from transformers import CLIPImageProcessor, CLIPModel

    model = CLIPModel.from_pretrained(clip_model_path).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    processor = CLIPImageProcessor.from_pretrained(clip_model_path)
    _CLIP_MODEL_CACHE[cache_key] = (model, processor)
    return model, processor


def _image_to_bchw(image: torch.Tensor, *, name: str) -> torch.Tensor:
    """Normalize a single image or image batch to BCHW layout."""
    if image.dim() == 3:
        if image.shape[0] in (1, 3):
            image = image.unsqueeze(0)
        elif image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Cannot infer channel dimension for {name} {image.shape}")
    elif image.dim() == 4:
        if image.shape[1] in (1, 3):
            pass
        elif image.shape[-1] in (1, 3):
            image = image.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Cannot infer channel dimension for {name} {image.shape}")
    else:
        raise ValueError(
            f"Expected {name} as [C,H,W], [H,W,C], [B,C,H,W], or "
            f"[B,H,W,C], got {image.shape}."
        )
    return image.contiguous()


def _normalize_image(
    image: torch.Tensor, *, name: str, range_policy: str
) -> torch.Tensor:
    """Normalize uint8/[0,1]/[-1,1] image data to float [0,1]."""
    if range_policy not in VIDEO_RANGE_POLICIES:
        raise ValueError(f"Unsupported video range policy {range_policy!r}.")
    image = _image_to_bchw(image, name=name)
    if not torch.is_floating_point(image):
        return image.to(torch.float32) / 255.0
    image = image.to(torch.float32)
    image_min = float(image.min().item())
    image_max = float(image.max().item())
    if 0.0 <= image_min and image_max <= 1.0:
        return image
    if -1.0 <= image_min and image_max <= 1.0:
        return (image + 1.0) / 2.0
    if range_policy == "clamp":
        return image.clamp(0.0, 1.0)
    raise ValueError(
        f"{name} has unsupported float range [{image_min}, {image_max}]."
    )


def compute_terminal_goal_reward(
    ctrl_world_video_chunk: torch.Tensor,
    goal_main_image: torch.Tensor,
    *,
    ctrl_world_video_chunk_wrist: torch.Tensor,
    ctrl_world_video_chunk_extra: torch.Tensor,
    goal_wrist_image: torch.Tensor,
    goal_extra_image: torch.Tensor,
    window_size: int,
    size: tuple[int, int] | list[int],
    view_weights: Mapping[str, float],
    range_policy: str = "error",
    reward_scale: float = 160.0,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare the final Ctrl-World window with a shared three-view goal."""
    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError("Terminal goal window_size must be positive.")
    if len(size) != 2 or any(int(value) <= 0 for value in size):
        raise ValueError(f"Terminal goal size must be positive, got {size}.")
    reward_scale = float(reward_scale)
    if not np.isfinite(reward_scale) or reward_scale <= 0.0:
        raise ValueError("Terminal goal reward_scale must be finite and positive.")
    weights = {
        name: float(view_weights[name]) for name in ("main", "wrist", "extra")
    }
    if any(not np.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError(f"Invalid terminal-goal view weights: {weights}.")
    weight_sum = sum(weights.values())
    if weight_sum <= 0.0:
        raise ValueError("Terminal-goal view weights must have positive sum.")
    weights = {name: value / weight_sum for name, value in weights.items()}

    videos = {
        "main": ctrl_world_video_chunk,
        "wrist": ctrl_world_video_chunk_wrist,
        "extra": ctrl_world_video_chunk_extra,
    }
    goals = {
        "main": goal_main_image,
        "wrist": goal_wrist_image,
        "extra": goal_extra_image,
    }
    per_view_mse: dict[str, torch.Tensor] = {}
    batch_size = None
    device = None
    for view_name in ("main", "wrist", "extra"):
        video = _normalize_video(
            videos[view_name],
            name=f"ctrl_world_video_chunk_{view_name}",
            range_policy=range_policy,
        )
        if video.shape[2] < window_size:
            raise ValueError(
                f"Terminal window {window_size} exceeds {view_name} length "
                f"{video.shape[2]}."
            )
        if batch_size is None:
            batch_size, device = int(video.shape[0]), video.device
        elif int(video.shape[0]) != batch_size:
            raise ValueError("Terminal-goal view batch sizes must match.")
        video = _resize_video(video[:, :, -window_size:], size)
        goal = _normalize_image(
            goals[view_name],
            name=f"goal_{view_name}_image",
            range_policy=range_policy,
        ).to(video.device)
        goal = F.interpolate(
            goal,
            size=(int(size[0]), int(size[1])),
            mode="bilinear",
            align_corners=False,
        )
        if goal.shape[0] == 1:
            goal = goal.expand(video.shape[0], -1, -1, -1)
        elif goal.shape[0] != video.shape[0]:
            raise ValueError("Terminal-goal image batch must be 1 or video batch.")
        per_view_mse[view_name] = (
            video - goal.unsqueeze(2)
        ).square().mean(dim=(1, 2, 3, 4))

    assert batch_size is not None and device is not None
    weighted_mse = torch.zeros(batch_size, dtype=torch.float32, device=device)
    for view_name, mse in per_view_mse.items():
        weighted_mse += weights[view_name] * mse.to(device)
    reward = -reward_scale * weighted_mse
    if not return_stats:
        return reward
    stats = {
        "per_episode_mse": weighted_mse,
        "per_episode_reward": reward,
        "mse_mean": weighted_mse.mean(),
        "mse_std": weighted_mse.std(unbiased=False),
        "reward_mean": reward.mean(),
        "reward_std": reward.std(unbiased=False),
        "reward_min": reward.min(),
        "reward_max": reward.max(),
    }
    for view_name, mse in per_view_mse.items():
        stats[f"{view_name}_per_episode_mse"] = mse
        stats[f"{view_name}_mse_mean"] = mse.mean()
        stats[f"{view_name}_weight"] = torch.tensor(weights[view_name], device=device)
    return reward, stats


def compute_video_clip_similarity_reward(
    imagined_video_chunk: torch.Tensor,
    ctrl_world_video_chunk: torch.Tensor,
    *,
    clip_model_path: str,
    frame_index: int | None = -1,
    range_policy: str = "error",
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Perceptual reward: CLIP image-embedding cosine similarity.

    Cosmos3 and Ctrl-World decode through unrelated VAEs (Wan2.2 vs SVD's own
    AutoencoderKLTemporalDecoder) trained independently of each other, so their
    internal pre-decode latents live in unrelated coordinate systems and are
    not directly comparable. This instead maps each model's already-decoded
    *pixel* frame(s) into a shared, frozen CLIP embedding space and compares
    there -- reward is genuine cosine similarity in [-1, 1] (higher is
    better), not a negative distance like the MSE reward.

    ``frame_index``: an int compares only that one frame (e.g. -1 for the
    plan doc's original "last frame" framing). ``None`` instead computes
    per-frame cosine similarity for every frame in the chunk and averages
    over time -- this matches the *actual* training reward's
    ``reward_type=chunk_level`` aggregation (which sums all per-frame pixel
    rewards, not just the last one) far more closely than the single-frame
    variant does, at the cost of T times more CLIP forward passes.
    """
    imagined = _normalize_video(
        imagined_video_chunk, name="imagined_video_chunk", range_policy=range_policy
    )
    actual = _normalize_video(
        ctrl_world_video_chunk, name="ctrl_world_video_chunk", range_policy=range_policy
    ).to(imagined.device)

    device = imagined.device
    model, processor = _load_clip_model(clip_model_path, device)
    clip_mean = torch.tensor(processor.image_mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    clip_std = torch.tensor(processor.image_std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    crop_size = getattr(processor, "crop_size", 224)
    if isinstance(crop_size, dict):
        target_size = (crop_size["height"], crop_size["width"])
    else:
        target_size = (int(crop_size), int(crop_size))

    def to_clip_input(frame: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(
            frame.to(torch.float32),
            size=target_size,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
        return (resized - clip_mean) / clip_std

    if frame_index is not None:
        # [B, C, T, H, W] -> [B, C, H, W]
        imagined_frame = imagined[:, :, frame_index]
        actual_frame = actual[:, :, frame_index]
        with torch.no_grad():
            feat_imagined = model.get_image_features(pixel_values=to_clip_input(imagined_frame))
            feat_actual = model.get_image_features(pixel_values=to_clip_input(actual_frame))
        feat_imagined = F.normalize(feat_imagined, dim=-1)
        feat_actual = F.normalize(feat_actual, dim=-1)
        reward = (feat_imagined * feat_actual).sum(dim=-1)  # [B] cosine similarity
    else:
        # Every frame, per-frame cosine similarity, averaged over time.
        bsz, _, timesteps = imagined.shape[:3]
        # [B, C, T, H, W] -> [B*T, C, H, W]
        imagined_flat = imagined.permute(0, 2, 1, 3, 4).reshape(
            bsz * timesteps, *imagined.shape[1:2], *imagined.shape[3:]
        )
        actual_flat = actual.permute(0, 2, 1, 3, 4).reshape(
            bsz * timesteps, *actual.shape[1:2], *actual.shape[3:]
        )
        with torch.no_grad():
            feat_imagined = model.get_image_features(pixel_values=to_clip_input(imagined_flat))
            feat_actual = model.get_image_features(pixel_values=to_clip_input(actual_flat))
        feat_imagined = F.normalize(feat_imagined, dim=-1).reshape(bsz, timesteps, -1)
        feat_actual = F.normalize(feat_actual, dim=-1).reshape(bsz, timesteps, -1)
        per_frame_similarity = (feat_imagined * feat_actual).sum(dim=-1)  # [B, T]
        reward = per_frame_similarity.mean(dim=-1)  # [B]

    if return_stats:
        valid = reward.reshape(-1)
        stats = {
            "reward_mean": _safe_stat(valid, "mean"),
            "reward_std": _safe_stat(valid, "std"),
            "reward_min": _safe_stat(valid, "min"),
            "reward_max": _safe_stat(valid, "max"),
            "per_env_reward": reward,
        }
        return reward, stats
    return reward


_DINO_MODEL_CACHE: dict[str, tuple] = {}


def _load_dino_model(dino_model_path: str, device: torch.device):
    """Lazily load and cache a frozen DINOv2 vision encoder for one process."""
    cache_key = f"{dino_model_path}:{device}"
    if cache_key in _DINO_MODEL_CACHE:
        return _DINO_MODEL_CACHE[cache_key]
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(dino_model_path).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(dino_model_path)
    _DINO_MODEL_CACHE[cache_key] = (model, processor)
    return model, processor


def compute_video_dino_similarity_reward(
    imagined_video_chunk: torch.Tensor,
    ctrl_world_video_chunk: torch.Tensor,
    *,
    dino_model_path: str,
    frame_index: int = -1,
    range_policy: str = "error",
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Perceptual reward: DINOv2 embedding cosine similarity of one frame.

    Same rationale and pixel-space entry point as
    :func:`compute_video_clip_similarity_reward`, but DINOv2 is a
    self-supervised *vision-only* encoder trained to preserve fine-grained
    spatial/structural detail (patch-level features), rather than CLIP's
    language-aligned *semantic* embedding -- worth comparing directly if CLIP
    similarity turns out too coarse-grained to track pose/position changes.
    """
    imagined = _normalize_video(
        imagined_video_chunk, name="imagined_video_chunk", range_policy=range_policy
    )
    actual = _normalize_video(
        ctrl_world_video_chunk, name="ctrl_world_video_chunk", range_policy=range_policy
    ).to(imagined.device)
    # [B, C, T, H, W] -> [B, C, H, W]
    imagined_frame = imagined[:, :, frame_index]
    actual_frame = actual[:, :, frame_index]

    device = imagined_frame.device
    model, processor = _load_dino_model(dino_model_path, device)
    dino_mean = torch.tensor(processor.image_mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    dino_std = torch.tensor(processor.image_std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    crop_size = getattr(processor, "crop_size", 224)
    if isinstance(crop_size, dict):
        target_size = (crop_size["height"], crop_size["width"])
    else:
        target_size = (int(crop_size), int(crop_size))

    def to_dino_input(frame: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(
            frame.to(torch.float32),
            size=target_size,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
        return (resized - dino_mean) / dino_std

    with torch.no_grad():
        out_imagined = model(pixel_values=to_dino_input(imagined_frame))
        out_actual = model(pixel_values=to_dino_input(actual_frame))
    feat_imagined = getattr(out_imagined, "pooler_output", None)
    if feat_imagined is None:
        feat_imagined = out_imagined.last_hidden_state[:, 0]
    feat_actual = getattr(out_actual, "pooler_output", None)
    if feat_actual is None:
        feat_actual = out_actual.last_hidden_state[:, 0]
    feat_imagined = F.normalize(feat_imagined, dim=-1)
    feat_actual = F.normalize(feat_actual, dim=-1)
    reward = (feat_imagined * feat_actual).sum(dim=-1)  # [B] cosine similarity

    if return_stats:
        valid = reward.reshape(-1)
        stats = {
            "reward_mean": _safe_stat(valid, "mean"),
            "reward_std": _safe_stat(valid, "std"),
            "reward_min": _safe_stat(valid, "min"),
            "reward_max": _safe_stat(valid, "max"),
            "per_env_reward": reward,
        }
        return reward, stats
    return reward


def render_comparison_grid_frames(

    imagined_video_chunk: torch.Tensor,
    ctrl_world_video_chunk: torch.Tensor,
    ctrl_world_video_chunk_wrist: torch.Tensor | None,
    ctrl_world_video_chunk_extra: torch.Tensor | None,
    size: tuple[int, int] | list[int] | None = None,
    *,
    range_policy: str = "error",
    frame_offset: int = 0,
    total_frames: int | None = None,
    camera_layout: str = "main_top",
    reference_is_composite: bool = False,
    reference_label: str = "Ctrl-World predicted",
) -> np.ndarray:
    """Render a Cosmos-vs-Ctrl-World comparison grid for the first sample in
    the batch, for a human to visually judge fidelity.

    Cosmos policies trained on a 3-camera concat layout (main view on top,
    wrist+extra views scaled to half size side-by-side on the bottom -- see
    e.g. cosmos-framework's UR5EEFLeRobotDataset) receive and imagine that
    same composite image, so "Cosmos plan" already carries all 3 views
    baked into a single frame, not just the main view. Layout, left to
    right:
      - "Cosmos plan": Cosmos's own predicted frame (the full 3-view
        composite for checkpoints trained that way), letterboxed (not
        stretched) to the right-hand composite's height so its native
        aspect ratio survives even when it differs a lot from `size`.
      - "Ctrl-World predicted": Ctrl-World's own 3-view composite — main on
        top spanning the full width, wrist bottom-left, extra bottom-right
        (matching the same training-time layout).

    Returns uint8 frames with shape ``[T, 2*H, 4*W, 3]``.
    """
    from PIL import Image

    from rlinf.envs.utils import put_text_on_image

    if camera_layout not in {"main_top", "droid", EDGE_FULL_FOV_LAYOUT}:
        raise ValueError(
            f"Unsupported comparison camera layout {camera_layout!r}; "
            "expected 'main_top', 'droid', or 'edge_full_fov'."
        )

    imagined = _normalize_video(
        imagined_video_chunk, name="imagined_video_chunk", range_policy=range_policy
    )
    ctrl_main = _normalize_video(
        ctrl_world_video_chunk, name="ctrl_world_video_chunk", range_policy=range_policy
    ).to(imagined.device)

    # Base the display resolution on Cosmos's own (already reasonably sharp)
    # native size, not the reward model's small `size` -- Ctrl-World's own
    # generated tiles are much smaller (its native working resolution, e.g.
    # 192x320) than Cosmos's, so upscaling Ctrl-World's tiles UP to match
    # keeps both panels comparably sharp, instead of squashing Cosmos DOWN
    # to Ctrl-World's small size (which visibly blurs only the Cosmos side).
    # `size` (the reward model's own comparison resolution) intentionally
    # plays no role in what gets displayed here.
    raw_h, raw_w = imagined.shape[-2], imagined.shape[-1]
    if camera_layout == EDGE_FULL_FOV_LAYOUT:
        # Display at the 720x640 training canvas; no source view is cropped.
        view_h, view_w = 240, 320
        panel_h, panel_w = 720, 640
    else:
        view_h = max(1, round(raw_h / 3))
        view_w = max(1, round(raw_w / 2))
        panel_h, panel_w = view_h * 3, view_w * 2

    def _to_uint8_thwc(video: torch.Tensor) -> np.ndarray:
        frame = video[0].clamp(0.0, 1.0)  # [C, T, H, W]
        frame = (frame * 255.0).round().to(torch.uint8)
        return frame.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]

    def _prep_optional(video: torch.Tensor | None) -> np.ndarray | None:
        if video is None:
            return None
        normalized = _normalize_video(
            video, name="ctrl_world_extra_view", range_policy=range_policy
        ).to(imagined.device)
        normalized = _resize_video(normalized, (view_h, view_w))
        return _to_uint8_thwc(normalized)

    # Letterbox (not stretch) Cosmos's own plan to the final panel size
    # (3*view_h, 2*view_w) in one step -- this is a close-to-1:1 resize
    # since view_h/view_w were derived from imagined's own size, so it stays
    # sharp instead of being squashed down and then re-upscaled.
    imagined_final = _resize_video_letterbox(imagined, (panel_h, panel_w))
    cosmos_frames = _to_uint8_thwc(imagined_final)
    ctrl_main = (
        _resize_video_letterbox(ctrl_main, (panel_h, panel_w))
        if reference_is_composite
        else _resize_video(ctrl_main, (view_h, view_w))
    )
    ctrl_main_frames = _to_uint8_thwc(ctrl_main)
    ctrl_wrist_frames = _prep_optional(ctrl_world_video_chunk_wrist)
    ctrl_extra_frames = _prep_optional(ctrl_world_video_chunk_extra)

    def _resample_frames(frames: np.ndarray | None, target_len: int) -> np.ndarray | None:
        if frames is None or frames.shape[0] == target_len:
            return frames
        if frames.shape[0] <= 0:
            return frames
        indices = np.linspace(0, frames.shape[0] - 1, target_len).round().astype(np.int64)
        return frames[indices]

    target_len = max(cosmos_frames.shape[0], ctrl_main_frames.shape[0])
    cosmos_frames = _resample_frames(cosmos_frames, target_len)
    ctrl_main_frames = _resample_frames(ctrl_main_frames, target_len)
    ctrl_wrist_frames = _resample_frames(ctrl_wrist_frames, target_len)
    ctrl_extra_frames = _resample_frames(ctrl_extra_frames, target_len)

    def _assemble_ctrl_world_panel(
        main_tile: np.ndarray, wrist_tile: np.ndarray, extra_tile: np.ndarray
    ) -> np.ndarray:
        """Ctrl-World panel: main tile uniformly upscaled 2x (no aspect
        distortion, width 2*view_w matches the bottom row), wrist/extra kept
        at native size. Total panel height is 3*view_h, matching the Cosmos
        panel's height above.
        """
        if camera_layout in {"droid", EDGE_FULL_FOV_LAYOUT}:
            main_tile, wrist_tile = wrist_tile, main_tile
        main_row = np.array(
            Image.fromarray(main_tile).resize((view_w * 2, view_h * 2))
        )
        bottom_row = np.concatenate([wrist_tile, extra_tile], axis=1)
        return np.concatenate([main_row, bottom_row], axis=0)

    blank_tile = np.zeros((view_h, view_w, 3), dtype=np.uint8)
    num_frames = target_len
    frames = []
    for t in range(num_frames):
        cosmos_panel = cosmos_frames[t]
        wrist_tile = ctrl_wrist_frames[t] if ctrl_wrist_frames is not None else blank_tile
        extra_tile = ctrl_extra_frames[t] if ctrl_extra_frames is not None else blank_tile
        ctrl_panel = (
            ctrl_main_frames[t]
            if reference_is_composite
            else _assemble_ctrl_world_panel(ctrl_main_frames[t], wrist_tile, extra_tile)
        )

        cosmos_panel = put_text_on_image(cosmos_panel, ["Cosmos plan"])
        ctrl_panel = put_text_on_image(ctrl_panel, [reference_label])
        combined = np.concatenate([cosmos_panel, ctrl_panel], axis=1)
        denominator = total_frames if total_frames is not None else num_frames
        frame_label_y = max(56, round(combined.shape[0] * 0.10))
        combined = put_text_on_image(
            combined,
            [f"frame {frame_offset + t + 1}/{denominator}"],
            origin=(8, frame_label_y),
        )
        frames.append(combined)

    return np.stack(frames, axis=0)
