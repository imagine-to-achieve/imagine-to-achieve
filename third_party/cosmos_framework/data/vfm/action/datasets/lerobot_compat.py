# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compatibility helpers for LeRobot package layout and video backend changes."""

from __future__ import annotations

from pathlib import Path

import torch

try:
    from lerobot.datasets.video_utils import decode_video_frames as _lerobot_decode_video_frames
except ModuleNotFoundError as exc:
    if exc.name != "lerobot.datasets":
        raise
    from lerobot.common.datasets.video_utils import decode_video_frames as _lerobot_decode_video_frames


def _decode_video_frames_pyav(video_path: Path | str, timestamps: list[float], tolerance_s: float) -> torch.Tensor:
    """Decode frames with PyAV when torchvision's VideoReader backend is unavailable."""
    import av
    import numpy as np

    if not timestamps:
        raise ValueError("timestamps must be non-empty")

    path = str(video_path)
    query_ts = np.asarray(timestamps, dtype=np.float64)
    min_ts = float(query_ts.min())
    max_ts = float(query_ts.max())
    slack_s = max(float(tolerance_s), 0.05)

    decoded_frames: list[torch.Tensor] = []
    decoded_ts: list[float] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        try:
            container.seek(max(0, int((min_ts - slack_s) * 1_000_000)), any_frame=False, backward=True)
        except Exception:
            pass

        for frame in container.decode(stream):
            frame_ts = frame.time
            if frame_ts is None and frame.pts is not None and stream.time_base is not None:
                frame_ts = float(frame.pts * stream.time_base)
            if frame_ts is None:
                continue
            frame_ts = float(frame_ts)
            if frame_ts + slack_s < min_ts:
                continue
            if frame_ts - slack_s > max_ts:
                break

            array = frame.to_ndarray(format="rgb24")
            tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
            decoded_frames.append(tensor)
            decoded_ts.append(frame_ts)

    if not decoded_frames:
        raise RuntimeError(f"Could not decode any frames from {path} around timestamps {timestamps}")

    loaded_ts = np.asarray(decoded_ts, dtype=np.float64)
    chosen: list[torch.Tensor] = []
    max_error = 0.0
    for ts in query_ts:
        idx = int(np.argmin(np.abs(loaded_ts - ts)))
        max_error = max(max_error, float(abs(loaded_ts[idx] - ts)))
        chosen.append(decoded_frames[idx])

    if max_error > float(tolerance_s):
        raise AssertionError(
            f"Closest decoded frame is outside tolerance ({max_error:.4f}s > {tolerance_s:.4f}s) for {path}; "
            f"queried timestamps={timestamps}, decoded range=({loaded_ts.min():.4f}, {loaded_ts.max():.4f})"
        )

    return torch.stack(chosen, dim=0)


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    try:
        return _lerobot_decode_video_frames(video_path, timestamps, tolerance_s, backend=backend)
    except AttributeError as exc:
        if "VideoReader" not in str(exc):
            raise
        return _decode_video_frames_pyav(video_path, timestamps, tolerance_s)


__all__ = ["decode_video_frames"]
