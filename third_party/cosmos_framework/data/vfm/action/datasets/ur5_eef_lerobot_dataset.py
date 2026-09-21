# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""UR5 absolute-EEF LeRobot dataset for Cosmos Action SFT.

The source dataset stores absolute EEF waypoints as
``[x, y, z, rotvec_x, rotvec_y, rotvec_z, gripper_qpos]``.  Cosmos action
policies are trained on relative pose actions, so each window is converted to
``[pos_delta(3), rot6d_delta(6), gripper(1)]`` using the first observation state
as the anchor pose and the following absolute action waypoints as targets.

Multi-view geometry is driven by an ordered :class:`CameraLayout`. The declared
region order is exactly the vertical stack order (and DROID region order).
Prompt text is derived from the same layout object so labels cannot drift from
pixels.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from cosmos_framework.data.vfm.action.datasets.lerobot_compat import decode_video_frames

from cosmos_framework.data.vfm.action.action_normalization import load_action_stats, normalize_action
from cosmos_framework.data.vfm.action.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.vfm.action.camera_layout import (
    CameraLayout,
    ViewLayout,
    resolve_camera_layout,
)
from cosmos_framework.data.vfm.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.vfm.action.pose_utils import build_abs_pose_from_components, pose_abs_to_rel

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["concat_view"]

_STATE_FEATURE = "observation.state"
_ACTION_FEATURE = "action"


class UR5EEFLeRobotDataset(ActionBaseDataset):
    """UR5 close-desktop LeRobot dataset with absolute EEF pose commands.

    Actions returned by this dataset are 10D relative-pose commands:
    ``[pos_delta(3), rot6d_delta(6), gripper(1)]``.  The absolute pose convention
    assumes the source rotation fields are axis-angle / rotation vectors.
    """

    def __init__(
        self,
        root: str,
        media_root: str | None = None,
        fps: float = 15.0,
        chunk_length: int = 32,
        mode: str = "joint",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "concat_view",
        action_normalization: str | None = None,
        action_stats_path: str | None = None,
        sample_stride: int = 1,
        check_media: bool = True,
        view_layout: ViewLayout = "droid",
        camera_layout: Sequence[Mapping[str, str]] | CameraLayout | None = None,
        include_episodes: list[int] | None = None,
        exclude_episodes: list[int] | None = None,
    ) -> None:
        if viewpoint != "concat_view":
            raise NotImplementedError("UR5EEFLeRobotDataset only supports concat_view.")
        if view_layout not in ("vertical", "droid"):
            raise ValueError(f"view_layout must be 'vertical' or 'droid', got {view_layout!r}")
        if isinstance(camera_layout, CameraLayout):
            if camera_layout.view_layout != view_layout:
                raise ValueError(
                    f"CameraLayout.view_layout={camera_layout.view_layout!r} does not match "
                    f"view_layout={view_layout!r}"
                )
            self._camera_layout = camera_layout
        else:
            self._camera_layout = resolve_camera_layout(
                view_layout=view_layout,
                camera_layout=camera_layout,
            )
        self._view_layout = view_layout
        self._action_stats_path = Path(action_stats_path).expanduser().resolve() if action_stats_path else None
        if action_normalization is not None and self._action_stats_path is None:
            raise ValueError("action_stats_path is required when action_normalization is enabled")
        super().__init__(
            root=root,
            domain_name="robomind-ur",
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        if include_episodes is not None and exclude_episodes is not None:
            raise ValueError("Only one of include_episodes or exclude_episodes may be set")
        available_episodes = {int(row["episode_index"]) for row in self._rows}
        requested = set(int(value) for value in (include_episodes or exclude_episodes or []))
        unknown = requested - available_episodes
        if unknown:
            raise ValueError(
                f"Unknown episode ids {sorted(unknown)}; available ids are {sorted(available_episodes)}"
            )
        if include_episodes is not None:
            selected_episodes = requested
        elif exclude_episodes is not None:
            selected_episodes = available_episodes - requested
        else:
            selected_episodes = available_episodes
        if not selected_episodes:
            raise ValueError("Episode filtering removed every UR5 EEF episode")
        self._rows = [
            row for row in self._rows if int(row["episode_index"]) in selected_episodes
        ]
        self.selected_episode_ids = tuple(sorted(selected_episodes))
        self._media_root = Path(media_root) if media_root else self._root

        self._rows_by_episode: dict[int, list[dict[str, Any]]] = {}
        for row in self._rows:
            self._rows_by_episode.setdefault(int(row["episode_index"]), []).append(row)
        self._timestamps_by_episode = {
            episode_id: np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
            for episode_id, rows in self._rows_by_episode.items()
        }
        self._index: list[tuple[int, int]] = self._build_index()

        # Fail before worker processes are launched if the stats artifact is
        # missing or incompatible with this 10D action representation.
        if self.action_normalization is not None:
            self._load_norm_stats()

        if check_media:
            self._check_media_files()

    @property
    def camera_layout(self) -> CameraLayout:
        return self._camera_layout

    @property
    def action_dim(self) -> int:
        return 10

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(Pos(), Rot("rot6d"), Gripper())

    @classmethod
    def _stats_path(cls) -> Path:
        return Path(__file__).parent / "stats/ur5_eef_lerobot_stats.json"

    @property
    def action_stats_path(self) -> Path | None:
        return self._action_stats_path

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is not None:
            return self._norm_stats
        if self._action_stats_path is None:
            raise ValueError("action_stats_path is required when action_normalization is enabled")

        raw_stats = load_action_stats(str(self._action_stats_path))
        required = {
            "quantile": ("q01", "q99"),
            "meanstd": ("mean", "std"),
            "minmax": ("min", "max"),
        }.get(self.action_normalization)
        if required is None:
            raise ValueError(f"Unknown normalization method: {self.action_normalization!r}")

        stats: dict[str, torch.Tensor] = {}
        for key in required:
            if key not in raw_stats:
                raise ValueError(f"Action stats {self._action_stats_path} is missing {key!r}")
            value = torch.from_numpy(raw_stats[key]).float()
            if value.shape != (self.action_dim,):
                raise ValueError(
                    f"Action stats {key!r} must have shape ({self.action_dim},), got {tuple(value.shape)}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"Action stats {key!r} contains non-finite values")
            stats[key] = value

        lo_key, hi_key = required
        if self.action_normalization in {"quantile", "minmax"}:
            if not torch.all(stats[hi_key] > stats[lo_key]):
                raise ValueError(f"Action stats require {hi_key} > {lo_key} in every channel")
        elif not torch.all(stats["std"] > 0):
            raise ValueError("Action stats require std > 0 in every channel")

        self._norm_stats = stats
        return self._norm_stats

    def _build_index(self) -> list[tuple[int, int]]:
        index: list[tuple[int, int]] = []
        horizon_s = self._chunk_length / self._fps
        for episode_id, rows in self._rows_by_episode.items():
            timestamps = self._timestamps_by_episode[episode_id]
            last_ts = float(timestamps[-1])
            for local_idx in range(0, len(rows), self._sample_stride):
                if float(timestamps[local_idx]) + horizon_s <= last_ts + self._tolerance_s:
                    index.append((episode_id, local_idx))
        return index

    def _check_media_files(self) -> None:
        if not self._episodes:
            raise ValueError(f"No episodes found under {self._root / 'meta' / 'episodes'}")
        first_episode = self._episodes[min(self._episodes)]
        missing = [
            str(self._video_path(first_episode, video_key))
            for video_key in self._camera_layout.ordered_keys
            if not self._video_path(first_episode, video_key).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "UR5 EEF policy training requires LeRobot mp4 media for all three cameras. "
                f"Checked MEDIA_ROOT={self._media_root}. Missing examples: {missing}"
            )

    def _video_path(self, episode: dict[str, Any], video_key: str) -> Path:
        rel = super()._video_path(episode, video_key).relative_to(self._root)
        return self._media_root / rel

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        episode_id, local_idx = self._index[int(idx)]
        observation_rows = self._select_observation_rows(episode_id, local_idx)
        episode = self._episodes[int(observation_rows[0]["episode_index"])]

        video = self._apply_view_layout(self._load_concat_video(episode, observation_rows))
        raw_action, initial_pose = self._build_relative_eef_action(observation_rows)
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice([part.strip() for part in task.split(" | ") if part.strip()] or [task])

        return self._build_result(
            mode=mode,
            video=video,
            action=raw_action,
            ai_caption=ai_caption,
            initial_pose=initial_pose,
            action_spec_names=self.action_names,
            additional_view_description=self._view_description(),
        )

    def _select_observation_rows(self, episode_id: int, local_idx: int) -> list[dict[str, Any]]:
        rows = self._rows_by_episode[episode_id]
        timestamps = self._timestamps_by_episode[episode_id]
        start_ts = float(timestamps[local_idx])
        target_ts = start_ts + np.arange(self._chunk_length + 1, dtype=np.float64) / self._fps
        indices = np.searchsorted(timestamps, target_ts, side="left")
        indices = np.minimum(indices, len(rows) - 1)
        prev = np.maximum(indices - 1, 0)
        choose_prev = np.abs(timestamps[prev] - target_ts) <= np.abs(timestamps[indices] - target_ts)
        indices = np.where(choose_prev, prev, indices)
        return [rows[int(i)] for i in indices]

    def _load_concat_video(self, episode: dict[str, Any], observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        """Decode cameras and stack them top→bottom in CameraLayout order."""
        timestamps = [float(row["timestamp"]) for row in observation_rows]
        frames: list[torch.Tensor] = []
        for video_key in self._camera_layout.ordered_keys:
            frames.append(
                decode_video_frames(
                    self._video_path(episode, video_key),
                    [
                        float(episode.get(f"videos/{video_key}/from_timestamp", 0.0)) + ts
                        for ts in timestamps
                    ],
                    self._tolerance_s,
                )
            )
        return torch.cat(frames, dim=-2)

    def _apply_view_layout(self, vertical_video: torch.Tensor) -> torch.Tensor:
        """Optionally retile the vertical stack into DROID geometry.

        Vertical thirds correspond 1:1 to CameraLayout order:
          third 0 → full-width top
          third 1 → bottom-left
          third 2 → bottom-right
        """
        if self._view_layout == "vertical":
            return vertical_video

        _, _, h, w = vertical_video.shape
        if h % 3 != 0:
            raise ValueError(f"Expected vertical 3-view video height divisible by 3, got {h}")
        view_h = h // 3
        top = vertical_video[:, :, :view_h, :]
        bottom_left = vertical_video[:, :, view_h : 2 * view_h, :]
        bottom_right = vertical_video[:, :, 2 * view_h :, :]
        half_h, half_w = view_h // 2, w // 2
        left_small = torch.nn.functional.interpolate(
            bottom_left, size=(half_h, half_w), mode="bilinear", align_corners=False
        )
        right_small = torch.nn.functional.interpolate(
            bottom_right, size=(half_h, half_w), mode="bilinear", align_corners=False
        )
        return torch.cat([top, torch.cat([left_small, right_small], dim=-1)], dim=-2)

    def _view_description(self) -> str:
        return self._camera_layout.description()

    def _build_relative_eef_action(self, observation_rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = np.asarray(observation_rows[0][_STATE_FEATURE], dtype=np.float32)
        future_actions = np.asarray([row[_ACTION_FEATURE] for row in observation_rows[1:]], dtype=np.float32)
        absolute_waypoints = np.concatenate([anchor[None, :], future_actions], axis=0)

        poses_abs = build_abs_pose_from_components(
            absolute_waypoints[:, 0:3],
            absolute_waypoints[:, 3:6],
            "axisangle",
        )
        initial_pose = torch.from_numpy(poses_abs[0].copy()).float()
        poses_rel = pose_abs_to_rel(
            poses_abs,
            rotation_format="rot6d",
            pose_convention=self._pose_convention,
        )
        gripper = future_actions[:, 6:7].astype(np.float32, copy=False)
        action = np.concatenate([poses_rel, gripper], axis=-1)
        return torch.from_numpy(action).float(), initial_pose

    def _build_result(
        self,
        *,
        mode: str,
        video: torch.Tensor,
        action: torch.Tensor,
        ai_caption: str,
        **extras: Any,
    ) -> dict[str, Any]:
        idle_frames = self._compute_idle_frames(action)
        if self.action_normalization is not None:
            action = normalize_action(action, self.action_normalization, self._load_norm_stats())
        formatted_video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)
        return {
            "ai_caption": ai_caption,
            "video": formatted_video,
            "action": action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
            **extras,
        }
