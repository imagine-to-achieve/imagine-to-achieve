# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""UR5 joint-position LeRobot dataset for Cosmos Action SFT.

The source dataset stores absolute UR5 joint waypoints as
``[joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, gripper_qpos]``.
This wrapper trains raw absolute joint-position actions, matching the DROID
``joint_pos`` convention more closely than the EEF relative-pose wrapper.

When ``use_state=True`` the first row is the current observed joint state and
is prepended to the action sequence, following the DROID joint_pos recipe.
The remaining rows are the next ``chunk_length`` commanded joint targets.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.data.vfm.action.action_normalization import normalize_action
from cosmos_framework.data.vfm.action.action_spec import ActionSpec, Gripper, Joint, build_action_spec
from cosmos_framework.data.vfm.action.datasets.ur5_eef_lerobot_dataset import UR5EEFLeRobotDataset
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id

_STATE_FEATURE = "observation.state"
_ACTION_FEATURE = "action"


class UR5JointLeRobotDataset(UR5EEFLeRobotDataset):
    """UR5 close-desktop LeRobot dataset with absolute joint-position actions.

    Actions returned by this dataset are 7D raw joint commands:
    ``[joint(6), gripper(1)]``. With ``use_state=True``, the first action row is
    the current observed state, so the returned action length is
    ``chunk_length + 1``; otherwise it is ``chunk_length``.
    """

    def __init__(
        self,
        root: str,
        media_root: str | None = None,
        fps: float = 15.0,
        chunk_length: int = 32,
        mode: str = "joint",
        pose_convention: str = "backward_framewise",
        tolerance_s: float = 2e-4,
        viewpoint: str = "concat_view",
        action_normalization: str | None = None,
        sample_stride: int = 1,
        check_media: bool = True,
        use_state: bool = True,
        view_layout: str = "vertical",
    ) -> None:
        self._use_state = bool(use_state)
        if view_layout not in ("vertical", "droid"):
            raise ValueError(f"view_layout must be 'vertical' or 'droid', got {view_layout!r}")
        self._view_layout = view_layout
        super().__init__(
            root=root,
            media_root=media_root,
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,  # accepted by the shared base class
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
            check_media=check_media,
            view_layout=view_layout,
        )
        self._domain_name = "robomind-ur-joint"
        self._domain_id = get_domain_id(self._domain_name)

    @property
    def action_dim(self) -> int:
        return 7

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(Joint(n=6, label="joint"), Gripper())

    @classmethod
    def _stats_path(cls) -> Path:
        return Path(__file__).parent / "stats/ur5_joint_lerobot_stats.json"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        episode_id, local_idx = self._index[int(idx)]
        observation_rows = self._select_observation_rows(episode_id, local_idx)
        episode = self._episodes[int(observation_rows[0]["episode_index"])]

        video = self._load_joint_video(episode, observation_rows)
        raw_action = self._build_joint_action(observation_rows)
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice([part.strip() for part in task.split(" | ") if part.strip()] or [task])

        return self._build_result(
            mode=mode,
            video=video,
            action=raw_action,
            ai_caption=ai_caption,
            action_spec_names=self.action_names,
            additional_view_description=self._view_description(),
        )

    def _load_joint_video(self, episode: dict[str, Any], observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        return self._apply_view_layout(self._load_concat_video(episode, observation_rows))

    def _view_description(self) -> str:
        return super()._view_description()

    def _build_joint_action(self, observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        future_actions = np.asarray([row[_ACTION_FEATURE] for row in observation_rows[1:]], dtype=np.float32)
        if future_actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected 7D UR5 joint action, got shape {future_actions.shape}")

        action = future_actions
        if self._use_state:
            initial_state = np.asarray(observation_rows[0][_STATE_FEATURE], dtype=np.float32)
            if initial_state.shape[-1] != self.action_dim:
                raise ValueError(f"Expected 7D UR5 joint state, got shape {initial_state.shape}")
            action = np.concatenate([initial_state[None, :], action], axis=0)
        return torch.from_numpy(action).float()

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
