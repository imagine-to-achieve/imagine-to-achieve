# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Map-style action SFT dataset: ``DROIDLeRobotDataset`` → ``ActionTransformPipeline``.

The base ``DROIDLeRobotDataset.__getitem__`` returns the raw sample
(``video``/``action``/``ai_caption``/``viewpoint``/``mode``/``domain_id``/
``idle_frames``). The model expects each sample to be passed through
``ActionTransformPipeline`` (spatial resize/pad, text tokenization, action
padding to ``max_action_dim``, and ``sequence_plan`` construction). This thin
wrapper composes the two so the experiment can hand a single map-style dataset
to ``RankPartitionedDataLoader`` (mirroring how the vision recipe uses
``get_sft_dataset``).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.camera_layout import CameraLayout
from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset
from cosmos_framework.data.vfm.action.datasets.ur5_eef_lerobot_dataset import UR5EEFLeRobotDataset
from cosmos_framework.data.vfm.action.datasets.ur5_joint_lerobot_dataset import UR5JointLeRobotDataset
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline


class ActionSFTDataset(Dataset):
    """Wraps a map-style action dataset and applies ``ActionTransformPipeline`` per sample."""

    def __init__(self, dataset: Dataset, transform: ActionTransformPipeline, resolution: str | int | None):
        super().__init__()
        self._dataset = dataset
        self._transform = transform
        self._resolution = resolution

    def _shard_info(self) -> tuple[int, int]:
        """Return the rank-local shard assigned by RankPartitionedDataLoader."""
        world_size = int(getattr(self, "shard_world_size", 1))
        rank = int(getattr(self, "shard_rank", 0))
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"Invalid action dataset shard: rank={rank}, world_size={world_size}")
        return world_size, rank

    def __len__(self) -> int:
        world_size, rank = self._shard_info()
        total = len(self._dataset)
        if rank >= total:
            return 0
        return (total - rank + world_size - 1) // world_size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        world_size, rank = self._shard_info()
        local_idx = int(idx)
        if local_idx < 0:
            local_idx += len(self)
        if not 0 <= local_idx < len(self):
            raise IndexError(local_idx)
        global_idx = rank + local_idx * world_size
        return self._transform(self._dataset[global_idx], self._resolution)


def get_action_droid_sft_dataset(
    *,
    root: str,
    fps: float = 15.0,
    chunk_length: int = 32,
    action_space: str = "joint_pos",
    use_state: bool = True,
    action_normalization: str | None = None,
    viewpoint: str = "concat_view",
    use_image_augmentation: bool = False,
    use_filter_dict: bool = False,
    filter_dict_path: str | None = None,
    resolution: str | int = "256",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
) -> ActionSFTDataset:
    """Build the DROID action SFT dataset (joint_pos 8D by default), matching the
    internal ``droid_lerobot_8b_policy`` data: ``action_space='joint_pos'`` +
    ``use_state`` (8D, raw/un-normalized), concat_view, chunk_length 32."""
    dataset = DROIDLeRobotDataset(
        root=root,
        fps=fps,
        chunk_length=chunk_length,
        viewpoint=viewpoint,
        action_space=action_space,
        use_state=use_state,
        action_normalization=action_normalization,
        use_image_augmentation=use_image_augmentation,
        use_filter_dict=use_filter_dict,
        filter_dict_path=filter_dict_path,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
    )
    return ActionSFTDataset(dataset, transform, resolution)


def get_action_ur5_eef_sft_dataset(
    *,
    root: str,
    media_root: str | None = None,
    fps: float = 15.0,
    chunk_length: int = 32,
    mode: str = "joint",
    action_normalization: str | None = None,
    action_stats_path: str | None = None,
    viewpoint: str = "concat_view",
    resolution: str | int = "480",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    check_media: bool = True,
    view_layout: str = "droid",
    camera_layout: Sequence[Mapping[str, str]] | CameraLayout | None = None,
    include_episodes: list[int] | None = None,
    exclude_episodes: list[int] | None = None,
) -> ActionSFTDataset:
    """Build the UR5 absolute-EEF action SFT dataset.

    The source LeRobot data stores absolute EEF waypoints; the dataset converts
    them to 10D relative pose actions before ``ActionTransformPipeline`` pads
    them to the model action width.

    ``camera_layout`` is an ordered list of ``{region, key, label}`` entries.
    Its order is the physical concat / DROID region order and drives prompt text.
    When null, ``CAMERA_LAYOUT_JSON`` may supply the same structure.
    """
    import json
    import os

    if camera_layout is None:
        env_layout = os.environ.get("CAMERA_LAYOUT_JSON")
        if env_layout:
            camera_layout = json.loads(env_layout)

    dataset = UR5EEFLeRobotDataset(
        root=root,
        media_root=media_root,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        viewpoint=viewpoint,
        action_normalization=action_normalization,
        action_stats_path=action_stats_path,
        check_media=check_media,
        view_layout=view_layout,
        camera_layout=camera_layout,
        include_episodes=include_episodes,
        exclude_episodes=exclude_episodes,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
    )
    return ActionSFTDataset(dataset, transform, resolution)


def get_action_ur5_joint_sft_dataset(
    *,
    root: str,
    media_root: str | None = None,
    fps: float = 15.0,
    chunk_length: int = 32,
    mode: str = "joint",
    action_normalization: str | None = None,
    viewpoint: str = "concat_view",
    resolution: str | int = "480",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    check_media: bool = True,
    use_state: bool = True,
    view_layout: str = "vertical",
) -> ActionSFTDataset:
    """Build the UR5 absolute joint-position action SFT dataset.

    The source LeRobot data stores absolute joint waypoints as 7D
    ``[joint_0..joint_5, gripper_qpos]``. This keeps the raw joint-position
    convention and optionally prepends the current observed joint state, matching
    the DROID ``joint_pos`` policy recipe more closely than the EEF 10D wrapper.
    """
    dataset = UR5JointLeRobotDataset(
        root=root,
        media_root=media_root,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        viewpoint=viewpoint,
        action_normalization=action_normalization,
        check_media=check_media,
        use_state=use_state,
        view_layout=view_layout,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
    )
    return ActionSFTDataset(dataset, transform, resolution)
