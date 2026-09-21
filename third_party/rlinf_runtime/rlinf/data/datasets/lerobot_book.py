# Copyright 2026 The RLinf Authors.
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

"""Helpers for local LeRobot-format video datasets."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from torch.utils.data import Dataset

BOOK_MAIN_IMAGE_KEY = "observation.images.d405_rgb"
BOOK_EXTRA_VIEW_IMAGE_KEY = "observation.images.d405_1_rgb"
BOOK_WRIST_IMAGE_KEY = "observation.images.d435_rgb"
BOOK_IMAGE_KEYS = (
    BOOK_MAIN_IMAGE_KEY,
    BOOK_EXTRA_VIEW_IMAGE_KEY,
    BOOK_WRIST_IMAGE_KEY,
)
VIDEO_TOLERANCE_S = 1e-4


def resolve_lerobot_dataset_root(repo_id: str) -> Path:
    """Resolve a local LeRobot dataset root from HF_LEROBOT_HOME and repo_id."""
    env_root = os.environ.get("HF_LEROBOT_HOME")
    candidates: list[Path] = []
    if env_root:
        base = Path(env_root).expanduser().resolve()
        candidates.append(base)
        candidates.append(base / repo_id)
    candidates.append(Path(repo_id).expanduser())

    for candidate in candidates:
        if (candidate / "meta").is_dir() and (candidate / "data").is_dir():
            return candidate.resolve()

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not resolve local LeRobot dataset for repo_id={repo_id!r}. Tried: {tried}"
    )


def load_lerobot_task_map(root: Path) -> dict[int, str]:
    tasks_table = pq.read_table(root / "meta" / "tasks.parquet")
    tasks: dict[int, str] = {}
    for row in tasks_table.to_pylist():
        task = row.get("task", row.get("__index_level_0__"))
        if task is None:
            raise KeyError(
                "Expected 'task' or '__index_level_0__' in tasks.parquet rows."
            )
        tasks[int(row["task_index"])] = str(task)
    return tasks


def load_book_image_frame(
    root: Path, episode_index: int, image_key: str, frame_index: int
) -> np.ndarray:
    image_path = (
        root
        / "images"
        / image_key
        / f"episode-{episode_index:06d}"
        / f"frame-{frame_index:06d}.png"
    )
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing frame for {image_key}: {image_path}")
    with Image.open(image_path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


def load_lerobot_video_frame(
    root: Path,
    episode: dict[str, Any],
    image_key: str,
    timestamp: float,
    video_path_template: str,
    tolerance_s: float = VIDEO_TOLERANCE_S,
    video_backend: str = "pyav",
) -> np.ndarray:
    chunk_idx_key = f"videos/{image_key}/chunk_index"
    file_idx_key = f"videos/{image_key}/file_index"
    from_ts_key = f"videos/{image_key}/from_timestamp"
    if chunk_idx_key not in episode or file_idx_key not in episode:
        raise FileNotFoundError(
            f"No image files or video metadata for {image_key} in dataset {root}"
        )

    video_path = root / video_path_template.format(
        video_key=image_key,
        chunk_index=int(episode[chunk_idx_key]),
        file_index=int(episode[file_idx_key]),
    )
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing video for {image_key}: {video_path}")

    # Same timestamp convention as CN/lerobot's DatasetReader._query_videos:
    # query the frame at episode video start + current row timestamp.
    query_timestamp = float(episode.get(from_ts_key, 0.0)) + timestamp
    frame = decode_video_frames(
        video_path,
        [query_timestamp],
        tolerance_s=tolerance_s,
        backend=video_backend,
        return_uint8=True,
    ).squeeze(0)
    return frame.numpy()


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    return_uint8: bool = False,
) -> torch.Tensor:
    """Decode video frames following CN/lerobot's video_utils.decode_video_frames."""
    video_path = str(video_path)
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True

    reader = torchvision.io.VideoReader(video_path, "video")
    first_ts = min(timestamps)
    last_ts = max(timestamps)
    reader.seek(first_ts, keyframes_only=keyframes_only)

    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise RuntimeError(
            f"One or several query timestamps unexpectedly violate the tolerance "
            f"({min_[~is_within_tol]} > tolerance_s={tolerance_s}). "
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
            f"\nbackend: {backend}"
        )

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    if len(timestamps) != len(closest_frames):
        raise RuntimeError(
            f"Number of retrieved frames ({len(closest_frames)}) does not match "
            f"number of queried timestamps ({len(timestamps)})"
        )

    if return_uint8:
        return closest_frames
    return closest_frames.type(torch.float32) / 255


def load_lerobot_image_or_video_frame(
    root: Path,
    episode: dict[str, Any],
    image_key: str,
    frame_index: int,
    timestamp: float,
    video_path_template: str,
) -> np.ndarray:
    episode_index = int(episode["episode_index"])
    image_path = (
        root
        / "images"
        / image_key
        / f"episode-{episode_index:06d}"
        / f"frame-{frame_index:06d}.png"
    )
    if image_path.is_file():
        with Image.open(image_path) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return load_lerobot_video_frame(
        root,
        episode,
        image_key,
        timestamp,
        video_path_template,
    )


@dataclass(frozen=True)
class LeRobotBookDatasetMetadata:
    repo_id: str

    def __post_init__(self) -> None:
        root = resolve_lerobot_dataset_root(self.repo_id)
        info = json.loads((root / "meta" / "info.json").read_text())

        object.__setattr__(self, "root", root)
        object.__setattr__(self, "info", info)
        object.__setattr__(self, "fps", int(info["fps"]))
        object.__setattr__(self, "tasks", load_lerobot_task_map(root))


class LeRobotBookDataset:
    """Random-access dataset for OpenPI SFT on local LeRobot datasets."""

    def __init__(
        self,
        repo_id: str,
        *,
        delta_timestamps: dict[str, list[float]] | None = None,
        frame_stride: int = 1,
        camera_keys: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.root = resolve_lerobot_dataset_root(repo_id)
        self.meta = LeRobotBookDatasetMetadata(repo_id)
        self.delta_timestamps = delta_timestamps or {}
        self.frame_stride = max(1, int(frame_stride))
        self.info = self.meta.info

        available_image_keys = [
            key
            for key, feature in self.info["features"].items()
            if key.startswith("observation.images.")
            and feature.get("dtype") in {"image", "video"}
        ]
        if camera_keys is None:
            self.image_keys = list(available_image_keys)
        else:
            self.image_keys = list(camera_keys)
            missing_keys = [
                image_key
                for image_key in self.image_keys
                if image_key not in available_image_keys
            ]
            if missing_keys:
                raise ValueError(
                    f"Requested camera_keys {missing_keys} are not available in dataset {self.root}"
                )

        action_key = next(iter(self.delta_timestamps), "action")
        action_offsets = self.delta_timestamps.get(action_key, [0.0])
        self.action_horizon = max(1, len(action_offsets))

        episodes_path = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episodes_table = pq.read_table(episodes_path)
        self.episodes = sorted(
            episodes_table.to_pylist(), key=lambda row: int(row["episode_index"])
        )

        self._episode_rows: dict[int, list[dict[str, Any]]] = {}
        self._samples: list[tuple[int, int]] = []
        for episode in self.episodes:
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            required_frames = 1 + (self.action_horizon - 1) * self.frame_stride
            valid_len = max(0, length - required_frames + 1)
            for frame_idx in range(valid_len):
                self._samples.append((episode_index, frame_idx))

    def __len__(self) -> int:
        return len(self._samples)

    def _get_data_file_for_episode(self, episode_index: int) -> Path:
        episode = self.episodes[episode_index]
        chunk_index = int(episode["data/chunk_index"])
        file_index = int(episode["data/file_index"])
        return self.root / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

    def _load_episode_rows(self, episode_index: int) -> list[dict[str, Any]]:
        if episode_index in self._episode_rows:
            return self._episode_rows[episode_index]

        data_file = self._get_data_file_for_episode(episode_index)
        table = pq.read_table(
            data_file,
            columns=[
                "observation.state",
                "action",
                "task_index",
                "episode_index",
                "frame_index",
                "timestamp",
            ],
            filters=[("episode_index", "=", episode_index)],
        )
        rows = sorted(table.to_pylist(), key=lambda row: int(row["frame_index"]))
        self._episode_rows[episode_index] = rows
        return rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, frame_index = self._samples[index]
        rows = self._load_episode_rows(episode_index)

        row = rows[frame_index]
        action_indices = range(
            frame_index,
            frame_index + self.action_horizon * self.frame_stride,
            self.frame_stride,
        )
        action_rows = [rows[action_idx] for action_idx in action_indices]
        row_frame_index = int(row["frame_index"])

        sample = {
            "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
            "action": np.asarray(
                [action_row["action"] for action_row in action_rows], dtype=np.float32
            ),
            "task_index": np.asarray(row["task_index"], dtype=np.int64),
        }
        episode = self.episodes[episode_index]
        for image_key in self.image_keys:
            sample[image_key] = load_lerobot_image_or_video_frame(
                self.root,
                episode,
                image_key,
                row_frame_index,
                float(row["timestamp"]),
                self.info.get(
                    "video_path",
                    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                ),
            )
        return sample


class LeRobotBookTrajectoryDatasetWrapper(Dataset):
    """Read reset states from the local book dataset for world-model rollout."""

    def __init__(
        self,
        data_dir: str,
        camera_heights: Optional[int] = None,
        camera_widths: Optional[int] = None,
        camera_keys: list[str] | tuple[str, ...] | None = None,
    ):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.meta_dir = self.data_dir / "meta"
        self.data_files_dir = self.data_dir / "data"
        if not self.meta_dir.is_dir() or not self.data_files_dir.is_dir():
            raise ValueError(
                f"LeRobot dataset root must contain meta/ and data/: {self.data_dir}"
            )

        episodes_meta_path = self.meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
        if not episodes_meta_path.exists():
            raise ValueError(f"Missing LeRobot episode metadata: {episodes_meta_path}")

        self.image_transforms = None
        if camera_heights is not None and camera_widths is not None:
            self.image_transforms = transforms.Compose(
                [transforms.Resize((camera_heights, camera_widths))]
            )

        episodes_table = pq.read_table(episodes_meta_path)
        self.episodes = sorted(
            episodes_table.to_pylist(), key=lambda row: int(row["episode_index"])
        )
        self.info = json.loads((self.meta_dir / "info.json").read_text())
        self.video_path_template = self.info.get(
            "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        self.task_map = load_lerobot_task_map(self.data_dir)
        self._episode_cache: dict[int, dict[str, Any]] = {}
        self._episode_rows: dict[int, list[dict[str, Any]]] = {}
        self.view_image_keys = self._resolve_view_image_keys(camera_keys)

    def _resolve_view_image_keys(
        self, camera_keys: list[str] | tuple[str, ...] | None
    ) -> tuple[str, str, str]:
        if camera_keys is None:
            return BOOK_IMAGE_KEYS

        keys = tuple(str(key) for key in camera_keys)
        if len(keys) == 2:
            return (keys[0], keys[0], keys[1])
        if len(keys) == 3:
            return keys
        raise ValueError(
            f"Expected 2 or 3 camera_keys for world-model reset data, got {keys}"
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def _get_data_file_for_episode(self, episode: dict[str, Any]) -> Path:
        chunk_index = int(episode["data/chunk_index"])
        file_index = int(episode["data/file_index"])
        return self.data_dir / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

    def _get_first_frame(self, episode: dict[str, Any]) -> dict[str, Any]:
        episode_index = int(episode["episode_index"])
        if episode_index in self._episode_cache:
            return self._episode_cache[episode_index]

        rows = self._load_episode_rows(episode)
        if len(rows) == 0:
            raise ValueError(f"Empty episode rows for episode {episode_index}")
        first_frame = rows[0]
        if int(first_frame["frame_index"]) != 0:
            raise ValueError(
                f"Expected first frame_index=0 for episode {episode_index}, "
                f"got {first_frame['frame_index']}"
            )
        self._episode_cache[episode_index] = first_frame
        return first_frame

    def _load_episode_rows(self, episode: dict[str, Any]) -> list[dict[str, Any]]:
        episode_index = int(episode["episode_index"])
        if episode_index in self._episode_rows:
            return self._episode_rows[episode_index]

        data_file = self._get_data_file_for_episode(episode)
        episode_table = pq.read_table(
            data_file,
            columns=[
                "observation.state",
                "action",
                "episode_index",
                "frame_index",
                "timestamp",
                "task_index",
            ],
            filters=[
                ("episode_index", "=", episode_index),
            ],
        )
        rows = sorted(episode_table.to_pylist(), key=lambda row: int(row["frame_index"]))
        self._episode_rows[episode_index] = rows
        return rows

    def _load_tensor_image(
        self,
        episode: dict[str, Any],
        episode_index: int,
        image_key: str,
        frame_index: int,
        timestamp: float,
    ) -> torch.Tensor:
        image_np = load_lerobot_image_or_video_frame(
            self.data_dir,
            episode,
            image_key,
            frame_index,
            timestamp,
            self.video_path_template,
        )
        if image_np.ndim == 3 and image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
        image_tensor = transforms.ToTensor()(image_np)
        if self.image_transforms is not None:
            image_tensor = self.image_transforms(image_tensor)
        return image_tensor

    def load_frame_views(
        self,
        episode_index: int = -1,
        frame_index: int = -1,
    ) -> dict[str, Any]:
        """Load an explicit multi-camera frame with Python-style negatives."""
        if not self.episodes:
            raise ValueError(f"No episodes found in LeRobot dataset {self.data_dir}.")
        if episode_index < 0:
            try:
                episode = self.episodes[episode_index]
            except IndexError as exc:
                raise IndexError(
                    f"Episode position {episode_index} is out of range."
                ) from exc
        else:
            episode = next(
                (
                    candidate
                    for candidate in self.episodes
                    if int(candidate["episode_index"]) == episode_index
                ),
                None,
            )
            if episode is None:
                raise KeyError(f"Episode {episode_index} is not present.")
        selected_episode_index = int(episode["episode_index"])
        rows = self._load_episode_rows(episode)
        if not rows:
            raise ValueError(f"Episode {selected_episode_index} has no rows.")
        if frame_index < 0:
            try:
                row = rows[frame_index]
            except IndexError as exc:
                raise IndexError(
                    f"Frame position {frame_index} is out of range."
                ) from exc
        else:
            row = next(
                (
                    candidate
                    for candidate in rows
                    if int(candidate["frame_index"]) == frame_index
                ),
                None,
            )
            if row is None:
                raise KeyError(
                    f"Frame {frame_index} is not present in episode "
                    f"{selected_episode_index}."
                )
        selected_frame_index = int(row["frame_index"])
        timestamp = float(row.get("timestamp", 0.0))
        images = {
            image_key: self._load_tensor_image(
                episode,
                selected_episode_index,
                image_key,
                selected_frame_index,
                timestamp,
            )
            for image_key in self.view_image_keys
        }
        return {
            "images": images,
            "camera_keys": self.view_image_keys,
            "episode_index": selected_episode_index,
            "frame_index": selected_frame_index,
            "timestamp": timestamp,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self.episodes[index]
        episode_index = int(episode["episode_index"])
        first_frame = self._get_first_frame(episode)
        episode_rows = self._load_episode_rows(episode)
        frame_index = int(first_frame["frame_index"])
        timestamp = float(first_frame.get("timestamp", 0.0))

        main_key, extra_view_key, wrist_key = self.view_image_keys
        main_image = self._load_tensor_image(
            episode, episode_index, main_key, frame_index, timestamp
        )
        extra_view_image = self._load_tensor_image(
            episode, episode_index, extra_view_key, frame_index, timestamp
        )
        wrist_image = self._load_tensor_image(
            episode, episode_index, wrist_key, frame_index, timestamp
        )
        state = torch.tensor(first_frame["observation.state"], dtype=torch.float32)

        tasks = episode.get("tasks", [])
        if tasks:
            task = str(tasks[0])
        else:
            task = self.task_map.get(int(first_frame["task_index"]), "")

        start_frame = {
            "image": main_image,
            "main_image_1": main_image,
            "main_image_2": extra_view_image,
            "wrist_image": wrist_image,
            "observation.state": state,
        }
        actions = torch.tensor(
            np.asarray([row["action"] for row in episode_rows], dtype=np.float32),
            dtype=torch.float32,
        )

        return {
            "start_items": [start_frame],
            "target_items": [],
            "actions": actions,
            "episode_index": episode_index,
            "task": task,
            "dataset_meta": {
                "episode_length": int(episode.get("length", 1)),
                "file_path": os.fspath(self.data_dir),
                "task_index": int(first_frame["task_index"]),
            },
        }
