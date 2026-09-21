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

"""Dataset wrappers for using local LeRobot-format data as world-model reset states."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class LeRobotTrajectoryDatasetWrapper(Dataset):
    """Read reset states from a local LeRobot-format dataset.

    The wrapper keeps the same high-level output contract as
    ``NpyTrajectoryDatasetWrapper``:

    - ``start_items``: list[dict]
    - ``target_items``: list[dict]
    - ``task``: str
    - ``dataset_meta``: dict

    For Ctrl-World, each start frame includes:
    - ``image``: the default main image
    - ``main_image_1``: first main-view slot for Ctrl-World
    - ``main_image_2``: second main-view slot for Ctrl-World
    - ``wrist_image``: wrist-view image
    - ``observation.state``: proprio/state tensor

    Current LeRobot LIBERO dataset only exposes one third-person image and one wrist
    image, so ``main_image_1`` and ``main_image_2`` are both mapped from the same
    main-view frame.
    """

    def __init__(
        self,
        data_dir: str,
        camera_heights: Optional[int] = None,
        camera_widths: Optional[int] = None,
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
                [
                    transforms.Resize((camera_heights, camera_widths)),
                ]
            )

        episodes_table = pq.read_table(episodes_meta_path)
        self.episodes = sorted(episodes_table.to_pylist(), key=lambda row: row["episode_index"])
        self._episode_cache: dict[int, dict[str, Any]] = {}

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

        data_file = self._get_data_file_for_episode(episode)
        first_frame_table = pq.read_table(
            data_file,
            columns=[
                "observation.images.image",
                "observation.images.wrist_image",
                "observation.state",
                "episode_index",
                "frame_index",
                "task_index",
            ],
            filters=[
                ("episode_index", "=", episode_index),
                ("frame_index", "=", 0),
            ],
        )
        rows = first_frame_table.to_pylist()
        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one first frame for episode {episode_index}, got {len(rows)}"
            )
        self._episode_cache[episode_index] = rows[0]
        return rows[0]

    def _decode_image(self, image_struct: dict[str, Any]) -> torch.Tensor:
        image_bytes = image_struct.get("bytes")
        image_path = image_struct.get("path")

        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif image_path:
            image = Image.open(self.data_dir / image_path).convert("RGB")
        else:
            raise ValueError("LeRobot image struct contains neither bytes nor path.")

        image_tensor = transforms.ToTensor()(image)
        if self.image_transforms is not None:
            image_tensor = self.image_transforms(image_tensor)
        return image_tensor

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self.episodes[index]
        episode_index = int(episode["episode_index"])
        first_frame = self._get_first_frame(episode)

        main_image = self._decode_image(first_frame["observation.images.image"])
        wrist_image = self._decode_image(first_frame["observation.images.wrist_image"])
        state = torch.tensor(first_frame["observation.state"], dtype=torch.float32)

        tasks = episode.get("tasks", [])
        task = str(tasks[0]) if tasks else ""

        start_frame = {
            "image": main_image,
            "main_image_1": main_image,
            "main_image_2": main_image.clone(),
            "wrist_image": wrist_image,
            "observation.state": state,
        }

        return {
            "start_items": [start_frame],
            "target_items": [],
            "episode_index": episode_index,
            "task": task,
            "dataset_meta": {
                "episode_length": int(episode.get("length", 1)),
                "file_path": os.fspath(self.data_dir),
                "task_index": int(first_frame["task_index"]),
            },
        }
