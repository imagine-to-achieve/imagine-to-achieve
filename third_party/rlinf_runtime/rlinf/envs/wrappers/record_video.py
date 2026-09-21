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

import json
import numbers
import os
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

import gymnasium as gym
import imageio
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

from rlinf.envs.utils import put_info_on_image, tile_images


class RecordVideo(gym.Wrapper):
    """
    A general video recording wrapper that owns the recording logic.

    ``RecordVideo`` centralizes frame collection and MP4 writing for both regular
    stepping and chunked stepping APIs. Frames are buffered in memory and flushed
    asynchronously to avoid blocking environment interaction.

    The wrapper supports multiple observation image layouts (single frame, batched
    frames, and temporal batches). For ``chunk_step()``, it correctly handles the
    terminal-to-reset transition by recording terminal observations (for the last
    step in the chunk) and then appending the corresponding reset observations.

    When ``video_cfg.info_on_video`` is enabled, per-frame text metadata is drawn
    through ``put_info_on_image()``. The overlay always includes reward and
    termination when available, and can include extra fields from environment
    ``info`` via ``video_cfg.extra_info_on_video``. Nested keys are supported with
    dot notation, for example
    ``["env_id", "episode.success_once", "episode.episode_len"]``.

    Args:
        env: Wrapped environment. It must expose a ``seed`` attribute and may
            optionally provide ``num_envs`` and metadata for FPS inference.
        video_cfg: Video configuration object/dict. Common fields:
            ``video_base_dir`` (output directory root),
            ``fps`` (optional FPS override),
            ``info_on_video`` (whether to render overlay text),
            ``extra_info_on_video`` (list of ``info`` keys to render).
        fps: Explicit FPS override. If ``None``, FPS is resolved from
            ``video_cfg.fps``, environment config/metadata, then fallback ``30``.
    """

    def __init__(self, env: gym.Env, video_cfg, fps: Optional[int] = None):
        """Initialize the wrapper and set FPS/config."""
        if isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env

        if not hasattr(env, "seed"):
            raise AttributeError("Environment must have 'seed' attribute")

        self.video_cfg = video_cfg
        self.render_images: list[np.ndarray] = []
        self.per_env_render_images: list[list[np.ndarray]] = [
            [] for _ in range(getattr(env, "num_envs", 1))
        ]
        self.success_images: list[dict[str, Any]] = []
        self.video_cnt = 0
        self.success_image_cnt = 0
        self._num_envs = getattr(env, "num_envs", 1)
        self._success_once_state = np.zeros(self._num_envs, dtype=bool)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._save_futures: list[Future] = []
        self._camera_layout = self._resolve_camera_layout(video_cfg)

        if fps is not None:
            self._fps = fps
        else:
            self._fps = self._get_fps_from_env(env)

    @property
    def is_start(self):
        return getattr(self.env, "is_start")

    @is_start.setter
    def is_start(self, value):
        setattr(self.env, "is_start", value)

    def _get_fps_from_env(self, env: gym.Env) -> int:
        """Resolve FPS from config/env metadata with fallback."""
        if hasattr(self.video_cfg, "fps") and self.video_cfg.fps is not None:
            return int(self.video_cfg.fps)
        if hasattr(env, "cfg") and hasattr(env.cfg, "init_params"):
            if hasattr(env.cfg.init_params, "sim_config"):
                if hasattr(env.cfg.init_params.sim_config, "control_freq"):
                    return int(env.cfg.init_params.sim_config.control_freq)
        metadata = getattr(env, "metadata", None)
        if isinstance(metadata, dict) and "render_fps" in metadata:
            return int(metadata["render_fps"])
        return 30

    @staticmethod
    def _resolve_camera_layout(video_cfg: Any) -> str:
        if hasattr(video_cfg, "get"):
            layout = video_cfg.get("camera_layout", "main_top")
        else:
            layout = getattr(video_cfg, "camera_layout", "main_top")
        layout = str(layout)
        if layout not in {"main_top", "droid", "edge_full_fov"}:
            raise ValueError(
                f"Unsupported video camera layout {layout!r}; "
                "expected 'main_top', 'droid', or 'edge_full_fov'."
            )
        return layout

    def _to_numpy(self, value: Any) -> np.ndarray:
        """Convert tensors/arrays to numpy."""
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.array(value)

    def _get_image_from_dict(self, obs: dict) -> Optional[Any]:
        """Pick the best image field from an observation dict."""
        for key in ("main_images", "images", "rgb", "full_image", "main_image"):
            if key in obs and obs[key] is not None:
                return obs[key]
        return None

    _MULTIVIEW_KEYS = ("main_images", "wrist_images", "extra_view_images")

    @staticmethod
    def _assemble_three_view_grid_tile(
        main_tile: np.ndarray,
        wrist_tile: np.ndarray,
        extra_tile: np.ndarray,
        *,
        layout: str = "main_top",
    ) -> np.ndarray:
        """Arrange one main/wrist/extra tile triple in the checkpoint layout.

        ``main_top`` is the historical 10D arrangement: main on top, wrist
        bottom-left, extra bottom-right.  ``droid`` is the 7D UR5-joint
        arrangement: wrist (D435) on top, front (D405_0) bottom-left, right
        (D405_1) bottom-right. ``edge_full_fov`` (Edge4B's full-FOV geometry:
        main_images=wrist/D435, wrist_images=front/D405_0,
        extra_view_images=right/D405_1) uses this same wrist-top arrangement,
        just at different absolute per-view resolutions, so it is handled
        identically to ``droid`` here. See
        ``_stitch_multiview_camera_grid_for_recording`` for the caller-level
        shape/missing-view handling this helper assumes has already passed.
        """
        if layout in ("droid", "edge_full_fov"):
            main_tile, wrist_tile = wrist_tile, main_tile
        elif layout != "main_top":
            raise ValueError(
                f"Unsupported video camera layout {layout!r}; "
                "expected 'main_top', 'droid', or 'edge_full_fov'."
            )
        view_h, view_w = wrist_tile.shape[0], wrist_tile.shape[1]
        # Ctrl-World's own per-view working resolution (e.g. 192x320,
        # aspect 0.6 H/W) differs from the real camera used at
        # training time (480x640, aspect 0.75 H/W). Applying the
        # main+halved-sides recipe directly to Ctrl-World's own
        # tiles would produce the right *shape* out of the wrong
        # *proportioned* ingredients (overall composite ends up
        # ~0.9 instead of the training target ~1.125) -- correct
        # each tile's aspect ratio first, same fix as
        # cosmos_backend.py's _stitch_three_camera_views_for_policy_input.
        target_ratio = 480 / 640
        if abs(view_h / view_w - target_ratio) > 1e-3:
            new_w = max(1, round(view_h / target_ratio))
            main_tile = np.array(Image.fromarray(main_tile).resize((new_w, view_h)))
            wrist_tile = np.array(Image.fromarray(wrist_tile).resize((new_w, view_h)))
            extra_tile = np.array(Image.fromarray(extra_tile).resize((new_w, view_h)))
            view_w = new_w
        # Uniformly upscale the main tile 2x (both dimensions, so its
        # aspect ratio is preserved) to span the same width as the
        # wrist+extra bottom row; do not stretch width only.
        main_row = np.array(Image.fromarray(main_tile).resize((view_w * 2, view_h * 2)))
        bottom_row = np.concatenate([wrist_tile, extra_tile], axis=1)
        return np.concatenate([main_row, bottom_row], axis=0)

    def _stitch_multiview_camera_grid_for_recording(self, obs: dict) -> Optional[list[list[np.ndarray]]]:
        """If obs carries all 3 camera views, arrange them into a grid per
        env: main on top spanning the full width, wrist bottom-left, extra
        bottom-right -- matching the training-time concat layout (e.g.
        cosmos-framework's UR5EEFLeRobotDataset._load_concat_video). `obs[key]`
        may itself span multiple timesteps (e.g. one `chunk_step()` call
        covers a whole action chunk) — every timestep is kept, not just the
        first, so callers get one grid frame per timestep, not one frame per
        call. Returns None when any view is missing so callers can fall back
        to the single-view path.
        """
        if not all(key in obs and obs[key] is not None for key in self._MULTIVIEW_KEYS):
            return None
        per_view_batches = []
        for key in self._MULTIVIEW_KEYS:
            batches = self._split_image_source(obs[key])
            if not batches:
                return None
            per_view_batches.append(batches)
        num_frames = min(len(batches) for batches in per_view_batches)
        if num_frames == 0:
            return None
        all_frames = []
        for t in range(num_frames):
            num_envs = min(len(per_view_batches[v][t]) for v in range(len(self._MULTIVIEW_KEYS)))
            if num_envs == 0:
                return None
            combined = []
            for env_id in range(num_envs):
                main_tile, wrist_tile, extra_tile = (
                    per_view_batches[v][t][env_id] for v in range(len(self._MULTIVIEW_KEYS))
                )
                shapes = {main_tile.shape[:2], wrist_tile.shape[:2], extra_tile.shape[:2]}
                if len(shapes) > 1:
                    warnings.warn(
                        f"Camera views have mismatched shapes {shapes}; skipping "
                        "multiview grid for this frame."
                    )
                    return None
                combined.append(
                    self._assemble_three_view_grid_tile(
                        main_tile,
                        wrist_tile,
                        extra_tile,
                        layout=self._camera_layout,
                    )
                )
            all_frames.append(combined)
        return all_frames

    def _extract_frame_batches(self, obs: Any) -> list[list[np.ndarray]]:
        """Extract a list of per-step image batches from obs."""
        if obs is None:
            return []

        if isinstance(obs, dict):
            multiview = self._stitch_multiview_camera_grid_for_recording(obs)
            if multiview is not None:
                return multiview
            image_src = self._get_image_from_dict(obs)
            if image_src is None:
                return []
            return self._split_image_source(image_src)

        if isinstance(obs, (list, tuple)):
            if len(obs) == 0:
                return []
            if isinstance(obs[0], dict):
                frames = []
                for item in obs:
                    multiview = self._stitch_multiview_camera_grid_for_recording(item)
                    if multiview is not None:
                        frames.extend(multiview)
                        continue
                    image_src = self._get_image_from_dict(item)
                    if image_src is None:
                        continue
                    batches = self._split_image_source(image_src)
                    if batches:
                        frames.append(batches[0])
                return frames
            images = []
            for item in obs:
                img = self._to_numpy(item)
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                images.append(img)
            return [images] if images else []

        if torch is not None and isinstance(obs, torch.Tensor):
            return self._split_image_source(obs)
        if isinstance(obs, np.ndarray):
            return self._split_image_source(obs)
        return []

    def _split_image_source(self, image_src: Any) -> list[list[np.ndarray]]:
        """Normalize common image tensor layouts into frame batches."""
        img = self._to_numpy(image_src)

        if img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            return [[img]]

        if img.ndim == 4:
            if img.shape[1] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (0, 2, 3, 1))
            images = []
            for i in range(img.shape[0]):
                single = img[i]
                if single.dtype != np.uint8:
                    single = single.astype(np.uint8)
                images.append(single)
            return [images]

        if img.ndim == 5:
            if img.shape[2] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (0, 1, 3, 4, 2))
            frames = []
            for t in range(img.shape[1]):
                images = []
                for i in range(img.shape[0]):
                    single = img[i, t]
                    if single.dtype != np.uint8:
                        single = single.astype(np.uint8)
                    images.append(single)
                frames.append(images)
            return frames

        return []

    def _value_for_env(self, value: Any, env_id: int):
        """Select a scalar/value for a specific env from batched inputs."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            if value.size == 1:
                return value.reshape(-1)[0].item()
            if value.shape[0] > env_id:
                return value[env_id]
            return value.reshape(-1)[0]
        if isinstance(value, (list, tuple)):
            if len(value) > env_id:
                return value[env_id]
            if len(value) > 0:
                return value[0]
        return value

    def _to_bool_array(self, value: Any) -> Optional[np.ndarray]:
        """Convert a scalar/list/tensor metric into a per-env boolean array."""
        if value is None:
            return None

        if torch is not None and isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        if isinstance(value, np.ndarray):
            if value.shape == ():
                return np.full(self._num_envs, bool(value.item()), dtype=bool)
            flat = value.astype(bool).reshape(-1)
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            flat = np.asarray(value, dtype=bool).reshape(-1)
        else:
            return np.full(self._num_envs, bool(value), dtype=bool)

        if flat.size == 1:
            return np.full(self._num_envs, bool(flat[0]), dtype=bool)
        if flat.size < self._num_envs:
            return None
        return flat[: self._num_envs].astype(bool, copy=False)

    def _extract_episode_metric(self, info: Any, key: str) -> Any:
        """Extract an episode metric, preferring final_info for done envs."""
        if not isinstance(info, dict):
            return None

        value = None
        episode = info.get("episode")
        if isinstance(episode, dict) and key in episode:
            value = episode[key]
        elif key in info:
            value = info[key]

        final_info = info.get("final_info")
        final_mask = self._to_bool_array(info.get("_final_info"))
        if isinstance(final_info, dict) and final_mask is not None and final_mask.any():
            final_value = None
            final_episode = final_info.get("episode")
            if isinstance(final_episode, dict) and key in final_episode:
                final_value = final_episode[key]
            elif key in final_info:
                final_value = final_info[key]

            final_value = self._to_bool_array(final_value)
            if final_value is not None:
                merged = self._to_bool_array(value)
                if merged is None:
                    merged = np.zeros(self._num_envs, dtype=bool)
                merged[final_mask] = final_value[final_mask]
                return merged

        return value

    def _get_task_description(self, obs: Any, env_id: int):
        """Get task description from obs or env attribute."""
        if isinstance(obs, dict) and "task_descriptions" in obs:
            task_desc = obs["task_descriptions"]
            if isinstance(task_desc, (list, tuple)) and len(task_desc) > env_id:
                return task_desc[env_id]
            return task_desc[0] if isinstance(task_desc, (list, tuple)) else task_desc
        if hasattr(self.env, "task_descriptions"):
            task_desc = self.env.task_descriptions
            if isinstance(task_desc, (list, tuple)) and len(task_desc) > env_id:
                return task_desc[env_id]
            return task_desc[0] if isinstance(task_desc, (list, tuple)) else task_desc
        return None

    def _get_video_info_keys(self) -> list[str]:
        """Get configured info keys to overlay on video frames."""
        if hasattr(self.video_cfg, "extra_info_on_video"):
            keys = getattr(self.video_cfg, "extra_info_on_video")
        else:
            keys = None

        if keys:
            if isinstance(keys, str):
                return [keys]
            return list(keys)
        return []

    def _lookup_info_value(self, info: Any, key: str) -> Any:
        """Read a key from info, supporting dotted access for nested dicts."""
        if not isinstance(info, dict):
            return None
        if key in info:
            return info[key]

        value = info
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            return value

        final_info = info.get("final_info")
        if not isinstance(final_info, dict):
            return None
        if key in final_info:
            return final_info[key]

        value = final_info
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    def _get_info_value(self, info: Any, key: str) -> Any:
        """Get an auxiliary info payload, preferring top-level info then final_info."""
        return self._lookup_info_value(info, key)

    def _build_info_item(
        self,
        infos: Optional[Any],
        rewards: Optional[Any],
        terminations: Optional[Any],
        env_id: int,
        time_idx: Optional[int] = None,
        num_frames: Optional[int] = None,
    ) -> dict:
        """Build a per-env info dict for overlay."""
        info_item: dict[str, Any] = {}

        def _select_time_value(value: Any) -> Any:
            if time_idx is None or not isinstance(value, (np.ndarray, list, tuple)):
                return value

            value_len = len(value)
            if value_len == 0:
                return value

            effective_idx = time_idx
            if num_frames == 1 and value_len > 1:
                effective_idx = value_len - 1
            elif effective_idx >= value_len:
                effective_idx = value_len - 1
            return value[effective_idx]

        if rewards is not None:
            value = self._value_for_env(rewards, env_id)
            value = _select_time_value(value)
            info_item["reward"] = float(value) if value is not None else value

        if terminations is not None:
            value = self._value_for_env(terminations, env_id)
            value = _select_time_value(value)
            info_item["termination"] = bool(value) if value is not None else value

        if infos is not None:
            for key in self._get_video_info_keys():
                value = self._lookup_info_value(infos, key)
                if value is None:
                    continue
                value = self._value_for_env(value, env_id)
                value = _select_time_value(value)
                if key == "policy_action_last":
                    action_value = np.asarray(value, dtype=np.float32).reshape(-1)
                    if action_value.size >= 7:
                        info_item["a_xyz"] = ",".join(
                            f"{x:.3f}" for x in action_value[:3]
                        )
                        info_item["a_rot"] = ",".join(
                            f"{x:.3f}" for x in action_value[3:6]
                        )
                        info_item["a_g"] = f"{action_value[6]:.3f}"
                        continue
                if isinstance(value, np.ndarray):
                    if value.shape == ():
                        value = value.item()
                    elif value.size == 1:
                        value = value.reshape(-1)[0].item()
                    else:
                        value = np.array2string(
                            value.astype(np.float32, copy=False),
                            precision=3,
                            separator=",",
                            suppress_small=False,
                        )
                elif isinstance(value, (list, tuple)):
                    value = np.array2string(
                        np.asarray(value, dtype=np.float32),
                        precision=3,
                        separator=",",
                        suppress_small=False,
                    )
                elif isinstance(value, numbers.Number):
                    pass
                else:
                    warnings.warn(f"Unsupported value type {type(value)} for key {key}")
                    continue
                info_item[key] = value

        return info_item

    def _extract_done_mask(self, info: Any) -> Optional[np.ndarray]:
        """Get done mask attached to final_info/final_observation."""
        if not isinstance(info, dict):
            return None
        return self._to_bool_array(info.get("_final_info"))

    def _capture_success_frames(
        self,
        images: list[np.ndarray],
        infos: Optional[Any],
        rewards: Optional[Any],
        terminations: Optional[Any],
        time_idx: Optional[int] = None,
        total_frame_count: Optional[int] = None,
    ) -> None:
        """Save the first frame where success_once becomes true for each env."""
        if not self.video_cfg.get("save_success_frame", False) or infos is None:
            return

        success_once = self._to_bool_array(self._extract_episode_metric(infos, "success_once"))
        if success_once is None:
            return

        new_success = success_once & ~self._success_once_state
        for env_id in np.flatnonzero(new_success):
            if env_id >= len(images):
                continue
            success_frame_images = self._get_info_value(infos, "success_frame_images")
            success_frame_raw_images = self._get_info_value(infos, "success_frame_raw_images")
            success_frame_raw_tensors = self._get_info_value(infos, "success_frame_raw_tensors")
            success_frame_meta = self._get_info_value(infos, "success_frame_meta")
            success_frame_wrist_images = self._get_info_value(infos, "success_frame_wrist_images")
            success_frame_time_idx = self._get_info_value(infos, "success_frame_time_idx")
            selected_time_idx = self._value_for_env(success_frame_time_idx, env_id)
            frame_value = self._value_for_env(success_frame_images, env_id)
            raw_frame_value = self._value_for_env(success_frame_raw_images, env_id)
            raw_tensor_value = self._value_for_env(success_frame_raw_tensors, env_id)
            raw_meta_value = self._value_for_env(success_frame_meta, env_id)
            if frame_value is not None:
                frame = self._to_numpy(frame_value)
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                frame = frame.copy()
            else:
                frame = images[env_id].copy()
            if raw_frame_value is not None:
                raw_frame = self._to_numpy(raw_frame_value)
                if raw_frame.dtype != np.uint8:
                    raw_frame = raw_frame.astype(np.uint8)
                raw_frame = raw_frame.copy()
            else:
                raw_frame = frame.copy()
            if torch is not None and isinstance(raw_tensor_value, torch.Tensor):
                raw_tensor = raw_tensor_value.detach().cpu().to(torch.float32).contiguous()
            elif raw_tensor_value is not None:
                raw_tensor = torch.as_tensor(np.asarray(raw_tensor_value), dtype=torch.float32)
            else:
                raw_tensor = None
            raw_meta = self._jsonable(raw_meta_value) if raw_meta_value is not None else {}
            if isinstance(raw_meta, dict):
                raw_meta.setdefault("env_id", int(env_id))
                raw_meta.setdefault("video_idx", int(self.video_cnt))
                raw_meta.setdefault("success_idx", int(self.success_image_cnt))
            if self.video_cfg.get("info_on_video", True):
                task_desc = self._get_task_description(infos, env_id)
                extras = [f"task: {task_desc}"] if task_desc else None
                frame = put_info_on_image(
                    frame,
                    self._build_info_item(
                        infos,
                        rewards,
                        terminations,
                        env_id,
                        selected_time_idx if selected_time_idx is not None else time_idx,
                        num_frames=total_frame_count,
                    ),
                    extras=extras,
                )
            self.success_images.append(
                {
                    "image": frame,
                    "video_idx": self.video_cnt,
                    "env_id": env_id,
                    "success_idx": self.success_image_cnt,
                    "image_kind": "main",
                    "raw_main_image": raw_frame,
                    "raw_tensor": raw_tensor,
                    "meta": raw_meta,
                }
            )
            wrist_frame_value = self._value_for_env(success_frame_wrist_images, env_id)
            if wrist_frame_value is not None:
                wrist_frame = self._to_numpy(wrist_frame_value)
                if wrist_frame.dtype != np.uint8:
                    wrist_frame = wrist_frame.astype(np.uint8)
                wrist_frame = wrist_frame.copy()
                if self.video_cfg.get("info_on_video", True):
                    task_desc = self._get_task_description(infos, env_id)
                    extras = [f"task: {task_desc}"] if task_desc else None
                    wrist_frame = put_info_on_image(
                        wrist_frame,
                        self._build_info_item(
                            infos,
                            rewards,
                            terminations,
                            env_id,
                            selected_time_idx if selected_time_idx is not None else time_idx,
                            num_frames=total_frame_count,
                        ),
                        extras=extras,
                    )
                self.success_images.append(
                    {
                        "image": wrist_frame,
                        "video_idx": self.video_cnt,
                        "env_id": env_id,
                        "success_idx": self.success_image_cnt,
                        "image_kind": "wrist",
                    }
                )
            self.success_image_cnt += 1

        next_state = success_once.copy()
        done_mask = self._extract_done_mask(infos)
        if done_mask is not None:
            next_state[done_mask] = False
        self._success_once_state = next_state

    def _append_frame(
        self,
        images: list[np.ndarray],
        infos: Optional[Any],
        rewards: Optional[Any],
        terminations: Optional[Any],
        time_idx: Optional[int] = None,
        total_frame_count: Optional[int] = None,
    ) -> None:
        """Overlay info (optional) and append a tiled frame."""
        if not images:
            return
        self._capture_success_frames(
            images,
            infos,
            rewards,
            terminations,
            time_idx,
            total_frame_count=total_frame_count,
        )
        if self.video_cfg.get("info_on_video", True):
            rendered_images = []
            for env_id, img in enumerate(images):
                task_desc = self._get_task_description(infos, env_id)
                extras = [f"task: {task_desc}"] if task_desc else None
                rendered_images.append(
                    put_info_on_image(
                        img,
                        self._build_info_item(
                            infos,
                            rewards,
                            terminations,
                            env_id,
                            time_idx,
                            num_frames=total_frame_count,
                        ),
                        extras=extras,
                    )
                )
            images = rendered_images
        if self.video_cfg.get("save_all_trajectories", False):
            for env_id, image in enumerate(images):
                if env_id < len(self.per_env_render_images):
                    self.per_env_render_images[env_id].append(image.copy())
        if len(images) > 1:
            nrows = int(np.sqrt(len(images)))
            full_image = tile_images(images, nrows=nrows)
            self.render_images.append(full_image)
        else:
            self.render_images.append(images[0])

    def add_new_frames(
        self,
        obs: Any,
        infos: Optional[Any] = None,
        rewards: Optional[Any] = None,
        terminations: Optional[Any] = None,
    ):
        """Extract frames from obs and append to the buffer."""
        frames = self._extract_frame_batches(obs)
        if not frames:
            warnings.warn(
                f"Failed to extract images from obs, obs type: {type(obs)}, obs keys: "
                f"{list(obs.keys()) if isinstance(obs, dict) else 'N/A'}"
            )
            return

        if isinstance(infos, (list, tuple)):
            total_frame_count = len(frames)
            for time_idx, images in enumerate(frames):
                info_item = infos[time_idx] if time_idx < len(infos) else None
                self._append_frame(
                    images,
                    info_item,
                    rewards,
                    terminations,
                    time_idx,
                    total_frame_count=total_frame_count,
                )
            return

        total_frame_count = len(frames)
        for time_idx, images in enumerate(frames):
            self._append_frame(
                images,
                infos,
                rewards,
                terminations,
                time_idx,
                total_frame_count=total_frame_count,
            )

    def reset(self, *args, **kwargs):
        """Reset env and record the initial frame."""
        obs, info = self.env.reset(*args, **kwargs)
        self._success_once_state[:] = False
        self.add_new_frames(obs, info)
        return obs, info

    def get_current_obs(self):
        """Read and record the current observation without resetting the env."""
        obs, info = self.env.get_current_obs()
        self._success_once_state[:] = False
        self.add_new_frames(obs, info)
        return obs, info

    def step(self, action):
        """Step env and record the resulting frame."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        terminations = (
            info.get("terminations", terminated)
            if isinstance(info, dict)
            else terminated
        )
        self.add_new_frames(obs, info, reward, terminations)
        return obs, reward, terminated, truncated, info

    def _chunk_video_source(self, info: Optional[dict]) -> Optional[dict]:
        """Build a synthetic obs dict from the full per-chunk camera tensors
        (neutral `world_model_video_chunk_*` first, then legacy
        `ctrl_world_video_chunk*`; each `[B, chunk, H, W, C]`) exposed in
        `infos`. `chunk_step()`'s own returned observation
        only ever carries the single LAST frame of the chunk (needed for the
        next policy decision) — recording from it would only ever capture 1
        frame per chunk-step call instead of the whole chunk, regardless of
        how many frames elapsed. Returns None when the chunk tensors aren't
        present so callers fall back to the plain per-step observation.
        """
        if not isinstance(info, dict):
            return None
        if info.get("world_model_video_chunk_main") is not None:
            return {
                "main_images": info["world_model_video_chunk_main"],
                "wrist_images": info.get("world_model_video_chunk_wrist"),
                "extra_view_images": info.get("world_model_video_chunk_extra"),
            }
        if info.get("world_model_video_chunk") is not None:
            return {"main_images": info["world_model_video_chunk"]}
        if info.get("ctrl_world_video_chunk_raw") is not None:
            return {
                "main_images": info["ctrl_world_video_chunk_raw"],
                "wrist_images": info.get("ctrl_world_video_chunk_raw_wrist"),
                "extra_view_images": info.get("ctrl_world_video_chunk_raw_extra"),
            }
        if info.get("ctrl_world_video_chunk") is None:
            return None
        return {
            "main_images": info["ctrl_world_video_chunk"],
            "wrist_images": info.get("ctrl_world_video_chunk_wrist"),
            "extra_view_images": info.get("ctrl_world_video_chunk_extra"),
        }

    def chunk_step(self, *args, **kwargs):
        """Step a chunk and record all frames from the chunk."""
        result = self.env.chunk_step(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 5:
            obs_list, rewards, terminations, _truncations, infos_list = result[:5]
            final_obs = None
            last_info = None
            if isinstance(infos_list, (list, tuple)) and len(infos_list) > 0:
                last_info = infos_list[-1]
                if isinstance(last_info, dict):
                    if last_info.get("final_obs") is not None:
                        final_obs = last_info["final_obs"]
                    elif last_info.get("final_observation") is not None:
                        final_obs = last_info["final_observation"]

            chunk_obs = self._chunk_video_source(last_info)

            if (
                final_obs is not None
                and isinstance(obs_list, (list, tuple))
                and len(obs_list) > 0
            ):
                reset_obs = obs_list[-1]
                obs_main = list(obs_list)
                obs_main[-1] = final_obs
                infos_main = (
                    list(infos_list)
                    if isinstance(infos_list, (list, tuple))
                    else infos_list
                )
                if chunk_obs is not None:
                    self.add_new_frames(chunk_obs, last_info, rewards, terminations)
                else:
                    self.add_new_frames(obs_main, infos_main, rewards, terminations)
                self.add_new_frames(reset_obs, None)
            elif chunk_obs is not None:
                self.add_new_frames(chunk_obs, last_info, rewards, terminations)
            else:
                self.add_new_frames(obs_list, infos_list, rewards, terminations)
        return result

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """Write buffered frames to an MP4 file (async)."""
        if not self.render_images:
            return

        output_dir = os.path.join(
            self.video_cfg.video_base_dir, f"seed_{self.env.seed}"
        )
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")

        os.makedirs(output_dir, exist_ok=True)
        mp4_path = os.path.join(output_dir, f"{self.video_cnt}.mp4")
        frames = list(self.render_images)
        per_env_frames = [list(items) for items in self.per_env_render_images]
        success_images = list(self.success_images)
        video_idx = self.video_cnt
        self.render_images = []
        self.per_env_render_images = [[] for _ in range(self._num_envs)]
        self.success_images = []
        self.video_cnt += 1
        self._submit_save(frames, mp4_path)
        if self.video_cfg.get("save_all_trajectories", False):
            for env_id, env_frames in enumerate(per_env_frames):
                if not env_frames:
                    continue
                env_dir = os.path.join(output_dir, "per_env", f"env_{env_id}")
                os.makedirs(env_dir, exist_ok=True)
                self._submit_save(
                    env_frames, os.path.join(env_dir, f"{video_idx}.mp4")
                )
        if success_images:
            success_dir = os.path.join(output_dir, "success_once")
            os.makedirs(success_dir, exist_ok=True)
            for item in success_images:
                image = item["image"]
                video_idx = item["video_idx"]
                env_id = item["env_id"]
                success_idx = item["success_idx"]
                image_kind = item["image_kind"]
                png_path = os.path.join(
                    success_dir,
                    f"{video_idx}_env{env_id}_success_once_{success_idx}_{image_kind}.png",
                )
                self._submit_save_image(image, png_path)
                if image_kind != "main":
                    continue
                raw_main_image = item.get("raw_main_image")
                if raw_main_image is not None:
                    raw_png_path = os.path.join(
                        success_dir,
                        f"{video_idx}_env{env_id}_success_once_{success_idx}_success_frame_raw_main.png",
                    )
                    self._submit_save_image(raw_main_image, raw_png_path)
                raw_tensor = item.get("raw_tensor")
                if raw_tensor is not None:
                    raw_tensor_path = os.path.join(
                        success_dir,
                        f"{video_idx}_env{env_id}_success_once_{success_idx}_success_frame_raw_tensor.pt",
                    )
                    self._submit_save_tensor(raw_tensor, raw_tensor_path)
                meta = item.get("meta")
                if meta is not None:
                    meta_path = os.path.join(
                        success_dir,
                        f"{video_idx}_env{env_id}_success_once_{success_idx}_success_frame_meta.json",
                    )
                    self._submit_save_json(meta, meta_path)

    def _submit_save(self, frames: list[np.ndarray], mp4_path: str) -> None:
        """Submit a background job to save the video."""
        self._prune_futures()
        future = self._executor.submit(self._save_video, frames, mp4_path)
        self._save_futures.append(future)

    def _submit_save_image(self, image: np.ndarray, png_path: str) -> None:
        """Submit a background job to save an image."""
        self._prune_futures()
        future = self._executor.submit(self._save_image, image, png_path)
        self._save_futures.append(future)

    def _submit_save_tensor(self, tensor: Any, tensor_path: str) -> None:
        """Submit a background job to save a tensor."""
        self._prune_futures()
        future = self._executor.submit(self._save_tensor, tensor, tensor_path)
        self._save_futures.append(future)

    def _submit_save_json(self, payload: Any, json_path: str) -> None:
        """Submit a background job to save a json file."""
        self._prune_futures()
        future = self._executor.submit(self._save_json, payload, json_path)
        self._save_futures.append(future)

    def _save_video(self, frames: list[np.ndarray], mp4_path: str) -> None:
        """Save frames to disk (runs in background)."""
        video_writer = None
        try:
            video_writer = imageio.get_writer(mp4_path, fps=self._fps)
            for img in frames:
                video_writer.append_data(img)
        except Exception as exc:
            warnings.warn(f"Failed to save video {mp4_path}: {exc}")
        finally:
            if video_writer is not None:
                video_writer.close()

    def _save_image(self, image: np.ndarray, png_path: str) -> None:
        """Save a success frame to disk (runs in background)."""
        try:
            imageio.imwrite(png_path, image)
        except Exception as exc:
            warnings.warn(f"Failed to save image {png_path}: {exc}")

    def _save_tensor(self, tensor: Any, tensor_path: str) -> None:
        """Save a tensor artifact to disk (runs in background)."""
        if torch is None:
            warnings.warn(f"Failed to save tensor {tensor_path}: torch is unavailable")
            return
        try:
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(np.asarray(tensor), dtype=torch.float32)
            torch.save(tensor.detach().cpu(), tensor_path)
        except Exception as exc:
            warnings.warn(f"Failed to save tensor {tensor_path}: {exc}")

    def _save_json(self, payload: Any, json_path: str) -> None:
        """Save a json artifact to disk (runs in background)."""
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self._jsonable(payload), f, ensure_ascii=False, indent=2)
        except Exception as exc:
            warnings.warn(f"Failed to save json {json_path}: {exc}")

    def _jsonable(self, value: Any) -> Any:
        """Convert nested tensors/arrays/scalars into JSON-serializable values."""
        if torch is not None and isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _prune_futures(self) -> None:
        """Remove finished futures to avoid unbounded growth."""
        self._save_futures = [f for f in self._save_futures if not f.done()]

    def wait_for_pending_saves(self) -> None:
        """Block until all queued video/image/tensor writes are fully flushed."""
        pending_futures = list(self._save_futures)
        for future in pending_futures:
            future.result()
        self._prune_futures()

    def close(self):
        """Wait for pending video writes before closing."""
        self.wait_for_pending_saves()
        self._executor.shutdown(wait=True)
        self._save_futures = []
        return super().close()

    def update_reset_state_ids(self):
        if hasattr(self.env, "update_reset_state_ids"):
            self.env.update_reset_state_ids()
