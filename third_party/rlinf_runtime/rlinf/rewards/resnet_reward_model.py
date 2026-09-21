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

"""Fallback loader for the frame-level ResNet reward checkpoint.

The Ctrl-World configs historically import ``diffsynth.models.reward_model``.
Some repo-local GH200 environments do not include that package, but the reward
checkpoint itself is a plain PyTorch state dict. This module mirrors the small
ResNet architecture encoded by that state dict so real reward weights can still
be used without creating a placeholder reward path.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn


DUCK_EPISODE_COLOR_RANGES: dict[str, tuple[int, int]] = {
    "brown": (0, 49),
    "red": (50, 99),
    "white": (100, 149),
    "yellow": (150, 199),
}


def classify_terminal_probabilities(
    probabilities: torch.Tensor,
    cfg: Mapping[str, Any],
) -> dict[str, torch.Tensor | str]:
    """Apply either the retained max-window rule or the native terminal rule.

    ``terminal_positive_ratio`` mirrors ``check_rm_rlinf_preprocess.py``:
    sample the last ``terminal_window_frames`` with ``frame_stride`` and append
    the actual last frame when the stride did not select it.
    """
    if probabilities.ndim == 1:
        probabilities = probabilities.unsqueeze(0)
    if probabilities.ndim != 2 or probabilities.shape[1] <= 0:
        raise ValueError("success probabilities must have shape [B,T] with T > 0")
    probabilities = probabilities.to(dtype=torch.float32)
    if not torch.isfinite(probabilities).all() or (
        (probabilities < 0).any() or (probabilities > 1).any()
    ):
        raise ValueError("success probabilities must be finite and in [0,1]")

    aggregation = str(cfg.get("aggregation", "legacy_max_window"))
    if aggregation == "terminal_positive_ratio":
        window = int(cfg.get("terminal_window_frames", 30))
        stride = int(cfg.get("frame_stride", 3))
        threshold = float(
            cfg.get("probability_threshold", cfg.get("threshold", 0.5))
        )
        minimum_ratio = float(cfg.get("minimum_terminal_positive_ratio", 0.8))
        requires_last = bool(cfg.get("requires_last_frame_positive", True))
        if window <= 0:
            raise ValueError("terminal_window_frames must be positive")
        if stride <= 0:
            raise ValueError("frame_stride must be positive")
        if not 0.0 <= minimum_ratio <= 1.0:
            raise ValueError("minimum_terminal_positive_ratio must be in [0,1]")
        start = max(0, int(probabilities.shape[1]) - window)
        indices = list(range(start, int(probabilities.shape[1]), stride))
        if not indices or indices[-1] != probabilities.shape[1] - 1:
            indices.append(int(probabilities.shape[1]) - 1)
        index_tensor = torch.tensor(
            indices, dtype=torch.long, device=probabilities.device
        )
        sampled = probabilities.index_select(1, index_tensor)
        positive = sampled >= threshold
        positive_ratio = positive.to(torch.float32).mean(dim=1)
        last_positive = positive[:, -1]
        success = positive_ratio >= minimum_ratio
        if requires_last:
            success = success & last_positive
        rule = "terminal_positive_ratio"
    else:
        window = int(cfg.get("window_size", 4))
        threshold = float(cfg.get("threshold", 0.5))
        if window <= 0 or window > probabilities.shape[1]:
            raise ValueError("invalid success diagnostic window_size")
        index_tensor = torch.arange(
            probabilities.shape[1] - window,
            probabilities.shape[1],
            dtype=torch.long,
            device=probabilities.device,
        )
        sampled = probabilities.index_select(1, index_tensor)
        positive = sampled >= threshold
        positive_ratio = positive.to(torch.float32).mean(dim=1)
        success = sampled.max(dim=1).values >= threshold
        minimum_ratio = 0.0
        requires_last = False
        rule = "legacy_max_window"

    probability_max = sampled.max(dim=1).values
    last_probability = sampled[:, -1]
    reported_probability = (
        last_probability if rule == "terminal_positive_ratio" else probability_max
    )
    return {
        "rule": rule,
        "sample_indices": index_tensor,
        "sampled_probabilities": sampled,
        "success": success,
        "reported_probability": reported_probability,
        "last_probability": last_probability,
        "positive_ratio": positive_ratio,
        "probability_max": probability_max,
        "threshold": torch.tensor(threshold, device=probabilities.device),
        "minimum_positive_ratio": torch.tensor(
            minimum_ratio, device=probabilities.device
        ),
        "requires_last_frame_positive": torch.tensor(
            requires_last, device=probabilities.device
        ),
    }


def duck_color_for_episode(
    episode_id: int,
    color_ranges: Mapping[str, Sequence[int]] | None = None,
) -> str:
    """Return the duck color assigned to an episode id."""
    ranges = color_ranges or DUCK_EPISODE_COLOR_RANGES
    matches = []
    for color, bounds in ranges.items():
        if len(bounds) != 2:
            raise ValueError(f"Color range for {color!r} must contain [min, max].")
        lower, upper = (int(bounds[0]), int(bounds[1]))
        if lower > upper:
            raise ValueError(f"Color range for {color!r} is reversed.")
        if lower <= int(episode_id) <= upper:
            matches.append(str(color))
    if len(matches) != 1:
        raise ValueError(
            f"Episode {episode_id} must match exactly one duck color; got {matches}."
        )
    return matches[0]


def resolve_eval_success_model_provenance(
    episode_id: int,
    *,
    active_variants: Sequence[str],
    runtime_provenance: Mapping[str, Mapping[str, Any]],
    configured_models: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, str]]:
    """Resolve one task-aware success-model identity before eval inference.

    The original post-update audit assumed every task used the four Duck color
    ranges. Single-task profiles such as ``nest_four_cups`` deliberately load
    one plain ``ResnetRewModel`` instead; that model has no routed
    ``provenance()`` method. Resolve its explicit configured artifact without
    changing the inference model or weakening the required model hash.
    """
    variants = tuple(str(value) for value in active_variants)
    if len(set(variants)) != len(variants):
        raise ValueError("active success-model variants must be unique")
    variant = (
        variants[0]
        if len(variants) == 1
        else duck_color_for_episode(int(episode_id))
    )

    runtime_model = dict(runtime_provenance.get(variant, {}) or {})
    configured_model = dict(configured_models.get(variant, {}) or {})
    path_value = runtime_model.get("path")
    if not path_value:
        root_value = configured_model.get("from_pretrained")
        if root_value:
            checkpoint_path = Path(str(root_value)).expanduser().resolve()
            if checkpoint_path.is_dir():
                checkpoint_path /= str(
                    configured_model.get("artifact_name", "resnet_rm.pth")
                )
            path_value = str(checkpoint_path)
    digest = str(
        runtime_model.get("sha256") or configured_model.get("sha256") or ""
    ).strip().lower()
    if not path_value or not Path(str(path_value)).expanduser().is_file():
        raise ValueError(
            f"Evaluation episode {episode_id} variant {variant!r} has no "
            "readable success-model artifact."
        )
    # Push-T is intentionally launched with integrity/hash gates disabled.
    # Keep strict validation for every existing routed/single-task profile.
    if variants == ("push_t",) and (not digest or digest == "unverified"):
        digest = "unverified"
    elif len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(
            f"Evaluation episode {episode_id} variant {variant!r} has no valid "
            "success-model SHA-256."
        )
    return variant, {
        "path": str(Path(str(path_value)).expanduser().resolve()),
        "sha256": digest,
    }


@lru_cache(maxsize=32)
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            if stride != 1 or in_channels != out_channels
            else None
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.conv3 is None else self.conv3(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class ResnetRewModel(nn.Module):
    """Frame-level binary reward model compatible with existing checkpoints."""

    def __init__(self, from_pretrained: str | Path) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ),
            nn.Sequential(
                _ResidualBlock(64, 64),
                _ResidualBlock(64, 64),
            ),
            nn.Sequential(
                _ResidualBlock(64, 128, stride=2),
                _ResidualBlock(128, 128),
            ),
            nn.Sequential(
                _ResidualBlock(128, 256, stride=2),
                _ResidualBlock(256, 256),
            ),
            nn.Sequential(
                _ResidualBlock(256, 512, stride=2),
                _ResidualBlock(512, 512),
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1),
        )
        state_dict = torch.load(
            str(Path(from_pretrained).expanduser()), map_location="cpu", weights_only=False
        )
        state_dict = self._extract_state_dict(state_dict)
        self.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> OrderedDict[str, torch.Tensor] | dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model", "module"):
                nested = checkpoint.get(key)
                if isinstance(nested, dict):
                    checkpoint = nested
                    break
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid ResnetRewModel checkpoint format.")
        if checkpoint and all(str(key).startswith("module.") for key in checkpoint):
            return OrderedDict((str(key)[7:], value) for key, value in checkpoint.items())
        return checkpoint

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.net(images).squeeze(-1)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_rew(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)


class ColorRoutedResnetRewModel(nn.Module):
    """Route duck frames to one frozen ResNet checkpoint per episode color."""

    def __init__(
        self,
        checkpoints: Mapping[str, str | Path],
        color_ranges: Mapping[str, Sequence[int]] | None = None,
        checkpoint_sha256: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.color_ranges = {
            str(color): (int(bounds[0]), int(bounds[1]))
            for color, bounds in (color_ranges or DUCK_EPISODE_COLOR_RANGES).items()
        }
        expected_colors = set(self.color_ranges)
        configured_colors = {str(color) for color in checkpoints}
        if configured_colors != expected_colors:
            raise ValueError(
                "Color-routed ResNet checkpoints must exactly match configured "
                f"colors: expected {sorted(expected_colors)}, got "
                f"{sorted(configured_colors)}."
            )
        configured_hashes = None
        if checkpoint_sha256 is not None:
            configured_hashes = {
                str(color): str(digest).strip().lower()
                for color, digest in checkpoint_sha256.items()
            }
            if set(configured_hashes) != expected_colors:
                raise ValueError(
                    "Color-routed ResNet hashes must exactly match configured "
                    f"colors: expected {sorted(expected_colors)}, got "
                    f"{sorted(configured_hashes)}."
                )
            for color, digest in configured_hashes.items():
                if len(digest) != 64 or any(
                    char not in "0123456789abcdef" for char in digest
                ):
                    raise ValueError(
                        f"Configured ResNet SHA-256 for {color!r} is invalid."
                    )

        self.checkpoint_paths: dict[str, str] = {}
        self.checkpoint_sha256: dict[str, str] = {}
        models = {}
        for color in self.color_ranges:
            checkpoint_path = Path(checkpoints[color]).expanduser().resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Reward-model checkpoint does not exist: {checkpoint_path}"
                )
            models[color] = ResnetRewModel(checkpoint_path)
            self.checkpoint_paths[color] = str(checkpoint_path)
            self.checkpoint_sha256[color] = (
                configured_hashes[color]
                if configured_hashes is not None
                else _sha256_file(checkpoint_path)
            )
        self.models = nn.ModuleDict(models)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def color_for_episode(self, episode_id: int) -> str:
        """Return the configured color for one episode."""
        return duck_color_for_episode(episode_id, self.color_ranges)

    def provenance(self) -> dict[str, dict[str, str]]:
        """Return checkpoint paths and hashes without model tensors."""
        return {
            color: {
                "path": self.checkpoint_paths[color],
                "sha256": self.checkpoint_sha256[color],
            }
            for color in self.color_ranges
        }

    @torch.no_grad()
    def predict_rew(
        self, images: torch.Tensor, episode_ids: torch.Tensor | Sequence[int]
    ) -> torch.Tensor:
        """Score images with the model selected by each row's episode id."""
        episode_ids = torch.as_tensor(episode_ids, dtype=torch.int64).reshape(-1)
        if episode_ids.numel() != images.shape[0]:
            raise ValueError(
                "episode_ids must contain one value per image: "
                f"{episode_ids.numel()} != {images.shape[0]}"
            )
        output = torch.empty(images.shape[0], device=images.device, dtype=torch.float32)
        for color, model in self.models.items():
            matching = torch.tensor(
                [
                    self.color_for_episode(int(episode_id)) == color
                    for episode_id in episode_ids.tolist()
                ],
                dtype=torch.bool,
                device=images.device,
            )
            if matching.any():
                output[matching] = model.predict_rew(images[matching]).to(torch.float32)
        return output
