"""Strict loader for the frozen four-color duck episode contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DuckEpisodeSplit:
    """Validated episode ids and color pools for one manifest split."""

    name: str
    episodes: tuple[int, ...]
    episodes_by_color: dict[str, tuple[int, ...]]
    color_order: tuple[str, ...]
    manifest_path: str
    manifest_sha256: str
    source_sha256: dict[str, tuple[str, str]]


def select_duck_episode_colors(
    contract: DuckEpisodeSplit,
    color_order: tuple[str, ...] | list[str] | None,
) -> tuple[tuple[int, ...], dict[str, tuple[int, ...]], tuple[str, ...]]:
    """Select an explicitly configured non-empty color subset."""
    selected_order = (
        contract.color_order
        if color_order is None
        else tuple(str(color) for color in color_order)
    )
    if not selected_order:
        raise ValueError("Duck reset episode color order must not be empty.")
    if len(set(selected_order)) != len(selected_order):
        raise ValueError("Duck reset episode color order must be unique.")
    unknown = set(selected_order) - set(contract.episodes_by_color)
    if unknown:
        raise ValueError(
            "Duck reset episode color order contains colors absent from the "
            f"validated manifest: {sorted(unknown)}"
        )
    selected_pools = {
        color: contract.episodes_by_color[color] for color in selected_order
    }
    selected_episodes = tuple(
        episode
        for color in selected_order
        for episode in selected_pools[color]
    )
    if len(set(selected_episodes)) != len(selected_episodes):
        raise ValueError("Selected Duck episode color pools overlap.")
    return selected_episodes, selected_pools, selected_order


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_list_sha256(values: tuple[int, ...]) -> str:
    payload = json.dumps(
        list(values), separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_split_name(split: str) -> str:
    aliases = {
        "train": "training",
        "training": "training",
        "eval": "evaluation",
        "evaluation": "evaluation",
        "val": "evaluation",
        "validation": "evaluation",
    }
    try:
        return aliases[str(split).lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported duck manifest split {split!r}.") from error


def _read_nested(payload: dict[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for part in str(dotted_key).split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Duck manifest does not contain {dotted_key!r}.")
        value = value[part]
    return value


@lru_cache(maxsize=16)
def load_duck_episode_split(
    manifest_path: str,
    split: str,
    verify_source_hashes: bool = True,
) -> DuckEpisodeSplit:
    """Load and validate one frozen split, including upstream source hashes."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Duck split manifest does not exist: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Duck split manifest schema_version must be 1.")
    split_name = _canonical_split_name(split)
    split_payload = payload.get(split_name)
    if not isinstance(split_payload, dict):
        raise KeyError(f"Duck split manifest is missing {split_name!r}.")

    episodes = tuple(int(value) for value in split_payload.get("episodes", []))
    if not episodes or len(set(episodes)) != len(episodes):
        raise ValueError(f"Duck {split_name} episodes must be non-empty and unique.")
    if int(split_payload.get("count", -1)) != len(episodes):
        raise ValueError(f"Duck {split_name} count does not match its episode list.")
    expected_episode_sha256 = str(split_payload.get("episodes_sha256", ""))
    actual_episode_sha256 = _episode_list_sha256(episodes)
    if expected_episode_sha256 != actual_episode_sha256:
        raise ValueError(
            f"Duck {split_name} episode-list SHA-256 changed: "
            f"{actual_episode_sha256} != {expected_episode_sha256}"
        )
    ranges = payload.get("color_ranges", {})
    episodes_by_color_payload = split_payload.get("episodes_by_color", {})
    color_order = tuple(str(color) for color in ranges)
    if set(episodes_by_color_payload) != set(color_order):
        raise ValueError(f"Duck {split_name} color pools do not match color_ranges.")
    episodes_by_color: dict[str, tuple[int, ...]] = {}
    flattened: list[int] = []
    for color in color_order:
        bounds = ranges[color]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"Duck color range {color!r} must contain [min, max].")
        lower, upper = int(bounds[0]), int(bounds[1])
        color_episodes = tuple(
            int(value) for value in episodes_by_color_payload[color]
        )
        if any(not lower <= episode <= upper for episode in color_episodes):
            raise ValueError(f"Duck {split_name} pool {color!r} violates its range.")
        episodes_by_color[color] = color_episodes
        flattened.extend(color_episodes)
    if set(flattened) != set(episodes) or len(flattened) != len(episodes):
        raise ValueError(
            f"Duck {split_name} color pools must partition the episode list exactly."
        )
    expected_counts = split_payload.get("counts_by_color")
    if expected_counts is not None:
        actual_counts = {
            color: len(color_episodes)
            for color, color_episodes in episodes_by_color.items()
        }
        if {str(key): int(value) for key, value in expected_counts.items()} != actual_counts:
            raise ValueError(f"Duck {split_name} counts_by_color is inconsistent.")

    training_episodes = set(
        int(value) for value in payload.get("training", {}).get("episodes", [])
    )
    evaluation_episodes = set(
        int(value) for value in payload.get("evaluation", {}).get("episodes", [])
    )
    if training_episodes & evaluation_episodes:
        raise ValueError("Duck training and evaluation episode lists overlap.")

    source_sha256 = {}
    for source_name, source in payload.get("source_splits", {}).items():
        source_path = Path(str(source.get("path", ""))).expanduser().resolve()
        expected_sha256 = str(source.get("sha256", ""))
        source_sha256[str(source_name)] = (str(source_path), expected_sha256)
        if verify_source_hashes:
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Duck split source {source_name!r} does not exist: {source_path}"
                )
            actual_sha256 = _sha256_file(source_path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Duck split source {source_name!r} SHA-256 changed: "
                    f"{actual_sha256} != {expected_sha256}"
                )

    return DuckEpisodeSplit(
        name=split_name,
        episodes=episodes,
        episodes_by_color=episodes_by_color,
        color_order=color_order,
        manifest_path=str(path),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        source_sha256=source_sha256,
    )


def assert_duck_episode_split_unchanged(contract: DuckEpisodeSplit) -> None:
    """Fail if the frozen manifest or either upstream split source changed."""
    manifest_path = Path(contract.manifest_path)
    if _sha256_file(manifest_path) != contract.manifest_sha256:
        raise ValueError(f"Duck split manifest changed during the run: {manifest_path}")
    for source_name, (source_path_value, expected_sha256) in contract.source_sha256.items():
        source_path = Path(source_path_value)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Duck split source {source_name!r} disappeared: {source_path}"
            )
        if _sha256_file(source_path) != expected_sha256:
            raise ValueError(
                f"Duck split source {source_name!r} changed during the run: "
                f"{source_path}"
            )


def load_duck_manifest_value(manifest_path: str, dotted_key: str) -> Any:
    """Read one dotted key after validating the parent split contract."""
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = str(dotted_key).split(".", 1)[0]
    load_duck_episode_split(str(path), split)
    return _read_nested(payload, dotted_key)
