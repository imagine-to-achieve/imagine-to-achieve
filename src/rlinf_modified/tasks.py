"""Task registry with fixed, auditable train/evaluation episode membership."""

from __future__ import annotations

import json
from pathlib import Path

from rlinf_modified.config import TrainConfig
from rlinf_modified.contracts import TaskSpec
from rlinf_modified.prompts import build_action_prompt


_CLOSE_TRAIN = (
    0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23,
    24, 25, 27, 28, 29, 30, 33, 34, 35, 36, 38, 39,
)
_CLOSE_EVAL = tuple(sorted(set(range(40)) - set(_CLOSE_TRAIN)))


def _load_manifest_split(
    path: str,
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Task split manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Task split manifest schema_version must be 1")
    training = manifest.get("training", {}).get("episodes_by_color", {})
    evaluation = manifest.get("evaluation", {}).get("episodes_by_color", {})
    return (
        {name: tuple(int(item) for item in ids) for name, ids in training.items()},
        {name: tuple(int(item) for item in ids) for name, ids in evaluation.items()},
    )


def build_task_spec(config: TrainConfig) -> TaskSpec:
    """Resolve prompts and episode membership from one task source of truth."""

    task = config.task
    prompts = {
        variant: build_action_prompt(
            task.variant_prompts[variant],
            camera_labels=task.camera_labels,
            num_frames=config.cosmos.action_chunk + 1,
            fps=config.cosmos.action_fps,
            resolution=(720, 640),
            append_viewpoint=task.append_viewpoint_info,
            append_duration_fps=task.append_duration_fps,
            append_resolution=task.append_resolution_info,
        )
        for variant in task.active_variants
    }
    if task.profile in {"close", "synthetic"}:
        train = {variant: _CLOSE_TRAIN for variant in task.active_variants}
        evaluation = {variant: _CLOSE_EVAL for variant in task.active_variants}
    elif task.profile in {"duck", "nest_four_cups", "push_t"}:
        if task.split_manifest_path is None:
            raise ValueError(f"{task.profile} requires task.split_manifest_path")
        all_train, all_eval = _load_manifest_split(task.split_manifest_path)
        train = {variant: all_train[variant] for variant in task.active_variants}
        evaluation = {variant: all_eval[variant] for variant in task.active_variants}
        for variant in task.active_variants:
            if len(train[variant]) != task.train_episodes_per_variant:
                raise ValueError(f"{task.profile} {variant} training split count mismatch")
            if len(evaluation[variant]) != task.eval_episodes_per_variant:
                raise ValueError(f"{task.profile} {variant} evaluation split count mismatch")
    else:  # TrainConfig.validate rejects this; retain a fail-closed local guard.
        raise ValueError(f"Unsupported task profile: {task.profile}")
    return TaskSpec(
        profile=task.profile,
        mode=task.mode,
        prompt=prompts[task.active_variants[0]],
        active_variants=task.active_variants,
        train_episode_ids=train,
        eval_episode_ids=evaluation,
    )


def aggregate_sparse_evaluation(
    rank_local: list[list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Merge empty/uneven rank-local eval results by unique episode ID."""

    by_episode: dict[int, dict[str, object]] = {}
    for local_results in rank_local:
        for item in local_results:
            if "episode_id" not in item:
                raise ValueError("evaluation item is missing episode_id")
            episode_id = int(item["episode_id"])
            if episode_id in by_episode:
                raise ValueError(f"duplicate evaluation episode_id={episode_id}")
            by_episode[episode_id] = item
    return [by_episode[key] for key in sorted(by_episode)]
