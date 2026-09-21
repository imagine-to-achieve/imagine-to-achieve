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
from pathlib import Path
from typing import Any

import torch


_VALIDATION_BATCH_KEYS = (
    "actions",
    "advantages",
    "dones",
    "loss_mask",
    "old_combined_rewards",
    "prev_logprobs",
    "reset_state_ids",
    "reward_model_probabilities",
    "rewards",
    "terminal_goal_rewards",
    "terminations",
    "truncations",
    "video_similarity_rewards",
)
_VALIDATION_TERMINAL_KEYS = ("dones", "terminations", "truncations")
_VALIDATION_FORWARD_KEYS = (
    "action",
    "action_chains",
    "action_denoise_timesteps",
    "action_mask",
    "action_sampling_seed",
    "action_sigmas",
    "action_transition_mode_id",
    "action_transition_means",
    "matched_transition_noise_seed",
    "model_action",
    "fpo_clean_action_normalized",
    "fpo_base_times",
    "fpo_sigmas",
    "fpo_timesteps",
    "fpo_noise_seeds",
)
_VALIDATION_REPLAY_FORWARD_KEYS = (
    *_VALIDATION_FORWARD_KEYS,
    "action_condition_main_images",
    "action_condition_states",
    "imagined_video_chunk",
    "native_full_chains",
    "fpo_clean_joint_latent",
)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _shape_dict(tensors: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in tensors.items()}


def _slice_tensor(value: torch.Tensor, count: int) -> torch.Tensor:
    return value[:count].detach().cpu().contiguous()


def _select_tensor_fields(
    values: dict[str, Any],
    keys: tuple[str, ...],
    *,
    chunk_steps: int,
    local_trajectories: int,
    indices: torch.Tensor | None = None,
    trim_terminal_fields: bool = False,
) -> dict[str, torch.Tensor]:
    selected = {}
    for key in keys:
        value = values.get(key)
        if not torch.is_tensor(value):
            continue
        if (
            trim_terminal_fields
            and key in _VALIDATION_TERMINAL_KEYS
            and int(value.shape[0]) == int(chunk_steps) + 1
        ):
            value = value[:-1]
        if value.ndim < 2 or tuple(value.shape[:2]) != (
            int(chunk_steps),
            int(local_trajectories),
        ):
            raise ValueError(
                f"P0 field {key!r} must begin with [chunk_steps, "
                f"local_trajectories]=[{chunk_steps}, "
                f"{local_trajectories}], got {tuple(value.shape)}."
            )
        flat_value = value.reshape(
            int(chunk_steps) * int(local_trajectories),
            *value.shape[2:],
        )
        if indices is not None:
            flat_value = flat_value[indices]
        selected[key] = flat_value.detach().cpu().contiguous()
    return selected


class CosmosArtifactWriter:
    """Bounded runtime artifact writer for Cosmos training diagnostics."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: str | Path,
        save_replay_tensors: bool = True,
        max_episodes: int = 2,
        max_chunks: int = 4,
    ) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.save_replay_tensors = save_replay_tensors
        self.max_episodes = max(0, int(max_episodes))
        self.max_chunks = max(0, int(max_chunks))
        self._saved_rollout_chunks = 0

    @classmethod
    def from_model_cfg(cls, model_cfg: Any) -> "CosmosArtifactWriter":
        cosmos_cfg = _cfg_get(model_cfg, "cosmos", {})
        artifact_cfg = _cfg_get(cosmos_cfg, "artifacts", {})
        output_dir = _cfg_get(artifact_cfg, "output_dir", "logs/cosmos_artifacts")
        return cls(
            enabled=_to_bool(_cfg_get(artifact_cfg, "enabled", False)),
            output_dir=output_dir,
            save_replay_tensors=_to_bool(
                _cfg_get(artifact_cfg, "save_replay_tensors", True)
            ),
            max_episodes=int(_cfg_get(artifact_cfg, "max_episodes", 2)),
            max_chunks=int(_cfg_get(artifact_cfg, "max_chunks", 4)),
        )

    def write_rollout_chunk(
        self,
        *,
        rank: int,
        stage_id: int,
        rollout_epoch: int,
        chunk_step: int,
        actions: torch.Tensor | None,
        prev_logprobs: torch.Tensor | None,
        rewards: torch.Tensor | None,
        dones: torch.Tensor | None,
        forward_inputs: dict[str, torch.Tensor],
        sample_metadata: dict[str, torch.Tensor] | None = None,
    ) -> Path | None:
        if (
            not self.enabled
            or not self.save_replay_tensors
            or self.max_episodes == 0
            or self._saved_rollout_chunks >= self.max_chunks
        ):
            return None
        if not forward_inputs:
            return None

        first_tensor = next(iter(forward_inputs.values()))
        sample_count = min(self.max_episodes, int(first_tensor.shape[0]))
        if sample_count <= 0:
            return None

        rollout_dir = (
            self.output_dir
            / "rollouts"
            / f"epoch_{rollout_epoch:06d}"
            / f"rank_{rank:04d}"
        )
        rollout_dir.mkdir(parents=True, exist_ok=True)
        file_path = (
            rollout_dir
            / f"stage_{stage_id:02d}_chunk_{chunk_step:04d}_samples.pt"
        )
        forward_slice = {
            key: _slice_tensor(value, sample_count)
            for key, value in forward_inputs.items()
        }
        sample_metadata_slice = {
            key: _slice_tensor(value, sample_count).to(torch.int64)
            for key, value in (sample_metadata or {}).items()
        }
        payload = {
            "metadata": {
                "rank": rank,
                "stage_id": stage_id,
                "rollout_epoch": rollout_epoch,
                "chunk_step": chunk_step,
                "sample_count": sample_count,
            },
            "actions": _slice_tensor(actions, sample_count)
            if torch.is_tensor(actions)
            else None,
            "prev_logprobs": _slice_tensor(prev_logprobs, sample_count)
            if torch.is_tensor(prev_logprobs)
            else None,
            "rewards": _slice_tensor(rewards, sample_count)
            if torch.is_tensor(rewards)
            else None,
            "dones": _slice_tensor(dones, sample_count)
            if torch.is_tensor(dones)
            else None,
            "forward_inputs": forward_slice,
            "sample_metadata": sample_metadata_slice,
        }
        torch.save(payload, file_path)
        self._saved_rollout_chunks += 1
        self._append_manifest(
            {
                "kind": "rollout_chunk",
                "path": file_path.relative_to(self.output_dir).as_posix(),
                "rank": rank,
                "stage_id": stage_id,
                "rollout_epoch": rollout_epoch,
                "chunk_step": chunk_step,
                "sample_count": sample_count,
                "forward_input_shapes": _shape_dict(forward_slice),
                "sample_metadata": {
                    key: value.reshape(-1).tolist()
                    for key, value in sample_metadata_slice.items()
                },
            }
        )
        return file_path

    def write_validation_batch(
        self,
        *,
        rank: int,
        global_step: int,
        batch: dict[str, Any],
        chunk_steps: int,
        local_trajectories: int,
        global_trajectory_offset: int,
        replay_probe_indices: (
            torch.Tensor | list[int] | tuple[int, ...]
        ) = (),
    ) -> dict[str, Path] | None:
        """Persist one compact P0 batch plus a bounded full replay probe."""

        if not self.enabled or not self.save_replay_tensors:
            return None
        probe_indices = torch.as_tensor(
            replay_probe_indices, dtype=torch.int64
        ).reshape(-1)
        if probe_indices.unique().numel() != probe_indices.numel():
            raise ValueError("replay_probe_indices must not contain duplicates.")

        chunk_steps = int(chunk_steps)
        local_trajectories = int(local_trajectories)
        global_trajectory_offset = int(global_trajectory_offset)
        if chunk_steps <= 0 or local_trajectories <= 0:
            raise ValueError(
                "chunk_steps and local_trajectories must be positive."
            )
        sample_count = chunk_steps * local_trajectories
        if probe_indices.numel() and (
            (probe_indices < 0).any()
            or (probe_indices >= sample_count).any()
        ):
            raise IndexError(
                "replay_probe_indices must lie in [0, sample_count); "
                f"sample_count={sample_count}, indices={probe_indices.tolist()}."
            )

        forward_inputs = batch.get("forward_inputs", {})
        if not isinstance(forward_inputs, dict):
            raise TypeError("batch['forward_inputs'] must be a dictionary.")
        compact_batch = _select_tensor_fields(
            batch,
            _VALIDATION_BATCH_KEYS,
            chunk_steps=chunk_steps,
            local_trajectories=local_trajectories,
            trim_terminal_fields=True,
        )
        compact_forward = _select_tensor_fields(
            forward_inputs,
            _VALIDATION_FORWARD_KEYS,
            chunk_steps=chunk_steps,
            local_trajectories=local_trajectories,
        )
        if not compact_batch and not compact_forward:
            raise ValueError(
                "P0 validation batch contains no batched allowlisted tensors."
            )
        flat_sample_index = torch.arange(sample_count, dtype=torch.int64)
        identifiers = {
            "flat_sample_index": flat_sample_index,
            "chunk_step_index": flat_sample_index // local_trajectories,
            "local_trajectory_index": (
                flat_sample_index % local_trajectories
            ),
            "global_candidate_id": (
                flat_sample_index % local_trajectories
            )
            + global_trajectory_offset,
        }
        selected_tensors = {
            **compact_batch,
            **{
                f"forward_inputs.{key}": value
                for key, value in compact_forward.items()
            },
        }
        mismatched = {
            key: list(value.shape)
            for key, value in selected_tensors.items()
            if value.ndim == 0 or int(value.shape[0]) != sample_count
        }
        if mismatched:
            raise ValueError(
                "P0 validation tensors disagree on their leading sample "
                f"dimension: expected={sample_count}, got={mismatched}."
            )

        capture_dir = (
            self.output_dir
            / "p0_capture"
            / f"global_step_{int(global_step):06d}"
            / f"rank_{int(rank):04d}"
        )
        capture_dir.mkdir(parents=True, exist_ok=True)
        compact_path = capture_dir / "compact_batch.pt"
        metadata = {
            "schema_version": 1,
            "rank": int(rank),
            "global_step": int(global_step),
            "sample_count": sample_count,
            "chunk_steps": chunk_steps,
            "local_trajectories": local_trajectories,
            "global_trajectory_offset": global_trajectory_offset,
            "flatten_order": "chunk_step_major",
            "identifier_shapes": _shape_dict(identifiers),
            "compact_batch_shapes": _shape_dict(compact_batch),
            "compact_forward_input_shapes": _shape_dict(compact_forward),
            "compact_excluded_forward_keys": sorted(
                set(forward_inputs).difference(compact_forward)
            ),
        }
        torch.save(
            {
                "metadata": metadata,
                "identifiers": identifiers,
                "batch": compact_batch,
                "forward_inputs": compact_forward,
            },
            compact_path,
        )
        paths = {"compact_batch": compact_path}

        probe_count = int(probe_indices.numel())
        replay_path = None
        replay_forward = {}
        if probe_count:
            replay_batch = _select_tensor_fields(
                batch,
                _VALIDATION_BATCH_KEYS,
                chunk_steps=chunk_steps,
                local_trajectories=local_trajectories,
                indices=probe_indices,
                trim_terminal_fields=True,
            )
            replay_forward = _select_tensor_fields(
                forward_inputs,
                _VALIDATION_REPLAY_FORWARD_KEYS,
                chunk_steps=chunk_steps,
                local_trajectories=local_trajectories,
                indices=probe_indices,
            )
            fpo_probe = "fpo_clean_action_normalized" in replay_forward
            required_replay_key = (
                "fpo_clean_joint_latent"
                if fpo_probe
                else "native_full_chains"
            )
            if required_replay_key not in replay_forward:
                raise KeyError(
                    "P0 replay probe requires forward_inputs."
                    f"{required_replay_key}."
                )
            replay_path = capture_dir / "replay_probe.pt"
            torch.save(
                {
                    "metadata": metadata
                    | {
                        "sample_count": probe_count,
                        "replay_forward_input_shapes": _shape_dict(
                            replay_forward
                        ),
                    },
                    "identifiers": {
                        key: value[probe_indices].contiguous()
                        for key, value in identifiers.items()
                    },
                    "batch": replay_batch,
                    "forward_inputs": replay_forward,
                },
                replay_path,
            )
            paths["replay_probe"] = replay_path

        self._append_manifest(
            {
                "kind": "p0_validation_batch",
                **metadata,
                "compact_path": compact_path.relative_to(
                    self.output_dir
                ).as_posix(),
                "replay_probe_count": probe_count,
                "replay_probe_indices": probe_indices.tolist(),
                "replay_probe_path": (
                    replay_path.relative_to(self.output_dir).as_posix()
                    if replay_path is not None
                    else None
                ),
                "replay_forward_input_shapes": _shape_dict(replay_forward),
            }
        )
        return paths

    def write_validation_parity_report(
        self,
        *,
        rank: int,
        global_step: int,
        report: dict[str, Any],
    ) -> Path | None:
        """Persist one rank-local E1 report without mutating the report."""

        if not self.enabled:
            return None
        report_dir = (
            self.output_dir
            / "e1_replay_parity"
            / f"global_step_{int(global_step):06d}"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"rank_{int(rank):04d}.json"
        payload = {
            "kind": "same_policy_replay_parity",
            "rank": int(rank),
            "global_step": int(global_step),
            **report,
        }
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        self._append_manifest(
            {
                "kind": "same_policy_replay_parity",
                "rank": int(rank),
                "global_step": int(global_step),
                "status": report.get("status"),
                "path": report_path.relative_to(
                    self.output_dir
                ).as_posix(),
            }
        )
        return report_path


    def write_validation_causal_report(
        self,
        *,
        rank: int,
        global_step: int,
        report: dict[str, Any],
    ) -> Path | None:
        """Persist one rank-local P1 causal-gate report."""

        if not self.enabled:
            return None
        report_dir = (
            self.output_dir
            / "p1_causal_gate"
            / f"global_step_{int(global_step):06d}"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"rank_{int(rank):04d}.json"
        payload = {
            "kind": "fpo_p1_causal_gate",
            "rank": int(rank),
            "global_step": int(global_step),
            **report,
        }
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        self._append_manifest(
            {
                "kind": "fpo_p1_causal_gate",
                "rank": int(rank),
                "global_step": int(global_step),
                "status": report.get("status"),
                "path": report_path.relative_to(
                    self.output_dir
                ).as_posix(),
            }
        )
        return report_path

    def write_actor_logprob_summary(
        self,
        *,
        rank: int,
        optimizer_step: int,
        microbatch_idx: int,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        global_step: int | None = None,
        action_sampling_seed: torch.Tensor | None = None,
        action_ratio_min: float | None = None,
        action_ratio_max: float | None = None,
        approx_kl_k2: float | None = None,
        failure_reason: str | None = None,
        single_action_dim: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        delta = (new_logprobs.detach() - old_logprobs.detach()).float()
        ratio = torch.clamp(delta, min=-20.0, max=20.0).exp()
        abs_delta = delta.abs()
        flat_index = int(abs_delta.reshape(-1).argmax().cpu())
        max_delta_index = [
            int(value)
            for value in torch.unravel_index(
                torch.tensor(flat_index), abs_delta.shape
            )
        ]
        summary = {
            "kind": "actor_logprob_summary",
            "rank": rank,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "microbatch_idx": microbatch_idx,
            "old_logprobs_shape": list(old_logprobs.shape),
            "new_logprobs_shape": list(new_logprobs.shape),
            "ratio_mean": float(ratio.mean().cpu()),
            "ratio_max": float(ratio.max().cpu()),
            "per_coordinate_ratio_mean": float(ratio.mean().cpu()),
            "per_coordinate_ratio_max": float(ratio.max().cpu()),
            "per_coordinate_logprob_abs_delta_max": float(abs_delta.max().cpu()),
            "per_coordinate_max_flat_index": flat_index,
            "per_coordinate_max_index": max_delta_index,
            "old_logprob_at_max_delta": float(
                old_logprobs.detach().float().reshape(-1)[flat_index].cpu()
            ),
            "new_logprob_at_max_delta": float(
                new_logprobs.detach().float().reshape(-1)[flat_index].cpu()
            ),
            "action_ratio_min": action_ratio_min,
            "action_ratio_max": action_ratio_max,
            "approx_kl_k2": approx_kl_k2,
            "failure_reason": failure_reason,
        }
        if single_action_dim is not None and int(single_action_dim) > 0:
            flat_action_coordinate = max_delta_index[-1]
            summary["max_delta_action_index"] = (
                flat_action_coordinate // int(single_action_dim)
            )
            summary["max_delta_action_coordinate"] = (
                flat_action_coordinate % int(single_action_dim)
            )
        if torch.is_tensor(action_sampling_seed):
            summary["action_sampling_seed"] = [
                int(value)
                for value in action_sampling_seed.detach().cpu().reshape(-1).tolist()
            ]
        actor_dir = self.output_dir / "actor_updates"
        actor_dir.mkdir(parents=True, exist_ok=True)
        rank_manifest = actor_dir / f"rank_{rank:04d}.jsonl"
        with rank_manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
        self._append_manifest(
            summary
            | {"path": rank_manifest.relative_to(self.output_dir).as_posix()}
        )

    def _append_manifest(self, record: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.output_dir / "manifest.jsonl"
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
