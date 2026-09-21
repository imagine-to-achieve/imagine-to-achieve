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

import csv
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import (
    append_metrics_history,
    compute_evaluate_metrics,
    print_metrics_table,
)
from rlinf.utils.runner_utils import check_progress

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union[
            "EmbodiedFSDPActor", "EmbodiedSACFSDPPolicy", "AsyncEmbodiedSACFSDPPolicy"
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"],
        env: Union["EnvWorker", "AsyncEnvWorker"],
        critic=None,
        reward=None,
        run_timer=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward
        self.weight_sync_interval = self.cfg.runner.weight_sync_interval
        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        actor_channel_cfg = self.cfg.runner.get("actor_channel", {})
        self.actor_channel = Channel.create(
            "Actor",
            maxsize=int(actor_channel_cfg.get("maxsize", 0)),
            distributed=bool(actor_channel_cfg.get("distributed", False)),
        )

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        self._post_update_frame_archive_metrics: dict[str, float] = {}
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)
        self.enable_per_worker_metric_log = bool(
            self.cfg.runner.get("per_worker_log", False)
        )

        # Async logging setup
        self.stop_logging = False
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()

    def _run_root(self) -> Path:
        return Path(str(self.cfg.runner.logger.log_path)) / str(
            self.cfg.runner.logger.experiment_name
        )

    @staticmethod
    def _atomic_write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)

    @staticmethod
    def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
        """Atomically publish a complete JSONL record set."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)

    def _post_update_evaluation_cfg(self):
        generic_cfg = self.cfg.get("post_update_evaluation", {})
        if bool(generic_cfg.get("enabled", False)):
            return generic_cfg
        duck_cfg = self.cfg.get("duck", {})
        evaluation_cfg = duck_cfg.get("evaluation", {})
        if not bool(duck_cfg.get("enabled", False)) or not bool(
            evaluation_cfg.get("enabled", False)
        ):
            return None
        return evaluation_cfg

    def _post_update_eval_seed_base(self, evaluation_cfg) -> int:
        """Cancel the worker step offset for a fixed held-out random stream."""
        seed_base = int(evaluation_cfg.get("seed_base", 42))
        if bool(evaluation_cfg.get("fixed_seeds", False)):
            seed_base -= len(evaluation_cfg.get("episode_ids", [])) * (int(self.global_step) - 1)
        return seed_base

    def _policy_state_identity(self, sync_metrics: dict) -> tuple[str, float | None]:
        """Build an auditable identifier for the synchronized action policy."""
        checksum = sync_metrics.get("sync/action_param_checksum")
        if hasattr(checksum, "item"):
            checksum = checksum.item()
        checksum_value = float(checksum) if checksum is not None else None
        base_manifest_sha256 = None
        manifest_value = self.cfg.algorithm.get("trajectory_records", {}).get(
            "checkpoint_checksum_manifest", None
        )
        if manifest_value:
            manifest_path = Path(str(manifest_value))
            if manifest_path.is_file():
                base_manifest_sha256 = hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest()
        payload = {
            "global_step": int(self.global_step),
            "action_param_checksum": checksum_value,
            "base_checkpoint_manifest_sha256": base_manifest_sha256,
        }
        state_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return state_id, checksum_value

    def _commit_post_update_eval_records(
        self,
        worker_records: list,
        *,
        sync_metrics: dict,
    ) -> list[dict]:
        """Validate and atomically commit exactly one record per eval episode."""
        self._post_update_frame_archive_metrics = {}
        evaluation_cfg = self._post_update_evaluation_cfg()
        if evaluation_cfg is None:
            return []
        records = []
        for worker_value in worker_records:
            if worker_value is None:
                continue
            if not isinstance(worker_value, list):
                raise TypeError(
                    "EnvWorker post-update evaluation records must be returned as lists."
                )
            records.extend(worker_value)

        expected_episodes = sorted(
            int(value) for value in evaluation_cfg.get("episode_ids", [])
        )
        expected_count = int(
            evaluation_cfg.get("expected_records_per_step", len(expected_episodes))
        )
        if len(records) != expected_count:
            raise RuntimeError(
                "Post-update evaluation is incomplete: "
                f"expected {expected_count} records, got {len(records)}."
            )
        policy_hash, action_checksum = self._policy_state_identity(sync_metrics)
        by_episode = {}
        seed_base = self._post_update_eval_seed_base(evaluation_cfg)
        for record in records:
            normalized = dict(record)
            episode = int(normalized["episode"])
            if episode in by_episode:
                raise RuntimeError(
                    f"Post-update evaluation duplicated episode {episode}."
                )
            normalized.update(
                {
                    "step": int(self.global_step - 1),
                    "global_step": int(self.global_step),
                    "update_index": int(self.global_step - 1),
                    "series": "eval_pre_training" if self.global_step == 0 else "eval_post_update",
                    "policy_stage": "pre_training" if self.global_step == 0 else "post_update",
                    "policy_hash": policy_hash,
                    "policy_action_param_checksum": action_checksum,
                    "model_hash": normalized.get("success_model_sha256"),
                    "success_probability_max": normalized.get(
                        "model_success_probability"
                    ),
                    "success": normalized.get("model_success"),
                    "trajectory_mse": normalized.get("video_mse"),
                    "lastframe_mse": normalized.get("terminal_goal_mse"),
                    "valid": bool(normalized.get("complete", False))
                    and normalized.get("error") is None,
                    "exception": normalized.get("error"),
                }
            )
            by_episode[episode] = normalized

        if sorted(by_episode) != expected_episodes:
            raise RuntimeError(
                "Post-update evaluation episode set changed: "
                f"expected {expected_episodes}, got {sorted(by_episode)}."
            )
        ordered_records = []
        for eval_index, episode in enumerate(expected_episodes):
            record = by_episode[episode]
            expected_seed = (
                seed_base
                + len(expected_episodes) * (int(self.global_step) - 1)
                + eval_index
            )
            if int(record.get("seed", -1)) != expected_seed:
                raise RuntimeError(
                    f"Episode {episode} eval seed {record.get('seed')} != "
                    f"{expected_seed}."
                )
            if not record["valid"] or not record.get("model_hash"):
                raise RuntimeError(
                    f"Episode {episode} produced an invalid evaluation record."
                )
            ordered_records.append(record)

        self._post_update_frame_archive_metrics = self._archive_final_side_frames(
            ordered_records, step=int(self.global_step), split="validation"
        )
        record_dir = Path(str(evaluation_cfg.record_dir))
        output_path = record_dir / ("baseline.jsonl" if self.global_step == 0 else f"step_{self.global_step - 1}.jsonl")
        self._atomic_write_jsonl(output_path, ordered_records)
        return ordered_records

    def _finalize_post_update_eval_mse_extremes(
        self, records: list[dict]
    ) -> dict[str, float]:
        """Copy the four global held-out success/MSE extrema into one directory."""
        evaluation_cfg = self._post_update_evaluation_cfg()
        if evaluation_cfg is None or not bool(
            evaluation_cfg.get("save_success_failure_mse_extremes", False)
        ):
            return {}
        if not records:
            raise ValueError("Eval MSE-extreme video selection received no records.")
        required_fields = {
            "episode",
            "rank",
            "env_id",
            "success",
            "trajectory_mse",
            "comparison_video_path",
        }
        for record in records:
            missing = required_fields - set(record)
            if missing:
                raise ValueError(
                    "Eval MSE-extreme record is missing fields "
                    f"{sorted(missing)} for episode {record.get('episode')}."
                )
            if not math.isfinite(float(record["trajectory_mse"])):
                raise ValueError(
                    "Eval MSE-extreme selection received non-finite MSE for "
                    f"episode {record['episode']}."
                )

        selected = self._select_success_failure_mse_extremes(records)
        expected_categories = (
            "success_mse_min",
            "success_mse_max",
            "failure_mse_min",
            "failure_mse_max",
        )
        missing_categories = [
            category for category in expected_categories if category not in selected
        ]
        if missing_categories and bool(
            evaluation_cfg.get("require_all_mse_extreme_categories", False)
        ):
            raise RuntimeError(
                "Held-out evaluation did not populate required MSE-extreme "
                f"categories: {missing_categories}."
            )

        default_base_dir = (
            self._run_root() / "videos" / "eval" / "mse_extremes"
        )
        base_dir = Path(
            str(evaluation_cfg.get("mse_extremes_base_dir", default_base_dir))
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        metrics = {
            "eval_success_trajectory_count": float(
                sum(bool(record["success"]) for record in records)
            ),
            "eval_failure_trajectory_count": float(
                sum(not bool(record["success"]) for record in records)
            ),
        }
        for category in expected_categories:
            winner = selected.get(category)
            if winner is None:
                metrics[f"{category}_video_saved"] = 0.0
                continue
            source_path = Path(str(winner["comparison_video_path"]))
            if not source_path.is_file() or source_path.stat().st_size <= 0:
                raise FileNotFoundError(
                    "Selected eval comparison video is missing or empty for "
                    f"{category}: {source_path}"
                )
            trajectory_mse = float(winner["trajectory_mse"])
            episode = int(winner["episode"])
            color = str(winner.get("color", "unknown"))
            final_name = (
                f"global_step_{self.global_step:06d}_{category}_"
                f"{color}_episode_{episode:06d}_"
                f"trajectory_mse_{trajectory_mse:.8f}.mp4"
            )
            final_path = base_dir / final_name
            temporary_path = base_dir / f".{final_name}.tmp"
            shutil.copy2(source_path, temporary_path)
            os.replace(temporary_path, final_path)
            metrics[category] = trajectory_mse
            metrics[f"{category}_video_saved"] = 1.0
            rows.append(
                {
                    "global_step": int(self.global_step),
                    "category": category,
                    "success": bool(winner["success"]),
                    "trajectory_mse": trajectory_mse,
                    "episode": episode,
                    "color": color,
                    "rank": int(winner["rank"]),
                    "env_id": int(winner["env_id"]),
                    "success_probability_max": winner.get(
                        "success_probability_max"
                    ),
                    "source_video_path": str(source_path),
                    "video_path": str(final_path),
                }
            )

        csv_path = base_dir / "mse_extremes_summary.csv"
        csv_temporary = csv_path.with_suffix(".csv.tmp")
        fieldnames = [
            "global_step",
            "category",
            "success",
            "trajectory_mse",
            "episode",
            "color",
            "rank",
            "env_id",
            "success_probability_max",
            "source_video_path",
            "video_path",
        ]
        with csv_temporary.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(csv_temporary, csv_path)
        self._atomic_write_json(
            base_dir / "mse_extremes_summary.json",
            {
                "global_step": int(self.global_step),
                "record_count": len(records),
                "success_count": int(metrics["eval_success_trajectory_count"]),
                "failure_count": int(metrics["eval_failure_trajectory_count"]),
                "selections": rows,
            },
        )
        metrics["eval_mse_extremes_videos_saved"] = float(len(rows))
        return metrics

    def _run_duck_analysis(self) -> bool:
        """Publish analysis and report whether this audit boundary committed."""
        duck_cfg = self.cfg.get("duck", {})
        analysis_cfg = duck_cfg.get("analysis", {})
        if not bool(duck_cfg.get("enabled", False)) or not bool(
            analysis_cfg.get("enabled", False)
        ):
            return True
        runner_cfg = self.cfg.get("runner", {})
        evaluation_cfg = duck_cfg.get("evaluation", {})
        default_interval = runner_cfg.get("val_check_interval", 1)
        evaluation_interval = int(
            analysis_cfg.get(
                "interval_updates",
                evaluation_cfg.get("interval_updates", default_interval),
            )
        )
        if evaluation_interval < 1:
            raise RuntimeError("Duck analysis evaluation interval must be positive")
        max_steps = int(getattr(self, "max_steps", self.global_step))
        if self.global_step % evaluation_interval != 0 and self.global_step != max_steps:
            self.logger.info(
                "Skipping duck analysis at non-evaluation boundary step %s (interval=%s).",
                self.global_step,
                evaluation_interval,
            )
            return True
        command = Path(str(analysis_cfg.command))
        args = [
            sys.executable,
            str(command),
            "--run-dir",
            str(self._run_root()),
            "--output-dir",
            str(analysis_cfg.output_root),
            "--through-step",
            str(self.global_step),
            "--expected-eval-per-step",
            str(
                self.cfg.duck.evaluation.get("expected_records_per_step", 8)
            ),
            "--evaluation-interval-updates",
            str(evaluation_interval),
        ]
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            if (
                completed.returncode == 3
                and bool(analysis_cfg.get("fail_on_evaluation_error", False))
            ):
                raise RuntimeError(
                    "Duck analysis input contract failed after step "
                    f"{self.global_step}: {detail}"
                )
            self.logger.warning(
                "Duck analysis failed after step %s (exit %s): %s",
                self.global_step,
                completed.returncode,
                detail,
            )
            return False
        if completed.stdout.strip():
            self.logger.info(completed.stdout.strip())
        return True

    def _write_run_status(self, state: str, **extra) -> None:
        status_path = Path(
            str(
                self.cfg.runner.get(
                    "status_file", self._run_root() / "status.json"
                )
            )
        )
        self._atomic_write_json(
            status_path,
            {
                "state": state,
                "updated_at_epoch": time.time(),
                "completed_updates": int(self.global_step),
                "max_updates": int(self.max_steps),
                "next_update_id": int(self.global_step),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                **extra,
            },
        )

    def _append_scalar_csv(self, name: str, step: int, metrics: dict) -> None:
        csv_cfg = self.cfg.runner.get("csv_metrics", {})
        if not bool(csv_cfg.get("enabled", False)):
            return
        path = self._run_root() / "metrics" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(("step", "metric", "value"))
            for metric, value in sorted(metrics.items()):
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
                writer.writerow((int(step), metric, scalar))

    def _remaining_walltime_seconds(self) -> float | None:
        walltime_seconds = float(
            self.cfg.runner.get("walltime_seconds", 0.0)
        )
        if walltime_seconds <= 0.0:
            return None
        allocation_start = float(
            self.cfg.runner.get("allocation_start_epoch", 0.0)
        )
        if allocation_start <= 0.0:
            allocation_start = getattr(self, "_runner_start_epoch", time.time())
        return walltime_seconds - (time.time() - allocation_start)

    def _should_stop_before_update(self) -> tuple[bool, float | None]:
        remaining = self._remaining_walltime_seconds()
        if remaining is None:
            return False, None
        reserve = float(
            self.cfg.runner.get(
                "min_remaining_before_update_seconds", 90 * 60
            )
        )
        return remaining < reserve, remaining

    def _duck_checkpoint_shard_count(self) -> int | None:
        """Return the strict Duck DCP shard count, or None for generic runs."""
        duck_cfg = self.cfg.get("duck", {})
        if not bool(duck_cfg.get("enabled", False)):
            return None
        topology_cfg = duck_cfg.get("topology", {})
        topology_shards = int(topology_cfg.get("total_gpus", 32))
        return int(
            self.cfg.runner.get(
                "checkpoint_expected_distcp_shards", topology_shards
            )
        )

    def _checkpoint_has_files(self, step: int) -> bool:
        checkpoint = self._run_root() / "checkpoints" / f"global_step_{step}"
        if not checkpoint.is_dir():
            return False
        manifest_path = checkpoint / "checkpoint_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if manifest.get("state") != "completed" or int(manifest.get("update", -1)) != step:
            return False
        expected_shards = self._duck_checkpoint_shard_count()
        if expected_shards is None:
            return any(
                candidate.is_file() for candidate in checkpoint.rglob("*")
            )
        if expected_shards <= 0:
            return False
        dcp = checkpoint / "actor" / "dcp_checkpoint"
        metadata = dcp / ".metadata"
        try:
            if not metadata.is_file() or metadata.stat().st_size <= 0:
                return False
            shards = tuple(dcp.glob("*.distcp"))
            return len(shards) == expected_shards and all(
                shard.is_file() and shard.stat().st_size > 0
                for shard in shards
            )
        except OSError:
            return False

    def _save_recovery_checkpoint_if_needed(self) -> bool:
        """Ensure the current recovery checkpoint is complete without pruning."""
        if self.global_step <= 0:
            return False
        if self._checkpoint_has_files(self.global_step):
            return True
        self._save_checkpoint()
        return self._checkpoint_has_files(self.global_step)

    def _commit_recovery_stop(
        self,
        stop_state: str,
        *,
        audit_committed: bool,
        **status_fields,
    ) -> bool:
        """Publish a recovery stop only with a complete, audited checkpoint."""
        if self.global_step <= 0 and stop_state == "walltime_stop":
            self._write_run_status(
                stop_state,
                recovery_checkpoint_ready=False,
                audit_committed=bool(audit_committed),
                no_completed_update=True,
                **status_fields,
            )
            return True
        checkpoint_ready = self._save_recovery_checkpoint_if_needed()
        if not checkpoint_ready or not audit_committed:
            reasons = []
            if not checkpoint_ready:
                reasons.append("current checkpoint is incomplete")
            if not audit_committed:
                reasons.append("current audit boundary is incomplete")
            self._write_run_status(
                "recovery_gate_incomplete",
                requested_stop_state=stop_state,
                recovery_checkpoint_ready=checkpoint_ready,
                audit_committed=bool(audit_committed),
                recovery_gate_failures=reasons,
                **status_fields,
            )
            return False
        self._write_run_status(
            stop_state,
            recovery_checkpoint_ready=True,
            audit_committed=True,
            **status_fields,
        )
        return self._prune_checkpoints_after_audit_commit(True)

    def _commit_training_success(self, *, audit_committed: bool) -> bool:
        """Publish terminal success only for a complete, audited checkpoint."""
        checkpoint_ready = self._checkpoint_has_files(self.global_step)
        if not checkpoint_ready or not audit_committed:
            reasons = []
            if not checkpoint_ready:
                reasons.append("final checkpoint is incomplete")
            if not audit_committed:
                reasons.append("final audit boundary is incomplete")
            self._write_run_status(
                "completion_gate_incomplete",
                requested_state="complete",
                final_checkpoint_ready=checkpoint_ready,
                audit_committed=bool(audit_committed),
                completion_gate_failures=reasons,
                first_update_validated=bool(self.global_step >= 1),
            )
            return False
        self._write_run_status(
            "complete",
            final_checkpoint_ready=True,
            audit_committed=True,
            first_update_validated=bool(self.global_step >= 1),
        )
        self._write_success_marker()
        return True

    def _write_success_marker(self) -> None:
        success_path = Path(
            str(
                self.cfg.runner.get(
                    "success_file", self._run_root() / "SUCCESS"
                )
            )
        )
        self._atomic_write_json(
            success_path,
            {
                "state": "complete",
                "completed_updates": int(self.global_step),
                "max_updates": int(self.max_steps),
                "completed_at_epoch": time.time(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
        )

    def _validate_first_update_smoke(self, actor_metrics: list[dict]) -> None:
        """Gate continuation on one full production update."""
        smoke_cfg = self.cfg.runner.get("first_update_smoke", {})
        if not bool(smoke_cfg.get("enabled", False)) or self.global_step != 1:
            return
        merged = self._aggregate_numeric_metrics(actor_metrics)
        required_metrics = tuple(
            smoke_cfg.get(
                "required_actor_metrics",
                (
                    "actor/total_loss",
                    "actor/grad_norm",
                    "action/ratio_mean",
                    "actor/optimizer_steps_per_update",
                    "hardware/gpu_peak_reserved_fraction_max",
                ),
            )
        )
        missing = [name for name in required_metrics if name not in merged]
        non_finite = [
            name
            for name in required_metrics
            if name in merged and not math.isfinite(float(merged[name]))
        ]
        expected_steps = int(
            smoke_cfg.get("expected_optimizer_steps_per_update", 4)
        )
        actual_steps = int(
            round(float(merged.get("actor/optimizer_steps_per_update", -1)))
        )
        peak = float(
            merged.get("hardware/gpu_peak_reserved_fraction_max", math.inf)
        )
        maximum_peak = float(
            smoke_cfg.get("max_peak_memory_fraction", 0.9)
        )
        failures = []
        if missing:
            failures.append(f"missing metrics: {missing}")
        if non_finite:
            failures.append(f"non-finite metrics: {non_finite}")
        if actual_steps != expected_steps:
            failures.append(
                f"optimizer steps {actual_steps} != {expected_steps}"
            )
        enforce_peak_limit = bool(
            smoke_cfg.get("enforce_peak_memory_limit", True)
        )
        if enforce_peak_limit and peak >= maximum_peak:
            failures.append(
                f"peak reserved GPU memory {peak:.4f} is not below "
                f"{maximum_peak:.4f}"
            )
        if failures:
            recovery_checkpoint_ready = (
                self._save_recovery_checkpoint_if_needed()
            )
            self._write_run_status(
                "first_update_smoke_failed",
                first_update_validated=False,
                recovery_checkpoint_ready=recovery_checkpoint_ready,
                peak_reserved_memory_fraction=peak,
                failures=failures,
            )
            raise RuntimeError(
                "First production update failed acceptance gate: "
                + "; ".join(failures)
            )
        self._write_run_status(
            "running",
            first_update_validated=True,
            peak_reserved_memory_fraction=peak,
        )

    def _log_worker(self):
        """Background thread for processing log messages."""
        while not self.stop_logging:
            try:
                # Wait for log message with timeout
                log_func, args = self.log_queue.get(timeout=0.1)
                log_func(*args)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Logging error: {e}")
                continue

    def print_metrics_table_async(
        self,
        step: int,
        total_steps: int,
        start_time: float,
        metrics: dict,
        start_step: int = 0,
    ):
        """Async version that puts table printing in queue."""
        self.log_queue.put(
            (print_metrics_table, (step, total_steps, start_time, metrics, start_step))
        )

    @staticmethod
    def _pop_best_rollout_records(env_results: list) -> list[dict[str, object]]:
        """Remove internal per-env scores and preserve exact terminal frames."""
        def _values(value):
            if value is None:
                return None
            if isinstance(value, list) and len(value) == 1:
                inner = value[0]
                if hasattr(inner, "detach"):
                    value = inner
                elif isinstance(inner, (list, tuple)):
                    return list(inner)
            if hasattr(value, "detach"):
                return value.detach().cpu().reshape(-1).tolist()
            return list(value)

        def _sha256_values(value):
            if value is None:
                return None
            if (
                isinstance(value, list)
                and len(value) == 1
                and hasattr(value[0], "detach")
            ):
                value = value[0]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            encoded = np.asarray(value)
            if encoded.ndim == 3 and int(encoded.shape[0]) == 1:
                encoded = encoded[0]
            if encoded.ndim == 1:
                encoded = encoded.reshape(1, -1)
            expected_width = hashlib.sha256().digest_size
            if encoded.ndim != 2 or int(encoded.shape[1]) != expected_width:
                raise ValueError(
                    "Final-side-frame SHA-256 batch must be [B,32] uint8, got "
                    f"{tuple(encoded.shape)}."
                )
            if encoded.dtype != np.uint8:
                raise ValueError(
                    "Final-side-frame SHA-256 batch must be uint8, got "
                    f"{encoded.dtype}."
                )
            return [bytes(row.tolist()).hex() for row in encoded]

        def _frame_batch(value):
            if value is None:
                return None
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            frames = np.asarray(value)
            if frames.ndim == 5 and int(frames.shape[0]) == 1:
                frames = frames[0]
            if frames.ndim != 4 or int(frames.shape[-1]) != 3:
                raise ValueError(
                    "Final-side-frame batch must be [B,H,W,3], got "
                    f"{tuple(frames.shape)}."
                )
            if frames.dtype != np.uint8:
                raise ValueError(
                    f"Final-side-frame batch must be uint8, got {frames.dtype}."
                )
            return np.ascontiguousarray(frames)

        records: list[dict[str, object]] = []
        for rank, rank_metrics in enumerate(env_results):
            if rank_metrics is None:
                continue
            rewards = rank_metrics.pop("_best_rollout_episode_rewards", None)
            if rewards is None:
                continue
            reward_values = _values(rewards)
            chunk_reward_max_values = _values(
                rank_metrics.pop("_best_rollout_chunk_reward_max", None)
            )
            chunk_reward_mean_values = _values(
                rank_metrics.pop("_best_rollout_chunk_reward_mean", None)
            )
            trajectory_mse_values = _values(
                rank_metrics.pop("_best_rollout_trajectory_mse", None)
            )
            success_values = _values(
                rank_metrics.pop("_best_rollout_success", None)
            )
            if (trajectory_mse_values is None) != (success_values is None):
                raise ValueError("Rollout MSE and success summaries must appear together.")
            rollout_uids = _values(
                rank_metrics.pop("_best_rollout_rollout_uid", None)
            )
            group_ids = _values(
                rank_metrics.pop("_best_rollout_global_group_id", None)
            )
            member_ids = _values(
                rank_metrics.pop("_best_rollout_group_member_id", None)
            )
            reset_episodes = _values(
                rank_metrics.pop("_best_rollout_reset_episode", None)
            )
            final_side_frames = _frame_batch(
                rank_metrics.pop("_best_rollout_final_side_frame", None)
            )
            final_side_sha256 = _sha256_values(
                rank_metrics.pop("_best_rollout_final_side_frame_sha256", None)
            )
            final_side_episodes = _values(
                rank_metrics.pop("_best_rollout_final_side_frame_episode", None)
            )
            frame_fields = (
                final_side_frames, final_side_sha256, final_side_episodes
            )
            if any(value is None for value in frame_fields) and not all(
                value is None for value in frame_fields
            ):
                raise ValueError(
                    "Final-side-frame tensor, hash, and episode must appear together."
                )
            batch_size = len(reward_values)
            named_values = {
                "chunk_reward_max": chunk_reward_max_values,
                "chunk_reward_mean": chunk_reward_mean_values,
                "trajectory_mse": trajectory_mse_values,
                "success": success_values,
                "rollout_uid": rollout_uids,
                "group_id": group_ids,
                "member_id": member_ids,
                "reset_episode": reset_episodes,
                "final_side_frame_sha256": final_side_sha256,
                "final_side_frame_episode": final_side_episodes,
            }
            for name, values in named_values.items():
                if values is not None and len(values) != batch_size:
                    raise ValueError(
                        f"Rollout field {name} has {len(values)} values; "
                        f"expected {batch_size}."
                    )
            if final_side_frames is not None and int(final_side_frames.shape[0]) != batch_size:
                raise ValueError(
                    "Final-side-frame batch size does not match rewards: "
                    f"{final_side_frames.shape[0]} != {batch_size}."
                )
            for env_id, reward in enumerate(reward_values):
                record: dict[str, object] = {
                    "rank": rank,
                    "env_id": env_id,
                    "reward": float(reward),
                }
                if chunk_reward_max_values is not None:
                    record["chunk_reward_max"] = float(
                        chunk_reward_max_values[env_id]
                    )
                    record["chunk_reward_mean"] = float(
                        chunk_reward_mean_values[env_id]
                    )
                if trajectory_mse_values is not None:
                    record["trajectory_mse"] = float(trajectory_mse_values[env_id])
                    record["success"] = bool(success_values[env_id])
                if rollout_uids is not None:
                    record["rollout_uid"] = int(rollout_uids[env_id])
                if group_ids is not None:
                    record["group_id"] = int(group_ids[env_id])
                if member_ids is not None:
                    record["member_id"] = int(member_ids[env_id])
                if reset_episodes is not None:
                    record["episode"] = int(reset_episodes[env_id])
                if final_side_frames is not None:
                    recorded_episode = int(final_side_episodes[env_id])
                    if (
                        "episode" in record
                        and int(record["episode"]) != recorded_episode
                    ):
                        raise ValueError(
                            "Final-side-frame episode does not match rollout metadata."
                        )
                    frame = np.ascontiguousarray(final_side_frames[env_id])
                    digest = hashlib.sha256(frame.tobytes(order="C")).hexdigest()
                    expected_digest = str(final_side_sha256[env_id])
                    if digest != expected_digest:
                        raise ValueError(
                            "Final-side-frame in-memory SHA-256 mismatch for "
                            f"rank={rank}, env={env_id}."
                        )
                    record.update(
                        {
                            "episode": recorded_episode,
                            "_final_side_frame": frame,
                            "final_side_frame_sha256": expected_digest,
                        }
                    )
                records.append(record)
        return records

    def _write_best_rollout_csv(
        self,
        *,
        step: int,
        group_summaries: list[dict[str, float | int]],
        global_best: float,
        global_mean: float,
        winner_rank: int,
        winner_env: int,
        winner_rollout_uid: int | None,
        winner_group_id: int | None,
        winner_member_id: int | None,
        video_path: str,
    ) -> None:
        """Write one idempotent set of per-group rows for a rollout step."""
        output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
        )
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "best_rollout_summary.csv")
        fieldnames = [
            "global_step",
            "group_id",
            "best_reward",
            "mean_reward",
            "chunk_reward_max",
            "chunk_reward_mean",
            "global_best_reward",
            "global_mean_reward",
            "winner_rank",
            "winner_env",
            "winner_rollout_uid",
            "winner_group_id",
            "winner_member_id",
            "video_path",
        ]
        previous_rows = []
        if os.path.isfile(csv_path):
            try:
                with open(csv_path, newline="") as csv_file:
                    previous_rows = [
                        row
                        for row in csv.DictReader(csv_file)
                        if int(row["global_step"]) != step
                    ]
            except Exception as exc:
                self.logger.warning(
                    "Ignoring an unreadable best-rollout CSV %s: %s", csv_path, exc
                )
                previous_rows = []

        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(previous_rows)
            for group in group_summaries:
                writer.writerow(
                    {
                        "global_step": step,
                        "group_id": group["group_id"],
                        "best_reward": group["best_reward"],
                        "mean_reward": group["mean_reward"],
                        "chunk_reward_max": group.get("chunk_reward_max", ""),
                        "chunk_reward_mean": group.get("chunk_reward_mean", ""),
                        "global_best_reward": global_best,
                        "global_mean_reward": global_mean,
                        "winner_rank": winner_rank,
                        "winner_env": winner_env,
                        "winner_rollout_uid": winner_rollout_uid,
                        "winner_group_id": winner_group_id,
                        "winner_member_id": winner_member_id,
                        "video_path": video_path,
                    }
                )

    def _finalize_best_rollout(
        self, records: list[dict[str, float | int]], step: int
    ) -> dict[str, float]:
        """Promote the global best candidate and compute GRPO reward metrics."""
        if not records:
            return {}
        group_size = int(self.cfg.algorithm.group_size)
        explicit_groups = all("group_id" in record for record in records)
        if explicit_groups:
            records = sorted(
                records,
                key=lambda item: (item["group_id"], item["member_id"]),
            )
        else:
            records = sorted(records, key=lambda item: (item["rank"], item["env_id"]))
        if len(records) % group_size != 0:
            raise ValueError(
                f"Best-rollout record count {len(records)} is not divisible by "
                f"group_size={group_size}."
            )

        rewards = [float(item["reward"]) for item in records]
        global_best = max(rewards)
        global_mean = sum(rewards) / len(rewards)
        winner = max(records, key=lambda item: float(item["reward"]))
        winner_rank = int(winner["rank"])
        winner_env = int(winner["env_id"])
        winner_rollout_uid = (
            int(winner["rollout_uid"]) if explicit_groups else None
        )
        winner_group_id = int(winner["group_id"]) if explicit_groups else None
        winner_member_id = (
            int(winner["member_id"]) if explicit_groups else None
        )
        group_summaries = []
        metrics = {
            "episode_reward_best": global_best,
            "episode_reward_mean": global_mean,
        }
        for group_index, start in enumerate(range(0, len(records), group_size)):
            group_records = records[start : start + group_size]
            if explicit_groups:
                group_ids = {int(item["group_id"]) for item in group_records}
                member_ids = [int(item["member_id"]) for item in group_records]
                if len(group_ids) != 1 or member_ids != list(range(group_size)):
                    raise ValueError(
                        "Best-rollout records contain an incomplete explicit group: "
                        f"group_ids={group_ids}, member_ids={member_ids}"
                    )
                group_id = next(iter(group_ids))
            else:
                group_id = group_index
            group_rewards = [float(item["reward"]) for item in group_records]
            group_best = max(group_rewards)
            group_mean = sum(group_rewards) / len(group_rewards)
            group_summary = {
                "group_id": group_id,
                "best_reward": group_best,
                "mean_reward": group_mean,
            }
            has_chunk_stats = all(
                "chunk_reward_max" in item for item in group_records
            )
            if has_chunk_stats:
                group_summary["chunk_reward_max"] = max(
                    float(item["chunk_reward_max"]) for item in group_records
                )
                group_chunk_means = [
                    float(item["chunk_reward_mean"]) for item in group_records
                ]
                group_summary["chunk_reward_mean"] = sum(
                    group_chunk_means
                ) / len(group_chunk_means)
            group_summaries.append(group_summary)
            metrics[f"group_{group_index}_reward_best"] = group_best
            metrics[f"group_{group_index}_reward_mean"] = group_mean

        video_cfg = self.cfg.env.train.video_cfg
        if not video_cfg.get("save_best_rollout", False):
            return metrics
        best_base_dir = str(
            video_cfg.get(
                "best_rollout_base_dir",
                os.path.join(str(video_cfg.video_base_dir), "best_rollout"),
            )
        )
        candidate_step_dir = os.path.join(
            f"{best_base_dir}_candidates", f"step_{step:06d}"
        )
        winner_candidate_name = f"rank_{winner_rank}.mp4"
        if winner_rollout_uid is not None:
            winner_candidate_name = (
                f"rank_{winner_rank}_uid_{winner_rollout_uid}.mp4"
            )
        winner_candidate = os.path.join(
            candidate_step_dir, winner_candidate_name
        )
        os.makedirs(best_base_dir, exist_ok=True)
        reward_label = f"{global_best:.6f}"
        if explicit_groups:
            artifact_label = (
                f"group_{winner_group_id}_member_{winner_member_id}_"
                f"uid_{winner_rollout_uid}_"
            )
        else:
            artifact_label = ""
        final_video_path = os.path.join(
            best_base_dir,
            f"step_{step:06d}_{artifact_label}best_reward_{reward_label}.mp4",
        )
        saved = 0.0
        try:
            if not os.path.isfile(winner_candidate):
                raise FileNotFoundError(winner_candidate)
            os.replace(winner_candidate, final_video_path)
            saved = 1.0
        except Exception as exc:
            self.logger.warning(
                "Failed to promote best-rollout video for step %s: %s", step, exc
            )
            final_video_path = ""
        finally:
            if os.path.isdir(candidate_step_dir):
                shutil.rmtree(candidate_step_dir)
        if saved:
            step_prefix = f"step_{step:06d}_"
            final_video_name = os.path.basename(final_video_path)
            for filename in os.listdir(best_base_dir):
                if (
                    filename.startswith(step_prefix)
                    and filename.endswith(".mp4")
                    and filename != final_video_name
                ):
                    stale_path = os.path.join(best_base_dir, filename)
                    try:
                        os.remove(stale_path)
                    except OSError as exc:
                        self.logger.warning(
                            "Failed to remove stale best-rollout video %s: %s",
                            stale_path,
                            exc,
                        )
        metrics["best_rollout_video_saved"] = saved

        self._write_best_rollout_csv(
            step=step,
            group_summaries=group_summaries,
            global_best=global_best,
            global_mean=global_mean,
            winner_rank=winner_rank,
            winner_env=winner_env,
            winner_rollout_uid=winner_rollout_uid,
            winner_group_id=winner_group_id,
            winner_member_id=winner_member_id,
            video_path=final_video_path,
        )
        return metrics

    def _finalize_worst_rollout(
        self, records: list[dict[str, float | int]], step: int
    ) -> dict[str, float]:
        """Promote the global worst candidate (mirrors _finalize_best_rollout)."""
        if not records or not self.cfg.env.train.video_cfg.get(
            "save_worst_rollout", False
        ):
            return {}

        rewards = [float(item["reward"]) for item in records]
        global_worst = min(rewards)
        loser = min(records, key=lambda item: float(item["reward"]))
        loser_rank = int(loser["rank"])
        explicit_groups = all("group_id" in record for record in records)
        loser_rollout_uid = (
            int(loser["rollout_uid"]) if explicit_groups else None
        )
        loser_group_id = int(loser["group_id"]) if explicit_groups else None
        loser_member_id = int(loser["member_id"]) if explicit_groups else None

        video_cfg = self.cfg.env.train.video_cfg
        worst_base_dir = str(
            video_cfg.get(
                "worst_rollout_base_dir",
                os.path.join(str(video_cfg.video_base_dir), "worst_rollout"),
            )
        )
        candidate_step_dir = os.path.join(
            f"{worst_base_dir}_candidates", f"step_{step:06d}"
        )
        loser_candidate_name = f"rank_{loser_rank}.mp4"
        if loser_rollout_uid is not None:
            loser_candidate_name = (
                f"rank_{loser_rank}_uid_{loser_rollout_uid}.mp4"
            )
        loser_candidate = os.path.join(candidate_step_dir, loser_candidate_name)
        os.makedirs(worst_base_dir, exist_ok=True)
        reward_label = f"{global_worst:.6f}"
        if explicit_groups:
            artifact_label = (
                f"group_{loser_group_id}_member_{loser_member_id}_"
                f"uid_{loser_rollout_uid}_"
            )
        else:
            artifact_label = ""
        final_video_path = os.path.join(
            worst_base_dir,
            f"step_{step:06d}_{artifact_label}worst_reward_{reward_label}.mp4",
        )
        saved = 0.0
        try:
            if not os.path.isfile(loser_candidate):
                raise FileNotFoundError(loser_candidate)
            os.replace(loser_candidate, final_video_path)
            saved = 1.0
        except Exception as exc:
            self.logger.warning(
                "Failed to promote worst-rollout video for step %s: %s", step, exc
            )
            final_video_path = ""
        finally:
            if os.path.isdir(candidate_step_dir):
                shutil.rmtree(candidate_step_dir)
        if saved:
            step_prefix = f"step_{step:06d}_"
            final_video_name = os.path.basename(final_video_path)
            for filename in os.listdir(worst_base_dir):
                if (
                    filename.startswith(step_prefix)
                    and filename.endswith(".mp4")
                    and filename != final_video_name
                ):
                    stale_path = os.path.join(worst_base_dir, filename)
                    try:
                        os.remove(stale_path)
                    except OSError as exc:
                        self.logger.warning(
                            "Failed to remove stale worst-rollout video %s: %s",
                            stale_path,
                            exc,
                        )
        return {
            "episode_reward_worst": global_worst,
            "worst_rollout_video_saved": saved,
        }

    @staticmethod
    def _select_success_failure_mse_extremes(
        records: list[dict],
    ) -> dict[str, dict]:
        """Select one deterministic global record for each populated category."""
        specs = (
            ("success_mse_min", True, True),
            ("success_mse_max", True, False),
            ("failure_mse_min", False, True),
            ("failure_mse_max", False, False),
        )
        selected = {}
        for category, success_value, use_minimum in specs:
            eligible = [
                record
                for record in records
                if bool(record["success"]) is success_value
            ]
            if not eligible:
                continue
            if use_minimum:
                winner = min(
                    eligible,
                    key=lambda item: (
                        float(item["trajectory_mse"]),
                        int(item["rank"]),
                        int(item["env_id"]),
                    ),
                )
            else:
                winner = min(
                    eligible,
                    key=lambda item: (
                        -float(item["trajectory_mse"]),
                        int(item["rank"]),
                        int(item["env_id"]),
                    ),
                )
            selected[category] = winner
        return selected

    @staticmethod
    def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> str:
        """Atomically publish one compressed, pickle-free NPZ archive."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".npz", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            np.savez_compressed(temporary, **arrays)
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise RuntimeError(f"NPZ archive was not written: {temporary}")
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _archive_final_side_frames(
        self,
        records: list[dict],
        *,
        step: int,
        split: str,
    ) -> dict[str, float]:
        """Pack every exact terminal side frame and all four MSE extrema."""
        artifact_cfg = self.cfg.algorithm.get("trajectory_records", {})
        if not bool(artifact_cfg.get("save_final_side_frames", False)):
            return {}
        if not records:
            raise ValueError("Final-side-frame archiving received no records.")
        required = {
            "rank",
            "env_id",
            "episode",
            "success",
            "trajectory_mse",
            "_final_side_frame",
            "final_side_frame_sha256",
        }
        if split == "training":
            required.add("rollout_uid")
        ordered = sorted(
            records,
            key=lambda item: (
                int(item.get("rollout_uid", item["episode"])),
                int(item["rank"]),
                int(item["env_id"]),
            ),
        )
        expected_key = (
            "expected_train_records_per_step"
            if split == "training"
            else "expected_validation_records_per_step"
        )
        expected_count = artifact_cfg.get(expected_key, None)
        if expected_count is not None and len(ordered) != int(expected_count):
            raise ValueError(
                f"{split} final-side-frame count {len(ordered)} != "
                f"configured {int(expected_count)}."
            )
        identities = set()
        frames = []
        for record in ordered:
            missing = required - set(record)
            if missing:
                raise ValueError(
                    "Final-side-frame record is missing "
                    "{} for rank={}, env={}.".format(
                        sorted(missing), record.get("rank"), record.get("env_id")
                    )
                )
            identity = (
                int(record["rank"]),
                int(record["env_id"]),
                int(record.get("rollout_uid", record["episode"])),
            )
            if identity in identities:
                raise ValueError(f"Duplicate final-side-frame identity: {identity}")
            identities.add(identity)
            mse = float(record["trajectory_mse"])
            if not math.isfinite(mse):
                raise ValueError("Final-side-frame record has non-finite MSE.")
            frame = np.asarray(record["_final_side_frame"])
            if frame.ndim != 3 or int(frame.shape[-1]) != 3:
                raise ValueError(
                    f"Terminal side frame must be [H,W,3], got {frame.shape}."
                )
            if frame.dtype != np.uint8:
                raise ValueError(
                    f"Terminal side frame must be uint8, got {frame.dtype}."
                )
            frame = np.ascontiguousarray(frame)
            digest = hashlib.sha256(frame.tobytes(order="C")).hexdigest()
            if digest != str(record["final_side_frame_sha256"]):
                raise ValueError(
                    "Final-side-frame SHA-256 mismatch for rank={}, env={}.".format(
                        record["rank"], record["env_id"]
                    )
                )
            frames.append(frame)
        frame_shapes = {tuple(frame.shape) for frame in frames}
        if len(frame_shapes) != 1:
            raise ValueError(
                f"Terminal side frames changed shape within one step: {frame_shapes}."
            )
        frame_array = np.stack(frames, axis=0)

        categories = (
            "success_mse_min",
            "success_mse_max",
            "failure_mse_min",
            "failure_mse_max",
        )
        save_extremes = bool(
            artifact_cfg.get("save_mse_extreme_final_frames", False)
        )
        selected = (
            self._select_success_failure_mse_extremes(ordered)
            if save_extremes
            else {}
        )
        record_indices = {id(record): index for index, record in enumerate(ordered)}
        extreme_available = np.zeros(len(categories), dtype=np.bool_)
        extreme_indices = np.full(len(categories), -1, dtype=np.int64)
        extreme_mse = np.full(len(categories), np.nan, dtype=np.float64)
        extreme_frames = np.zeros(
            (len(categories), *frame_array.shape[1:]), dtype=np.uint8
        )
        extreme_frame_sha256 = np.full(len(categories), "", dtype="U64")
        metrics: dict[str, float] = {
            "rollout_final_side_frames_saved": float(len(ordered))
        }
        for category_index, category in enumerate(categories):
            winner = selected.get(category)
            if winner is None:
                metrics[f"{category}_final_frame_saved"] = 0.0
                continue
            archive_index = int(record_indices[id(winner)])
            extreme_available[category_index] = True
            extreme_indices[category_index] = archive_index
            extreme_mse[category_index] = float(winner["trajectory_mse"])
            extreme_frames[category_index] = frame_array[archive_index]
            extreme_frame_sha256[category_index] = str(
                winner["final_side_frame_sha256"]
            )
            metrics[category] = float(winner["trajectory_mse"])
            metrics[f"{category}_final_frame_saved"] = 1.0
        metrics["mse_extreme_final_frames_saved"] = float(
            extreme_available.sum()
        )

        root = Path(
            str(
                artifact_cfg.get(
                    "final_side_frame_dir",
                    self._run_root() / "rollout_final_side_frames",
                )
            )
        )
        step_dir = root / split / f"step_{int(step):06d}"
        archive_path = (step_dir / "final_side_frames.npz").resolve()

        def int_values(name: str, default: int = -1) -> np.ndarray:
            return np.asarray(
                [int(record.get(name, default)) for record in ordered],
                dtype=np.int64,
            )

        archive_sha256 = self._atomic_write_npz(
            archive_path,
            schema_version=np.asarray([1], dtype=np.int64),
            global_step=np.asarray([int(step)], dtype=np.int64),
            split=np.asarray([split]),
            camera_key=np.asarray(["observation.images.d405_1_rgb"]),
            ctrl_world_view_index=np.asarray([2], dtype=np.int64),
            frame_encoding=np.asarray(["lossless_uint8_rgb_hwc"]),
            frame_sha256_scope=np.asarray(["contiguous_uint8_rgb_bytes"]),
            frames=frame_array,
            frame_sha256=np.asarray(
                [str(record["final_side_frame_sha256"]) for record in ordered],
                dtype="U64",
            ),
            rank=int_values("rank"),
            env_id=int_values("env_id"),
            episode=int_values("episode"),
            rollout_uid=int_values("rollout_uid"),
            group_id=int_values("group_id"),
            member_id=int_values("member_id"),
            success=np.asarray(
                [bool(record["success"]) for record in ordered], dtype=np.bool_
            ),
            trajectory_mse=np.asarray(
                [float(record["trajectory_mse"]) for record in ordered],
                dtype=np.float64,
            ),
            reward=np.asarray(
                [float(record.get("reward", np.nan)) for record in ordered],
                dtype=np.float64,
            ),
            extreme_categories=np.asarray(categories),
            extreme_available=extreme_available,
            extreme_archive_index=extreme_indices,
            extreme_trajectory_mse=extreme_mse,
            extreme_frames=extreme_frames,
            extreme_frame_sha256=extreme_frame_sha256,
        )
        for archive_index, record in enumerate(ordered):
            record.pop("_final_side_frame", None)
            record.update(
                {
                    "final_side_frame_archive_path": str(archive_path),
                    "final_side_frame_archive_sha256": archive_sha256,
                    "final_side_frame_archive_index": int(archive_index),
                    "final_side_frame_camera_key": (
                        "observation.images.d405_1_rgb"
                    ),
                    "final_side_frame_ctrl_world_view_index": 2,
                    "final_side_frame_encoding": "lossless_uint8_rgb_hwc_npz",
                }
            )
        metrics["rollout_final_side_frame_archive_bytes"] = float(
            archive_path.stat().st_size
        )
        metrics["rollout_final_side_frame_archives_saved"] = 1.0
        return metrics

    def _finalize_rollout_final_side_frames(
        self, records: list[dict], step: int
    ) -> dict[str, float]:
        return self._archive_final_side_frames(
            records, step=step, split="training"
        )

    def _finalize_success_failure_mse_extremes(
        self, records: list[dict], step: int
    ) -> dict[str, float]:
        """Promote global success/failure trajectory-MSE extrema and audit them."""
        video_cfg = self.cfg.env.train.video_cfg
        if not video_cfg.get("save_success_failure_mse_extremes", False):
            return {}
        if not records:
            raise ValueError("MSE-extreme video selection received no rollout records.")
        if any(
            "trajectory_mse" not in record or "success" not in record
            for record in records
        ):
            raise ValueError(
                "MSE-extreme video selection requires MSE and success for every trajectory."
            )
        if any(not math.isfinite(float(record["trajectory_mse"])) for record in records):
            raise ValueError("MSE-extreme video selection received non-finite MSE.")

        specs = (
            ("success_mse_min", True, "min"),
            ("success_mse_max", True, "max"),
            ("failure_mse_min", False, "min"),
            ("failure_mse_max", False, "max"),
        )
        selected = self._select_success_failure_mse_extremes(records)
        default_video_base = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            "videos",
            "train",
        )
        base_dir = str(
            video_cfg.get(
                "mse_extremes_base_dir",
                os.path.join(
                    str(video_cfg.get("video_base_dir", default_video_base)),
                    "mse_extremes",
                ),
            )
        )
        candidate_step_dir = os.path.join(
            f"{base_dir}_candidates", f"step_{step:06d}"
        )
        os.makedirs(base_dir, exist_ok=True)

        success_count = sum(bool(record["success"]) for record in records)
        metrics = {
            "success_trajectory_count": float(success_count),
            "failure_trajectory_count": float(len(records) - success_count),
        }
        rows = []
        kept_filenames = set()
        saved_count = 0
        try:
            for category, success_value, extreme in specs:
                winner = selected.get(category)
                if winner is None:
                    metrics[f"{category}_video_saved"] = 0.0
                    rows.append(
                        {
                            "global_step": step,
                            "category": category,
                            "success": success_value,
                            "extreme": extreme,
                            "available": False,
                            "trajectory_mse": "",
                            "rank": "",
                            "env_id": "",
                            "rollout_uid": "",
                            "group_id": "",
                            "member_id": "",
                            "video_saved": False,
                            "video_path": "",
                        }
                    )
                    continue

                winner_rank = int(winner["rank"])
                winner_env = int(winner["env_id"])
                trajectory_mse = float(winner["trajectory_mse"])
                rollout_uid = winner.get("rollout_uid")
                candidate_name = f"{category}_rank_{winner_rank}.mp4"
                if rollout_uid is not None:
                    candidate_name = (
                        f"{category}_rank_{winner_rank}_uid_{int(rollout_uid)}.mp4"
                    )
                candidate_path = os.path.join(candidate_step_dir, candidate_name)
                identity_label = ""
                if rollout_uid is not None:
                    identity_label = (
                        f"group_{int(winner['group_id'])}_"
                        f"member_{int(winner['member_id'])}_"
                        f"uid_{int(rollout_uid)}_"
                    )
                final_name = (
                    f"step_{step:06d}_{category}_{identity_label}"
                    f"trajectory_mse_{trajectory_mse:.8f}.mp4"
                )
                final_path = os.path.join(base_dir, final_name)
                saved = 0.0
                try:
                    if not os.path.isfile(candidate_path):
                        raise FileNotFoundError(candidate_path)
                    os.replace(candidate_path, final_path)
                    saved = 1.0
                    saved_count += 1
                    kept_filenames.add(final_name)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to promote {category} video for step {step}: {exc}"
                    ) from exc

                metrics[category] = trajectory_mse
                metrics[f"{category}_video_saved"] = saved
                rows.append(
                    {
                        "global_step": step,
                        "category": category,
                        "success": success_value,
                        "extreme": extreme,
                        "available": True,
                        "trajectory_mse": trajectory_mse,
                        "rank": winner_rank,
                        "env_id": winner_env,
                        "rollout_uid": rollout_uid if rollout_uid is not None else "",
                        "group_id": winner.get("group_id", ""),
                        "member_id": winner.get("member_id", ""),
                        "video_saved": bool(saved),
                        "video_path": final_path,
                    }
                )
        finally:
            if os.path.isdir(candidate_step_dir):
                shutil.rmtree(candidate_step_dir)

        step_prefix = f"step_{step:06d}_"
        for filename in os.listdir(base_dir):
            if (
                filename.startswith(step_prefix)
                and filename.endswith(".mp4")
                and filename not in kept_filenames
            ):
                try:
                    os.remove(os.path.join(base_dir, filename))
                except OSError as exc:
                    self.logger.warning(
                        "Failed to remove stale MSE-extreme video %s: %s",
                        filename,
                        exc,
                    )

        csv_path = os.path.join(base_dir, "mse_extremes_summary.csv")
        fieldnames = [
            "global_step",
            "category",
            "success",
            "extreme",
            "available",
            "trajectory_mse",
            "rank",
            "env_id",
            "rollout_uid",
            "group_id",
            "member_id",
            "video_saved",
            "video_path",
        ]
        previous_rows = []
        if os.path.isfile(csv_path):
            try:
                with open(csv_path, newline="") as csv_file:
                    previous_rows = [
                        row
                        for row in csv.DictReader(csv_file)
                        if int(row["global_step"]) != step
                    ]
            except Exception as exc:
                self.logger.warning(
                    "Ignoring unreadable MSE-extreme CSV %s: %s", csv_path, exc
                )
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(previous_rows)
            writer.writerows(rows)
        metrics["mse_extremes_videos_saved"] = float(saved_count)
        return metrics

    def _write_selected_mse_extreme_candidates(
        self, records: list[dict]
    ) -> None:
        """Select globally by scalars, then ask only the four winners to encode."""
        video_cfg = self.cfg.env.train.video_cfg
        if not video_cfg.get("global_mse_extremes_only", False):
            return
        selected = self._select_success_failure_mse_extremes(records)
        selections = {
            category: {
                key: winner[key]
                for key in (
                    "rank",
                    "env_id",
                    "success",
                    "trajectory_mse",
                    "rollout_uid",
                    "group_id",
                    "member_id",
                )
                if key in winner
            }
            for category, winner in selected.items()
        }
        results = self.env.write_selected_mse_extreme_candidates(selections).wait()
        written = sum(
            int(result.get("mse_extreme_candidate_videos_written", 0))
            for result in results
            if result
        )
        if written != len(selected):
            raise RuntimeError(
                "Expected "
                f"{len(selected)} globally selected MSE-extreme candidate videos, "
                f"wrote {written}."
            )

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        if self.cfg.runner.get("replay_only", False):
            self.env.init_worker().wait()
            return

        self.actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.logger.info(f"Resuming training from checkpoint directory {resume_dir}.")
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_sync_metrics = actor_handle.wait()
        rollout_handle.wait()
        if isinstance(actor_sync_metrics, dict):
            return actor_sync_metrics
        if isinstance(actor_sync_metrics, list):
            return self._aggregate_numeric_metrics(actor_sync_metrics)
        return {}

    def evaluate(self, *, collect_post_update_records: bool = False):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results]
        if not eval_metrics_list:
            raise RuntimeError("Evaluation returned no active worker metrics.")
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        if not collect_post_update_records:
            return eval_metrics
        worker_records = self.env.consume_post_update_eval_records().wait()
        return eval_metrics, worker_records

    def _post_update_eval_is_committed(self, global_step: int) -> bool:
        """Return whether an atomic, complete Duck eval record already exists."""
        evaluation_cfg = self._post_update_evaluation_cfg()
        if evaluation_cfg is None:
            return True
        record_path = (
            Path(str(evaluation_cfg.record_dir))
            / f"step_{int(global_step) - 1}.jsonl"
        )
        try:
            records = [
                json.loads(line)
                for line in record_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        expected_episodes = sorted(
            int(value) for value in evaluation_cfg.get("episode_ids", [])
        )
        return (
            len(records) == int(
                evaluation_cfg.get(
                    "expected_records_per_step", len(expected_episodes)
                )
            )
            and sorted(int(record.get("episode", -1)) for record in records)
            == expected_episodes
            and all(
                int(record.get("global_step", -1)) == int(global_step)
                and bool(record.get("valid", False))
                for record in records
            )
        )

    def _resume_requires_post_update_eval(self) -> bool:
        """Detect a checkpoint saved before its fallible validation committed."""
        if self.cfg.runner.get("resume_dir", None) is None:
            return False
        if self._post_update_evaluation_cfg() is None or self.global_step <= 0:
            return False
        interval = int(self.cfg.runner.val_check_interval)
        if interval <= 0:
            return False
        is_eval_boundary = (
            self.global_step % interval == 0 or self.global_step == self.max_steps
        )
        return is_eval_boundary and not self._post_update_eval_is_committed(
            self.global_step
        )

    def _run_post_update_evaluation(self) -> dict:
        """Synchronize the updated policy and run one held-out evaluation."""
        post_update_sync_metrics = self.update_rollout_weights()
        evaluation_cfg = self._post_update_evaluation_cfg()
        if evaluation_cfg is None:
            eval_metrics = self.evaluate()
            return {f"eval/{key}": value for key, value in eval_metrics.items()}

        episode_ids = sorted(
            int(value) for value in evaluation_cfg.get("episode_ids", [])
        )
        execution_episode_ids = [
            int(value)
            for value in evaluation_cfg.get("execution_episode_ids", episode_ids)
        ]
        shard_degree = int(
            evaluation_cfg.get("data_parallel_shard_degree", 0)
        )
        expected_execution = (
            episode_ids
            + [
                episode_ids[index % len(episode_ids)]
                for index in range((-len(episode_ids)) % shard_degree)
            ]
            if episode_ids and shard_degree > 0
            else []
        )
        if execution_episode_ids != expected_execution:
            raise RuntimeError("Duck eval execution IDs are not shard-aligned padding")
        seed_base = self._post_update_eval_seed_base(evaluation_cfg)
        self.rollout.set_post_update_eval_context(
            self.global_step,
            episode_ids,
            seed_base,
            execution_episode_ids,
        ).wait()
        self.env.set_post_update_eval_context(
            self.global_step,
            episode_ids,
            seed_base,
            execution_episode_ids,
        ).wait()
        eval_metrics, worker_records = self.evaluate(
            collect_post_update_records=True
        )
        committed_records = self._commit_post_update_eval_records(
            worker_records,
            sync_metrics=post_update_sync_metrics,
        )
        artifact_metrics = dict(self._post_update_frame_archive_metrics)
        artifact_metrics.update(
            self._finalize_post_update_eval_mse_extremes(committed_records)
        )
        result = {f"eval/{key}": value for key, value in eval_metrics.items()}
        result.update(
            {f"eval/{key}": value for key, value in artifact_metrics.items()}
        )
        return result

    def run_evaluation_only(self) -> None:
        """Evaluate one resumed checkpoint and exit without a training rollout."""
        if self.cfg.runner.get("resume_dir", None) is None:
            raise ValueError("runner.only_eval=true requires runner.resume_dir")
        if self.global_step <= 0:
            raise ValueError("runner.only_eval requires a positive checkpoint step")
        self._write_run_status(
            "evaluation_running", evaluated_global_step=int(self.global_step)
        )
        try:
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)
            self.env.set_global_step(self.global_step)
            eval_metrics = self._run_post_update_evaluation()
            self.metric_logger.log(data=eval_metrics, step=self.global_step - 1)
            self._write_run_status(
                "evaluation_complete",
                evaluated_global_step=int(self.global_step),
                training_updates_executed=0,
            )
        except Exception as exc:
            self._write_run_status(
                "evaluation_failed",
                evaluated_global_step=int(self.global_step),
                training_updates_executed=0,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            self.metric_logger.finish()
            self.stop_logging = True
            self.log_queue.join()
            self.log_thread.join(timeout=1.0)

    def _log_ranked_metrics(
        self,
        metrics_list: list[dict] | None,
        step: int,
        prefix: str,
        worker_group_name: str,
        add_prefix: bool = True,
    ):
        if not self.enable_per_worker_metric_log or not metrics_list:
            return
        for rank, metrics in enumerate(metrics_list):
            if not metrics:
                continue
            metrics_to_log = (
                {f"{prefix}/{k}": v for k, v in metrics.items()}
                if add_prefix
                else metrics
            )
            self.metric_logger.log(
                data=metrics_to_log,
                step=step,
                worker_group_name=worker_group_name,
                rank=rank,
            )

    def _aggregate_numeric_metrics(self, metrics_list: list[dict] | None) -> dict:
        if not metrics_list:
            return {}
        merged_metrics = defaultdict(list)
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key].append(value)
        return {
            key: (sum(values) / len(values))
            for key, values in merged_metrics.items()
            if values
        }

    def _sum_numeric_metrics(self, metrics_list: list[dict] | None) -> dict[str, float]:
        if not metrics_list:
            return {}
        merged_metrics: dict[str, float] = defaultdict(float)
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key] += float(value)
        return dict(merged_metrics)

    def _sum_ranked_numeric_metrics(
        self, metrics_lists: list[list[dict]] | None
    ) -> list[dict]:
        if not metrics_lists:
            return []
        max_rank = max((len(metrics) for metrics in metrics_lists), default=0)
        ranked_metrics: list[dict] = []
        for rank in range(max_rank):
            ranked_metrics.append(
                self._sum_numeric_metrics(
                    [
                        metrics[rank]
                        for metrics in metrics_lists
                        if rank < len(metrics) and metrics[rank]
                    ]
                )
            )
        return ranked_metrics

    def _process_ranked_numeric_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = self._aggregate_numeric_metrics(metric_list)
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = self._aggregate_numeric_metrics(
                    metrics_list
                )
        return aggregated_metrics, ranked_metrics_list

    def _process_ranked_eval_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = (
            compute_evaluate_metrics(metric_list) if metric_list else {}
        )
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = compute_evaluate_metrics(metrics_list)
        return aggregated_metrics, ranked_metrics_list

    def _get_replay_buffer_warmup_status(self) -> dict[str, int | bool]:
        if not self.cfg.runner.get("rollout_until_replay_buffer_ready", False):
            return {
                "enabled": False,
                "has_replay_buffer": False,
                "is_ready": True,
                "buffer_size": 0,
                "min_buffer_size": 0,
            }

        status_list = self.actor.get_replay_buffer_warmup_status().wait()
        replay_statuses = [
            status
            for status in status_list
            if status is not None and status.get("has_replay_buffer", False)
        ]
        if not replay_statuses:
            return {
                "enabled": True,
                "has_replay_buffer": False,
                "is_ready": True,
                "buffer_size": 0,
                "min_buffer_size": 0,
            }

        return {
            "enabled": True,
            "has_replay_buffer": True,
            "is_ready": all(
                bool(status.get("is_ready", True)) for status in replay_statuses
            ),
            "buffer_size": min(
                int(status.get("buffer_size", 0)) for status in replay_statuses
            ),
            "min_buffer_size": max(
                int(status.get("min_buffer_size", 0)) for status in replay_statuses
            ),
        }

    def run(self):
        if self.cfg.runner.get("replay_only", False):
            self.run_replay_only()
            return
        if self.cfg.runner.get("only_eval", False):
            self.run_evaluation_only()
            return

        start_step = self.global_step
        start_time = time.time()
        self._runner_start_epoch = start_time
        stopped_for_walltime = False
        stopped_after_operator_gate = False
        # A resumed prefix was selected as auditable before this runner starts.
        last_update_audit_committed = True
        self._write_run_status("running")
        initial_eval_cfg = self._post_update_evaluation_cfg()
        if self.global_step == 0 and initial_eval_cfg is not None and bool(
            initial_eval_cfg.get("before_training", False)
        ):
            self.logger.info("Evaluating the initial SFT policy before training.")
            initial_metrics = self._run_post_update_evaluation()
            self.metric_logger.log(
                data={key.replace("eval/", "eval_initial/", 1): value
                      for key, value in initial_metrics.items()},
                step=0,
            )
        if self._resume_requires_post_update_eval():
            self.logger.info(
                "Checkpoint global_step_%s has no committed post-update "
                "evaluation; completing it before the next training update.",
                self.global_step,
            )
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)
            self.env.set_global_step(self.global_step)
            recovered_eval_metrics = self._run_post_update_evaluation()
            self.metric_logger.log(
                data=recovered_eval_metrics,
                step=self.global_step - 1,
            )
            last_update_audit_committed = self._run_duck_analysis()
        for _step in range(start_step, self.max_steps):
            should_stop, remaining = self._should_stop_before_update()
            if should_stop:
                self.logger.info(
                    "Remaining allocation time %.1f minutes is below the "
                    "configured reserve; not starting another production "
                    "update.",
                    max(float(remaining or 0.0), 0.0) / 60.0,
                )
                if not self._commit_recovery_stop(
                    "walltime_stop",
                    audit_committed=last_update_audit_committed,
                    remaining_walltime_seconds=float(remaining or 0.0),
                ):
                    raise RuntimeError(
                        "Cannot enter walltime_stop without a complete, "
                        "audited recovery checkpoint."
                    )
                stopped_for_walltime = True
                break
            # set global step
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)
            self.env.set_global_step(self.global_step)

            sync_metrics = {}
            with self.timer("step"):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        sync_metrics = self.update_rollout_weights()
                env_handles: list[Handle] = []
                rollout_handles: list[Handle] = []
                env_results_list = []
                best_rollout_records = []
                ranked_env_results = []
                rollout_phase = 0

                with self.timer("generate_rollouts"):
                    while True:
                        rollout_phase += 1
                        self.env.set_rollout_context(
                            self.global_step, rollout_phase
                        )
                        env_handle = self.env.interact(
                            input_channel=self.rollout_channel,
                            output_channel=self.env_channel,
                            actor_channel=self.actor_channel,
                        )
                        rollout_handle = self.rollout.generate(
                            input_channel=self.env_channel,
                            output_channel=self.rollout_channel,
                        )
                        self.actor.recv_rollout_trajectories(
                            input_channel=self.actor_channel
                        ).wait()
                        rollout_handle.wait()

                        env_handles.append(env_handle)
                        rollout_handles.append(rollout_handle)

                        env_results = env_handle.wait()
                        phase_best_rollout_records = self._pop_best_rollout_records(
                            env_results
                        )
                        self._write_selected_mse_extreme_candidates(
                            phase_best_rollout_records
                        )
                        best_rollout_records.extend(phase_best_rollout_records)
                        env_results_list.extend(
                            results for results in env_results if results is not None
                        )
                        ranked_env_results.extend(
                            {
                                "rank": rank,
                                "env": rank_metrics,
                            }
                            for rank, rank_metrics in enumerate(env_results)
                            if rank_metrics is not None
                        )

                        warmup_status = self._get_replay_buffer_warmup_status()
                        if (
                            warmup_status["is_ready"]
                            or not warmup_status["has_replay_buffer"]
                        ):
                            break

                        self.logger.info(
                            "Replay buffer warmup: collected %s/%s trajectories after rollout phase %s; continuing rollout collection before training.",
                            warmup_status["buffer_size"],
                            warmup_status["min_buffer_size"],
                            rollout_phase,
                        )

                best_rollout_metrics = self._finalize_best_rollout(
                    best_rollout_records, _step + 1
                )
                best_rollout_metrics.update(
                    self._finalize_worst_rollout(best_rollout_records, _step + 1)
                )
                best_rollout_metrics.update(
                    self._finalize_success_failure_mse_extremes(best_rollout_records, _step + 1)
                )
                best_rollout_metrics.update(
                    self._finalize_rollout_final_side_frames(
                        best_rollout_records, _step + 1
                    )
                )

                # compute advantages and returns once after rollout warmup completes.
                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                # actor training, or an explicit rollout-only smoke gate.
                if self.cfg.runner.get("rollout_only", False):
                    actor_training_handle: Handle = (
                        self.actor.discard_rollout_batch()
                    )
                else:
                    actor_training_handle = self.actor.run_training()

                actor_training_metrics = actor_training_handle.wait()

                self.global_step += 1
                self._validate_first_update_smoke(actor_training_metrics)

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )
                # Commit the recoverable training state before
                # potentially fallible Duck evaluation collectives.
                if save_model:
                    self._save_checkpoint()

                eval_metrics = {}
                if run_val:
                    with self.timer("eval"):
                        eval_metrics = self._run_post_update_evaluation()
                        self.metric_logger.log(data=eval_metrics, step=_step)

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            env_timing_data = [
                env_handle.consume_durations(return_per_rank=True)
                for env_handle in env_handles
            ]
            env_time_metrics = self._sum_numeric_metrics(
                [timing[0] for timing in env_timing_data]
            )
            env_time_metrics_per_rank = self._sum_ranked_numeric_metrics(
                [timing[1] for timing in env_timing_data]
            )
            rollout_timing_data = [
                rollout_handle.consume_durations(return_per_rank=True)
                for rollout_handle in rollout_handles
            ]
            rollout_time_metrics = self._sum_numeric_metrics(
                [timing[0] for timing in rollout_timing_data]
            )
            rollout_time_metrics_per_rank = self._sum_ranked_numeric_metrics(
                [timing[1] for timing in rollout_timing_data]
            )
            actor_time_metrics, actor_time_metrics_per_rank = (
                actor_training_handle.consume_durations(return_per_rank=True)
            )
            time_metrics.update(
                {f"time/env/{k}": v for k, v in env_time_metrics.items()}
            )
            time_metrics.update(
                {f"time/rollout/{k}": v for k, v in rollout_time_metrics.items()}
            )
            time_metrics.update(
                {f"time/actor/{k}": v for k, v in actor_time_metrics.items()}
            )

            env_metrics = compute_evaluate_metrics(env_results_list)
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            _, env_metrics_per_rank = self._process_ranked_eval_results(
                ranked_env_results, metric_field="env"
            )

            actor_rollout_metric_values = self._aggregate_numeric_metrics(
                actor_rollout_metrics
            )
            cross_rank_training_metrics = {
                key: value
                for key, value in actor_rollout_metric_values.items()
                if key.startswith("train/")
            }
            rollout_metrics = {
                f"rollout/{key}": value
                for key, value in actor_rollout_metric_values.items()
                if not key.startswith("train/")
            }
            rollout_metrics.update(
                {f"rollout/{key}": value for key, value in best_rollout_metrics.items()}
            )

            actor_training_metric_values = self._aggregate_numeric_metrics(
                actor_training_metrics
            )
            actor_training_metric_values.update(sync_metrics)
            training_metrics = {
                f"train/{k}": v for k, v in actor_training_metric_values.items()
            }
            training_metrics.update(cross_rank_training_metrics)

            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)
            self._append_scalar_csv("train", self.global_step, training_metrics)
            self._append_scalar_csv(
                "rollout", self.global_step, rollout_metrics
            )
            self._write_run_status(
                "running",
                first_update_validated=bool(self.global_step >= 1),
            )
            self._log_ranked_metrics(
                metrics_list=actor_rollout_metrics,
                step=_step,
                prefix="rollout",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=actor_training_metrics,
                step=_step,
                prefix="train",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=actor_time_metrics_per_rank,
                step=_step,
                prefix="time/actor",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=rollout_time_metrics_per_rank,
                step=_step,
                prefix="time/rollout",
                worker_group_name=self.rollout.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=env_time_metrics_per_rank,
                step=_step,
                prefix="time/env",
                worker_group_name=self.env.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=env_metrics_per_rank,
                step=_step,
                prefix="env",
                worker_group_name=self.env.worker_group_name,
            )

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)

            self.print_metrics_table_async(
                _step, self.max_steps, start_time, logging_metrics, start_step
            )
            append_metrics_history(
                os.path.join(
                    self.cfg.runner.logger.log_path,
                    self.cfg.runner.logger.experiment_name,
                ),
                _step,
                self.max_steps,
                time.time() - start_time,
                logging_metrics,
            )

            duck_analysis_committed = self._run_duck_analysis()
            last_update_audit_committed = duck_analysis_committed
            stop_after_updates = int(
                self.cfg.runner.get("stop_after_updates", 0)
            )
            if (
                stop_after_updates > 0
                and self.global_step >= stop_after_updates
                and self.global_step < self.max_steps
            ):
                if not self._commit_recovery_stop(
                    "operator_gate_stop",
                    audit_committed=duck_analysis_committed,
                    first_update_validated=bool(self.global_step >= 1),
                    stop_after_updates=stop_after_updates,
                ):
                    raise RuntimeError(
                        "Cannot enter operator_gate_stop without a complete, "
                        "audited recovery checkpoint."
                    )
                stopped_after_operator_gate = True
                break

            if save_model:
                self._prune_checkpoints_after_audit_commit(
                    duck_analysis_committed
                )

        if (
            not stopped_for_walltime
            and not stopped_after_operator_gate
            and self.global_step >= self.max_steps
        ):
            if not self._commit_training_success(
                audit_committed=last_update_audit_committed
            ):
                raise RuntimeError(
                    "Cannot mark training complete without a complete, "
                    "audited final checkpoint."
                )
        self.metric_logger.finish()

        # Stop logging thread
        self.stop_logging = True
        self.log_queue.join()  # Wait for all queued logs to be processed
        self.log_thread.join(timeout=1.0)

    def run_replay_only(self):
        start_time = time.time()
        start_step = self.global_step
        for _step in range(start_step, self.max_steps):
            with self.timer("step"):
                with self.timer("generate_rollouts"):
                    env_handle = self.env.replay_only_interact()
                    env_results = env_handle.wait()

                self.global_step += 1

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            env_timing_data = env_handle.consume_durations(return_per_rank=True)
            env_time_metrics, env_time_metrics_per_rank = env_timing_data
            time_metrics.update(
                {f"time/env/{k}": v for k, v in env_time_metrics.items()}
            )

            env_results_list = [
                results for results in env_results if results is not None
            ]
            ranked_env_results = [
                {"rank": rank, "env": rank_metrics}
                for rank, rank_metrics in enumerate(env_results)
                if rank_metrics is not None
            ]
            env_metrics = compute_evaluate_metrics(env_results_list)
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            _, env_metrics_per_rank = self._process_ranked_eval_results(
                ranked_env_results, metric_field="env"
            )

            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self._log_ranked_metrics(
                metrics_list=env_time_metrics_per_rank,
                step=_step,
                prefix="time/env",
                worker_group_name=self.env.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=env_metrics_per_rank,
                step=_step,
                prefix="env",
                worker_group_name=self.env.worker_group_name,
            )

            logging_metrics = time_metrics
            logging_metrics.update(env_metrics)
            self.print_metrics_table_async(
                _step, self.max_steps, start_time, logging_metrics, start_step
            )

        self.metric_logger.finish()
        self.stop_logging = True
        self.log_queue.join()
        self.log_thread.join(timeout=1.0)

    def _save_checkpoint(self):
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        checkpoint_root = self._run_root() / "checkpoints"
        target = checkpoint_root / f"global_step_{self.global_step}"
        if self._checkpoint_has_files(self.global_step):
            return
        if target.exists():
            raise RuntimeError(f"refusing incomplete existing checkpoint: {target}")
        staging = checkpoint_root / (
            f".staging-global_step_{self.global_step}-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}"
        )
        if staging.exists():
            raise RuntimeError(f"checkpoint staging path already exists: {staging}")
        actor_save_path = staging / "actor"
        actor_save_path.mkdir(parents=True)
        self.actor.save_checkpoint(str(actor_save_path), self.global_step).wait()
        dcp_path = actor_save_path / "dcp_checkpoint"
        metadata_path = dcp_path / ".metadata"
        shards = tuple(sorted(dcp_path.glob("*.distcp")))
        expected_shards = self._duck_checkpoint_shard_count()
        if not metadata_path.is_file() or not shards:
            raise RuntimeError(f"incomplete staged DCP checkpoint: {dcp_path}")
        if expected_shards is not None and len(shards) != expected_shards:
            raise RuntimeError(
                f"staged DCP shard count {len(shards)} != expected {expected_shards}"
            )
        staging_manifest = {
            "schema_version": 1,
            "state": "staging",
            "update": int(self.global_step),
            "dcp_shards": len(shards),
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        }
        self._atomic_write_json(staging / "checkpoint_manifest.json", staging_manifest)
        # Only a fully flushed DCP directory may become a visible checkpoint.
        for file_path in (metadata_path, *shards, staging / "checkpoint_manifest.json"):
            with file_path.open("rb") as stream:
                os.fsync(stream.fileno())
        for directory in (dcp_path, actor_save_path, staging, checkpoint_root):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.replace(staging, target)
        completed_manifest = {**staging_manifest, "state": "completed"}
        self._atomic_write_json(target / "checkpoint_manifest.json", completed_manifest)
        descriptor = os.open(checkpoint_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _prune_checkpoints_after_audit_commit(
        self, audit_committed: bool
    ) -> bool:
        """Prune only after the current checkpoint has an auditable prefix."""
        if not audit_committed:
            self.logger.warning(
                "Checkpoint pruning deferred at step %s because the audit "
                "boundary did not commit.",
                self.global_step,
            )
            return False
        if not self._checkpoint_has_files(self.global_step):
            self.logger.warning(
                "Checkpoint pruning deferred at step %s because the current "
                "checkpoint is incomplete.",
                self.global_step,
            )
            return False
        self._prune_checkpoints()
        return True

    def _prune_checkpoints(self) -> None:
        """Keep milestones plus the newest temporary recovery checkpoint."""
        milestone_interval = int(
            self.cfg.runner.get("checkpoint_milestone_interval", 0)
        )
        keep_latest_recovery = int(
            self.cfg.runner.get("checkpoint_keep_latest_recovery", -1)
        )
        if milestone_interval <= 0 or keep_latest_recovery < 0:
            return

        checkpoint_root = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            "checkpoints",
        )
        checkpoints = []
        for name in os.listdir(checkpoint_root):
            if not name.startswith("global_step_"):
                continue
            try:
                step = int(name.removeprefix("global_step_"))
            except ValueError:
                continue
            path = os.path.join(checkpoint_root, name)
            if os.path.isdir(path):
                checkpoints.append((step, path))

        recovery_checkpoints = sorted(
            (step, path)
            for step, path in checkpoints
            if step % milestone_interval != 0
        )
        keep_recovery = (
            0
            if self.global_step % milestone_interval == 0
            else keep_latest_recovery
        )
        kept_paths = (
            {path for _, path in recovery_checkpoints[-keep_recovery:]}
            if keep_recovery
            else set()
        )
        for step, path in recovery_checkpoints:
            if path in kept_paths:
                continue
            self.logger.info(
                "Removing superseded recovery checkpoint at step %s: %s",
                step,
                path,
            )
            shutil.rmtree(path)

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
