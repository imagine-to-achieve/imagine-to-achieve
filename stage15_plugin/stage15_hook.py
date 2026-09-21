"""Runtime-only Stage 1.5 hooks for exact FPO replay and paired evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

_MODE = os.environ.get("RLINF_STAGE15_MODE", "").strip()
_VALID_MODES = {"baseline", "a_exact", "d_exact"}
_WEIGHT_DIAGNOSTICS = (
    os.environ.get("RLINF_STAGE16_WEIGHT_DIAGNOSTICS", "").strip() == "1"
)
_WEIGHT_DIAGNOSTICS_SCHEMA = 1
_DELTA_CHUNK_ELEMENTS = 1_048_576


def _replace_override(values: list[str], key: str, value: Any) -> None:
    prefix = f"{key}="
    replacement = f"{key}={value}"
    for index, item in enumerate(values):
        if item.startswith(prefix):
            values[index] = replacement
            return
    values.append(replacement)


def _patch_real_bridge() -> None:
    try:
        from rlinf_modified.engine.real import RealTrainer
    except ModuleNotFoundError:
        return
    if hasattr(RealTrainer, "_stage15_original_overrides"):
        return
    original = RealTrainer._overrides
    RealTrainer._stage15_original_overrides = original

    def stage15_overrides(self, normalized_checkpoint_config: str) -> list[str]:
        values = list(original(self, normalized_checkpoint_config))
        rollout_only = _MODE == "baseline"
        _replace_override(values, "runner.rollout_only", str(rollout_only).lower())
        _replace_override(
            values,
            "runner.first_update_smoke.enabled",
            str(not rollout_only).lower(),
        )
        _replace_override(values, "runner.save_interval", -1)
        _replace_override(values, "runner.checkpoint_milestone_interval", -1)
        _replace_override(values, "runner.val_check_interval", 1)
        _replace_override(values, "duck.evaluation.enabled", "true")
        _replace_override(
            values,
            "duck.evaluation.record_dir",
            self.run_dir / "evaluation_records",
        )
        return values

    RealTrainer._overrides = stage15_overrides


def _patch_runner_completion() -> None:
    try:
        from rlinf.runners.embodied_runner import EmbodiedRunner
    except ModuleNotFoundError:
        return
    if hasattr(EmbodiedRunner, "_stage15_original_commit_training_success"):
        return
    original = EmbodiedRunner._commit_training_success
    EmbodiedRunner._stage15_original_commit_training_success = original

    def stage15_commit_training_success(
        self, *, audit_committed: bool
    ) -> bool:
        if not audit_committed:
            return False
        self._write_run_status(
            "completed",
            final_checkpoint_ready=False,
            stage15_plugin=True,
            stage15_mode=_MODE,
        )
        self._write_success_marker()
        return True

    EmbodiedRunner._commit_training_success = stage15_commit_training_success


def _patch_sparse_eval_batching() -> None:
    """Make ranks outside the sparse eval prefix truly inactive.

    The vendored worker computes a uniform eval batch when the padded episode
    count is divisible by the active prefix size. That value is also retained
    on ranks outside the prefix, even though their episode assignment is empty.
    Keep the vendored implementation intact and correct the rank-local value at
    the runtime boundary immediately before the context is validated.
    """
    try:
        from rlinf.workers.rollout.hf.huggingface_worker import (
            MultiStepRolloutWorker,
        )
    except ModuleNotFoundError:
        return
    if hasattr(
        MultiStepRolloutWorker,
        "_stage15_original_set_post_update_eval_context",
    ):
        return
    original = MultiStepRolloutWorker.set_post_update_eval_context
    MultiStepRolloutWorker._stage15_original_set_post_update_eval_context = original

    def stage15_set_post_update_eval_context(self, *args, **kwargs):
        if self._rank >= int(self.eval_mapping_world_size):
            self.eval_batch_size = 0
        return original(self, *args, **kwargs)

    MultiStepRolloutWorker.set_post_update_eval_context = (
        stage15_set_post_update_eval_context
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _weight_diagnostics_run_dir(worker) -> Path:
    record_cfg = worker.cfg.algorithm.get("trajectory_records", {})
    output_dir = record_cfg.get("output_dir", None)
    if not output_dir:
        raise RuntimeError(
            "Stage 1.6 weight diagnostics require trajectory_records.output_dir"
        )
    return Path(str(output_dir)).parent / "weight_diagnostics"


def _local_parameter_tensor(parameter):
    from torch.distributed.tensor import DTensor

    tensor = parameter.to_local() if isinstance(parameter, DTensor) else parameter
    return tensor.detach().contiguous().cpu().clone()


def _snapshot_action_parameters(worker) -> dict[str, Any]:
    named_parameters = worker.model._native_trainable_named_parameters()
    return {
        str(name): _local_parameter_tensor(parameter)
        for name, parameter in named_parameters
    }


def _update_tensor_digest(digest, name: str, tensor) -> None:
    import torch

    value = tensor.detach().contiguous().cpu()
    header = json.dumps(
        {
            "name": str(name),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    raw = value.view(torch.uint8).reshape(-1).numpy()
    digest.update(memoryview(raw))


def _snapshot_identity(snapshot: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(snapshot):
        _update_tensor_digest(digest, name, snapshot[name])
    return digest.hexdigest()


def _compare_action_snapshots(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    import torch

    if set(before) != set(after):
        raise RuntimeError(
            "Stage 1.6 action parameter names changed while comparing "
            f"{phase}: before_only={sorted(set(before) - set(after))}, "
            f"after_only={sorted(set(after) - set(before))}"
        )

    total_elements = 0
    changed_elements = 0
    changed_tensors = 0
    nonfinite_elements = 0
    abs_sum = 0.0
    square_sum = 0.0
    signed_sum = 0.0
    max_abs = 0.0
    dtype_counts: dict[str, int] = {}
    first_changed_names: list[str] = []

    for name in sorted(before):
        left = before[name]
        right = after[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(
                "Stage 1.6 action parameter metadata changed for "
                f"{name}: {(left.shape, left.dtype)} != {(right.shape, right.dtype)}"
            )
        dtype_key = str(left.dtype)
        dtype_counts[dtype_key] = dtype_counts.get(dtype_key, 0) + int(left.numel())
        left_flat = left.reshape(-1)
        right_flat = right.reshape(-1)
        tensor_changed = 0
        for start in range(0, int(left_flat.numel()), _DELTA_CHUNK_ELEMENTS):
            stop = min(start + _DELTA_CHUNK_ELEMENTS, int(left_flat.numel()))
            left_chunk = left_flat[start:stop]
            right_chunk = right_flat[start:stop]
            changed = int(torch.count_nonzero(left_chunk != right_chunk).item())
            tensor_changed += changed
            changed_elements += changed
            total_elements += int(stop - start)
            delta = right_chunk.float() - left_chunk.float()
            finite = torch.isfinite(delta)
            nonfinite_elements += int(torch.count_nonzero(~finite).item())
            if bool(finite.any().item()):
                finite_delta = delta[finite].double()
                absolute = finite_delta.abs()
                abs_sum += float(absolute.sum().item())
                square_sum += float((finite_delta * finite_delta).sum().item())
                signed_sum += float(finite_delta.sum().item())
                max_abs = max(max_abs, float(absolute.max().item()))
        if tensor_changed:
            changed_tensors += 1
            if len(first_changed_names) < 20:
                first_changed_names.append(name)

    denominator = max(total_elements - nonfinite_elements, 1)
    return {
        "schema_version": _WEIGHT_DIAGNOSTICS_SCHEMA,
        "phase": str(phase),
        "tensor_count": len(before),
        "element_count": total_elements,
        "changed_tensor_count": changed_tensors,
        "changed_element_count": changed_elements,
        "changed_element_fraction": changed_elements / max(total_elements, 1),
        "nonfinite_delta_element_count": nonfinite_elements,
        "mean_abs_delta": abs_sum / denominator,
        "rms_delta": math.sqrt(square_sum / denominator),
        "max_abs_delta": max_abs,
        "signed_delta_sum": signed_sum,
        "parameter_element_dtypes": dtype_counts,
        "before_sha256": _snapshot_identity(before),
        "after_sha256": _snapshot_identity(after),
        "first_changed_parameter_names": first_changed_names,
    }


def _optimizer_state_summary(optimizer) -> dict[str, Any]:
    import torch

    dtype_counts: dict[str, int] = {}
    step_values: list[float] = []
    exp_avg_abs_max = 0.0
    exp_avg_sq_abs_max = 0.0
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            dtype_key = f"{key}:{value.dtype}"
            dtype_counts[dtype_key] = dtype_counts.get(dtype_key, 0) + int(
                value.numel()
            )
            if key == "step" and value.numel() == 1:
                step_values.append(float(value.detach().cpu().item()))
                continue
            if key not in {"exp_avg", "exp_avg_sq"}:
                continue
            flat = value.detach().reshape(-1)
            target_max = 0.0
            for start in range(0, int(flat.numel()), _DELTA_CHUNK_ELEMENTS):
                stop = min(start + _DELTA_CHUNK_ELEMENTS, int(flat.numel()))
                chunk = flat[start:stop].float()
                if chunk.numel():
                    target_max = max(
                        target_max, float(chunk.abs().max().detach().cpu().item())
                    )
            if key == "exp_avg":
                exp_avg_abs_max = max(exp_avg_abs_max, target_max)
            else:
                exp_avg_sq_abs_max = max(exp_avg_sq_abs_max, target_max)
    return {
        "state_tensor_element_dtypes": dtype_counts,
        "state_step_min": min(step_values) if step_values else None,
        "state_step_max": max(step_values) if step_values else None,
        "exp_avg_abs_max": exp_avg_abs_max,
        "exp_avg_sq_abs_max": exp_avg_sq_abs_max,
    }


def _append_weight_diagnostic(worker, record: dict[str, Any]) -> None:
    output_dir = _weight_diagnostics_run_dir(worker) / ".rank_shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"rank_{int(worker._rank):05d}.jsonl"
    payload = {
        "schema_version": _WEIGHT_DIAGNOSTICS_SCHEMA,
        "mode": _MODE,
        "rank": int(worker._rank),
        "world_size": int(worker._world_size),
        "timestamp_unix": time.time(),
        "learning_rates": [
            float(group["lr"]) for group in worker.optimizer.param_groups
        ],
        **record,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _initialize_weight_diagnostics(worker) -> None:
    path = (
        _weight_diagnostics_run_dir(worker)
        / ".rank_shards"
        / f"rank_{int(worker._rank):05d}.jsonl"
    )
    if not path.exists():
        return
    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Stage 1.6 could not validate existing diagnostic shard: {path}"
        ) from exc
    expected_initial_sync = (
        len(records) == 1
        and records[0].get("phase") == "sync_materialized_state"
        and int(records[0].get("rank", -1)) == int(worker._rank)
        and int(records[0].get("world_size", -1)) == int(worker._world_size)
        and records[0].get("mode") == _MODE
    )
    if expected_initial_sync:
        return
    raise RuntimeError(
        "Stage 1.6 weight diagnostic shard contains records beyond this run's "
        f"initial sync; refusing to mix runs: {path}"
    )


def _state_dict_identity(state_dict: dict[str, Any]) -> dict[str, Any]:
    import torch
    from torch.distributed.tensor import DTensor

    digest = hashlib.sha256()
    tensor_count = 0
    element_count = 0
    dtype_counts: dict[str, int] = {}
    value_sum = 0.0
    abs_sum = 0.0
    square_sum = 0.0
    for name in sorted(state_dict):
        value = state_dict[name]
        if not torch.is_tensor(value):
            continue
        tensor = value.full_tensor() if isinstance(value, DTensor) else value
        tensor = tensor.detach().contiguous().cpu()
        _update_tensor_digest(digest, str(name), tensor)
        tensor_count += 1
        element_count += int(tensor.numel())
        dtype_key = str(tensor.dtype)
        dtype_counts[dtype_key] = dtype_counts.get(dtype_key, 0) + int(
            tensor.numel()
        )
        flat = tensor.reshape(-1)
        for start in range(0, int(flat.numel()), _DELTA_CHUNK_ELEMENTS):
            stop = min(start + _DELTA_CHUNK_ELEMENTS, int(flat.numel()))
            chunk = flat[start:stop].double()
            value_sum += float(chunk.sum().item())
            abs_sum += float(chunk.abs().sum().item())
            square_sum += float((chunk * chunk).sum().item())
    return {
        "schema_version": _WEIGHT_DIAGNOSTICS_SCHEMA,
        "phase": "sync_materialized_state",
        "state_sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
        "element_count": element_count,
        "parameter_element_dtypes": dtype_counts,
        "value_sum_float64": value_sum,
        "value_abs_sum_float64": abs_sum,
        "value_l2_float64": math.sqrt(square_sum),
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    denom = denom_x * denom_y
    if denom == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denom


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["valid"]]
    advantages = [float(record["advantage"]) for record in valid]
    deltas = [float(record["score_delta"]) for record in valid]
    directionals = [
        float(record["advantage_times_score_delta"]) for record in valid
    ]
    positive = [
        delta for advantage, delta in zip(advantages, deltas) if advantage > 0.0
    ]
    negative = [
        delta for advantage, delta in zip(advantages, deltas) if advantage < 0.0
    ]
    aligned = [
        math.copysign(1.0, advantage) == math.copysign(1.0, delta)
        for advantage, delta in zip(advantages, deltas)
        if advantage != 0.0 and delta != 0.0
    ]
    mean_delta = sum(deltas) / len(deltas)
    delta_std = math.sqrt(
        sum((value - mean_delta) ** 2 for value in deltas) / len(deltas)
    )
    return {
        "schema_version": 1,
        "mode": _MODE,
        "records": len(records),
        "valid_records": len(valid),
        "trajectories": len({int(record["rollout_uid"]) for record in valid}),
        "score_delta_mean": mean_delta,
        "score_delta_std": delta_std,
        "score_abs_delta_mean": sum(abs(value) for value in deltas) / len(deltas),
        "score_delta_min": min(deltas),
        "score_delta_max": max(deltas),
        "advantage_times_score_delta_mean": (
            sum(directionals) / len(directionals)
        ),
        "advantage_times_score_delta_sum": sum(directionals),
        "advantage_score_delta_corr": _pearson(advantages, deltas),
        "adv_positive_score_delta_mean": (
            sum(positive) / len(positive) if positive else 0.0
        ),
        "adv_negative_score_delta_mean": (
            sum(negative) / len(negative) if negative else 0.0
        ),
        "alignment_fraction_nonzero": (
            sum(aligned) / len(aligned) if aligned else 0.0
        ),
        "ratio_min": math.exp(max(min(deltas), -20.0)),
        "ratio_max": math.exp(min(max(deltas), 20.0)),
    }


def _run_exact_replay(worker, module) -> dict[str, Any]:
    import torch

    if worker.rollout_batch is None:
        raise RuntimeError("Stage 1.5 exact replay lost the frozen rollout batch")
    cosmos_cfg = worker.cfg.actor.model.get("cosmos", {})
    per_mc = (
        worker.cfg.algorithm.get("fpo_ratio_granularity", "per_action")
        == "per_mc"
    )
    rollout_size = int(worker.rollout_batch["prev_logprobs"].shape[0])
    micro_batch_size = int(worker.cfg.actor.micro_batch_size)
    if rollout_size % micro_batch_size:
        raise ValueError("exact replay batch is not divisible by micro batch size")
    batches = module.split_dict_to_chunk(
        worker.rollout_batch, rollout_size // micro_batch_size
    )
    device = (
        f"{module.Worker.torch_device_type}:"
        f"{int(os.environ['LOCAL_RANK'])}"
    )
    local_records: list[dict[str, Any]] = []
    was_training = worker.model.training
    worker.model.eval()
    started_at = time.monotonic()
    try:
        for batch_index, batch in enumerate(batches, start=1):
            batch = module.put_tensor_device(batch, device)
            torch.set_grad_enabled(True)
            with worker.amp_context:
                output = worker.model(
                    forward_inputs=batch.get("forward_inputs"),
                    compute_logprobs=True,
                    compute_entropy=False,
                    compute_values=False,
                    use_cache=False,
                    track_grad=True,
                    return_replay_diagnostics=per_mc,
                )
            if per_mc:
                old_values = batch["prev_fpo_pair_scores"].detach().float()
                new_values = -output["fpo_pair_losses"].detach().float()
                old_scores = old_values.mean(dim=-1)
                new_scores = new_values.mean(dim=-1)
            else:
                old_values = None
                new_values = None
                old_scores = (
                    batch["prev_logprobs"].detach().float().reshape(
                        batch["prev_logprobs"].shape[0], -1
                    ).sum(dim=-1)
                )
                new_scores = (
                    output["logprobs"].detach().float().reshape(
                        output["logprobs"].shape[0], -1
                    ).sum(dim=-1)
                )
            batch_size = int(old_scores.shape[0])
            advantages = (
                batch["advantages"].detach().float().reshape(batch_size, -1).mean(dim=-1)
            )
            loss_mask = batch.get("loss_mask")
            if loss_mask is None:
                valid_mask = torch.ones(
                    batch_size, dtype=torch.bool, device=old_scores.device
                )
            else:
                valid_mask = (
                    loss_mask.detach().reshape(batch_size, -1).amax(dim=-1).bool()
                )
            uid_tensor = batch["rollout_uid"]
            chunk_tensor = batch["chunk_id"]
            group_tensor = batch.get("group_id")
            for sample_index in range(batch_size):
                before = float(old_scores[sample_index].item())
                after = float(new_scores[sample_index].item())
                advantage = float(advantages[sample_index].item())
                delta = after - before
                record = {
                    "schema_version": 1,
                    "mode": _MODE,
                    "rank": int(worker._rank),
                    "rollout_uid": int(
                        uid_tensor[sample_index].reshape(-1)[0].item()
                    ),
                    "chunk_id": int(
                        chunk_tensor[sample_index].reshape(-1)[0].item()
                    ),
                    "group_id": (
                        int(group_tensor[sample_index].reshape(-1)[0].item())
                        if torch.is_tensor(group_tensor)
                        else None
                    ),
                    "advantage": advantage,
                    "score_before": before,
                    "score_after": after,
                    "score_delta": delta,
                    "advantage_times_score_delta": advantage * delta,
                    "valid": bool(valid_mask[sample_index].item()),
                    "per_mc_score_before": (
                        [float(value) for value in old_values[sample_index].cpu().tolist()]
                        if per_mc
                        else None
                    ),
                    "per_mc_score_after": (
                        [float(value) for value in new_values[sample_index].cpu().tolist()]
                        if per_mc
                        else None
                    ),
                }
                local_records.append(record)
            del output, batch
            if batch_index % 10 == 0 or batch_index == len(batches):
                worker.log_on_first_rank(
                    "Stage 1.5 exact FPO replay progress: "
                    f"{batch_index}/{len(batches)} microbatches"
                )
    finally:
        worker.model.train(was_training)

    # A grad-enabled FSDP2 forward normally completes its parameter lifecycle
    # during backward. This diagnostic detaches scores and deliberately has no
    # backward, so explicitly restore the native sharded parameter bindings.
    from torch.distributed.fsdp import FSDPModule

    service_model = worker.model.ensure_native_trainable_state()
    resharded = 0
    for child in service_model.modules():
        if isinstance(child, FSDPModule):
            child.reshard()
            resharded += 1
    validator = getattr(worker.model, "validate_native_trainable_bindings", None)
    if callable(validator):
        validator(worker.optimizer)
    worker.log_on_first_rank(
        "Stage 1.5 exact replay restored native FSDP2 shards: "
        f"modules={resharded}."
    )
    record_cfg = worker.cfg.algorithm.get("trajectory_records", {})
    run_dir = Path(str(record_cfg.output_dir)).parent
    output_dir = run_dir / "exact_score_replay"
    shard_dir = output_dir / ".rank_shards"
    update_id = int(getattr(worker, "global_step", worker.version))
    shard_path = shard_dir / (
        f"rank_{int(worker._rank):05d}_update_{update_id:04d}.jsonl"
    )
    _atomic_jsonl(shard_path, local_records)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    summary: dict[str, Any] = {}
    if int(worker._rank) == 0:
        merged: list[dict[str, Any]] = []
        for rank in range(int(worker._world_size)):
            path = shard_dir / f"rank_{rank:05d}_update_{update_id:04d}.jsonl"
            with path.open(encoding="utf-8") as handle:
                merged.extend(
                    json.loads(line) for line in handle if line.strip()
                )
        expected = (
            int(record_cfg.get("expected_trajectories", 128))
            * int(worker.cfg.algorithm.get("trajectory_chunks", 5))
        )
        if len(merged) != expected:
            raise RuntimeError(
                f"exact replay record count {len(merged)} != {expected}"
            )
        keys = {
            (int(record["rollout_uid"]), int(record["chunk_id"]))
            for record in merged
        }
        if len(keys) != expected:
            raise RuntimeError("exact replay contains duplicate chunk keys")
        merged.sort(
            key=lambda record: (
                int(record["rollout_uid"]), int(record["chunk_id"])
            )
        )
        summary = _summary(merged)
        summary.update(
            {
                "ratio_granularity": str(
                    worker.cfg.algorithm.get(
                        "fpo_ratio_granularity", "per_action"
                    )
                ),
                "score_parameterization": str(
                    cosmos_cfg.get("fpo_score_parameterization", "velocity")
                ),
                "num_mc_samples": int(cosmos_cfg.get("fpo_num_mc_samples", 0)),
                "elapsed_seconds": time.monotonic() - started_at,
            }
        )
        _atomic_jsonl(output_dir / f"update_{update_id:04d}.jsonl", merged)
        _atomic_json(output_dir / f"summary_{update_id:04d}.json", summary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return summary


def _install_optimizer_step_diagnostics(worker):
    _initialize_weight_diagnostics(worker)
    original_step = worker.optimizer.step
    context: dict[str, Any] = {
        "step_index": 0,
        "first_before": None,
        "last_after": None,
    }

    def diagnostic_step(*args, **kwargs):
        step_index = int(context["step_index"]) + 1
        before = _snapshot_action_parameters(worker)
        previous_after = context.get("last_after")
        if previous_after is not None:
            lifecycle = _compare_action_snapshots(
                previous_after,
                before,
                phase=f"between_optimizer_step_{step_index - 1}_and_{step_index}",
            )
            _append_weight_diagnostic(worker, lifecycle)
        if context.get("first_before") is None:
            context["first_before"] = before
        state_before = _optimizer_state_summary(worker.optimizer)
        try:
            result = original_step(*args, **kwargs)
        except Exception as exc:
            _append_weight_diagnostic(
                worker,
                {
                    "phase": f"optimizer_step_{step_index}_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "optimizer_state_before": state_before,
                },
            )
            raise
        after = _snapshot_action_parameters(worker)
        comparison = _compare_action_snapshots(
            before,
            after,
            phase=f"optimizer_step_{step_index}",
        )
        comparison["optimizer_state_before"] = state_before
        comparison["optimizer_state_after"] = _optimizer_state_summary(
            worker.optimizer
        )
        _append_weight_diagnostic(worker, comparison)
        context["step_index"] = step_index
        context["last_after"] = after
        worker.log_on_first_rank(
            "Stage 1.6 weight delta after optimizer step "
            f"{step_index}: changed={comparison['changed_element_count']}/"
            f"{comparison['element_count']}, max_abs_delta="
            f"{comparison['max_abs_delta']:.8g}."
        )
        return result

    worker.optimizer.step = diagnostic_step
    return original_step, context


def _record_post_training_weight_diagnostics(worker, context) -> dict[str, Any]:
    post_training = _snapshot_action_parameters(worker)
    last_after = context.get("last_after")
    first_before = context.get("first_before")
    if last_after is None or first_before is None:
        raise RuntimeError(
            "Stage 1.6 observed no optimizer steps in a required update"
        )
    _append_weight_diagnostic(
        worker,
        _compare_action_snapshots(
            last_after,
            post_training,
            phase="after_last_optimizer_to_training_return",
        ),
    )
    _append_weight_diagnostic(
        worker,
        _compare_action_snapshots(
            first_before,
            post_training,
            phase="training_total_before_exact_replay",
        ),
    )
    _append_weight_diagnostic(
        worker,
        {
            "phase": "optimizer_step_contract",
            "observed_optimizer_steps": int(context["step_index"]),
            "optimizer_state_after_training": _optimizer_state_summary(
                worker.optimizer
            ),
        },
    )
    return post_training


def _record_post_exact_replay_weight_diagnostics(worker, post_training) -> None:
    post_exact = _snapshot_action_parameters(worker)
    comparison = _compare_action_snapshots(
        post_training,
        post_exact,
        phase="exact_replay_and_reshard",
    )
    _append_weight_diagnostic(worker, comparison)
    worker.log_on_first_rank(
        "Stage 1.6 exact replay/reshard delta: "
        f"changed={comparison['changed_element_count']}/"
        f"{comparison['element_count']}."
    )


def _patch_actor_exact_replay() -> None:
    if _MODE == "baseline":
        return
    try:
        import rlinf.workers.actor.fsdp_actor_worker as module
    except ModuleNotFoundError:
        return
    actor_class = module.EmbodiedFSDPActor
    if hasattr(actor_class, "_stage15_original_run_training"):
        return
    original = actor_class.run_training
    actor_class._stage15_original_run_training = original

    def stage15_run_training(self, *args, **kwargs):
        finalize = self._finalize_trajectory_records
        release = self._release_rollout_batch
        offload = self._offload_actor_state
        self._finalize_trajectory_records = lambda: None
        self._release_rollout_batch = lambda: None
        self._offload_actor_state = lambda: None
        finalized = False
        original_optimizer_step = None
        diagnostic_context = None
        try:
            if _WEIGHT_DIAGNOSTICS:
                original_optimizer_step, diagnostic_context = (
                    _install_optimizer_step_diagnostics(self)
                )
            try:
                result = original(self, *args, **kwargs)
            finally:
                if original_optimizer_step is not None:
                    self.optimizer.step = original_optimizer_step
            post_training = None
            if _WEIGHT_DIAGNOSTICS:
                post_training = _record_post_training_weight_diagnostics(
                    self, diagnostic_context
                )
            _run_exact_replay(self, module)
            if _WEIGHT_DIAGNOSTICS:
                _record_post_exact_replay_weight_diagnostics(
                    self, post_training
                )
            finalize()
            finalized = True
            return result
        finally:
            if original_optimizer_step is not None:
                self.optimizer.step = original_optimizer_step
            self._finalize_trajectory_records = finalize
            self._release_rollout_batch = release
            self._offload_actor_state = offload
            if not finalized:
                self.log_on_first_rank(
                    "Stage 1.5 exact replay did not complete; releasing batch"
                )
            release()
            offload()

    actor_class.run_training = stage15_run_training


def _patch_actor_sync_diagnostics() -> None:
    try:
        import rlinf.workers.actor.fsdp_actor_worker as module
    except ModuleNotFoundError:
        return
    actor_class = module.EmbodiedFSDPActor
    if hasattr(actor_class, "_stage16_original_sync_model_to_rollout"):
        return
    original_sync = actor_class.sync_model_to_rollout
    actor_class._stage16_original_sync_model_to_rollout = original_sync

    def stage16_sync_model_to_rollout(self, *args, **kwargs):
        if not _WEIGHT_DIAGNOSTICS:
            return original_sync(self, *args, **kwargs)
        model_class = type(self.model)
        original_builder = getattr(
            model_class, "native_rollout_state_dict", None
        )
        if not callable(original_builder):
            return original_sync(self, *args, **kwargs)

        def capturing_builder(model_self, *builder_args, **builder_kwargs):
            state_dict = original_builder(
                model_self, *builder_args, **builder_kwargs
            )
            if model_self is self.model:
                _append_weight_diagnostic(
                    self, _state_dict_identity(state_dict)
                )
            return state_dict

        model_class.native_rollout_state_dict = capturing_builder
        try:
            return original_sync(self, *args, **kwargs)
        finally:
            model_class.native_rollout_state_dict = original_builder

    actor_class.sync_model_to_rollout = stage16_sync_model_to_rollout


def install() -> None:
    if _MODE not in _VALID_MODES:
        raise ValueError(
            f"RLINF_STAGE15_MODE must be one of {sorted(_VALID_MODES)}, got {_MODE!r}"
        )
    _patch_real_bridge()
    _patch_runner_completion()
    _patch_sparse_eval_batching()
    _patch_actor_exact_replay()
    _patch_actor_sync_diagnostics()
