"""Dependency-light trajectory analysis for standalone RLinf runs.

The training process atomically publishes one ``update_XXXX.jsonl`` file per
completed update.  This module consumes only those committed files and the
scalar metrics/config/status files in a run directory.  It never opens videos
or checkpoints and is therefore safe to run on a login node while training is
still active.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


CORRELATION_SPECS = (
    ("trajectory_mse", "Trajectory MSE", "trajectory_mse"),
    ("lastframe_mse", "Terminal-goal MSE", "terminal_goal_mse"),
    ("combined_reward", "Combined reward", "combined_reward"),
    (
        "success_probability_max",
        "Success-classifier probability (last-4 max)",
        "success_probability",
    ),
)

BINARY_PLOT_SPECS = (
    ("trajectory_mse", "Trajectory MSE", "trajectory_mse"),
    ("lastframe_mse", "Last-frame MSE", "last_frame_mse"),
    ("combined_reward", "Combined reward", "combined"),
)

TRAINING_METRICS = {
    "step_time_s": "time/step",
    "rollout_time_s": "time/generate_rollouts",
    "actor_training_time_s": "time/actor/run_training",
    "policy_loss": "train/actor/policy_loss",
    "total_loss": "train/actor/total_loss",
    "approx_kl": "train/actor/approx_kl",
    "actor_clip_fraction": "train/actor/clip_fraction",
    "action_ratio_mean_metric": "train/action/ratio_mean",
    "grad_norm": "train/actor/grad_norm",
    "optimizer_steps": "train/actor/optimizer_steps_per_update",
    "gpu_peak_reserved_fraction": "train/hardware/gpu_peak_reserved_fraction_max",
    "invalid_trajectory_count_metric": "train/group_invalid_trajectory_count",
}

REQUIRED_TRAJECTORY_FIELDS = (
    "update",
    "group_id",
    "member_id",
    "rollout_uid",
    "success",
    "success_probability_max",
    "success_threshold",
    "success_last4_probabilities",
    "trajectory_reward",
    "trajectory_mse",
    "lastframe_reward",
    "lastframe_mse",
    "combined_reward",
    "chunk_rewards",
    "chunk_advantages",
    "old_chunk_logprobs",
    "new_chunk_logprobs",
    "chunk_ratios",
    "chunk_clip_fractions",
    "group_mean",
    "group_std",
    "normalized_advantage",
    "ratio",
    "clip_fraction",
    "valid",
)


@dataclass(frozen=True)
class RunPaths:
    """Resolved input and output paths for one training run."""

    run_dir: Path
    output_dir: Path
    trajectory_dir: Path
    metrics_history: Path
    status_file: Path
    config_file: Path

    @classmethod
    def create(cls, run_dir: Path, output_dir: Path | None = None) -> RunPaths:
        resolved = run_dir.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {resolved}")
        target = (output_dir or resolved / "analysis").expanduser().resolve()
        return cls(
            run_dir=resolved,
            output_dir=target,
            trajectory_dir=resolved / "trajectory_records",
            metrics_history=resolved / "metrics_history.jsonl",
            status_file=resolved / "status.json",
            config_file=resolved / "resolved_config.yaml",
        )


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else math.nan


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - half_width, center + half_width


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 3e-14 else 3e-14)
    result = d
    for iteration in range(1, 201):
        iteration2 = 2 * iteration
        coefficient = (
            iteration * (b - iteration) * x / ((qam + iteration2) * (a + iteration2))
        )
        d = 1 + coefficient * d
        c = 1 + coefficient / c
        d = 1.0 / (d if abs(d) > 3e-14 else 3e-14)
        result *= d * c
        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + iteration2) * (qap + iteration2))
        )
        d = 1 + coefficient * d
        c = 1 + coefficient / c
        d = 1.0 / (d if abs(d) > 3e-14 else 3e-14)
        delta = d * c
        result *= delta
        if abs(delta - 1) < 3e-12:
            break
    return result


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1 - factor * _beta_continued_fraction(b, a, 1 - x) / b


def pearson_with_p_value(
    x_values: Sequence[float], y_values: Sequence[float]
) -> tuple[float, float, bool]:
    """Return Pearson r and its two-sided Student-t p-value."""
    if (
        len(x_values) != len(y_values)
        or len(x_values) < 3
        or max(x_values) == min(x_values)
        or max(y_values) == min(y_values)
    ):
        return math.nan, math.nan, False
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    )
    denominator = math.sqrt(
        sum((value - x_mean) ** 2 for value in x_values)
        * sum((value - y_mean) ** 2 for value in y_values)
    )
    correlation = max(-1.0, min(1.0, numerator / denominator))
    degrees_freedom = len(x_values) - 2
    t_squared = (
        correlation * correlation * degrees_freedom
        / max(1 - correlation * correlation, 1e-15)
    )
    p_value = _regularized_incomplete_beta(
        degrees_freedom / (degrees_freedom + t_squared),
        degrees_freedom / 2,
        0.5,
    )
    return correlation, p_value, True


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"resolved config is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"resolved config must contain a mapping: {path}")
    return value


def _read_metrics(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
            step = int(record["step"])
            records[step] = record
    return records


def _finite_float(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def _success_from_record(record: dict[str, Any]) -> bool:
    """Recompute success using the decision rule that produced the record."""
    threshold = float(record["success_threshold"])
    rule = str(record.get("success_decision_rule", "legacy_max_window"))
    if rule != "terminal_positive_ratio":
        return float(record["success_probability_max"]) >= threshold

    probabilities = record.get("success_terminal_probabilities")
    if not isinstance(probabilities, list) or not probabilities:
        raise ValueError(
            "terminal_positive_ratio records require "
            "success_terminal_probabilities"
        )
    positive = [float(value) >= threshold for value in probabilities]
    positive_ratio = sum(positive) / len(positive)
    minimum_ratio = float(record.get("success_minimum_positive_ratio", 0.8))
    requires_last = bool(
        record.get("success_requires_last_frame_positive", True)
    )
    return positive_ratio >= minimum_ratio and (
        positive[-1] or not requires_last
    )


def _read_trajectories(
    path: Path,
    *,
    expected_trajectories: int,
    expected_chunks: int,
    expected_group_size: int,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    if not path.is_dir():
        raise RuntimeError(
            f"trajectory directory does not exist yet: {path}; "
            "recording begins with the first update run after it was enabled"
        )
    by_step: dict[int, list[dict[str, Any]]] = {}
    duplicate_uids: list[int] = []
    incomplete_steps: dict[int, int] = {}
    for record_path in sorted(path.glob("update_*.jsonl")):
        rows: list[dict[str, Any]] = []
        with record_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON in {record_path}:{line_number}: {error}"
                    ) from error
                missing = [name for name in REQUIRED_TRAJECTORY_FIELDS if name not in row]
                if missing:
                    raise ValueError(
                        f"trajectory {record_path}:{line_number} is missing fields: {missing}"
                    )
                for field in (
                    "trajectory_reward",
                    "trajectory_mse",
                    "lastframe_reward",
                    "lastframe_mse",
                    "combined_reward",
                    "success_probability_max",
                    "group_mean",
                    "group_std",
                    "normalized_advantage",
                    "ratio",
                    "clip_fraction",
                ):
                    _finite_float(row[field], label=f"{record_path.name}:{field}")
                for field in (
                    "chunk_rewards",
                    "chunk_advantages",
                    "old_chunk_logprobs",
                    "new_chunk_logprobs",
                    "chunk_ratios",
                    "chunk_clip_fractions",
                ):
                    if len(row[field]) != expected_chunks:
                        raise ValueError(
                            f"{record_path.name}:{field} has {len(row[field])} values; "
                            f"expected {expected_chunks}"
                        )
                if len(row["success_last4_probabilities"]) != 4:
                    raise ValueError(
                        f"{record_path.name}: success_last4_probabilities must have length 4"
                    )
                rows.append(row)
        if not rows:
            continue
        update_ids = {int(row["update"]) for row in rows}
        if len(update_ids) != 1:
            raise ValueError(f"mixed update IDs in {record_path}: {sorted(update_ids)}")
        step = update_ids.pop() + 1
        if step in by_step:
            raise ValueError(f"multiple committed trajectory files map to global step {step}")
        uids = [int(row["rollout_uid"]) for row in rows]
        if len(set(uids)) != len(uids):
            duplicate_uids.extend(uid for uid in uids if uids.count(uid) > 1)
        if expected_trajectories and len(rows) != expected_trajectories:
            incomplete_steps[step] = len(rows)
        groups: dict[int, int] = defaultdict(int)
        for row in rows:
            groups[int(row["group_id"])] += 1
        invalid_groups = {
            group_id: count
            for group_id, count in groups.items()
            if count != expected_group_size
        }
        if invalid_groups:
            raise ValueError(
                f"global step {step} has non-{expected_group_size} trajectory groups: "
                f"{invalid_groups}"
            )
        by_step[step] = rows
    if not by_step:
        raise RuntimeError(f"no committed update_*.jsonl trajectory records found in {path}")
    if duplicate_uids:
        raise ValueError(f"duplicate rollout UIDs found: {sorted(set(duplicate_uids))[:10]}")
    if incomplete_steps:
        raise ValueError(
            "committed trajectory counts violate the batch contract: "
            f"{incomplete_steps}; expected {expected_trajectories} per update"
        )
    return by_step, {
        "committed_files": len(by_step),
        "incomplete_steps": incomplete_steps,
        "duplicate_rollout_uids": len(duplicate_uids),
    }


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None or value == "":
        return math.nan
    return float(value)


def _summarize_rows(
    step: int,
    rows: list[dict[str, Any]],
    *,
    expected: int,
    metrics: dict[str, Any] | None,
    variant: str | None = None,
) -> dict[str, Any]:
    valid = [row for row in rows if bool(row.get("valid", True))]
    if not valid:
        raise ValueError(f"global step {step} has no valid trajectory records")
    successes = [row for row in valid if bool(row["success"])]
    failures = [row for row in valid if not bool(row["success"])]
    rewards = [float(row["combined_reward"]) for row in valid]
    group_success: dict[int, list[float]] = defaultdict(list)
    for row in valid:
        group_success[int(row["group_id"])].append(float(bool(row["success"])))
    group_rates = [mean(values) for values in group_success.values()]
    ci_low, ci_high = wilson_interval(len(successes), len(valid))
    summary = {
        "step": step,
        "update_id": step - 1,
        "variant": variant or "all",
        "data_status": "complete",
        "n_trajectories": len(rows),
        "n_expected_trajectories": expected,
        "n_valid": len(valid),
        "n_invalid": len(rows) - len(valid),
        "n_success": len(successes),
        "n_failure": len(failures),
        "success_rate": len(successes) / len(valid),
        "success_ci95_low": ci_low,
        "success_ci95_high": ci_high,
        "success_probability_max_mean": mean(
            [float(row["success_probability_max"]) for row in valid]
        ),
        "success_probability_last4_mean": mean(
            [
                mean([float(value) for value in row["success_last4_probabilities"]])
                for row in valid
            ]
        ),
        "reward_mean": mean(rewards),
        "reward_std": sample_std(rewards),
        "reward_median": quantile(rewards, 0.5),
        "reward_p10": quantile(rewards, 0.1),
        "reward_p25": quantile(rewards, 0.25),
        "reward_p75": quantile(rewards, 0.75),
        "reward_p90": quantile(rewards, 0.9),
        "reward_best": max(rewards),
        "reward_worst": min(rewards),
        "trajectory_reward_mean": mean(
            [float(row["trajectory_reward"]) for row in valid]
        ),
        "terminal_reward_mean": mean([float(row["lastframe_reward"]) for row in valid]),
        "trajectory_mse_mean": mean([float(row["trajectory_mse"]) for row in valid]),
        "terminal_goal_mse_mean": mean([float(row["lastframe_mse"]) for row in valid]),
        "success_reward_mean": mean(
            [float(row["combined_reward"]) for row in successes]
        ),
        "failure_reward_mean": mean(
            [float(row["combined_reward"]) for row in failures]
        ),
        "group_success_rate_std": sample_std(group_rates),
        "group_success_rate_min": min(group_rates),
        "group_success_rate_max": max(group_rates),
        "n_groups": len(group_rates),
        "ratio_mean": mean([float(row["ratio"]) for row in valid]),
        "ratio_min": min(float(row["ratio"]) for row in valid),
        "ratio_max": max(float(row["ratio"]) for row in valid),
        "clip_fraction_mean": mean([float(row["clip_fraction"]) for row in valid]),
        "normalized_advantage_mean": mean(
            [float(row["normalized_advantage"]) for row in valid]
        ),
        "normalized_advantage_std": sample_std(
            [float(row["normalized_advantage"]) for row in valid]
        ),
        "n_retry_trajectories": sum(bool(row.get("retry_count")) for row in rows),
        "n_exceptions": sum(row.get("exception") is not None for row in rows),
    }
    metric_values = (metrics or {}).get("metrics", {})
    for output_name, metric_name in TRAINING_METRICS.items():
        summary[output_name] = _metric_value(metric_values, metric_name)
    return summary


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _trajectory_csv_rows(
    trajectories_by_step: dict[int, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    preferred = (
        "logical_round",
        "physical_wave",
        "group_slot",
        "group_id",
        "member_id",
        "rollout_uid",
        "actor_rank",
        "source_env_rank",
        "local_env_id",
        "node",
        "gpu",
        "reset_episode",
        "color",
        "success",
        "success_probability_max",
        "success_threshold",
        "success_last4_probabilities",
        "trajectory_reward",
        "trajectory_mse",
        "trajectory_similarity",
        "lastframe_reward",
        "lastframe_mse",
        "lastframe_similarity",
        "combined_reward",
        "group_mean",
        "group_std",
        "normalized_advantage",
        "old_logprob",
        "new_logprob",
        "raw_log_ratio",
        "log_ratio",
        "ratio",
        "action_ratio_mean",
        "clip_indicator",
        "clip_fraction",
        "chunk_rewards",
        "chunk_advantages",
        "old_chunk_logprobs",
        "new_chunk_logprobs",
        "chunk_raw_log_ratios",
        "chunk_log_ratios",
        "chunk_ratios",
        "chunk_clip_indicators",
        "chunk_clip_fractions",
        "reset_seed",
        "ctrl_seed",
        "shuffle_seed",
        "cosmos_joint_seeds",
        "cosmos_seed_nonces",
        "retry_count",
        "valid",
        "validity_status",
        "exception",
        "provenance_mode",
    )
    for step, trajectories in sorted(trajectories_by_step.items()):
        for trajectory in trajectories:
            row = {"global_step": step, "update_id": int(trajectory["update"])}
            row.update({name: trajectory.get(name) for name in preferred})
            rows.append(row)
    return rows


def _correlation_record(
    values: Sequence[float], successes: Sequence[int], step: int | str
) -> dict[str, Any]:
    successful = [value for value, success in zip(values, successes) if success]
    failed = [value for value, success in zip(values, successes) if not success]
    correlation, p_value, defined = pearson_with_p_value(values, successes)
    success_mean = mean(successful)
    failure_mean = mean(failed)
    return {
        "global_step": step,
        "n_trajectories": len(values),
        "n_success": len(successful),
        "n_failure": len(failed),
        "success_rate": mean(successes),
        "metric_mean": mean(values),
        "metric_std": sample_std(values),
        "success_mean": success_mean,
        "failure_mean": failure_mean,
        "success_minus_failure": success_mean - failure_mean,
        "pearson_point_biserial_r": correlation,
        "p_value": p_value,
        "correlation_defined": int(defined),
    }


def _write_correlations(
    output_dir: Path,
    trajectories_by_step: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    correlation_dir = output_dir / "correlations"
    overview: list[dict[str, Any]] = []
    for source, label, stem in CORRELATION_SPECS:
        records: list[dict[str, Any]] = []
        all_values: list[float] = []
        all_successes: list[int] = []
        for step, raw_rows in sorted(trajectories_by_step.items()):
            rows = [row for row in raw_rows if bool(row.get("valid", True))]
            values = [float(row[source]) for row in rows]
            successes = [int(bool(row["success"])) for row in rows]
            records.append(_correlation_record(values, successes, step))
            all_values.extend(values)
            all_successes.extend(successes)
        overall = _correlation_record(all_values, all_successes, "ALL")
        records.append(overall)
        _write_csv(correlation_dir / f"{stem}_vs_success.csv", records)
        overview.append(
            {
                "metric": label,
                "source_column": source,
                "steps": len(trajectories_by_step),
                "trajectories": len(all_values),
                "overall_success_rate": overall["success_rate"],
                "overall_r": overall["pearson_point_biserial_r"],
                "overall_p_value": overall["p_value"],
                "success_mean": overall["success_mean"],
                "failure_mean": overall["failure_mean"],
            }
        )
    _write_csv(correlation_dir / "correlation_overview.csv", overview)
    return overview


def _gnuplot_number(value: float) -> str:
    return "NaN" if math.isnan(value) else repr(value)


def _run_gnuplot(script: str, *, required: bool) -> bool:
    executable = shutil.which("gnuplot")
    if executable is None:
        if required:
            raise RuntimeError("gnuplot is required for PNG generation but was not found")
        print("[analysis] gnuplot not found; CSV/JSON/Markdown/SVG outputs were generated")
        return False
    subprocess.run([executable], input=script, text=True, check=True)
    return True


def _write_binary_correlation_suite(
    output_dir: Path,
    trajectories: list[dict[str, Any]],
    *,
    threshold: float,
    include_success_curve: bool,
    require_plots: bool,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        grouped[int(row["update"]) + 1].append(row)
    steps = sorted(grouped)
    step_min, step_max = min(steps), max(steps)
    palette_min = step_min if step_min != step_max else step_min - 0.5
    palette_max = step_max if step_min != step_max else step_max + 0.5
    palette_mid = (palette_min + palette_max) / 2
    step_tick = max(1, math.ceil(len(steps) / 10))
    overview: list[dict[str, Any]] = []
    for source, label, stem in BINARY_PLOT_SPECS:
        records: list[dict[str, Any]] = []
        all_values: list[float] = []
        all_successes: list[int] = []
        point_lines: list[str] = []
        point_index = 0
        for step in steps:
            rows = grouped[step]
            values = [float(row[source]) for row in rows]
            successes = [
                int(float(row["success_probability_max"]) >= threshold) for row in rows
            ]
            records.append(_correlation_record(values, successes, step))
            all_values.extend(values)
            all_successes.extend(successes)
            for value, success in zip(values, successes):
                jitter = ((((point_index * 37) % 101) / 100) - 0.5) * 0.11
                point_lines.append(f"{success + jitter} {value} {step}")
                point_index += 1
        overall = _correlation_record(all_values, all_successes, "ALL")
        records.append(overall)
        _write_csv(output_dir / f"{stem}_vs_success.csv", records)
        points = output_dir / f".{stem}_vs_success_points.dat"
        trends = output_dir / f".{stem}_vs_success_trends.dat"
        _atomic_write_text(points, "\n".join(point_lines) + "\n")
        _atomic_write_text(
            trends,
            "\n".join(
                f"{record['global_step']} "
                f"{_gnuplot_number(float(record['pearson_point_biserial_r']))} "
                f"{record['success_rate']}"
                for record in records[:-1]
            )
            + "\n",
        )
        image = output_dir / f"{stem}_vs_success.png"
        _run_gnuplot(
            f'''
set terminal pngcairo size 1800,720 font "Sans,12"
set output "{image.as_posix()}"
set multiplot layout 1,2 title "{label} vs binary success at threshold {threshold:g} | overall r={_gnuplot_number(float(overall['pearson_point_biserial_r']))}"
set title "All trajectories (n={len(all_values)})"
set xrange [-0.2:1.2]
set xtics ("Failure" 0, "Success" 1)
set ylabel "{label}"
set grid ytics lc rgb "#d9d9d9"
set palette defined ({palette_min} "#440154", {palette_mid:g} "#21918c", {palette_max} "#fde725")
set cbrange [{palette_min}:{palette_max}]
set cblabel "RL step"
plot "{points.as_posix()}" using 1:2:3 with points pt 7 ps 0.48 palette notitle
unset colorbox
unset xtics
set xrange [{step_min - 0.5}:{step_max + 0.5}]
set xtics {step_min},{step_tick},{step_max}
set title "Per-step correlation and success rate"
set xlabel "RL step"
set ylabel "Point-biserial r"
set yrange [-1.05:1.05]
set y2label "Success rate"
set y2range [-0.03:1.03]
set y2tics
plot 0 with lines lc rgb "#777777" notitle, \
     "{trends.as_posix()}" using 1:2 with linespoints lw 2 title "correlation", \
     "{trends.as_posix()}" using 1:3 axes x1y2 with linespoints lw 2 title "success rate"
unset multiplot
''',
            required=require_plots,
        )
        overview.append(
            {
                "metric": label,
                "source_column": source,
                "success_threshold": threshold,
                "steps": len(steps),
                "trajectories": len(all_values),
                "overall_success_rate": overall["success_rate"],
                "overall_r": overall["pearson_point_biserial_r"],
                "overall_p_value": overall["p_value"],
                "success_mean": overall["success_mean"],
                "failure_mean": overall["failure_mean"],
            }
        )
    _write_csv(output_dir / "correlation_overview.csv", overview)
    if include_success_curve:
        rate_rows = []
        for step in steps:
            rows = grouped[step]
            successes = sum(
                float(row["success_probability_max"]) >= threshold for row in rows
            )
            rate_rows.append(
                {
                    "global_step": step,
                    "average_success_rate": successes / len(rows),
                    "n_success": successes,
                    "n_trajectories": len(rows),
                }
            )
        table = output_dir / "average_success_rate_by_rl_step.csv"
        image = output_dir / "average_success_rate_by_rl_step.png"
        _write_csv(table, rate_rows)
        _run_gnuplot(
            f'''
set terminal pngcairo size 1400,800 font "Sans,14"
set output "{image.as_posix()}"
set datafile separator comma
set title "Average success rate vs RL step (threshold {threshold:g})"
set xlabel "RL step"
set xrange [{step_min - 0.5}:{step_max + 0.5}]
set xtics {step_min},{step_tick},{step_max}
set ylabel "Average success rate"
set yrange [0:1.05]
set grid
plot "{table.as_posix()}" every ::1 using 1:2 with linespoints lw 2.5 title "success rate"
''',
            required=require_plots,
        )
    return overview


def _write_probability_correlations(
    output_dir: Path,
    trajectories: list[dict[str, Any]],
    *,
    require_plots: bool,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = [float(row["success_probability_max"]) for row in trajectories]
    steps = [int(row["update"]) + 1 for row in trajectories]
    step_min, step_max = min(steps), max(steps)
    palette_min = step_min if step_min != step_max else step_min - 0.5
    palette_max = step_max if step_min != step_max else step_max + 0.5
    palette_mid = (palette_min + palette_max) / 2
    probability_mean = mean(probabilities)
    probability_variance = sum(
        (probability - probability_mean) ** 2 for probability in probabilities
    )
    overview = []
    for source, label, stem in BINARY_PLOT_SPECS:
        values = [float(row[source]) for row in trajectories]
        correlation, p_value, defined = pearson_with_p_value(values, probabilities)
        value_mean = mean(values)
        if probability_variance:
            slope = sum(
                (probability - probability_mean) * (value - value_mean)
                for probability, value in zip(probabilities, values)
            ) / probability_variance
            intercept = value_mean - slope * probability_mean
        else:
            slope = intercept = math.nan
        points = output_dir / f".{stem}_vs_success_probability_points.dat"
        _atomic_write_text(
            points,
            "\n".join(
                f"{probability} {value} {step}"
                for probability, value, step in zip(
                    probabilities, values, steps
                )
            )
            + "\n",
        )
        image = output_dir / f"{stem}_vs_success_probability.png"
        fit_definition = fit_clause = ""
        if math.isfinite(slope):
            fit_definition = f"fit_line(x) = {intercept} + {slope} * x"
            fit_clause = (
                ", fit_line(x) with lines lw 3 lc rgb \"#d62728\" title \"linear fit\""
            )
        _run_gnuplot(
            f'''
set terminal pngcairo size 1400,900 font "Sans,14"
set output "{image.as_posix()}"
set title "{label} vs success probability | Pearson r={_gnuplot_number(correlation)}"
set xlabel "Success-classifier probability (last-4 max)"
set ylabel "{label}"
set xrange [0:1]
set grid
set palette defined ({palette_min} "#440154", {palette_mid:g} "#21918c", {palette_max} "#fde725")
set cbrange [{palette_min}:{palette_max}]
set cblabel "RL step"
{fit_definition}
plot "{points.as_posix()}" using 1:2:3 with points pt 7 ps 0.55 palette title "trajectories"{fit_clause}
''',
            required=require_plots,
        )
        overview.append(
            {
                "metric": label,
                "source_column": source,
                "trajectories": len(values),
                "pearson_r": correlation,
                "p_value": p_value,
                "correlation_defined": int(defined),
                "linear_fit_intercept": intercept,
                "linear_fit_slope": slope,
            }
        )
    _write_csv(output_dir / "correlation_overview.csv", overview)
    return overview


def _write_learning_curve_png(
    output_dir: Path,
    step_rows: list[dict[str, Any]],
    color_rows: list[dict[str, Any]],
    *,
    require_plots: bool,
) -> None:
    curve_table = output_dir / ".learning_curves.dat"
    _atomic_write_text(
        curve_table,
        "\n".join(
            f"{row['step']} {row['success_rate']} {row['success_ci95_low']} "
            f"{row['success_ci95_high']} {row['reward_mean']} {row['reward_p10']} "
            f"{row['reward_p90']}"
            for row in step_rows
        )
        + "\n",
    )
    image = output_dir / "success_reward_curves.png"
    _run_gnuplot(
        f'''
set terminal pngcairo size 1800,760 font "Sans,13"
set output "{image.as_posix()}"
set multiplot layout 1,2 title "Training success and reward"
set grid
set xlabel "RL step"
set ylabel "Success rate"
set yrange [0:1.05]
plot "{curve_table.as_posix()}" using 1:2:3:4 with yerrorlines lw 2 pt 7 title "success (95% Wilson CI)"
set ylabel "Combined reward (higher is better)"
set autoscale y
plot "{curve_table.as_posix()}" using 1:5 with linespoints lw 2 pt 7 title "mean reward", \
     "{curve_table.as_posix()}" using 1:6 with lines lw 1 title "P10", \
     "{curve_table.as_posix()}" using 1:7 with lines lw 1 title "P90"
unset multiplot
''',
        required=require_plots,
    )
    if not color_rows:
        return
    variants = sorted({str(row["variant"]) for row in color_rows})
    data_files: list[Path] = []
    for variant in variants:
        path = output_dir / f".color_{variant}_curves.dat"
        selected = [row for row in color_rows if row["variant"] == variant]
        _atomic_write_text(
            path,
            "\n".join(
                f"{row['step']} {row['success_rate']} {row['reward_mean']}" for row in selected
            )
            + "\n",
        )
        data_files.append(path)
    success_plots = ", \\\n     ".join(
        f'"{path.as_posix()}" using 1:2 with linespoints lw 2 title "{variant}"'
        for path, variant in zip(data_files, variants)
    )
    reward_plots = ", \\\n     ".join(
        f'"{path.as_posix()}" using 1:3 with linespoints lw 2 title "{variant}"'
        for path, variant in zip(data_files, variants)
    )
    image = output_dir / "per_color_success_reward_curves.png"
    _run_gnuplot(
        f'''
set terminal pngcairo size 1800,760 font "Sans,13"
set output "{image.as_posix()}"
set multiplot layout 1,2 title "Duck per-color training metrics"
set grid
set xlabel "RL step"
set ylabel "Success rate"
set yrange [0:1.05]
plot {success_plots}
set ylabel "Combined reward (higher is better)"
set autoscale y
plot {reward_plots}
unset multiplot
''',
        required=require_plots,
    )


def _svg_chart(
    title: str,
    x: float,
    y: float,
    width: float,
    height: float,
    series: list[tuple[str, str, list[tuple[float, float]]]],
    *,
    y_domain: tuple[float, float] | None = None,
    percent: bool = False,
) -> str:
    clean_series = [
        (label, color, [(px, py) for px, py in points if math.isfinite(py)])
        for label, color, points in series
    ]
    points = [point for _, _, values in clean_series for point in values]
    if not points:
        return ""
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_domain is None:
        y_min, y_max = min(y_values), max(y_values)
        span = y_max - y_min
        padding = span * 0.12 if span else max(abs(y_min) * 0.1, 1e-3)
        y_min, y_max = y_min - padding, y_max + padding
    else:
        y_min, y_max = y_domain
    plot_x, plot_y = x + 58, y + 40
    plot_width, plot_height = width - 80, height - 82

    def sx(value: float) -> float:
        return plot_x + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return plot_y + (y_max - value) / (y_max - y_min) * plot_height

    parts = [
        f'<g><rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" '
        'fill="#fff" stroke="#d9dee7"/>',
        f'<text x="{x + 16}" y="{y + 25}" font-size="16" font-weight="600">'
        f"{html.escape(title)}</text>",
    ]
    for grid_index in range(5):
        fraction = grid_index / 4
        grid_y = plot_y + fraction * plot_height
        value = y_max - fraction * (y_max - y_min)
        label = f"{value * 100:.0f}%" if percent else f"{value:.3g}"
        parts.extend(
            [
                f'<line x1="{plot_x}" y1="{grid_y}" x2="{plot_x + plot_width}" '
                'y2="{grid_y}" stroke="#e9edf3"/>',
                f'<text x="{plot_x - 8}" y="{grid_y + 4}" text-anchor="end" '
                f'font-size="11" fill="#596273">{label}</text>',
            ]
        )
    for step in sorted(set(x_values)):
        parts.append(
            f'<text x="{sx(step)}" y="{plot_y + plot_height + 20}" text-anchor="middle" '
            f'font-size="11" fill="#596273">{int(step)}</text>'
        )
    for index, (label, color, values) in enumerate(clean_series):
        if not values:
            continue
        coordinates = " ".join(f"{sx(px):.1f},{sy(py):.1f}" for px, py in values)
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="2.2" stroke-linejoin="round"/>'
        )
        for px, py in values:
            parts.append(
                f'<circle cx="{sx(px):.1f}" cy="{sy(py):.1f}" r="3.5" '
                f'fill="{color}" stroke="#fff"/>'
            )
        parts.append(
            f'<text x="{x + width - 14}" y="{y + 21 + index * 17}" text-anchor="end" '
            f'font-size="11" fill="{color}">{html.escape(label)}</text>'
        )
    parts.append("</g>")
    return "".join(parts)


def _write_overview_svg(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    charts = [
        _svg_chart(
            "Task success rate",
            30,
            70,
            570,
            330,
            [("Success rate", "#1677c8", [(row["step"], row["success_rate"]) for row in rows])],
            y_domain=(0.0, 1.0),
            percent=True,
        ),
        _svg_chart(
            "Combined reward (higher is better)",
            620,
            70,
            570,
            330,
            [
                ("Mean", "#d95f02", [(row["step"], row["reward_mean"]) for row in rows]),
                ("P10", "#9aa1ad", [(row["step"], row["reward_p10"]) for row in rows]),
                ("P90", "#3a9d5d", [(row["step"], row["reward_p90"]) for row in rows]),
            ],
        ),
        _svg_chart(
            "Reward MSE components (lower is better)",
            30,
            420,
            570,
            330,
            [
                (
                    "Trajectory MSE",
                    "#6a51a3",
                    [(row["step"], row["trajectory_mse_mean"]) for row in rows],
                ),
                (
                    "Terminal MSE",
                    "#e6550d",
                    [(row["step"], row["terminal_goal_mse_mean"]) for row in rows],
                ),
            ],
        ),
        _svg_chart(
            "PPO ratio and clip fraction",
            620,
            420,
            570,
            330,
            [
                ("Ratio mean", "#252525", [(row["step"], row["ratio_mean"]) for row in rows]),
                (
                    "Clip fraction",
                    "#31a354",
                    [(row["step"], row["clip_fraction_mean"]) for row in rows],
                ),
            ],
        ),
    ]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1220" height="790" '
        'viewBox="0 0 1220 790"><rect width="1220" height="790" fill="#f5f7fa"/>'
        '<text x="30" y="34" font-size="24" font-weight="700" fill="#1f2937">'
        'RL run overview</text>'
        f'<text x="30" y="55" font-size="12" fill="#596273">Committed steps: '
        f"{', '.join(str(row['step']) for row in rows)}</text>"
        + "".join(charts)
        + "</svg>"
    )
    _atomic_write_text(output_dir / "run_overview.svg", svg)


def _write_relationship_plot(
    output_dir: Path,
    trajectories: list[dict[str, Any]],
    *,
    require_plots: bool,
) -> dict[str, Any]:
    successes = [int(bool(row["success"])) for row in trajectories]
    steps = [int(row["update"]) + 1 for row in trajectories]
    step_min, step_max = min(steps), max(steps)
    palette_min = step_min if step_min != step_max else step_min - 0.5
    palette_max = step_max if step_min != step_max else step_max + 0.5
    palette_mid = (palette_min + palette_max) / 2
    specs = (
        ("success_probability_max", "Success probability", "reward_model_probability_max"),
        ("trajectory_mse", "Trajectory MSE", "video_similarity_mse_mean"),
        ("lastframe_mse", "Terminal-goal MSE", "terminal_goal_mse"),
        ("combined_reward", "Combined reward", "old_combined_episode_reward"),
    )
    columns: list[list[float]] = []
    panels: dict[str, Any] = {}
    for source, _label, legacy_name in specs:
        values = [float(row[source]) for row in trajectories]
        columns.append(values)
        correlation, p_value, _defined = pearson_with_p_value(values, successes)
        successful = [value for value, success in zip(values, successes) if success]
        failed = [value for value, success in zip(values, successes) if not success]
        panels[legacy_name] = {
            "mean_failure": mean(failed),
            "mean_success": mean(successful),
            "pearson_r": correlation,
            "p_value": p_value,
        }
    points = output_dir / ".success_similarity_relationship_points.dat"
    lines = []
    for index, (success, step) in enumerate(zip(successes, steps)):
        jitter = ((((index * 37) % 101) / 100) - 0.5) * 0.11
        values = " ".join(str(column[index]) for column in columns)
        lines.append(f"{success + jitter} {values} {step}")
    _atomic_write_text(points, "\n".join(lines) + "\n")
    image = output_dir / "success_similarity_relationship.png"
    titles = [
        f"{label} | r={_gnuplot_number(float(panels[legacy]['pearson_r']))}"
        for _source, label, legacy in specs
    ]
    _run_gnuplot(
        f'''
set terminal pngcairo size 2340,1620 font "Sans,15"
set output "{image.as_posix()}"
set multiplot layout 2,2 rowsfirst title "Reward metrics and diagnostic success (n={len(trajectories)})"
set xrange [-0.2:1.2]
set xtics ("Failure" 0, "Success" 1)
set grid ytics
set palette defined ({palette_min} "#440154", {palette_mid:g} "#21918c", {palette_max} "#fde725")
set cbrange [{palette_min}:{palette_max}]
set cblabel "RL step"
set title "{titles[0]}"
set ylabel "{specs[0][1]}"
plot "{points.as_posix()}" using 1:2:6 with points pt 7 ps 0.5 palette notitle
set title "{titles[1]}"
set ylabel "{specs[1][1]}"
plot "{points.as_posix()}" using 1:3:6 with points pt 7 ps 0.5 palette notitle
set title "{titles[2]}"
set ylabel "{specs[2][1]}"
plot "{points.as_posix()}" using 1:4:6 with points pt 7 ps 0.5 palette notitle
set title "{titles[3]}"
set ylabel "{specs[3][1]}"
plot "{points.as_posix()}" using 1:5:6 with points pt 7 ps 0.5 palette notitle
unset multiplot
''',
        required=require_plots,
    )
    summary = {
        "failure_count": successes.count(0),
        "num_trajectories": len(trajectories),
        "panels": panels,
        "success_count": successes.count(1),
        "success_rate": mean(successes),
    }
    _atomic_write_text(
        output_dir / "success_similarity_relationship.json",
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
    )
    return summary


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float) and math.isnan(value):
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _build_report(
    *,
    paths: RunPaths,
    config: dict[str, Any],
    status: dict[str, Any],
    step_rows: list[dict[str, Any]],
    color_rows: list[dict[str, Any]],
    cumulative: dict[str, Any],
    correlations: list[dict[str, Any]],
    validation: dict[str, Any],
) -> str:
    first, latest = step_rows[0], step_rows[-1]
    best_success = max(step_rows, key=lambda row: row["success_rate"])
    best_reward = max(step_rows, key=lambda row: row["reward_mean"])
    task = config["task"]
    batch = config["batch"]
    reward_source = str(
        config.get("reward", {}).get(
            "source", "ctrl_world_aligned_mse_plus_terminal"
        )
    )
    reward_equations = {
        "ctrl_world_aligned_mse_plus_terminal": (
            "combined_reward - trajectory_reward - terminal_reward"
        ),
        "trajectory_mse": "combined_reward - trajectory_reward",
        "terminal_goal_mse": "combined_reward - terminal_reward",
        "success_binary": "combined_reward - success_reward",
    }
    reward_equation = reward_equations.get(reward_source, "unknown")
    classifier_role = (
        "the sole training reward"
        if reward_source == "success_binary"
        else "diagnostic only"
    )
    lines = [
        f"# RL run analysis: {config.get('experiment_name', paths.run_dir.name)}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')} (UTC)",
        "",
        "## Snapshot",
        "",
        (
            f"Profile **{task['profile']}**, active variants "
            f"**{', '.join(str(value) for value in task['active_variants'])}**. "
            f"Committed trajectory data covers steps **{first['step']}–{latest['step']}** "
            f"({cumulative['n_valid']} valid trajectories)."
        ),
        "",
        (
            f"Cumulative success is **{cumulative['n_success']}/{cumulative['n_valid']} = "
            f"{cumulative['success_rate']:.2%}** (95% Wilson CI "
            f"{cumulative['success_ci95_low']:.2%}–{cumulative['success_ci95_high']:.2%}). "
            f"Combined reward is **{cumulative['reward_mean']:.6f} ± "
            f"{cumulative['reward_std']:.6f}**."
        ),
        "",
        (
            f"Best success: **step {best_success['step']} ({best_success['success_rate']:.2%})**. "
            f"Best reward: **step {best_reward['step']} ({best_reward['reward_mean']:.6f})**."
        ),
        "",
        "## Per-step results",
        "",
        "| Step | Valid/expected | Success (95% CI) | Mean probability | Reward mean ± SD | P10 / P90 | Trajectory MSE | Terminal MSE | PPO ratio | Clip fraction | Time |",
        "|---:|---:|:---|---:|:---|:---|---:|---:|---:|---:|---:|",
    ]
    for row in step_rows:
        lines.append(
            f"| {row['step']} | {row['n_valid']}/{row['n_expected_trajectories']} | "
            f"{row['success_rate']:.2%} ({row['success_ci95_low']:.2%}–"
            f"{row['success_ci95_high']:.2%}) | {row['success_probability_max_mean']:.4f} | "
            f"{row['reward_mean']:.6f} ± {_fmt(row['reward_std'], 6)} | "
            f"{row['reward_p10']:.6f} / {row['reward_p90']:.6f} | "
            f"{row['trajectory_mse_mean']:.8f} | {row['terminal_goal_mse_mean']:.8f} | "
            f"{row['ratio_mean']:.6f} | {row['clip_fraction_mean']:.6f} | "
            f"{_fmt(row['step_time_s'] / 60 if math.isfinite(row['step_time_s']) else math.nan, 1)} min |"
        )
    if color_rows:
        lines.extend(
            [
                "",
                "## Duck per-color results",
                "",
                "| Step | Color | Valid | Success | Mean probability | Mean reward | Trajectory MSE | Terminal MSE |",
                "|---:|:---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in color_rows:
            lines.append(
                f"| {row['step']} | {row['variant']} | {row['n_valid']} | "
                f"{row['success_rate']:.2%} | {row['success_probability_max_mean']:.4f} | "
                f"{row['reward_mean']:.6f} | {row['trajectory_mse_mean']:.8f} | "
                f"{row['terminal_goal_mse_mean']:.8f} |"
            )
    lines.extend(
        [
            "",
            "## Success/reward relationship",
            "",
            "| Metric | r | p-value | Mean on success | Mean on failure |",
            "|:---|---:|---:|---:|---:|",
        ]
    )
    for record in correlations:
        lines.append(
            f"| {record['metric']} | {_fmt(record['overall_r'], 4)} | "
            f"{_fmt(record['overall_p_value'], 3)} | {_fmt(record['success_mean'], 6)} | "
            f"{_fmt(record['failure_mean'], 6)} |"
        )
    lines.extend(
        [
            "",
            "## Contract and integrity",
            "",
            f"- Batch contract: {batch['trajectories_per_update']} trajectories/update, "
            f"{batch['chunks_per_trajectory']} chunks/trajectory, group size "
            f"{config['algorithm']['group_size']}.",
            f"- Status: `{status.get('state', 'unknown')}`, completed updates: "
            f"{status.get('completed_updates', 'unknown')}.",
            f"- Invalid trajectories: {validation['n_invalid']}; retries: "
            f"{validation['n_retry_trajectories']}; exceptions: "
            f"{validation['n_exceptions']}.",
            f"- Success-threshold mismatches: {validation['success_definition_mismatches']}.",
            f"- Training reward source: `{reward_source}`.",
            f"- Maximum `{reward_equation}` absolute error: "
            f"{validation['max_reward_component_abs_error']:.3g}.",
            f"- Success classifier output is {classifier_role}.",
            "- Full-precision values are in the CSV and JSON artifacts in this directory.",
            "",
        ]
    )
    return "\n".join(lines)


def _cumulative_summary(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in trajectories if bool(row.get("valid", True))]
    successes = [row for row in valid if bool(row["success"])]
    rewards = [float(row["combined_reward"]) for row in valid]
    ci_low, ci_high = wilson_interval(len(successes), len(valid))
    return {
        "n_trajectories": len(trajectories),
        "n_valid": len(valid),
        "n_success": len(successes),
        "success_rate": len(successes) / len(valid),
        "success_ci95_low": ci_low,
        "success_ci95_high": ci_high,
        "reward_mean": mean(rewards),
        "reward_std": sample_std(rewards),
        "reward_median": quantile(rewards, 0.5),
        "reward_p10": quantile(rewards, 0.1),
        "reward_p90": quantile(rewards, 0.9),
        "reward_best": max(rewards),
        "reward_worst": min(rewards),
        "trajectory_mse_mean": mean([float(row["trajectory_mse"]) for row in valid]),
        "terminal_goal_mse_mean": mean([float(row["lastframe_mse"]) for row in valid]),
    }


def analyze_run(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    require_plots: bool = False,
) -> dict[str, Any]:
    """Analyze all atomically committed trajectory updates in ``run_dir``."""
    paths = RunPaths.create(run_dir, output_dir)
    config = _read_config(paths.config_file)
    batch = config["batch"]
    algorithm = config["algorithm"]
    expected = int(batch["trajectories_per_update"])
    chunks = int(batch["chunks_per_trajectory"])
    group_size = int(algorithm["group_size"])
    trajectories_by_step, file_validation = _read_trajectories(
        paths.trajectory_dir,
        expected_trajectories=expected,
        expected_chunks=chunks,
        expected_group_size=group_size,
    )
    metrics = _read_metrics(paths.metrics_history)
    status = _read_json(paths.status_file)
    step_rows = [
        _summarize_rows(
            step,
            rows,
            expected=expected,
            metrics=metrics.get(step),
        )
        for step, rows in sorted(trajectories_by_step.items())
    ]
    all_trajectories = [row for rows in trajectories_by_step.values() for row in rows]
    all_valid = [row for row in all_trajectories if bool(row.get("valid", True))]
    variants = sorted(
        {
            str(row["color"])
            for row in all_valid
            if row.get("color") not in (None, "", "close")
        }
    )
    color_rows: list[dict[str, Any]] = []
    for step, rows in sorted(trajectories_by_step.items()):
        for variant in variants:
            selected = [row for row in rows if str(row.get("color")) == variant]
            if selected:
                color_rows.append(
                    _summarize_rows(
                        step,
                        selected,
                        expected=len(selected),
                        metrics=None,
                        variant=variant,
                    )
                )
    color_summary_rows = []
    for variant in variants:
        selected = [row for row in all_valid if str(row.get("color")) == variant]
        result = _cumulative_summary(selected)
        color_summary_rows.append({"variant": variant, **result})
    threshold_mismatches = sum(
        bool(row["success"]) != _success_from_record(row)
        for row in all_valid
    )
    reward_source = str(
        config.get("reward", {}).get(
            "source", "ctrl_world_aligned_mse_plus_terminal"
        )
    )
    expected_reward_fields = {
        "trajectory_mse": ("trajectory_reward",),
        "terminal_goal_mse": ("lastframe_reward",),
        "success_binary": ("success_reward",),
        "ctrl_world_aligned_mse_plus_terminal": (
            "trajectory_reward",
            "lastframe_reward",
        ),
    }
    if reward_source not in expected_reward_fields:
        raise ValueError(f"unsupported analysis reward source: {reward_source!r}")
    reward_errors = [
        abs(
            float(row["combined_reward"])
            - sum(
                float(row[field_name])
                for field_name in expected_reward_fields[reward_source]
            )
        )
        for row in all_valid
    ]
    advantage_errors = [
        abs(
            float(row["normalized_advantage"])
            - (
                (float(row["combined_reward"]) - float(row["group_mean"]))
                / (float(row["group_std"]) + 1e-6)
            )
        )
        for row in all_valid
    ]
    validation = {
        **file_validation,
        "n_invalid": sum(not bool(row.get("valid", True)) for row in all_trajectories),
        "n_retry_trajectories": sum(
            bool(row.get("retry_count")) for row in all_trajectories
        ),
        "n_exceptions": sum(row.get("exception") is not None for row in all_trajectories),
        "success_definition_mismatches": threshold_mismatches,
        "max_reward_component_abs_error": max(reward_errors),
        "max_normalized_advantage_abs_error": max(advantage_errors),
        "complete_step_record_counts": {
            str(step): len(rows) for step, rows in sorted(trajectories_by_step.items())
        },
    }
    if threshold_mismatches:
        raise ValueError(f"found {threshold_mismatches} success-threshold mismatches")
    if validation["max_reward_component_abs_error"] > 1e-5:
        raise ValueError(
            "reward components are inconsistent; max absolute error "
            f"{validation['max_reward_component_abs_error']}"
        )
    if validation["max_normalized_advantage_abs_error"] > 1e-5:
        raise ValueError(
            "normalized advantages are inconsistent; max absolute error "
            f"{validation['max_normalized_advantage_abs_error']}"
        )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    correlations = _write_correlations(paths.output_dir, trajectories_by_step)
    binary_05 = _write_binary_correlation_suite(
        paths.output_dir / "success_correlation_analysis",
        all_valid,
        threshold=0.5,
        include_success_curve=True,
        require_plots=require_plots,
    )
    binary_001 = _write_binary_correlation_suite(
        paths.output_dir / "success_correlation_analysis_thr001",
        all_valid,
        threshold=0.01,
        include_success_curve=False,
        require_plots=require_plots,
    )
    probability = _write_probability_correlations(
        paths.output_dir / "success_probability_correlation",
        all_valid,
        require_plots=require_plots,
    )
    relationship = _write_relationship_plot(
        paths.output_dir, all_valid, require_plots=require_plots
    )
    _write_learning_curve_png(
        paths.output_dir,
        step_rows,
        color_rows,
        require_plots=require_plots,
    )
    _write_overview_svg(paths.output_dir, step_rows)
    _write_csv(paths.output_dir / "trajectories.csv", _trajectory_csv_rows(trajectories_by_step))
    _write_csv(paths.output_dir / "per_step_summary.csv", step_rows)
    _write_csv(paths.output_dir / "ppo_per_step_summary.csv", step_rows)
    _write_csv(paths.output_dir / "per_color_per_step_summary.csv", color_rows)
    _write_csv(paths.output_dir / "per_color_summary.csv", color_summary_rows)
    cumulative = _cumulative_summary(all_trajectories)
    report = _build_report(
        paths=paths,
        config=config,
        status=status,
        step_rows=step_rows,
        color_rows=color_rows,
        cumulative=cumulative,
        correlations=correlations,
        validation=validation,
    )
    _atomic_write_text(paths.output_dir / "REPORT.md", report)
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(paths.run_dir),
        "analysis_dir": str(paths.output_dir),
        "experiment_name": config.get("experiment_name"),
        "reward_source": reward_source,
        "task": {
            "profile": config["task"]["profile"],
            "active_variants": config["task"]["active_variants"],
        },
        "status": status,
        "batch_contract": {
            "trajectories_per_update": expected,
            "chunks_per_trajectory": chunks,
            "group_size": group_size,
            "global_minibatch_chunks": batch["global_minibatch_chunks"],
            "micro_batch_size_per_gpu": batch["micro_batch_size_per_gpu"],
            "gradient_accumulation_steps": batch["gradient_accumulation_steps"],
            "expected_optimizer_steps": batch["expected_optimizer_steps"],
        },
        "complete_steps": sorted(trajectories_by_step),
        "cumulative_complete_steps": cumulative,
        "per_color": color_summary_rows,
        "validation": validation,
        "correlations": correlations,
        "plot_analyses": {
            "binary_success_threshold_0_5": binary_05,
            "binary_success_threshold_0_01": binary_001,
            "success_probability": probability,
            "relationship": relationship,
        },
    }
    _atomic_write_text(
        paths.output_dir / "summary.json",
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
    )
    print(
        f"Analyzed {len(step_rows)} committed steps and {len(all_valid)} valid "
        f"trajectories: {paths.output_dir}"
    )
    return summary


def _run_dir_from_config(path: Path) -> Path:
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    try:
        return Path(config["runtime"]["output_dir"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"config has no runtime.output_dir: {path}") from error


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CSV, report, learning curves, and correlation plots from RL runs."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        type=Path,
        help="Run output directory; may be supplied more than once.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        type=Path,
        help="Training YAML whose runtime.output_dir should be analyzed; repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override analysis output directory (only valid for one run).",
    )
    parser.add_argument(
        "--require-plots",
        action="store_true",
        help="Fail if gnuplot cannot produce all PNGs.",
    )
    arguments = parser.parse_args(argv)
    run_count = len(arguments.run_dir) + len(arguments.config)
    if run_count == 0:
        parser.error("at least one --run-dir or --config is required")
    if arguments.output_dir is not None and run_count != 1:
        parser.error("--output-dir may only be used when analyzing one run")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    run_dirs = [*arguments.run_dir]
    run_dirs.extend(_run_dir_from_config(path) for path in arguments.config)
    failures = []
    for run_dir in run_dirs:
        try:
            analyze_run(
                run_dir,
                output_dir=arguments.output_dir,
                require_plots=arguments.require_plots,
            )
        except Exception as error:  # noqa: BLE001 - multi-run CLI must report every run.
            failures.append((run_dir, error))
            print(f"[analysis] FAILED {run_dir}: {error}", file=sys.stderr)
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
