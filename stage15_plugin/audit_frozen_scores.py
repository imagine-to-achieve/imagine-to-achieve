#!/usr/bin/env python3
"""Compare FPO score parameterizations on one immutable frozen rollout.

The input exact-replay records were produced with epsilon scoring.  For the
linear CFM path used here,

    epsilon_mse = (1 - sigma) ** 2 * velocity_mse,

so raw velocity losses can be recovered exactly from the persisted MC scores
and deterministic action seeds without another model forward.  The third
candidate keeps the checkpoint's qnorm action coordinates and removes
timestep-dependent loss scale by dividing raw-u loss by the median raw-u loss
in an equal-frequency sigma bin.  This normalization is unsupervised: reward
labels are never used to estimate its scales.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in dx)
        * sum(value * value for value in dy)
    )
    if denominator == 0.0:
        return float("nan")
    return sum(x * y for x, y in zip(dx, dy)) / denominator


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0 + 1.0
        for position in range(cursor, end):
            result[order[position]] = average_rank
        cursor = end
    return result


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_correlation(
    units: dict[int, list[tuple[float, float]]],
    *,
    method: str,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    unit_ids = sorted(units)
    rng = random.Random(seed)
    estimates: list[float] = []
    correlation = _spearman if method == "spearman" else _pearson
    for _ in range(samples):
        xs: list[float] = []
        ys: list[float] = []
        for _position in unit_ids:
            selected = unit_ids[rng.randrange(len(unit_ids))]
            for x_value, y_value in units[selected]:
                xs.append(x_value)
                ys.append(y_value)
        value = correlation(xs, ys)
        if math.isfinite(value):
            estimates.append(value)
    estimates.sort()
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def _fpo_sigma(
    sampling_seed: int,
    mc_index: int,
    *,
    distribution: str,
    training_shift: float,
) -> float:
    payload = f"{int(sampling_seed)}:{int(mc_index)}".encode("ascii")
    digest = hashlib.blake2b(
        payload, digest_size=16, person=b"rlinf-fpo-v1"
    ).digest()
    uniform_bits = int.from_bytes(digest[8:], byteorder="little", signed=False)
    uniform = (uniform_bits + 0.5) / float(1 << 64)
    if distribution == "uniform":
        sampled_train_time = uniform
    elif distribution == "waver":
        sampled_train_time = 1.0 - uniform - 1.29 * (
            math.cos(math.pi * 0.5 * uniform) ** 2 - 1.0 + uniform
        )
    else:
        raise ValueError(f"unsupported FPO time distribution: {distribution}")
    base_time = min(
        max(1.0 - sampled_train_time, 1.0e-6), 1.0 - 1.0e-6
    )
    return training_shift * base_time / (
        1.0 + (training_shift - 1.0) * base_time
    )


def _bin_index(edges: list[float], value: float) -> int:
    lower = 0
    upper = len(edges)
    while lower < upper:
        middle = (lower + upper) // 2
        if value < edges[middle]:
            upper = middle
        else:
            lower = middle + 1
    return lower


def _validate_cross_run(
    primary_exact: list[dict[str, Any]],
    primary_trajectories: list[dict[str, Any]],
    cross_run_dir: Path,
) -> dict[str, Any]:
    other_exact = _load_jsonl(
        cross_run_dir / "exact_score_replay" / "update_0000.jsonl"
    )
    other_trajectories = _load_jsonl(
        cross_run_dir / "trajectory_records" / "update_0000.jsonl"
    )
    exact_key = lambda row: (int(row["rollout_uid"]), int(row["chunk_id"]))
    primary_exact_by_key = {exact_key(row): row for row in primary_exact}
    other_exact_by_key = {exact_key(row): row for row in other_exact}
    primary_traj_by_key = {
        int(row["rollout_uid"]): row for row in primary_trajectories
    }
    other_traj_by_key = {
        int(row["rollout_uid"]): row for row in other_trajectories
    }
    if set(primary_exact_by_key) != set(other_exact_by_key):
        raise RuntimeError("cross-check exact-replay chunk keys differ")
    if set(primary_traj_by_key) != set(other_traj_by_key):
        raise RuntimeError("cross-check trajectory keys differ")
    per_mc_matches = sum(
        primary_exact_by_key[key]["per_mc_score_before"]
        == other_exact_by_key[key]["per_mc_score_before"]
        for key in primary_exact_by_key
    )
    reward_matches = sum(
        primary_traj_by_key[key]["combined_reward"]
        == other_traj_by_key[key]["combined_reward"]
        for key in primary_traj_by_key
    )
    seed_matches = sum(
        primary_traj_by_key[key]["cosmos_joint_seeds"]
        == other_traj_by_key[key]["cosmos_joint_seeds"]
        for key in primary_traj_by_key
    )
    expected_exact = len(primary_exact_by_key)
    expected_trajectories = len(primary_traj_by_key)
    if per_mc_matches != expected_exact:
        raise RuntimeError("cross-check per-MC frozen scores are not identical")
    if reward_matches != expected_trajectories:
        raise RuntimeError("cross-check frozen rewards are not identical")
    if seed_matches != expected_trajectories:
        raise RuntimeError("cross-check MC action seeds are not identical")
    return {
        "cross_run_dir": str(cross_run_dir),
        "exact_chunks": expected_exact,
        "trajectories": expected_trajectories,
        "per_mc_score_before_exact_matches": per_mc_matches,
        "combined_reward_exact_matches": reward_matches,
        "joint_seed_exact_matches": seed_matches,
    }


def _method_summary(
    rows: list[dict[str, Any]],
    method: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    chunk_scores = [float(row["scores"][method]["chunk_score"]) for row in rows]
    advantages = [float(row["advantage"]) for row in rows]
    rewards_by_uid: dict[int, float] = {}
    groups_by_uid: dict[int, int] = {}
    trajectory_scores: dict[int, list[float]] = defaultdict(list)
    for row, chunk_score in zip(rows, chunk_scores):
        uid = int(row["rollout_uid"])
        rewards_by_uid[uid] = float(row["combined_reward"])
        groups_by_uid[uid] = int(row["group_id"])
        trajectory_scores[uid].append(chunk_score)
    trajectory_ids = sorted(trajectory_scores)
    reward_values = [rewards_by_uid[uid] for uid in trajectory_ids]
    score_values = [
        sum(trajectory_scores[uid]) / len(trajectory_scores[uid])
        for uid in trajectory_ids
    ]

    reward_units = {
        uid: [(rewards_by_uid[uid], score_values[index])]
        for index, uid in enumerate(trajectory_ids)
    }
    advantage_units: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row, score in zip(rows, chunk_scores):
        advantage_units[int(row["rollout_uid"])].append(
            (float(row["advantage"]), score)
        )
    reward_ci = _bootstrap_correlation(
        reward_units,
        method="spearman",
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    advantage_ci = _bootstrap_correlation(
        advantage_units,
        method="spearman",
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )

    group_correlations: list[float] = []
    for group_id in sorted(set(groups_by_uid.values())):
        members = [
            index
            for index, uid in enumerate(trajectory_ids)
            if groups_by_uid[uid] == group_id
        ]
        group_correlations.append(
            _spearman(
                [reward_values[index] for index in members],
                [score_values[index] for index in members],
            )
        )

    mc_reward_spearman: list[float] = []
    mc_advantage_spearman: list[float] = []
    num_mc = len(rows[0]["scores"][method]["per_mc_scores"])
    for mc_index in range(num_mc):
        mc_trajectory_scores: dict[int, list[float]] = defaultdict(list)
        mc_chunk_scores: list[float] = []
        for row in rows:
            value = float(row["scores"][method]["per_mc_scores"][mc_index])
            mc_chunk_scores.append(value)
            mc_trajectory_scores[int(row["rollout_uid"])].append(value)
        mc_score_values = [
            sum(mc_trajectory_scores[uid]) / len(mc_trajectory_scores[uid])
            for uid in trajectory_ids
        ]
        mc_reward_spearman.append(_spearman(reward_values, mc_score_values))
        mc_advantage_spearman.append(_spearman(advantages, mc_chunk_scores))

    ranked = sorted(range(len(reward_values)), key=lambda index: reward_values[index])
    quartile = max(1, len(ranked) // 4)
    bottom = ranked[:quartile]
    top = ranked[-quartile:]
    top_bottom_gap = (
        sum(score_values[index] for index in top) / len(top)
        - sum(score_values[index] for index in bottom) / len(bottom)
    )
    score_std = statistics.stdev(score_values)

    reward_spearman = _spearman(reward_values, score_values)
    advantage_spearman = _spearman(advantages, chunk_scores)
    within_group_mean = sum(group_correlations) / len(group_correlations)
    mc_reward_positive_fraction = sum(
        value > 0.0 for value in mc_reward_spearman
    ) / len(mc_reward_spearman)
    gate_checks = {
        "trajectory_reward_spearman_positive": reward_spearman > 0.0,
        "trajectory_reward_spearman_bootstrap_lower_positive": reward_ci[0] > 0.0,
        "chunk_advantage_spearman_positive": advantage_spearman > 0.0,
        "within_group_reward_spearman_mean_positive": within_group_mean > 0.0,
        "top_minus_bottom_reward_quartile_score_positive": top_bottom_gap > 0.0,
        "mc_reward_positive_fraction_at_least_0_75": (
            mc_reward_positive_fraction >= 0.75
        ),
    }
    return {
        "trajectory_reward_pearson": _pearson(reward_values, score_values),
        "trajectory_reward_spearman": reward_spearman,
        "trajectory_reward_spearman_cluster_bootstrap_95ci": list(reward_ci),
        "chunk_advantage_pearson": _pearson(advantages, chunk_scores),
        "chunk_advantage_spearman": advantage_spearman,
        "chunk_advantage_spearman_cluster_bootstrap_95ci": list(advantage_ci),
        "within_group_reward_spearman": group_correlations,
        "within_group_reward_spearman_mean": within_group_mean,
        "within_group_reward_spearman_median": statistics.median(group_correlations),
        "within_group_positive_groups": sum(value > 0.0 for value in group_correlations),
        "within_group_total_groups": len(group_correlations),
        "mc_reward_spearman": mc_reward_spearman,
        "mc_reward_spearman_median": statistics.median(mc_reward_spearman),
        "mc_reward_spearman_min": min(mc_reward_spearman),
        "mc_reward_spearman_max": max(mc_reward_spearman),
        "mc_reward_positive_fraction": mc_reward_positive_fraction,
        "mc_advantage_spearman": mc_advantage_spearman,
        "mc_advantage_spearman_median": statistics.median(mc_advantage_spearman),
        "top_minus_bottom_reward_quartile_score_gap": top_bottom_gap,
        "top_minus_bottom_gap_in_score_std": (
            top_bottom_gap / score_std if score_std > 0.0 else float("nan")
        ),
        "trajectory_score_mean": sum(score_values) / len(score_values),
        "trajectory_score_std": score_std,
        "selection_gate_checks": gate_checks,
        "passes_selection_gate": all(gate_checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cross-check-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sigma-bins", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.sigma_bins < 2:
        raise ValueError("--sigma-bins must be at least 2")
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")

    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "frozen_score_audit"
    )
    exact_summary = json.loads(
        (run_dir / "exact_score_replay" / "summary_0000.json").read_text(
            encoding="utf-8"
        )
    )
    if exact_summary.get("score_parameterization") != "epsilon":
        raise RuntimeError("frozen score audit requires persisted epsilon scores")
    exact_rows = _load_jsonl(
        run_dir / "exact_score_replay" / "update_0000.jsonl"
    )
    trajectories = _load_jsonl(
        run_dir / "trajectory_records" / "update_0000.jsonl"
    )
    trajectory_by_uid = {
        int(row["rollout_uid"]): row for row in trajectories
    }
    config = yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    fpo_config = config["fpo"]
    action_normalization = str(config["cosmos"]["action_normalization"])
    if action_normalization != "qnorm":
        raise RuntimeError(
            "normalized-u audit is defined for qnorm action coordinates"
        )
    num_mc = int(fpo_config["num_mc_samples"])
    distribution = str(fpo_config["time_distribution"])
    training_shift = float(fpo_config["training_shift"])

    cross_check = None
    if args.cross_check_run_dir is not None:
        cross_check = _validate_cross_run(
            exact_rows, trajectories, args.cross_check_run_dir.resolve()
        )

    rows: list[dict[str, Any]] = []
    reconstruction_errors: list[float] = []
    probes: list[tuple[float, float]] = []
    for exact in exact_rows:
        uid = int(exact["rollout_uid"])
        chunk_id = int(exact["chunk_id"])
        trajectory = trajectory_by_uid[uid]
        action_seeds = trajectory["cosmos_joint_seeds"]
        sampling_seed = int(action_seeds[chunk_id])
        epsilon_scores = [
            float(value) for value in exact["per_mc_score_before"]
        ]
        if len(epsilon_scores) != num_mc:
            raise RuntimeError("persisted MC score count differs from config")
        sigmas = [
            _fpo_sigma(
                sampling_seed,
                mc_index,
                distribution=distribution,
                training_shift=training_shift,
            )
            for mc_index in range(num_mc)
        ]
        epsilon_losses = [-value for value in epsilon_scores]
        raw_u_losses = [
            loss / max((1.0 - sigma) ** 2, 1.0e-30)
            for loss, sigma in zip(epsilon_losses, sigmas)
        ]
        reconstructed = -sum(epsilon_losses) / num_mc
        reconstruction_error = abs(
            reconstructed - float(exact["score_before"])
        )
        reconstruction_errors.append(reconstruction_error)
        if reconstruction_error > 1.0e-8:
            raise RuntimeError("per-MC epsilon scores do not reconstruct score_before")
        for sigma, raw_loss in zip(sigmas, raw_u_losses):
            probes.append((sigma, raw_loss))
        rows.append(
            {
                "rollout_uid": uid,
                "chunk_id": chunk_id,
                "group_id": int(trajectory["group_id"]),
                "combined_reward": float(trajectory["combined_reward"]),
                "advantage": float(exact["advantage"]),
                "sampling_seed": sampling_seed,
                "sigmas": sigmas,
                "epsilon_losses": epsilon_losses,
                "raw_u_losses": raw_u_losses,
            }
        )

    sorted_sigmas = sorted(sigma for sigma, _loss in probes)
    edges = [
        _quantile(sorted_sigmas, index / args.sigma_bins)
        for index in range(1, args.sigma_bins)
    ]
    bin_losses: list[list[float]] = [
        [] for _ in range(args.sigma_bins)
    ]
    for sigma, raw_loss in probes:
        bin_losses[_bin_index(edges, sigma)].append(raw_loss)
    scales = [statistics.median(values) for values in bin_losses]
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise RuntimeError("normalized-u sigma-bin scale is non-positive")

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_losses = [
            loss / scales[_bin_index(edges, sigma)]
            for sigma, loss in zip(row["sigmas"], row["raw_u_losses"])
        ]
        method_losses = {
            "raw_u": row["raw_u_losses"],
            "qnorm_sigma_normalized_u": normalized_losses,
            "epsilon": row["epsilon_losses"],
        }
        output_rows.append(
            {
                "schema_version": 1,
                "rollout_uid": row["rollout_uid"],
                "chunk_id": row["chunk_id"],
                "group_id": row["group_id"],
                "combined_reward": row["combined_reward"],
                "advantage": row["advantage"],
                "sampling_seed": row["sampling_seed"],
                "sigmas": row["sigmas"],
                "scores": {
                    method: {
                        "per_mc_scores": [-float(loss) for loss in losses],
                        "chunk_score": -sum(losses) / len(losses),
                    }
                    for method, losses in method_losses.items()
                },
            }
        )

    method_summaries = {
        method: _method_summary(
            output_rows,
            method,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + method_index * 100,
        )
        for method_index, method in enumerate(
            ("raw_u", "qnorm_sigma_normalized_u", "epsilon")
        )
    }
    eligible = [
        method
        for method, summary in method_summaries.items()
        if summary["passes_selection_gate"]
    ]
    selected = (
        max(
            eligible,
            key=lambda method: method_summaries[method][
                "trajectory_reward_spearman"
            ],
        )
        if eligible
        else None
    )
    summary = {
        "schema_version": 1,
        "analysis": "frozen_rollout_score_alignment",
        "run_dir": str(run_dir),
        "cross_check": cross_check,
        "input_contract": {
            "trajectories": len(trajectories),
            "chunks": len(exact_rows),
            "num_mc_samples": num_mc,
            "action_normalization": action_normalization,
            "time_distribution": distribution,
            "training_shift": training_shift,
            "frozen_score_source": "per_mc_score_before",
            "reward_source": "trajectory combined_reward",
            "per_mc_mean_reconstruction_tolerance": 1.0e-8,
            "per_mc_mean_reconstruction_max_abs_error": max(
                reconstruction_errors
            ),
        },
        "parameterizations": {
            "raw_u": "-mean(raw velocity MSE)",
            "qnorm_sigma_normalized_u": (
                "raw-u in checkpoint qnorm action coordinates, divided by "
                "the unsupervised median raw-u loss in one of 16 equal-frequency sigma bins"
            ),
            "epsilon": "-mean((1-sigma)^2 * raw velocity MSE)",
        },
        "normalization": {
            "sigma_bins": args.sigma_bins,
            "sigma_bin_edges": edges,
            "raw_u_median_loss_by_sigma_bin": scales,
            "uses_reward_labels": False,
        },
        "selection_gate": {
            "rule": "all six directional and stability checks must pass",
            "selected_score": selected,
            "eligible_scores": eligible,
        },
        "methods": method_summaries,
    }
    _atomic_jsonl(output_dir / "records.jsonl", output_rows)
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
