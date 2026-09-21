#!/usr/bin/env python3
"""Strict paired analysis for the pluggable Stage 1.5 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


RUN_NAMES = {
    "baseline": "stage15_baseline_paired_rollout",
    "A": "stage15_a_exact_u_mc4",
    "D": "stage15_d_exact_eps_mc8",
}
STAGE1_NAMES = {
    "A": "stage1_a_per_action_u_mc4",
    "D": "stage1_d_per_mc_eps_mc8",
}
EVAL_METRICS = {
    "combined_reward": 1.0,
    "terminal_goal_mse": -1.0,
    "video_mse": -1.0,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx, my = mean(xs), mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denominator = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denominator == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(dx, dy)) / denominator


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for index in order[cursor:end]:
            result[index] = rank
        cursor = end
    return result


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def quantile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(values: list[float], label: str) -> list[float]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    count = len(values)
    boot = sorted(
        mean([values[rng.randrange(count)] for _ in range(count)])
        for _ in range(20000)
    )
    return [quantile(boot, 0.025), quantile(boot, 0.975)]


def paired_summary(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    baseline_by_key = {
        (str(record["color"]), int(record["episode"]), int(record["seed"])): record
        for record in baseline
    }
    candidate_by_key = {
        (str(record["color"]), int(record["episode"]), int(record["seed"])): record
        for record in candidate
    }
    if baseline_by_key.keys() != candidate_by_key.keys():
        raise RuntimeError(f"{label}: evaluation key sets are not identical")
    keys = sorted(baseline_by_key)
    if len(keys) != 10:
        raise RuntimeError(f"{label}: expected 10 paired evaluations, got {len(keys)}")
    rng_mismatches = [
        key
        for key in keys
        if baseline_by_key[key].get("rng_manifest")
        != candidate_by_key[key].get("rng_manifest")
    ]
    if rng_mismatches:
        raise RuntimeError(f"{label}: RNG manifests differ for {rng_mismatches}")
    if not all(
        bool(baseline_by_key[key].get("valid"))
        and bool(candidate_by_key[key].get("valid"))
        for key in keys
    ):
        raise RuntimeError(f"{label}: invalid evaluation record")

    metrics: dict[str, Any] = {}
    for field, orientation in EVAL_METRICS.items():
        base_values = [float(baseline_by_key[key][field]) for key in keys]
        candidate_values = [float(candidate_by_key[key][field]) for key in keys]
        deltas = [value - base for base, value in zip(base_values, candidate_values)]
        improvements = [orientation * value for value in deltas]
        metrics[field] = {
            "baseline_mean": mean(base_values),
            "candidate_mean": mean(candidate_values),
            "candidate_minus_baseline_mean": mean(deltas),
            "paired_delta_median": statistics.median(deltas),
            "paired_delta_sample_sd": statistics.stdev(deltas),
            "improvement_mean": mean(improvements),
            "improvement_bootstrap_95_ci": bootstrap_mean_ci(
                improvements, f"{label}:{field}"
            ),
            "improved": sum(value > 0.0 for value in improvements),
            "tied": sum(value == 0.0 for value in improvements),
            "worsened": sum(value < 0.0 for value in improvements),
        }
    return {
        "pairs": len(keys),
        "keys": [
            {"color": color, "episode": episode, "seed": seed}
            for color, episode, seed in keys
        ],
        "rng_manifest_exact_match": True,
        "baseline_policy_hashes": sorted(
            {str(baseline_by_key[key]["policy_hash"]) for key in keys}
        ),
        "candidate_policy_hashes": sorted(
            {str(candidate_by_key[key]["policy_hash"]) for key in keys}
        ),
        "baseline_action_checksums": sorted(
            {float(baseline_by_key[key]["policy_action_param_checksum"]) for key in keys}
        ),
        "candidate_action_checksums": sorted(
            {float(candidate_by_key[key]["policy_action_param_checksum"]) for key in keys}
        ),
        "metrics": metrics,
    }


def exact_score_analysis(run_dir: Path) -> dict[str, Any]:
    summaries = sorted((run_dir / "exact_score_replay").glob("summary_*.json"))
    records_paths = sorted((run_dir / "exact_score_replay").glob("update_*.jsonl"))
    if len(summaries) != 1 or len(records_paths) != 1:
        raise RuntimeError(f"{run_dir}: expected one exact summary and one record file")
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    records = read_jsonl(records_paths[0])
    if len(records) != 640 or sum(bool(record["valid"]) for record in records) != 640:
        raise RuntimeError(f"{run_dir}: exact replay must contain 640 valid chunks")
    by_uid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_uid[int(record["rollout_uid"])].append(record)
    if len(by_uid) != 128 or any(len(chunks) != 5 for chunks in by_uid.values()):
        raise RuntimeError(f"{run_dir}: expected 128 trajectories with five chunks each")

    trajectory_path = run_dir / "trajectory_records" / "update_0000.jsonl"
    trajectories = read_jsonl(trajectory_path)
    trajectory_by_uid = {int(record["rollout_uid"]): record for record in trajectories}
    if trajectory_by_uid.keys() != by_uid.keys():
        raise RuntimeError(f"{run_dir}: exact-score and trajectory UID sets differ")

    rows = []
    for uid, chunks in by_uid.items():
        chunks.sort(key=lambda record: int(record["chunk_id"]))
        trajectory = trajectory_by_uid[uid]
        rows.append(
            {
                "uid": uid,
                "reward": float(trajectory["combined_reward"]),
                "advantage": float(trajectory["normalized_advantage"]),
                "before": sum(float(record["score_before"]) for record in chunks),
                "after": sum(float(record["score_after"]) for record in chunks),
                "delta": sum(float(record["score_delta"]) for record in chunks),
            }
        )
    rewards = [row["reward"] for row in rows]
    advantages = [row["advantage"] for row in rows]
    before = [row["before"] for row in rows]
    after = [row["after"] for row in rows]
    delta = [row["delta"] for row in rows]
    ordered = sorted(rows, key=lambda row: row["reward"])
    quartile = len(ordered) // 4
    low = ordered[:quartile]
    high = ordered[-quartile:]
    return {
        "summary": summary,
        "trajectory_level": {
            "records": len(rows),
            "reward_score_before_pearson": pearson(rewards, before),
            "reward_score_after_pearson": pearson(rewards, after),
            "reward_score_before_spearman": spearman(rewards, before),
            "reward_score_after_spearman": spearman(rewards, after),
            "reward_score_delta_pearson": pearson(rewards, delta),
            "advantage_score_delta_pearson": pearson(advantages, delta),
            "high_minus_low_reward_quartile_score_before": (
                mean([row["before"] for row in high])
                - mean([row["before"] for row in low])
            ),
            "high_minus_low_reward_quartile_score_after": (
                mean([row["after"] for row in high])
                - mean([row["after"] for row in low])
            ),
            "high_minus_low_reward_quartile_score_delta": (
                mean([row["delta"] for row in high])
                - mean([row["delta"] for row in low])
            ),
        },
    }


def rollout_reproducibility(
    current: list[dict[str, Any]], reference: list[dict[str, Any]], label: str
) -> dict[str, Any]:
    current_by_uid = {int(record["rollout_uid"]): record for record in current}
    reference_by_uid = {int(record["rollout_uid"]): record for record in reference}
    identity_fields = [
        "episode",
        "reset_episode",
        "reset_seed",
        "group_id",
        "group_slot",
        "member_id",
        "logical_round",
        "physical_wave",
        "ctrl_seed",
        "cosmos_joint_seeds",
        "cosmos_seed_nonces",
        "seed_derivation_version",
        "input_checksums",
    ]
    shared = sorted(current_by_uid.keys() & reference_by_uid.keys())
    field_mismatches = {
        field: sum(
            current_by_uid[uid].get(field) != reference_by_uid[uid].get(field)
            for uid in shared
        )
        for field in identity_fields
    }
    numeric_fields = [
        "combined_reward",
        "lastframe_mse",
        "trajectory_mse",
        "normalized_advantage",
    ]
    max_abs_difference = {}
    for field in numeric_fields:
        differences = [
            abs(
                float(current_by_uid[uid][field])
                - float(reference_by_uid[uid][field])
            )
            for uid in shared
        ]
        max_abs_difference[field] = max(differences) if differences else None
    return {
        "label": label,
        "current_records": len(current_by_uid),
        "reference_records": len(reference_by_uid),
        "shared_uids": len(shared),
        "identical_uid_set": current_by_uid.keys() == reference_by_uid.keys(),
        "identity_field_mismatches": field_mismatches,
        "numeric_max_abs_difference": max_abs_difference,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage 1.5 paired FPO report",
        "",
        "Positive improvement means better: higher combined reward or lower MSE.",
        "",
        "## Exact frozen-batch score movement",
        "",
        "| Variant | adv×Δscore mean | corr(adv, Δscore) | aligned | reward-score r before | reward-score r after |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("A", "D"):
        exact = report["exact_score"][label]
        summary = exact["summary"]
        trajectory = exact["trajectory_level"]
        lines.append(
            f"| {label} | {summary['advantage_times_score_delta_mean']:.6g} | "
            f"{summary['advantage_score_delta_corr']:.6g} | "
            f"{summary['alignment_fraction_nonzero']:.3f} | "
            f"{trajectory['reward_score_before_pearson']:.6g} | "
            f"{trajectory['reward_score_after_pearson']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Paired environment evaluation",
            "",
            "| Variant | Metric | Baseline | Candidate | Candidate−baseline | Improvement | 95% bootstrap CI | Wins/10 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("A", "D"):
        paired = report["paired_evaluation"][label]
        for metric, values in paired["metrics"].items():
            ci = values["improvement_bootstrap_95_ci"]
            lines.append(
                f"| {label} | {metric} | {values['baseline_mean']:.6g} | "
                f"{values['candidate_mean']:.6g} | "
                f"{values['candidate_minus_baseline_mean']:.6g} | "
                f"{values['improvement_mean']:.6g} | "
                f"[{ci[0]:.6g}, {ci[1]:.6g}] | {values['improved']}/10 |"
            )
    lines.extend(
        [
            "",
            "## Pairing and reproducibility",
            "",
            f"- A RNG manifests exact match: {report['paired_evaluation']['A']['rng_manifest_exact_match']}",
            f"- D RNG manifests exact match: {report['paired_evaluation']['D']['rng_manifest_exact_match']}",
            f"- A exact chunks: {report['exact_score']['A']['summary']['valid_records']}",
            f"- D exact chunks: {report['exact_score']['D']['summary']['valid_records']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.root.resolve()
    outputs = root / "outputs"
    run_dirs = {label: outputs / name for label, name in RUN_NAMES.items()}

    eval_records = {
        label: read_jsonl(run_dir / "evaluation_records" / "step_0.jsonl")
        for label, run_dir in run_dirs.items()
    }
    exact = {
        label: exact_score_analysis(run_dirs[label])
        for label in ("A", "D")
    }
    current_trajectories = {
        label: read_jsonl(run_dirs[label] / "trajectory_records" / "update_0000.jsonl")
        for label in ("A", "D")
    }
    references = {
        label: read_jsonl(
            outputs / STAGE1_NAMES[label] / "trajectory_records" / "update_0000.jsonl"
        )
        for label in ("A", "D")
    }
    report = {
        "schema_version": 1,
        "runs": {label: str(path) for label, path in run_dirs.items()},
        "paired_evaluation": {
            label: paired_summary(eval_records["baseline"], eval_records[label], label)
            for label in ("A", "D")
        },
        "exact_score": exact,
        "rollout_reproducibility": {
            label: rollout_reproducibility(
                current_trajectories[label], references[label], f"{label}:stage15-vs-stage1"
            )
            for label in ("A", "D")
        },
        "stage15_a_vs_d_rollout": rollout_reproducibility(
            current_trajectories["A"], current_trajectories["D"], "stage15:A-vs-D"
        ),
    }

    output_dir = outputs / "stage15_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(json_path), "markdown": str(markdown_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
