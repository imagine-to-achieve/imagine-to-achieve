#!/usr/bin/env python3
"""Plot per-RL-step trajectory-MSE violins for three Duck reward ablations.

The binary-success reward run is intentionally excluded.  Each violin is built
from the 128 valid trajectory records committed for one RL update.  The script
also writes the exact plotted samples, per-step summary statistics, and a
source manifest so that a later refresh can be audited and reproduced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np


EXPECTED_SAMPLES_PER_STEP = 128
RUNS = (
    (
        "terminal_goal_mse",
        "Terminal-goal MSE reward",
        "#D55E00",
        "duck4_b128_mb32_u15_terminal_mse_reward_8n4g",
    ),
    (
        "trajectory_mse",
        "Trajectory MSE reward",
        "#0072B2",
        "duck4_b128_mb32_u15_traj_reward_8n4g",
    ),
    (
        "combined_mse",
        "Combined MSE reward",
        "#CC79A7",
        "duck4_b128_mb32_u15_combined_mse_reward_8n4g",
    ),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=repo_root / "outputs",
        help="Directory containing the three Duck training runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repo_root
            / "outputs"
            / "duck4_b128_mb32_u15_reward_comparison"
            / "analysis"
        ),
        help="Destination for plots and their audit metadata.",
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def read_run(path: Path) -> tuple[dict[int, list[dict[str, Any]]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing trajectory metadata: {path}")

    payload = path.read_bytes()
    source_sha256 = sha256_bytes(payload)
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    required = {
        "global_step",
        "trajectory_mse",
        "rollout_uid",
        "color",
        "valid",
    }
    missing = required.difference(reader.fieldnames or ())
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    by_step: dict[int, list[dict[str, Any]]] = {}
    invalid_rows = 0
    for row in reader:
        if row["valid"].strip().lower() != "true":
            invalid_rows += 1
            continue
        step = int(row["global_step"])
        mse = float(row["trajectory_mse"])
        if step < 1 or not math.isfinite(mse):
            raise ValueError(f"Invalid step/MSE in {path}: step={step}, mse={mse}")
        by_step.setdefault(step, []).append(
            {
                "trajectory_mse": mse,
                "rollout_uid": row["rollout_uid"],
                "color": row["color"],
            }
        )

    if invalid_rows:
        raise ValueError(f"Expected no invalid records in {path}; found {invalid_rows}")
    if not by_step:
        raise ValueError(f"No valid trajectory records in {path}")

    steps = sorted(by_step)
    expected_steps = list(range(1, steps[-1] + 1))
    if steps != expected_steps:
        raise ValueError(f"Non-contiguous steps in {path}: {steps}")
    for step, rows in by_step.items():
        if len(rows) != EXPECTED_SAMPLES_PER_STEP:
            raise ValueError(
                f"Expected {EXPECTED_SAMPLES_PER_STEP} records at step {step} "
                f"in {path}; found {len(rows)}"
            )
        rollout_uids = [row["rollout_uid"] for row in rows]
        if len(set(rollout_uids)) != len(rollout_uids):
            raise ValueError(f"Duplicate rollout_uid at step {step} in {path}")
    return by_step, source_sha256


def save_plot(
    output_path: Path,
    reward_label: str,
    color: str,
    steps: list[int],
    values_by_step: dict[int, list[dict[str, Any]]],
    y_limits: tuple[float, float],
) -> None:
    values = [
        np.asarray(
            [row["trajectory_mse"] for row in values_by_step[step]],
            dtype=np.float64,
        )
        for step in steps
    ]
    quartiles = np.asarray([np.quantile(group, [0.25, 0.5, 0.75]) for group in values])

    fig, ax = plt.subplots(figsize=(20, 8), dpi=180)
    parts = ax.violinplot(
        values,
        positions=steps,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=160,
        bw_method="scott",
    )
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.68)
        body.set_linewidth(0.8)

    q1, medians, q3 = quartiles.T
    ax.vlines(steps, q1, q3, color="#202020", linewidth=2.0, zorder=3)
    ax.plot(
        steps,
        medians,
        color="#202020",
        linewidth=1.25,
        alpha=0.78,
        zorder=4,
    )
    ax.scatter(
        steps,
        medians,
        s=27,
        facecolor="white",
        edgecolor="#202020",
        linewidth=1.05,
        zorder=5,
    )

    ax.set_xlim(0.35, steps[-1] + 0.65)
    ax.set_ylim(*y_limits)
    ax.set_xticks(steps)
    ax.set_xlabel("RL update step", fontsize=15, labelpad=10)
    ax.set_ylabel("Trajectory MSE (lower is better)", fontsize=15, labelpad=10)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Four-Color Duck: Trajectory MSE Distribution by RL Step",
        fontsize=22,
        fontweight="semibold",
        y=0.975,
    )
    ax.set_title(
        f"{reward_label}  |  n={EXPECTED_SAMPLES_PER_STEP} valid trajectories per step",
        fontsize=15,
        color="#333333",
        pad=16,
    )
    legend_items = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=9,
            alpha=0.68,
            label="MSE density (violin)",
        ),
        Line2D(
            [0],
            [0],
            color="#202020",
            marker="o",
            markerfacecolor="white",
            markersize=6,
            linewidth=1.5,
            label="Median; vertical bar = IQR",
        ),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper right",
        frameon=False,
        fontsize=11,
        ncol=2,
    )
    fig.tight_layout(rect=(0.025, 0.025, 0.995, 0.94))

    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    fig.savefig(temp_path, format="png", dpi=180, facecolor="white")
    plt.close(fig)
    temp_path.replace(output_path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    outputs_root = args.outputs_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, dict[str, Any]] = {}
    for reward_definition, reward_label, color, run_name in RUNS:
        source = outputs_root / run_name / "analysis" / "trajectories.csv"
        by_step, source_sha256 = read_run(source)
        loaded[reward_definition] = {
            "reward_label": reward_label,
            "color": color,
            "run_name": run_name,
            "source": source,
            "source_sha256": source_sha256,
            "by_step": by_step,
        }

    step_sets = {tuple(sorted(item["by_step"])) for item in loaded.values()}
    if len(step_sets) != 1:
        raise ValueError(f"Runs do not have the same committed steps: {sorted(step_sets)}")
    steps = list(next(iter(step_sets)))

    all_values = np.asarray(
        [
            row["trajectory_mse"]
            for item in loaded.values()
            for step in steps
            for row in item["by_step"][step]
        ],
        dtype=np.float64,
    )
    y_min = float(all_values.min())
    y_max = float(all_values.max())
    padding = max((y_max - y_min) * 0.045, 0.001)
    y_limits = (max(0.0, y_min - padding), y_max + padding)

    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    generated_plots: list[str] = []
    for reward_definition, _, _, _ in RUNS:
        item = loaded[reward_definition]
        for step in steps:
            step_rows = item["by_step"][step]
            step_values = np.asarray(
                [row["trajectory_mse"] for row in step_rows], dtype=np.float64
            )
            q0, q1, median, q3, q4 = np.quantile(
                step_values, [0.0, 0.25, 0.5, 0.75, 1.0]
            )
            summary_rows.append(
                {
                    "reward_definition": reward_definition,
                    "reward_label": item["reward_label"],
                    "global_step": step,
                    "n": len(step_rows),
                    "mean": float(step_values.mean()),
                    "std": float(step_values.std(ddof=0)),
                    "min": float(q0),
                    "q1": float(q1),
                    "median": float(median),
                    "q3": float(q3),
                    "max": float(q4),
                }
            )
            for sample_index, row in enumerate(step_rows):
                sample_rows.append(
                    {
                        "reward_definition": reward_definition,
                        "reward_label": item["reward_label"],
                        "global_step": step,
                        "sample_index": sample_index,
                        "trajectory_mse": row["trajectory_mse"],
                        "rollout_uid": row["rollout_uid"],
                        "color": row["color"],
                        "source_path": str(item["source"]),
                        "source_sha256": item["source_sha256"],
                    }
                )

        plot_name = f"trajectory_mse_violin_{reward_definition}.png"
        save_plot(
            output_dir / plot_name,
            item["reward_label"],
            item["color"],
            steps,
            item["by_step"],
            y_limits,
        )
        generated_plots.append(plot_name)

    sample_path = output_dir / "trajectory_mse_violin_samples.csv"
    write_csv_atomic(
        sample_path,
        [
            "reward_definition",
            "reward_label",
            "global_step",
            "sample_index",
            "trajectory_mse",
            "rollout_uid",
            "color",
            "source_path",
            "source_sha256",
        ],
        sample_rows,
    )
    summary_path = output_dir / "trajectory_mse_violin_summary.csv"
    write_csv_atomic(
        summary_path,
        [
            "reward_definition",
            "reward_label",
            "global_step",
            "n",
            "mean",
            "std",
            "min",
            "q1",
            "median",
            "q3",
            "max",
        ],
        summary_rows,
    )
    manifest = {
        "excluded_reward_definition": "binary_success",
        "expected_samples_per_step": EXPECTED_SAMPLES_PER_STEP,
        "generated_plots": generated_plots,
        "metric": "trajectory_mse",
        "plot_semantics": {
            "density": "Gaussian KDE with Scott bandwidth, clipped to data range",
            "dot": "median",
            "vertical_bar": "interquartile range",
            "y_limits_shared_across_plots": list(y_limits),
        },
        "reward_definitions": list(loaded),
        "sample_count": len(sample_rows),
        "source_files": {
            reward_definition: {
                "path": str(item["source"]),
                "sha256": item["source_sha256"],
            }
            for reward_definition, item in loaded.items()
        },
        "steps": steps,
    }
    manifest_path = output_dir / "trajectory_mse_violin_manifest.json"
    write_json_atomic(manifest_path, manifest)

    print(
        json.dumps(
            {
                "plots": [str(output_dir / name) for name in generated_plots],
                "steps": [steps[0], steps[-1]],
                "samples": len(sample_rows),
                "samples_csv": str(sample_path),
                "summary_csv": str(summary_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
