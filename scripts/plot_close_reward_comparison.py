#!/usr/bin/env python3
"""Plot reward ablations for close-laptop or four-color Duck runs.

The script reads each run's committed ``analysis/per_step_summary.csv`` and
produces:

* ``success_rate_vs_step.png``: one training curve per reward definition.
* ``mse_vs_step.png``: trajectory MSE and terminal-goal MSE in two panels,
  with one curve per reward definition in each panel.
* ``comparison_per_step.csv``: the exact values used by the plots.
* ``validation_success_rate_vs_step.png`` and ``validation_comparison.csv``
  from the saved independent validation sets, when available.

Only Python's standard library and gnuplot are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable, Optional


BASELINE_PREFIX_STEPS = list(range(1, 6))
VALIDATION_INTERVAL = 5

VALIDATION_CONFIGS = {
    "close": (
        8,
        ("terminal_goal_mse", "trajectory_mse", "combined_mse"),
    ),
    "duck4": (
        40,
        ("terminal_goal_mse", "trajectory_mse", "binary_success", "combined_mse"),
    ),
}

CLOSE_RUNS = (
    (
        "terminal_goal_mse",
        "Terminal-goal MSE reward",
        "#D55E00",
        "close_b128_mb32_u15_terminal_mse_reward_8n4g",
        None,
    ),
    (
        "trajectory_mse",
        "Trajectory MSE reward",
        "#0072B2",
        "close_b128_mb32_u15_traj_reward_8n4g",
        None,
    ),
    (
        "combined_mse",
        "Combined MSE reward",
        "#CC79A7",
        "close_b128_mb32_u50_combined_mse_reward_matched_rollout_8n4g",
        None,
    ),
)

DUCK4_RUNS = (
    (
        "terminal_goal_mse",
        "Terminal-goal MSE reward",
        "#D55E00",
        "duck4_b128_mb32_u15_terminal_mse_reward_8n4g",
        None,
    ),
    (
        "trajectory_mse",
        "Trajectory MSE reward",
        "#0072B2",
        "duck4_b128_mb32_u15_traj_reward_8n4g",
        None,
    ),
    (
        "binary_success",
        "Binary-success reward",
        "#009E73",
        "duck4_b128_mb32_u15_success_reward_8n4g",
        None,
    ),
    (
        "combined_mse",
        "Combined MSE reward",
        "#CC79A7",
        "duck4_b128_mb32_u15_combined_mse_reward_8n4g",
        None,
    ),
)

PRESETS = {
    "close": (
        "Close Laptop",
        "close_b128_mb32_u15_reward_comparison",
        CLOSE_RUNS,
    ),
    "duck4": (
        "Four-Color Duck",
        "duck4_b128_mb32_u15_reward_comparison",
        DUCK4_RUNS,
    ),
}

NUMERIC_FIELDS = (
    "step",
    "n_valid",
    "success_rate",
    "success_ci95_low",
    "success_ci95_high",
    "trajectory_mse_mean",
    "terminal_goal_mse_mean",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="close",
        help="Experiment family to compare (default: close).",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=repo_root / "outputs",
        help="Directory containing the configured experiment run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination for the comparison plots and their source table.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        _, output_name, _ = PRESETS[args.preset]
        args.output_dir = repo_root / "outputs" / output_name / "analysis"
    return args


def wilson_interval(success_rate: float, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    denominator = 1.0 + z * z / n
    center = (success_rate + z * z / (2.0 * n)) / denominator
    half_width = (
        z
        * math.sqrt(
            success_rate * (1.0 - success_rate) / n + z * z / (4.0 * n * n)
        )
        / denominator
    )
    return center - half_width, center + half_width


def read_logged_prefix(path: Path) -> list[dict[str, float]]:
    """Recover baseline steps 1--5 that predate trajectory-record persistence."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing baseline prefix log: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    success_rates = [
        float(value)
        for value in re.findall(r"cosmos/success_rate=([0-9.eE+-]+)", text)
    ]
    trajectory_mses = [
        float(value)
        for value in re.findall(r"reward/video_mse=([0-9.eE+-]+)", text)
    ]
    terminal_mses = [
        float(value)
        for value in re.findall(r"reward/terminal_goal_mse=([0-9.eE+-]+)", text)
    ]
    if not (len(success_rates) == len(trajectory_mses) == len(terminal_mses) == 5):
        raise ValueError(
            "Expected exactly five baseline metric tables in "
            f"{path}; got success={len(success_rates)}, "
            f"trajectory_mse={len(trajectory_mses)}, terminal_mse={len(terminal_mses)}"
        )

    rows: list[dict[str, float]] = []
    for step, success_rate, trajectory_mse, terminal_mse in zip(
        BASELINE_PREFIX_STEPS, success_rates, trajectory_mses, terminal_mses
    ):
        success_count = round(success_rate * 128)
        if not math.isclose(success_count / 128, success_rate, abs_tol=1e-12):
            raise ValueError(f"Step {step} success rate is not based on 128 samples")
        ci_low, ci_high = wilson_interval(success_rate, 128)
        rows.append(
            {
                "step": float(step),
                "n_valid": 128.0,
                "success_rate": success_rate,
                "success_ci95_low": ci_low,
                "success_ci95_high": ci_high,
                "trajectory_mse_mean": trajectory_mse,
                "terminal_goal_mse_mean": terminal_mse,
            }
        )
    return rows


def read_run(
    path: Path, prefix_log: Optional[Path] = None
) -> list[dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-step summary: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(NUMERIC_FIELDS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows: list[dict[str, float]] = []
        for raw in reader:
            if raw.get("data_status") != "complete":
                continue
            row = {field: float(raw[field]) for field in NUMERIC_FIELDS}
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError(f"Non-finite plotted value in {path}: {row}")
            rows.append(row)

    steps = [int(row["step"]) for row in rows]
    if prefix_log is not None and steps and steps[0] == 6:
        rows = read_logged_prefix(prefix_log) + rows
        steps = [int(row["step"]) for row in rows]
    if not steps:
        raise ValueError(f"No committed steps in {path}")
    expected_steps = list(range(1, steps[-1] + 1))
    if steps != expected_steps:
        raise ValueError(
            f"Expected contiguous committed steps {expected_steps}, got {steps} in {path}"
        )
    if any(int(row["n_valid"]) != 128 for row in rows):
        raise ValueError(f"Expected 128 valid trajectories per step in {path}")
    return rows


def read_validation_run(
    path: Path,
    expected_steps: list[int],
    expected_records_per_step: int,
) -> list[dict[str, float]]:
    """Read and validate saved independent-evaluation checkpoints."""
    if not path.is_dir():
        raise FileNotFoundError(f"Missing validation directory: {path}")

    rows: list[dict[str, float]] = []
    for expected_step in expected_steps:
        # Evaluation files use the zero-based update index in their names,
        # while ``global_step`` is the one-based RL step shown in plots.
        jsonl_path = path / f"step_{expected_step - 1}.jsonl"
        if not jsonl_path.is_file():
            raise FileNotFoundError(f"Missing validation records: {jsonl_path}")

        records: list[dict[str, object]] = []
        with jsonl_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {jsonl_path}:{line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected an object in {jsonl_path}:{line_number}"
                    )
                records.append(record)

        if len(records) != expected_records_per_step:
            raise ValueError(
                f"Expected {expected_records_per_step} validation records in "
                f"{jsonl_path}, got {len(records)}"
            )

        global_steps = {int(record["global_step"]) for record in records}
        if global_steps != {expected_step}:
            raise ValueError(
                f"Expected global step {expected_step} in {jsonl_path}, "
                f"got {sorted(global_steps)}"
            )

        keys = [(record["color"], int(record["episode"])) for record in records]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate validation (color, episode) in {jsonl_path}")

        for record in records:
            if not record.get("valid", False):
                raise ValueError(f"Invalid validation record in {jsonl_path}: {record}")
            if record.get("complete") is not True:
                raise ValueError(f"Incomplete validation record in {jsonl_path}: {record}")
            if record.get("error") not in (None, "") or record.get(
                "exception"
            ) not in (None, ""):
                raise ValueError(f"Validation error in {jsonl_path}: {record}")
            expected_success = (
                float(record["success_probability_max"])
                >= float(record["success_threshold"])
            )
            if bool(record["success"]) != expected_success:
                raise ValueError(
                    f"Success/threshold mismatch in {jsonl_path}: {record}"
                )

        success_count = sum(bool(record["success"]) for record in records)
        rows.append(
            {
                "step": float(expected_step),
                "n_valid": float(len(records)),
                "n_success": float(success_count),
                "success_rate": success_count / len(records),
            }
        )

    return rows


def gnuplot_quote(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')


def write_dat(path: Path, rows: Iterable[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# step success_rate ci95_low ci95_high trajectory_mse_mean "
            "terminal_goal_mse_mean\n"
        )
        for row in rows:
            handle.write(
                f"{int(row['step'])} {row['success_rate']:.12g} "
                f"{row['success_ci95_low']:.12g} "
                f"{row['success_ci95_high']:.12g} "
                f"{row['trajectory_mse_mean']:.12g} "
                f"{row['terminal_goal_mse_mean']:.12g}\n"
            )


def write_validation_dat(
    path: Path, rows: Iterable[dict[str, float]]
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# step success_rate n_valid n_success\n")
        for row in rows:
            handle.write(
                f"{int(row['step'])} {row['success_rate']:.12g} "
                f"{int(row['n_valid'])} {int(row['n_success'])}\n"
            )


def run_gnuplot(script: str, script_path: Path) -> None:
    script_path.write_text(script.strip() + "\n", encoding="utf-8")
    executable = shutil.which("gnuplot")
    if executable is None:
        raise RuntimeError("gnuplot is required but was not found on PATH")
    subprocess.run(
        [executable, str(script_path)],
        check=True,
        text=True,
    )


def plot_clause(
    data_paths: dict[str, Path],
    runs: tuple[tuple[str, str, str, str, Optional[str]], ...],
    *,
    value_column: int,
    confidence_intervals: bool = False,
) -> str:
    clauses: list[str] = []
    for run_key, label, color, _, _ in runs:
        data_path = gnuplot_quote(data_paths[run_key])
        if confidence_intervals:
            clauses.append(
                f'"{data_path}" using 1:2:3:4 with yerrorbars '
                f'lw 1.2 pt 0 lc rgb "{color}" notitle'
            )
        clauses.append(
            f'"{data_path}" using 1:{value_column} with linespoints '
            f'lw 3 pt 7 ps 1.05 lc rgb "{color}" title "{label}"'
        )
    return ", \\\n     ".join(clauses)


def main() -> None:
    args = parse_args()
    task_title, _, runs = PRESETS[args.preset]
    outputs_root = args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, list[dict[str, float]]] = {}
    for run_key, _, _, run_dir, prefix_log in runs:
        run_root = outputs_root / run_dir
        rows = read_run(
            run_root / "analysis" / "per_step_summary.csv",
            run_root / prefix_log if prefix_log is not None else None,
        )
        data[run_key] = rows

    common_max_step = min(int(rows[-1]["step"]) for rows in data.values())
    common_steps = list(range(1, common_max_step + 1))
    x_tick_step = 1 if common_max_step <= 20 else 5

    data_paths: dict[str, Path] = {}
    for run_key, _, _, _, _ in runs:
        rows = [
            row for row in data[run_key]
            if int(row["step"]) <= common_max_step
        ]
        if [int(row["step"]) for row in rows] != common_steps:
            raise ValueError(
                f"{run_key} does not contain the common committed prefix "
                f"1--{common_max_step}"
            )
        data[run_key] = rows
        data_path = output_dir / f".{run_key}_per_step.dat"
        write_dat(data_path, rows)
        data_paths[run_key] = data_path

    success_y_max = 0.3 if args.preset == "duck4" else 1.05
    success_tick_step = 5 if args.preset == "duck4" else 10
    success_tick_limit = 30 if args.preset == "duck4" else 100
    success_y_ticks = ", ".join(
        f'"{percent}%%" {percent / 100:.2f}'
        for percent in range(0, success_tick_limit + 1, success_tick_step)
    )

    comparison_csv = output_dir / "comparison_per_step.csv"
    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ("reward_definition",) + NUMERIC_FIELDS
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run_key, _, _, _, _ in runs:
            for row in data[run_key]:
                writer.writerow({"reward_definition": run_key, **row})

    success_image = output_dir / "success_rate_vs_step.png"
    success_script = f'''
set terminal pngcairo size 1800,1100 enhanced font "Sans,18"
set output "{gnuplot_quote(success_image)}"
set title "{task_title}: Success Rate vs RL Step"
set xlabel "RL update step"
set ylabel "Success rate"
set xrange [0.5:{common_max_step + 0.5}]
set yrange [0:{success_y_max}]
set xtics {x_tick_step}
set ytics ({success_y_ticks})
set grid xtics ytics lc rgb "#D9D9D9"
set border lw 1.5
set key outside center top horizontal maxrows 1
set pointintervalbox 0
plot {plot_clause(data_paths, runs, value_column=2)}
'''
    run_gnuplot(success_script, output_dir / "success_rate_vs_step.gnuplot")

    validation_image: Optional[Path] = None
    validation_csv: Optional[Path] = None
    validation_episode_count, validation_run_keys = VALIDATION_CONFIGS[args.preset]
    validation_runs = tuple(run for run in runs if run[0] in validation_run_keys)
    if tuple(run[0] for run in validation_runs) != validation_run_keys:
        raise ValueError(
            f"Validation run configuration mismatch for {args.preset}: "
            f"expected {validation_run_keys}, got "
            f"{tuple(run[0] for run in validation_runs)}"
        )
    if validation_runs:
        validation_steps = list(
            range(VALIDATION_INTERVAL, common_max_step + 1, VALIDATION_INTERVAL)
        )
        if not validation_steps:
            raise ValueError(
                f"No validation boundary in common steps 1--{common_max_step}"
            )
        validation_data: dict[str, list[dict[str, float]]] = {}
        validation_paths: dict[str, Path] = {}
        for run_key, _, _, run_dir, _ in validation_runs:
            rows = read_validation_run(
                outputs_root / run_dir / "evaluation_records",
                validation_steps,
                validation_episode_count,
            )
            validation_data[run_key] = rows
            data_path = output_dir / f".{run_key}_validation.dat"
            write_validation_dat(data_path, rows)
            validation_paths[run_key] = data_path

        validation_csv = output_dir / "validation_comparison.csv"
        with validation_csv.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = (
                "reward_definition",
                "step",
                "n_valid",
                "n_success",
                "success_rate",
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for run_key, _, _, _, _ in validation_runs:
                for row in validation_data[run_key]:
                    writer.writerow({"reward_definition": run_key, **row})

        validation_image = output_dir / "validation_success_rate_vs_step.png"
        validation_xtics = ", ".join(
            f'"{step}" {step}' for step in validation_steps
        )
        validation_script = f'''
set terminal pngcairo size 1800,1100 enhanced font "Sans,18"
set output "{gnuplot_quote(validation_image)}"
set title "{task_title}: Validation Success Rate vs RL Step (n={validation_episode_count})"
set xlabel "RL update step"
set ylabel "Validation success rate"
set xrange [{validation_steps[0] - 0.5}:{validation_steps[-1] + 0.5}]
set yrange [0:{success_y_max}]
set xtics ({validation_xtics})
set ytics ({success_y_ticks})
set grid xtics ytics lc rgb "#D9D9D9"
set border lw 1.5
set key outside center top horizontal maxrows 1
set pointintervalbox 0
plot {plot_clause(validation_paths, validation_runs, value_column=2)}
'''
        run_gnuplot(
            validation_script,
            output_dir / "validation_success_rate_vs_step.gnuplot",
        )

    mse_image = output_dir / "mse_vs_step.png"
    trajectory_clause = plot_clause(data_paths, runs, value_column=5)
    terminal_clause = plot_clause(data_paths, runs, value_column=6)
    mse_script = f'''
set terminal pngcairo size 2400,1000 enhanced font "Sans,17"
set output "{gnuplot_quote(mse_image)}"
set multiplot layout 1,2 title "{task_title}: MSE vs RL Step" font ",22"
set xlabel "RL update step"
set xrange [0.5:{common_max_step + 0.5}]
set xtics {x_tick_step}
set format y "%.3f"
set grid xtics ytics lc rgb "#D9D9D9"
set border lw 1.5
set key outside center top horizontal maxrows 2
set ylabel "Trajectory MSE (lower is better)"
set title "Trajectory MSE"
plot {trajectory_clause}
set ylabel "Terminal-goal MSE (lower is better)"
set title "Terminal-goal MSE"
plot {terminal_clause}
unset multiplot
'''
    run_gnuplot(mse_script, output_dir / "mse_vs_step.gnuplot")

    print(success_image)
    print(mse_image)
    print(comparison_csv)
    print(f"common committed steps: 1-{common_max_step}")
    if validation_image is not None and validation_csv is not None:
        print(validation_image)
        print(validation_csv)


if __name__ == "__main__":
    main()
