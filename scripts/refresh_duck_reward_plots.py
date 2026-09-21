#!/usr/bin/env python3
"""Refresh per-run and common-prefix plots for the three Duck reward runs."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RUNS = (
    (
        "combined_mse",
        "Combined MSE",
        "#CC79A7",
        OUT / "duck4_combined_mse_gs15_to16_principal_ctrl50_extreme_videos_8n4g_base50_s5",
    ),
    (
        "trajectory_only",
        "Trajectory-only",
        "#0072B2",
        OUT / "duck4_combined_mse_gs15_to16_principal_ctrl50_extreme_videos_8n4g_trajectory_only50_s5",
    ),
    (
        "terminal_only",
        "Terminal-only",
        "#D55E00",
        OUT / "duck4_combined_mse_gs15_to16_principal_ctrl50_extreme_videos_8n4g_terminal_only50_s5",
    ),
)


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return max(0.0, center - half), min(1.0, center + half)


def read_source(run: Path) -> str:
    path = run / "analysis" / "summary.json"
    if path.is_file():
        return str(json.loads(path.read_text(encoding="utf-8"))["reward_source"])
    return "ctrl_world_aligned_mse_plus_terminal"


def source_reward(source: str) -> tuple[str, str]:
    if source == "trajectory_mse":
        return "trajectory_reward_mean", "trajectory_reward"
    if source == "terminal_goal_mse":
        return "terminal_reward_mean", "lastframe_reward"
    return "reward_mean", "combined_reward"


def read_summary(run: Path, source: str) -> list[dict[str, float]]:
    path = run / "analysis" / "per_step_summary.csv"
    reward_field, _ = source_reward(source)
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("data_status") != "complete":
                continue
            row: dict[str, float] = {"step": int(float(raw["step"]))}
            for field in (
                "n_valid",
                "success_rate",
                "success_ci95_low",
                "success_ci95_high",
                "reward_mean",
                "reward_p10",
                "reward_p90",
                "trajectory_mse_mean",
                "terminal_goal_mse_mean",
                "trajectory_reward_mean",
                "terminal_reward_mean",
            ):
                row[field] = float(raw[field])
            row["training_reward"] = row[reward_field]
            rows.append(row)
    rows.sort(key=lambda row: int(row["step"]))
    if not rows:
        raise RuntimeError(f"no complete steps in {path}")
    return rows


def read_training_values(run: Path, source: str) -> dict[int, list[float]]:
    _, field = source_reward(source)
    path = run / "analysis" / "trajectories.csv"
    values: dict[int, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if str(raw.get("valid", "true")).lower() not in {"true", "1"}:
                continue
            if not finite(raw.get(field)):
                continue
            values.setdefault(int(float(raw["global_step"])), []).append(float(raw[field]))
    return values


def read_probability_values(run: Path) -> dict[int, list[float]]:
    values: dict[int, list[float]] = {}
    path = run / "analysis" / "trajectories.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if str(raw.get("valid", "true")).lower() not in {"true", "1"}:
                continue
            if not finite(raw.get("success_probability_max")):
                continue
            values.setdefault(int(float(raw["global_step"])), []).append(
                float(raw["success_probability_max"])
            )
    return values


def read_validation(run: Path) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    root = run / "evaluation_records"
    for path in sorted(root.glob("step_*.jsonl"), key=lambda item: int(re.search(r"step_(\d+)", item.name).group(1))):
        step = int(re.search(r"step_(\d+)", path.name).group(1)) + 1
        records = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("valid") and record.get("complete") and not record.get("exception") and not record.get("error"):
                        records.append(record)
        if not records:
            continue
        total = len(records)
        successes = sum(bool(record.get("success")) for record in records)
        low, high = wilson(successes, total)
        result.append(
            {
                "step": step,
                "n_valid": total,
                "success_rate": successes / total,
                "success_ci95_low": low,
                "success_ci95_high": high,
                "reward_mean": mean(float(record["combined_reward"]) for record in records),
                "trajectory_mse_mean": mean(float(record["trajectory_mse"]) for record in records),
                "terminal_goal_mse_mean": mean(float(record["terminal_goal_mse"]) for record in records),
            }
        )
    return result


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def gnuplot(base: Path, body: str, width: int = 1800, height: int = 1100) -> None:
    scripts = []
    for extension, terminal in (
        ("png", f'pngcairo size {width},{height} enhanced font "Sans,18"'),
        ("svg", f'svg size {width},{height} enhanced font "Sans,18"'),
    ):
        output = base.with_suffix(f".{extension}")
        script = f'set terminal {terminal}\nset output "{output.as_posix()}"\n{body.strip()}\n'
        subprocess.run(["gnuplot"], input=script, text=True, check=True)
        scripts.append(script)
    write(base.with_suffix(".gnuplot"), scripts[0])


def write_data(path: Path, rows: list[dict[str, float]]) -> None:
    lines = ["# step success low high training_reward reward_p10 reward_p90 trajectory_mse terminal_mse"]
    for row in rows:
        lines.append(
            " ".join(
                f"{row[field]:.12g}"
                for field in (
                    "step",
                    "success_rate",
                    "success_ci95_low",
                    "success_ci95_high",
                    "training_reward",
                    "reward_p10",
                    "reward_p90",
                    "trajectory_mse_mean",
                    "terminal_goal_mse_mean",
                )
            )
        )
    write(path, "\n".join(lines) + "\n")


def convert_png(svg: Path, png: Path) -> None:
    subprocess.run(["convert", "-background", "white", str(svg), str(png)], check=True)


def violin(path: Path, values: dict[int, list[float]], title: str, ylabel: str, color: str, *, probability: bool = False) -> None:
    steps = sorted(values)
    all_values = [value for step in steps for value in values[step]]
    low = 0.0 if probability else min(all_values)
    high = 1.0 if probability else max(all_values)
    if high <= low:
        high = low + 1.0
    padding = (high - low) * 0.08
    ylow, yhigh = low - padding, high + padding
    width, height = 2400, 1200
    left, right, top, bottom = 145, 70, 110, 155
    plot_width, plot_height = width - left - right, height - top - bottom

    def xcoord(step: int) -> float:
        return left + (step - steps[0]) / max(1, steps[-1] - steps[0]) * plot_width

    def ycoord(value: float) -> float:
        return top + (yhigh - value) / (yhigh - ylow) * plot_height

    def label(value: float) -> str:
        return f"{value * 100:.0f}%" if probability else f"{value:.1f}"

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="55" text-anchor="middle" font-family="DejaVu Sans" font-size="30" fill="#202020">{title}</text>',
        f'<text x="38" y="{height / 2}" transform="rotate(-90 38 {height / 2})" text-anchor="middle" font-family="DejaVu Sans" font-size="21" fill="#202020">{ylabel}</text>',
        f'<text x="{width / 2}" y="{height - 50}" text-anchor="middle" font-family="DejaVu Sans" font-size="21" fill="#202020">RL update step</text>',
    ]
    for index in range(7):
        value = ylow + (yhigh - ylow) * index / 6.0
        y = ycoord(value)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#D9D9D9" stroke-width="2"/><text x="{left-16}" y="{y+6:.2f}" text-anchor="end" font-family="DejaVu Sans" font-size="17" fill="#333">{label(value)}</text>')
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#202020" stroke-width="3"/><line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#202020" stroke-width="3"/>')

    for step in steps:
        sample = sorted(values[step])
        n = len(sample)
        q1 = sample[int(0.25 * (n - 1))]
        med = sample[int(0.50 * (n - 1))]
        q3 = sample[int(0.75 * (n - 1))]
        avg = sum(sample) / n
        std = pstdev(sample) if n > 1 else 0.0
        bandwidth = max(std * 1.06 * n ** -0.2, (yhigh - ylow) / 600.0)
        support = [ylow + (yhigh - ylow) * index / 100.0 for index in range(101)]
        density = [sum(math.exp(-0.5 * ((value - item) / bandwidth) ** 2) for item in sample) for value in support]
        maximum = max(density) or 1.0
        half_width = min(plot_width / max(4, len(steps) * 2.2), 105.0)
        x = xcoord(step)
        points = [f"{x + half_width * value / maximum:.2f},{ycoord(value):.2f}" for value, value_density in zip(support, density) for _ in ()]
        points = [f"{x + half_width * d / maximum:.2f},{ycoord(value):.2f}" for value, d in zip(support, density)]
        points.extend(f"{x - half_width * d / maximum:.2f},{ycoord(value):.2f}" for value, d in reversed(list(zip(support, density))))
        elements.append(f'<polygon points="{" ".join(points)}" fill="{color}" fill-opacity="0.68" stroke="#202020" stroke-width="2"/>')
        elements.append(f'<line x1="{x:.2f}" y1="{ycoord(q1):.2f}" x2="{x:.2f}" y2="{ycoord(q3):.2f}" stroke="#202020" stroke-width="5"/><line x1="{x-13:.2f}" y1="{ycoord(med):.2f}" x2="{x+13:.2f}" y2="{ycoord(med):.2f}" stroke="white" stroke-width="5"/><circle cx="{x:.2f}" cy="{ycoord(avg):.2f}" r="5" fill="#202020"/>')
        elements.append(f'<line x1="{x:.2f}" y1="{height-bottom}" x2="{x:.2f}" y2="{height-bottom+9}" stroke="#202020" stroke-width="2"/><text x="{x:.2f}" y="{height-bottom+37}" text-anchor="middle" font-family="DejaVu Sans" font-size="17" fill="#333">{step}</text>')
    elements.append(f'<text x="{width-540}" y="78" font-family="DejaVu Sans" font-size="15" fill="#444">n=128/step; line=IQR; white=median; dot=mean</text></svg>')
    write(path, "\n".join(elements))


def plot_run(slug: str, label: str, color: str, run: Path) -> tuple[list[dict[str, float]], str, list[dict[str, float]]]:
    analysis = run / "analysis"
    source = read_source(run)
    rows = read_summary(run, source)
    write_data(analysis / ".three_reward_plot.dat", rows)
    fields = ["step", "n_valid", "success_rate", "success_ci95_low", "success_ci95_high", "training_reward", "reward_mean", "reward_std", "reward_p10", "reward_p90", "trajectory_reward_mean", "terminal_reward_mean", "trajectory_mse_mean", "terminal_goal_mse_mean"]
    with (analysis / "reward_by_step.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    data = (analysis / ".three_reward_plot.dat").as_posix()
    last = int(rows[-1]["step"])
    xtics = 1 if last <= 20 else 5
    gnuplot(analysis / "success_rate_vs_step", f'''
set title "Four-Color Duck ({label}): Success Rate vs RL Step"
set xlabel "RL update step"; set ylabel "Success rate"; set xrange [0.5:{last + 0.5}]; set yrange [0:1]; set xtics {xtics}; set ytics 0.1; set format y "%.0f%%"; set grid xtics ytics lc rgb "#D9D9D9"; set key outside center top horizontal
plot "{data}" using 1:2:3:4 with yerrorbars lw 1.2 pt 0 lc rgb "{color}" notitle, "{data}" using 1:2 with linespoints lw 3 pt 7 ps 1.05 lc rgb "{color}" title "{label}"
''')
    gnuplot(analysis / "reward_vs_step", f'''
set title "Four-Color Duck ({label}): Training Reward vs RL Step"
set xlabel "RL update step"; set ylabel "Mean training reward"; set xrange [0.5:{last + 0.5}]; set xtics {xtics}; set grid xtics ytics lc rgb "#D9D9D9"; set key outside center top horizontal
plot "{data}" using 1:5 with linespoints lw 3 pt 7 ps 1.05 lc rgb "{color}" title "{label}"
''')
    gnuplot(analysis / "mse_vs_step", f'''
set multiplot layout 1,2 title "Four-Color Duck ({label}): MSE vs RL Step" font ",22"; set xlabel "RL update step"; set xrange [0.5:{last + 0.5}]; set xtics {xtics}; set grid xtics ytics lc rgb "#D9D9D9"; set key outside center top horizontal
set ylabel "Trajectory MSE (lower is better)"; set title "Trajectory MSE"; plot "{data}" using 1:8 with linespoints lw 3 pt 7 ps 1.05 lc rgb "#0072B2" title "Trajectory MSE"
set ylabel "Terminal-goal MSE (lower is better)"; set title "Terminal-goal MSE"; plot "{data}" using 1:9 with linespoints lw 3 pt 7 ps 1.05 lc rgb "#D55E00" title "Terminal-goal MSE"
unset multiplot
''', width=2400, height=1000)
    reward_values = read_training_values(run, source)
    violin(analysis / "reward_violin_vs_step.svg", reward_values, f"Four-Color Duck ({label}): Training Reward Distribution vs RL Step", "Mean training reward", color)
    convert_png(analysis / "reward_violin_vs_step.svg", analysis / "reward_violin_vs_step.png")
    probability_values = read_probability_values(run)
    violin(analysis / "success_probability_violin_vs_step.svg", probability_values, f"Four-Color Duck ({label}): Success Probability Distribution vs RL Step", "Success probability", "#6A51A3", probability=True)
    convert_png(analysis / "success_probability_violin_vs_step.svg", analysis / "success_probability_violin_vs_step.png")
    validation = read_validation(run)
    if validation:
        with (analysis / "validation_metrics_by_step.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["step", "n_valid", "success_rate", "success_ci95_low", "success_ci95_high", "reward_mean", "trajectory_mse_mean", "terminal_goal_mse_mean"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(validation)
        vdata = analysis / ".three_validation.dat"
        write(vdata, "# step success low high reward trajectory_mse terminal_mse\n" + "\n".join(" ".join(f"{row[field]:.12g}" for field in ("step", "success_rate", "success_ci95_low", "success_ci95_high", "reward_mean", "trajectory_mse_mean", "terminal_goal_mse_mean")) for row in validation) + "\n")
        vq = vdata.as_posix()
        vmax = int(validation[-1]["step"])
        gnuplot(analysis / "validation_success_rate_vs_step", f'''
set title "Four-Color Duck ({label}): Validation Success Rate"; set xlabel "RL update step"; set ylabel "Validation success rate"; set xrange [0.5:{vmax + 0.5}]; set yrange [0:1]; set xtics 5; set ytics 0.1; set format y "%.0f%%"; set grid xtics ytics lc rgb "#D9D9D9"; set key outside center top horizontal
plot "{vq}" using 1:2:3:4 with yerrorbars lw 1.2 pt 0 lc rgb "{color}" notitle, "{vq}" using 1:2 with linespoints lw 3 pt 7 ps 1.05 lc rgb "{color}" title "{label}"
''')
        gnuplot(analysis / "validation_reward_mse_vs_step", f'''
set multiplot layout 1,2 title "Four-Color Duck ({label}): Validation Reward and MSE" font ",22"; set xlabel "RL update step"; set xtics 5; set grid xtics ytics lc rgb "#D9D9D9"; set key outside center top horizontal
set ylabel "Mean combined reward"; set title "Validation Reward"; plot "{vq}" using 1:5 with linespoints lw 3 pt 7 ps 1.05 lc rgb "{color}" title "{label}"
set ylabel "MSE (lower is better)"; set title "Validation MSE"; plot "{vq}" using 1:6 with linespoints lw 3 pt 7 ps 1.05 lc rgb "#0072B2" title "Trajectory MSE", "{vq}" using 1:7 with linespoints lw 3 pt 7 ps 1.05 lc rgb "#D55E00" title "Terminal-goal MSE"
unset multiplot
''', width=2400, height=1000)
    manifest = {"source_records": str(run / "trajectory_records"), "latest_step": last, "samples_per_step": 128, "reward_source": source, "training_reward_metric": source_reward(source)[0], "plots_refreshed": ["success_rate_vs_step", "reward_vs_step", "mse_vs_step", "reward_violin_vs_step", "success_probability_violin_vs_step", "validation_success_rate_vs_step", "validation_reward_mse_vs_step"]}
    write(analysis / "plot_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return rows, source, validation


def comparison(items: list[tuple[list[dict[str, float]], str, list[dict[str, float]]]]) -> tuple[Path, int]:
    common = min(len(item[0]) for item in items)
    out = OUT / "duck4_three_reward_comparison" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    datasets = []
    for (slug, label, color, _), (rows, _, _) in zip(RUNS, items):
        path = out / f".{slug}.dat"
        write_data(path, rows[:common])
        datasets.append((slug, label, color, path))

    def clauses(column: int) -> str:
        return ", \\\n+     ".join(f'"{path.as_posix()}" using 1:{column} with linespoints lw 3 pt 7 ps 1.0 lc rgb "{color}" title "{label}"' for _, label, color, path in datasets)

    xmax = common + 0.5
    gnuplot(out / "success_rate_vs_step", f'''set title "Four-Color Duck: Three Reward Sources — Success Rate"; set xlabel "RL update step"; set ylabel "Success rate"; set xrange [0.5:{xmax}]; set yrange [0:1]; set xtics 1; set ytics 0.1; set format y "%.0f%%"; set grid xtics ytics lc rgb "#D9D9D9"; set key outside center top horizontal
plot {clauses(2)}
''')
    gnuplot(out / "reward_vs_step", f'''set title "Four-Color Duck: Three Reward Sources — Training Reward"; set xlabel "RL update step"; set ylabel "Mean training reward"; set xrange [0.5:{xmax}]; set xtics 1; set grid xtics ytics lc rgb "#D9D9D9"; set key outside center top horizontal
plot {clauses(5)}
''')
    gnuplot(out / "mse_vs_step", f'''set multiplot layout 1,2 title "Four-Color Duck: Three Reward Sources — MSE" font ",22"; set xlabel "RL update step"; set xrange [0.5:{xmax}]; set xtics 1; set grid xtics ytics lc rgb "#D9D9D9"; set key outside center top horizontal
set ylabel "Trajectory MSE (lower is better)"; set title "Trajectory MSE"; plot {clauses(8)}
set ylabel "Terminal-goal MSE (lower is better)"; set title "Terminal-goal MSE"; plot {clauses(9)}
unset multiplot
''', width=2400, height=1000)
    with (out / "comparison_per_step.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["reward_definition", "step", "success_rate", "training_reward", "trajectory_mse_mean", "terminal_goal_mse_mean"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (slug, _, _, _), (rows, _, _) in zip(RUNS, items):
            for row in rows[:common]:
                writer.writerow({"reward_definition": slug, "step": row["step"], "success_rate": row["success_rate"], "training_reward": row["training_reward"], "trajectory_mse_mean": row["trajectory_mse_mean"], "terminal_goal_mse_mean": row["terminal_goal_mse_mean"]})
    write(out / "plot_manifest.json", json.dumps({"runs": [item[0] for item in RUNS], "common_steps": [1, common]}, indent=2) + "\n")
    return out, common


def main() -> None:
    items = [plot_run(*spec) for spec in RUNS]
    out, common = comparison(items)
    print(json.dumps({"runs": [{"name": spec[0], "latest_step": item[0][-1]["step"], "reward_source": item[1], "validation_steps": [row["step"] for row in item[2]]} for spec, item in zip(RUNS, items)], "comparison_dir": str(out), "comparison_common_steps": common}, indent=2))


if __name__ == "__main__":
    main()
