#!/usr/bin/env python3
"""Collect completed and partial demo-chain diagnostics without changing them."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    rows = []
    reports = []
    for directory in sorted(root.glob("episode_*")):
        if not directory.is_dir():
            continue
        complete = (directory / "summary.json").is_file()
        path = directory / ("summary.json" if complete else "partial_summary.json")
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        reports.append((directory, report, complete))
        for name, case in [("real", report), *report["cases"].items()]:
            before = case["real_at_original_horizon"] if name == "real" else case["at_original_horizon"]
            end = case["real_at_end"] if name == "real" else case["at_demonstration_end"]
            rows.append({"episode": report["episode"], "case": name,
                         "demo_duration_seconds": report["source_duration_seconds"],
                         "success_at_5_chunks": bool(before["success"][0]),
                         "probability_at_5_chunks": before["reported_probability"][0],
                         "success_at_demo_end": bool(end["success"][0]),
                         "probability_at_demo_end": end["reported_probability"][0],
                         "episode_all_cases_complete": complete})
    if not rows:
        print("No scored real references yet")
        return
    with (root / "result_table.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    text = ["# Demonstration replay results", "", f"Fully completed episodes: {sum(c for _,_,c in reports)} / {len(reports)}. Partial results do not indicate job completion.", "",
            "| Episode | Condition | Demo length/s | Success at 5 chunks | Success at demo end | Terminal probability |", "|---|---|---:|---|---|---:|"]
    for row in rows:
        text.append(f"| {row['episode']} | {row['case']} | {row['demo_duration_seconds']:.2f} | {row['success_at_5_chunks']} | {row['success_at_demo_end']} | {row['probability_at_demo_end']:.6f} |")
    text.extend(["", "Success is defined as in the original experiment: maximum classifier probability >= 0.5 over the last 4 native 30 FPS frames. The real demonstration endpoint and the generated endpoint padded to a whole chunk are stored separately; the table uses the generated endpoint matching the real demonstration time.",
                 "", "Pipeline validation on a fixed sample does not represent the overall success rate. The classifier must be combined with an object-state check on the main / side videos.", "", "[Experimental protocol](README.md)"])
    (root / "RESULTS.md").write_text("\n".join(text) + "\n")
    fig, axes = plt.subplots(len(reports), 1, figsize=(11, 2.8 * len(reports)), squeeze=False)
    colors = {"real": "black", "states": "#0072b2", "actions": "#009e73", "hold": "#d55e00"}
    for ax, (directory, report, complete) in zip(axes[:, 0], reports):
        for name in ["real", "states", "actions", "hold"]:
            path = directory / f"{name}_probabilities.npz"
            if path.exists():
                probabilities = np.load(path)["probabilities"]
                ax.plot(np.arange(len(probabilities)) / 30, probabilities, label=name, color=colors[name], linewidth=1.2, alpha=.8)
        ax.axhline(.5, color="grey", linestyle=":", linewidth=.8)
        ax.axvline(320/30, color="grey", linestyle="--", label="original horizon")
        ax.axvline(report["source_duration_seconds"], color="purple", linestyle=":", label="demo end")
        ax.set_ylim(-.03, 1.03)
        ax.set_title(f"Episode {report['episode']} ({'complete' if complete else 'partial'})")
        ax.set_ylabel("P(success)"); ax.set_xlabel("Time / seconds"); ax.grid(alpha=.15)
        ax.legend(loc="upper left", fontsize=8, ncol=6)
    fig.tight_layout()
    fig.savefig(root / "success_probability.png", dpi=170)
    fig.savefig(root / "success_probability.pdf")
    print(json.dumps({"complete_episodes": sum(c for _,_,c in reports), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
