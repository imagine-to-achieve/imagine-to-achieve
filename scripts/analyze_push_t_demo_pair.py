#!/usr/bin/env python3
"""Compare saved original and FP32-fixed Push-T demo/hold diagnostics."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def scalar(decision, key):
    return decision[key][0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--fixed", type=Path, required=True)
    args = p.parse_args()
    rows, weights, payload = [], [], {}
    for ep in (0, 6, 53, 123):
        old_dir = args.original / f"episode_{ep:03d}"
        old = json.loads((old_dir / "partial_summary.json").read_text())
        entries = {"original": old["cases"]["states"]}
        for condition in ("states", "hold"):
            folder = args.fixed / condition / f"episode_{ep:03d}"
            summary_path = folder / "summary.json"
            if not summary_path.exists():
                print(f"Waiting for {summary_path}")
                return
            summary = json.loads(summary_path.read_text())
            entries[condition] = summary["cases"][condition]
        payload[str(ep)] = {"real": old, **entries}
        for name, result in entries.items():
            error = result["chunks"][-1]["endpoint_error"]
            rows.append({"episode": ep, "condition": name,
                         "duration_seconds": old["source_duration_seconds"],
                         "success_at_original_horizon": scalar(result["at_original_horizon"], "success"),
                         "p_at_original_horizon": scalar(result["at_original_horizon"], "reported_probability"),
                         "success_at_demo_end": scalar(result["at_demonstration_end"], "success"),
                         "p_at_demo_end": scalar(result["at_demonstration_end"], "reported_probability"),
                         "endpoint_position_error_mm": 1000*error["position_max_m"],
                         "endpoint_rotation_error_deg": np.degrees(error["rotation_max_rad"])})
        for setting in ("current", "wrist_010", "wrist_000"):
            expert = entries["states"]["offline_terminal_reweighting"][setting]
            hold = entries["hold"]["offline_terminal_reweighting"][setting]
            weights.append({"episode": ep, "setting": setting, "expert_terminal_mse": expert,
                            "hold_terminal_mse": hold, "hold_minus_expert_mse": hold-expert,
                            "expert_preferred": hold > expert})
    for name, values in [("comparison_table.csv", rows), ("terminal_weight_comparison.csv", weights)]:
        with (args.fixed / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    summary = {"episodes": [0,6,53,123], "rows": rows, "offline_terminal_weights": weights,
               "expert_preferred_count": {setting: sum(r["expert_preferred"] for r in weights if r["setting"] == setting) for setting in ("current", "wrist_010", "wrist_000")},
               "success_counts": {name: {horizon: sum(r[key] for r in rows if r["condition"] == name) for horizon,key in [("original_horizon", "success_at_original_horizon"), ("demo_end", "success_at_demo_end")]} for name in ("original", "states", "hold")},
               "limitations": ["Four fixed episodes; diagnostic, not a population success estimate.",
                               "Success is the original classifier rule; inspect object geometry separately.",
                               "Terminal reweighting uses diagnostic resized uint8 references (antialias=True), unlike production goal resize (antialias=False, float). It is an approximate offline pixel comparison, not exact production rewards or a training result.",
                               "Terminal reweighting compares padded full demonstration endpoints, not the original 5-chunk training horizon or Cosmos-CtrlWorld trajectory MSE."]}
    (args.fixed / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 11), sharey=True)
    for ax, ep in zip(axes, (0,6,53,123)):
        old_dir = args.original / f"episode_{ep:03d}"
        files = [("Real demo", old_dir/"real_probabilities.npz", "black"),
                 ("Original adapter", old_dir/"states_probabilities.npz", "#d55e00"),
                 ("FP32 adapter", args.fixed/"states"/f"episode_{ep:03d}"/"states_probabilities.npz", "#0072b2"),
                 ("Hold control", args.fixed/"hold"/f"episode_{ep:03d}"/"hold_probabilities.npz", "#009e73")]
        for label, path, color in files:
            probs = np.load(path)["probabilities"]
            ax.plot(np.arange(len(probs))/30, probs, label=label, color=color, linewidth=1.3, alpha=.85)
        ax.axvline(320/30, color="grey", linestyle="--", linewidth=1)
        ax.axvline(payload[str(ep)]["real"]["source_duration_seconds"], color="purple", linestyle=":", linewidth=1)
        ax.axhline(.5, color="grey", linestyle=":", linewidth=.8)
        ax.set(title=f"Episode {ep}", ylabel="P(success)", xlabel="Time / seconds", ylim=(-.02,1.03))
        ax.grid(alpha=.15); ax.legend(ncol=4, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(args.fixed/"success_comparison.png", dpi=170); fig.savefig(args.fixed/"success_comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for condition, color in [("original", "#d55e00"), ("states", "#0072b2")]:
        selected = [r for r in rows if r["condition"] == condition]
        axes[0].plot(range(4), [r["endpoint_position_error_mm"] for r in selected], "o-", label=condition, color=color)
    axes[0].set_yscale("log"); axes[0].set_xticks(range(4), [0,6,53,123]); axes[0].set(xlabel="Episode", ylabel="Endpoint position error / mm", title="SE(3) precision correction")
    axes[0].legend(); axes[0].grid(alpha=.2)
    for i, setting in enumerate(("current", "wrist_010", "wrist_000")):
        selected = [r for r in weights if r["setting"] == setting]
        axes[1].bar(np.arange(4)+(i-1)*.24, [r["hold_minus_expert_mse"] for r in selected], width=.23, label=setting)
    axes[1].axhline(0, color="black", linewidth=.8); axes[1].set_xticks(range(4), [0,6,53,123]); axes[1].set(xlabel="Episode", ylabel="Hold MSE - expert MSE", title="Approximate terminal reweighting\nPositive means expert is preferred")
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(args.fixed/"precision_and_terminal_weights.png", dpi=170); fig.savefig(args.fixed/"precision_and_terminal_weights.pdf")
    plt.close(fig)

    # All four full-horizon endpoints, real / old / fixed / hold; main+side.
    canvas = Image.new("RGB", (1280, 1712), "white")
    for slot, ep in enumerate((0,6,53,123)):
        old_dir = args.original / f"episode_{ep:03d}"
        panels = [("Real", old_dir/"states_comparison.jpg", 856),
                  ("Original", old_dir/"states_comparison.jpg", 1070),
                  ("FP32", args.fixed/"states"/f"episode_{ep:03d}"/"states_comparison.jpg", 1070),
                  ("Hold", args.fixed/"hold"/f"episode_{ep:03d}"/"hold_comparison.jpg", 1070)]
        for line, (label, path, y) in enumerate(panels):
            tile = Image.open(path).crop((0,y,640,y+214))
            draw = ImageDraw.Draw(tile); draw.rectangle((0,0,640,21), fill="white"); draw.text((5,5), f"Episode {ep} | {label} | main / side", fill="black")
            canvas.paste(tile, ((slot%2)*640,(slot//2)*856+line*214))
    canvas.save(args.fixed/"terminal_comparison_main_side.jpg", quality=94)
    print(json.dumps({k:v for k,v in summary.items() if k not in {"rows","offline_terminal_weights"}}, indent=2))


if __name__ == "__main__":
    main()
