# Training-run analysis

This directory contains the dependency-light analysis entry point for the
standalone training library. Raw logs, videos, model weights, and historical
analysis outputs are not copied into the repository.

The trainer atomically commits scalar-only trajectory records under
`<run>/trajectory_records/update_XXXX.jsonl`. The analyzer reads only committed
JSONL files and writes all derived artifacts under `<run>/analysis/`.

All real YAML profiles set `trajectory_records.auto_analyze: true` and
`require_plots: true`, so the same strict analysis runs automatically after
each successful Slurm segment. The commands below are safe manual refreshes.

Analyze the three current formal runs:

```bash
./analysis/analyze_runs.sh
```

Analyze one run or one configured output directory:

```bash
python -m rlinf_modified.analysis --run-dir /absolute/path/to/run --require-plots
python -m rlinf_modified.analysis --config configs/close_b128_mb32_u25.yaml
```

The command fails rather than publishing misleading results when a committed
update violates the configured trajectory count, group size, chunk count,
reward composition, normalized-advantage, success-threshold, uniqueness, or
finite-value contracts.

Generated artifacts include:

- `trajectories.csv`, `per_step_summary.csv`, and `ppo_per_step_summary.csv`;
- Duck-only `per_color_per_step_summary.csv` and `per_color_summary.csv`;
- `REPORT.md`, `summary.json`, and `run_overview.svg`;
- `success_reward_curves.png` and Duck per-color learning curves;
- binary-success and continuous-probability correlation CSV/PNG suites;
- the four-panel `success_similarity_relationship.png` reference plot.

CSV/JSON/Markdown/SVG generation requires only the package dependencies.
PNG generation uses the system `gnuplot`; `--require-plots` makes a missing or
failed plotting backend a hard error.
