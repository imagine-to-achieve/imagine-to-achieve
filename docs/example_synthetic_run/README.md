# Example output: synthetic smoke run

These files are the unmodified output of the asset-free deterministic smoke path:

```bash
python -m rlinf_modified.train --config configs/synthetic.yaml
```

They come from the synthetic single-process backend (`configs/synthetic.yaml`), which trains a
randomly initialized policy against a synthetic world model for two updates. No real checkpoints,
datasets, or robot data are involved, and the reported metrics carry no scientific meaning; this
directory only shows the run artifacts and their schema so the pipeline can be sanity-checked
without a GPU allocation or real assets.

- `resolved_config.yaml` — the fully resolved config the trainer ran with.
- `status.json` / `summary.json` — final run state.
- `metrics.jsonl` — per-update optimization metrics (KL, clip fraction, grad norm, ...).
- `telemetry.jsonl` — per-phase RSS/disk telemetry.
- `heartbeats/rank_00000.json` — last recorded liveness heartbeat.
- `checkpoints/update_*/manifest.json` — atomic checkpoint manifests (the corresponding `state.pt`
  weight files are not included; this repository does not ship any weights).

Real training profiles (`configs/close.yaml`, `configs/duck*.yaml`, ...) additionally produce
`trajectory_records/`, CSV summaries, `REPORT.md`, and success/reward plots under
`<output_dir>/analysis/` once real Cosmos/Ctrl-World assets are configured; see
`analysis/README.md`.
