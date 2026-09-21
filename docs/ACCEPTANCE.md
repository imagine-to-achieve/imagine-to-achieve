# Acceptance gates

Run static and unit gates with the pinned environment:

```bash
python -m compileall -q src tests third_party
pytest -q
bash -n train.sh slurm/submit.sbatch scripts/*.sh
shellcheck train.sh slurm/submit.sbatch scripts/*.sh
python scripts/verify_no_weights.py
```

`pytest` covers strict config parsing, batch equations, T+1 boundaries, CameraBundle, FPO/GRPO/PPO,
reward composition, arbitrary Duck subsets, sparse evaluation, target normalization, atomic
checkpoint and synthetic save/resume. Real checkpoint gates require an aarch64 GPU allocation and
are executed with `./train.sh preflight` followed by the two 2x4 profiles.

The close 2x4 and Duck 2x4 profiles submit one update per allocation, then resume update 2. Duck has
four complete color groups and basic eval. The 15-update close stress run must show no OOM/cgroup
kill and no monotonic cross-update RSS/IPC/GPU curve; peak GPU reserved must stay below 90%.

These gates validate the two-update close and Duck smoke profiles. They do not by themselves cover
the separate 15-update stress test, SIGTERM/fault injection, a real 32-rank canary, or a full
production 50-update run; those require a dedicated allocation and should be run explicitly before
relying on them.
