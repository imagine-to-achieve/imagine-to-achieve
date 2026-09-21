# 🎯 Achieve What You Imagined: Learning to Align Actions with Visual Plans

**Anonymous submission — ICRA 2027**

[📄 Paper](https://imagine-to-achieve.github.io/paper/anonymous-paper.pdf) &nbsp;|&nbsp;
[🌐 Project page](https://imagine-to-achieve.github.io/) &nbsp;|&nbsp;
[🎬 Video overview](https://imagine-to-achieve.github.io/#video) &nbsp;|&nbsp;
[🤖 Real-robot demos](https://imagine-to-achieve.github.io/#demos)

This repository is the standalone training runtime for the paper. It implements critic-free
**Flow Policy Optimization (FPO)** on the action head of a world-action model, using a **frozen**
action-conditioned world model to score the consequences of the generated actions.

**TL;DR:** A world-action model can already *imagine* task completion before its actions can
achieve it. We measure that imagination–consequence gap with a frozen world model, turn it into a
dense reward, and post-train only the action head. No online robot interaction, no task-specific
reward model. Mean real-robot success over four UR5 tasks rises from **43.4% to 75.1%**.

<p align="center">
  <img src="https://imagine-to-achieve.github.io/assets/pipeline.webp" width="95%">
</p>

> The video generator and the action-conditioned world model stay frozen throughout; only the
> action head is updated. Trajectory consistency between the two predicted futures, together with
> alignment of the final predicted frame to a goal frame, forms the reward.

---

## Contents

- Critic-free action-head FPO with GRPO discounted suffix advantages and PPO clipping.
- Closed-loop cross-model rollout: the world-action model proposes a video and an action chunk,
  and a frozen Ctrl-World predicts what those actions actually produce and supplies the next
  observation.
- Two reward components — trajectory consistency and terminal-goal alignment — selectable
  independently, so the paper's reward ablation can be reproduced.
- An asset-free deterministic smoke path that exercises the full
  rollout → reward → replay → backward → save pipeline.
- Slurm submission with segmented `afterok` chains and DCP checkpoint resume.
- Analysis that regenerates CSV summaries, `REPORT.md`, and reward/success/correlation plots.

**Not included:** SAC, SFT, critic training, the legacy offline audit suite, and video promotion.
No weights, datasets or historical logs are stored in this repository. At runtime the package
imports only its own `src/` and the audited sources under `third_party/`; it never adds the
original RLinf, Cosmos or Ctrl-World trees to `PYTHONPATH`.

---

## Installation 🛠️

Requirements: an **aarch64 GPU compute node**, CUDA 13.0, Python ≥3.11,<3.12.

```bash
export BUILDENV_MODULE=<your cluster's CUDA build-environment module>
./scripts/create_env.sh
```

This creates the pinned environment in `.venv_aarch64` from `requirements-aarch64-cu130.txt`,
then runs `scripts/validate_env.sh`.

Two site-specific placeholders must be filled in before submitting to Slurm:

| Placeholder | Where | What to set |
| --- | --- | --- |
| `<SLURM_ACCOUNT>` | `slurm/*.sbatch`, `stage15_plugin/*.sbatch` | Your Slurm accounting project |
| `BUILDENV_MODULE` | environment variable read by every sbatch | The `module load` target for your CUDA toolchain |

---

## Checkpoints and assets 📷

No weights ship with this repository. Point the `assets:` block of a config at your own copies:

| Asset | Role | Source |
| --- | --- | --- |
| Cosmos world-action model | Video generator + action head; **only the action head is trained** | [NVIDIA Cosmos](https://github.com/nvidia-cosmos) |
| Ctrl-World | Frozen action-conditioned world model that predicts action consequences | [Robert-gyj/Ctrl-World](https://github.com/Robert-gyj/Ctrl-World) |
| Task success classifier | Model-space monitoring only; **not** a training reward by default | Trained per task |
| UR5 demonstrations | 40 / 200 / 150 / 100 for laptop / duck / Push-T / cup | Collected for this paper |

Paths and expected checksums live under `assets:` in each config. `preflight` verifies them
before any GPU is allocated.

---

## Quick start 📊

### (1) Asset-free smoke run

The deterministic synthetic profile runs the complete pipeline end to end, with no real assets:

```bash
python -m rlinf_modified.train --config configs/synthetic.yaml
# or, inside the pinned environment:
./train.sh run --config configs/synthetic.yaml
```

A reference output of this exact path is checked in under
[`docs/example_synthetic_run/`](docs/example_synthetic_run/) — metrics, telemetry, status,
resolved config and two checkpoint manifests.

### (2) Validate real assets without allocating workers

```bash
./train.sh preflight --config configs/close.yaml
./train.sh preflight --config configs/duck.yaml
```

Preflight checks asset paths and checksums, storage quota and checkpoint layout, and exits before
Ray workers or CUDA contexts are created.

### (3) Submit a production run

```bash
./train.sh submit --config configs/close.yaml            # production profile
./train.sh submit --config configs/duck_smoke_2n4g.yaml  # 2x4 acceptance profile
./train.sh submit-dry-run --config configs/close.yaml    # print the plan only
```

Production profiles submit **ten `afterok` jobs of five updates each**. Every allocation starts a
fresh Ray and CUDA process, restores exactly the preceding `global_step_N` DCP checkpoint, and
only the final segment may publish global success. Use `--start-update` / `--until-update` to
resume a partial chain.

`configs/close_stress_2n4g.yaml` deliberately runs 15 updates in a single process as a
long-running memory-stability stress profile.

---

## Configuration profiles

| Profile | Config | Scale |
| --- | --- | --- |
| Synthetic smoke | `synthetic.yaml` | 1 GPU, no assets |
| Close the laptop | `close.yaml` | 8 nodes × 4 GPUs, 512 trajectories/update |
| Place the duck | `duck.yaml` | 8 nodes × 4 GPUs |
| Push-T | `push_t_*.yaml` | 8 nodes × 4 GPUs |
| Nest four cups | `nest_four_cups_*.yaml` | 32 nodes × 4 GPUs (longest horizon) |
| Acceptance / stress | `*_smoke_2n4g.yaml`, `close_stress_2n4g.yaml` | 2 nodes × 4 GPUs |

Each trajectory contains 5 action chunks; discounted suffix returns are normalized within each
group at every chunk position with γ = 0.95.

### Reward ablation

Set `reward.source` to reproduce the ablation in the paper:

| `reward.source` | Paper setting |
| --- | --- |
| `ctrl_world_aligned_mse_plus_terminal` | Combined reward (default) |
| `trajectory_mse` | Trajectory reward only |
| `terminal_goal_mse` | Terminal-goal reward only |
| `success_binary` | Classifier success reward (comparison only) |

---

## Analysis and plots

Real profiles record each update atomically and regenerate analysis under
`<output_dir>/analysis/` whenever a five-update segment completes. To run it by hand:

```bash
python -m rlinf_modified.analysis --help
```

`scripts/` holds the figure generators used for the paper: per-step reward and success curves,
reward distributions for successful versus failed rollouts, paired evaluation with bootstrap
intervals, and the reward / predicted-success correlation plots.

See [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) for the acceptance gates each profile must pass.

---

## Results

Real-robot success rate (%) on four UR5 manipulation tasks:

| Method | Laptop | Duck | Push-T | Cup | **Avg.** |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ours (SFT) | 22.0 | 95.0 | 23.3 | 33.3 | 43.4 |
| Fast-WAM | 16.0 | 0.0 | 0.0 | 0.0 | 4.0 |
| VLA-JEPA | 15.0 | 20.0 | 0.0 | 0.0 | 8.8 |
| π<sub>0.5</sub> | 56.7 | 85.0 | 33.3 | 70.6 | 61.4 |
| **Ours (RL)** | **68.0** | **100.0** | **48.9** | **83.3** | **75.1** |

Fast-WAM and VLA-JEPA were developed with substantially larger adaptation budgets than the
40–200 demonstrations per task available here.

> **Model-space success is not real-robot success.** The success curves produced during training
> come from a classifier applied to the frozen world model's predicted terminal states. They are
> used only to compare optimization signals; every number in the table above comes from physical
> UR5 trials. See the [Analysis section](https://imagine-to-achieve.github.io/#analysis) of the
> project page.

---

## Repository layout

```
src/rlinf_modified/   training runtime: FPO, advantages, rewards, rollout engines, checkpointing
configs/              run profiles (synthetic, per-task production, acceptance, ablations)
slurm/                sbatch entry points and regression jobs
scripts/              environment setup, validation, and the paper's figure generators
analysis/             analysis outputs
tests/                unit and regression tests
docs/                 acceptance gates and a reference synthetic run
third_party/          vendored Cosmos, Ctrl-World and RLinf sources, with their licenses
```

Run the tests with:

```bash
python -m pytest tests/ -q
```

---

## Acknowledgement

This work builds on several open-source projects:

- [**Cosmos**](https://github.com/nvidia-cosmos) — the world-action model whose action head we
  post-train (`third_party/cosmos_framework`, OpenMDW 1.1).
- [**Ctrl-World**](https://github.com/Robert-gyj/Ctrl-World) — the action-conditioned world model
  we use frozen as an external consequence predictor (`third_party/ctrl_world`, MIT).
- [**RLinf**](https://github.com/RLinf/RLinf) — the RL infrastructure this runtime is derived from
  (`third_party/rlinf_runtime`, Apache-2.0).
- **Flow Policy Optimization** — the surrogate we adapt to a high-order multi-step sampler.

We thank the authors for releasing their work. Vendored third-party licenses are reproduced under
`third_party/licenses/`.

---

## Citation

This is an anonymous submission under review. A citation entry will be added once the paper is
de-anonymized.

```bibtex
@article{anonymous2027achieve,
  title   = {Achieve What You Imagined: Learning to Align Actions with Visual Plans},
  author  = {Anonymous},
  journal = {Under review at ICRA},
  year    = {2027}
}
```

## License

This repository's own code is released under Apache-2.0 (see [`LICENSE`](LICENSE)).
Vendored third-party code retains its original license.
