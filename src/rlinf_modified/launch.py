"""Config-driven Slurm submission and allocation checks (never uses eval)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from rlinf_modified.config import TrainConfig, load_config
from rlinf_modified.preflight import repository_root


def _segments(
    config: TrainConfig,
    *,
    start_update: int = 0,
    until_update: int | None = None,
) -> list[tuple[int, int]]:
    width = config.runtime.max_updates if config.runtime.continuous_stress else config.runtime.segment_updates
    stop = config.runtime.max_updates if until_update is None else until_update
    if not 0 <= start_update < stop <= config.runtime.max_updates:
        raise ValueError(
            "submission range must satisfy "
            f"0 <= start_update < until_update <= {config.runtime.max_updates}"
        )
    if start_update % width or stop % width:
        raise ValueError(
            f"submission range {start_update}:{stop} must align to segment width {width}"
        )
    return [
        (start, min(start + width, stop))
        for start in range(start_update, stop, width)
    ]


def submit(
    config_path: str,
    *,
    dry_run: bool,
    start_update: int = 0,
    until_update: int | None = None,
) -> list[dict[str, Any]]:
    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    if config.runtime.backend != "ray_fsdp":
        raise ValueError("Slurm submission is only valid for runtime.backend=ray_fsdp")
    root = repository_root()
    script = root / "slurm" / "submit.sbatch"
    log_dir = Path(config.runtime.output_dir).expanduser().resolve() / "slurm"
    log_dir.mkdir(parents=True, exist_ok=True)
    dependency: str | None = None
    records: list[dict[str, Any]] = []
    for segment_start, segment_end in _segments(
        config,
        start_update=start_update,
        until_update=until_update,
    ):
        exports = [
            "ALL",
            f"RLINF_CONFIG={config_file}",
            f"RLINF_SEGMENT_START={segment_start}",
            f"RLINF_SEGMENT_END={segment_end}",
        ]
        command = [
            "sbatch",
            "--parsable",
            f"--account={config.slurm.account}",
            f"--partition={config.slurm.partition}",
            f"--nodes={config.slurm.nodes}",
            f"--ntasks-per-node={config.slurm.ntasks_per_node}",
            f"--cpus-per-task={config.slurm.cpus_per_task}",
            f"--gpus-per-node={config.slurm.gpus_per_node}",
            f"--mem={config.slurm.memory}",
            f"--time={config.slurm.time_limit}",
            f"--signal=B:USR1@{config.slurm.signal_seconds}",
            f"--job-name={config.experiment_name}-{segment_start}-{segment_end}",
            f"--output={log_dir}/%x-%j.out",
            f"--export={','.join(exports)}",
        ]
        if dependency is not None:
            command.append(f"--dependency=afterok:{dependency}")
        command.extend([str(script), "--config", str(config_file)])
        record: dict[str, Any] = {
            "segment_start": segment_start,
            "segment_end": segment_end,
            "argv": command,
        }
        if dry_run:
            dependency = f"DRYRUN_{segment_end}"
            record["job_id"] = dependency
        else:
            completed = subprocess.run(command, check=True, text=True, capture_output=True)
            dependency = completed.stdout.strip().split(";", maxsplit=1)[0]
            if not dependency:
                raise RuntimeError("sbatch returned an empty job id")
            record["job_id"] = dependency
        records.append(record)
    print(json.dumps(records, indent=2, sort_keys=True))
    return records


def validate_allocation(config: TrainConfig, *, check_gpus: bool) -> None:
    allocated_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", config.slurm.nodes))
    if allocated_nodes != config.slurm.nodes:
        raise RuntimeError(f"Slurm nodes {allocated_nodes} != config nodes {config.slurm.nodes}")
    if int(os.environ.get("SLURM_NTASKS_PER_NODE", "1").split("(", maxsplit=1)[0]) != 1:
        raise RuntimeError("exactly one Slurm task per node is required")
    if check_gpus:
        import torch

        visible = torch.cuda.device_count()
        if visible != config.slurm.gpus_per_node:
            raise RuntimeError(
                f"visible CUDA devices {visible} != config gpus_per_node {config.slurm.gpus_per_node}"
            )


def wait_ray(*, nodes: int, timeout: int) -> None:
    import ray

    ray.init(address="auto", ignore_reinit_error=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [node for node in ray.nodes() if node.get("Alive")]
        if len(alive) == nodes:
            return
        time.sleep(2)
    raise TimeoutError(f"Ray cluster did not reach {nodes} alive nodes within {timeout}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("submit", "validate-allocation", "value"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", required=True)
        if name == "submit":
            child.add_argument("--dry-run", action="store_true")
            child.add_argument("--start-update", type=int, default=0)
            child.add_argument("--until-update", type=int)
        elif name == "validate-allocation":
            child.add_argument("--check-gpus", action="store_true")
        else:
            child.add_argument("--field", required=True)
    wait = subparsers.add_parser("wait-ray")
    wait.add_argument("--nodes", type=int, required=True)
    wait.add_argument("--timeout", type=int, required=True)
    return parser


def _field(config: TrainConfig, dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not hasattr(value, part):
            raise ValueError(f"unknown config field: {dotted}")
        value = getattr(value, part)
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "wait-ray":
        wait_ray(nodes=args.nodes, timeout=args.timeout)
        return 0
    config = load_config(args.config)
    if args.command == "submit":
        submit(
            args.config,
            dry_run=args.dry_run,
            start_update=args.start_update,
            until_update=args.until_update,
        )
    elif args.command == "validate-allocation":
        validate_allocation(config, check_gpus=args.check_gpus)
    elif args.command == "value":
        value = _field(config, args.field)
        print(str(value).lower() if isinstance(value, bool) else value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
