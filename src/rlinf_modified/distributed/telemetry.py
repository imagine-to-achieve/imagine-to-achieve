"""Low-overhead phase telemetry for memory and distributed stalls."""

from __future__ import annotations

import json
import os
import resource
import shutil
import time
import uuid
from pathlib import Path
from typing import Any


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
        return None if text == "max" else int(text)
    except (OSError, ValueError):
        return None


def memory_snapshot(path: str | Path) -> dict[str, Any]:
    """Capture process, cgroup, disk and CUDA allocator memory."""

    output_path = Path(path)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    disk = shutil.disk_usage(output_path.parent if output_path.parent.exists() else Path.cwd())
    payload: dict[str, Any] = {
        "time_unix": time.time(),
        "pid": os.getpid(),
        "rss_max_kib": int(usage.ru_maxrss),
        "cgroup_memory_current": _read_int(Path("/sys/fs/cgroup/memory.current")),
        "cgroup_memory_max": _read_int(Path("/sys/fs/cgroup/memory.max")),
        "disk_free_bytes": disk.free,
    }
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            payload["cuda"] = {
                "device": device,
                "allocated": torch.cuda.memory_allocated(device),
                "reserved": torch.cuda.memory_reserved(device),
                "max_allocated": torch.cuda.max_memory_allocated(device),
                "max_reserved": torch.cuda.max_memory_reserved(device),
                "total": torch.cuda.get_device_properties(device).total_memory,
            }
    except ImportError:
        pass
    return payload


def write_heartbeat(
    directory: str | Path,
    *,
    rank: int,
    phase: str,
    update: int,
    shapes: dict[str, tuple[int, ...]] | None = None,
) -> Path:
    """Atomically publish one rank's phase and tensor shapes."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"rank_{rank:05d}.json"
    temporary = root / f".{target.name}.{uuid.uuid4().hex}.tmp"
    payload = {
        "schema_version": 1,
        "rank": rank,
        "phase": phase,
        "update": update,
        "time_unix": time.time(),
        "shapes": {key: list(value) for key, value in (shapes or {}).items()},
    }
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()

