"""Crash-safe atomic checkpoints and run state reconciliation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from rlinf_modified.contracts import CheckpointState


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


@dataclass(frozen=True)
class CheckpointManifest:
    schema_version: int
    state: str
    update: int
    state_file: str
    state_sha256: str
    semantics_version: str


class AtomicCheckpointer:
    """Write staging directories that become visible only after fsync+rename."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.root = self.run_dir / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)

    def write_status(self, state: CheckpointState, **details: Any) -> None:
        _atomic_json(
            self.run_dir / "status.json",
            {"schema_version": 1, "state": state.value, **details},
        )

    def save(
        self,
        *,
        update: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        extra: dict[str, Any],
    ) -> Path:
        target = self.root / f"update_{update:06d}"
        if target.exists():
            manifest = self._read_manifest(target)
            if manifest.update == update and manifest.state == CheckpointState.COMPLETED.value:
                return target
            raise FileExistsError(f"non-completed checkpoint already exists: {target}")
        staging = self.root / f".staging-update-{update:06d}-{uuid.uuid4().hex}"
        staging.mkdir(parents=False)
        state_path = staging / "state.pt"
        payload = {
            "update": int(update),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "extra": extra,
        }
        torch.save(payload, state_path)
        with state_path.open("rb") as stream:
            os.fsync(stream.fileno())
        state_hash = _sha256(state_path)
        manifest = CheckpointManifest(
            schema_version=1,
            state=CheckpointState.STAGING.value,
            update=update,
            state_file=state_path.name,
            state_sha256=state_hash,
            semantics_version="fpo_action_head_chunk_v1",
        )
        _atomic_json(staging / "manifest.json", asdict(manifest))
        _fsync_directory(staging)
        # Rename only after model, optimizer, manifest and directory are durable.
        staging.replace(target)
        _fsync_directory(self.root)
        completed = dataclass_replace(manifest, state=CheckpointState.COMPLETED.value)
        _atomic_json(target / "manifest.json", asdict(completed))
        return target

    def latest(self) -> Path | None:
        candidates = []
        for candidate in sorted(self.root.glob("update_*")):
            try:
                manifest = self._read_manifest(candidate)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if manifest.state == CheckpointState.COMPLETED.value:
                candidates.append((manifest.update, candidate))
        return max(candidates, default=(None, None))[1]

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[int, dict[str, Any]]:
        latest = self.latest()
        if latest is None:
            return 0, {}
        manifest = self._read_manifest(latest)
        state_path = latest / manifest.state_file
        if _sha256(state_path) != manifest.state_sha256:
            raise ValueError(f"checkpoint checksum mismatch: {state_path}")
        # CPU RNG state must never be remapped onto CUDA during resume.
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        return int(payload["update"]), dict(payload.get("extra", {}))

    @staticmethod
    def _read_manifest(directory: Path) -> CheckpointManifest:
        payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        return CheckpointManifest(**payload)


def dataclass_replace(instance: CheckpointManifest, **changes: Any) -> CheckpointManifest:
    payload = asdict(instance)
    payload.update(changes)
    return CheckpointManifest(**payload)


def prune_staging_directories(run_dir: str | Path) -> list[Path]:
    """Remove only uncommitted staging directories from this run."""

    removed = []
    root = Path(run_dir) / "checkpoints"
    for path in root.glob(".staging-update-*"):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path)
    return removed
