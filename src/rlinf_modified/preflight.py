"""Fail-closed real-asset and runtime compatibility checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from rlinf_modified.config import ConfigError, TrainConfig


VALIDATION_TARGET = (
    "cosmos_framework.callbacks.close_desktop_validation."
    "CloseDesktopValidationOmniMoTModel"
)
BASE_MODEL_TARGET = "cosmos_framework.model.vfm.omni_mot_model.OmniMoTModel"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_target(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _replace_target(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_target(item) for item in value]
    return BASE_MODEL_TARGET if value == VALIDATION_TARGET else value


def normalize_cosmos_checkpoint_config(config: TrainConfig, run_dir: Path) -> Path:
    """Write a run-owned config with the validation-only model class removed."""

    if config.cosmos.checkpoint_config_path is None:
        raise ConfigError("Cosmos checkpoint config path is missing")
    source = Path(config.cosmos.checkpoint_config_path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    normalized = _replace_target(payload)
    model = normalized.get("model", {}) if isinstance(normalized, dict) else {}
    if model.get("_target_") != BASE_MODEL_TARGET:
        raise ConfigError(
            "normalized checkpoint model target is not the parameter-equivalent OmniMoTModel"
        )
    output = run_dir / "inputs" / "cosmos" / "checkpoint_config.normalized.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    temporary.replace(output)
    return output

def validate_checkpoint_parallelism(
    config: TrainConfig, normalized_config: Path
) -> dict[str, int]:
    """Require the configured HSDP shard degree to match the checkpoint recipe."""
    payload = yaml.safe_load(normalized_config.read_text(encoding="utf-8"))
    try:
        actual = payload["model"]["config"]["parallelism"][
            "data_parallel_shard_degree"
        ]
    except (KeyError, TypeError) as exc:
        raise ConfigError(
            "Cosmos checkpoint config has no model.config.parallelism."
            "data_parallel_shard_degree"
        ) from exc
    if isinstance(actual, bool) or not isinstance(actual, int):
        raise ConfigError("Cosmos checkpoint data_parallel_shard_degree must be an integer")
    expected = config.cosmos.data_parallel_shard_degree
    # Sparse Duck eval must align with the model's actual HSDP
    # shard degree; node-local GPU count is not an interchangeable assumption.
    if actual != expected:
        raise ConfigError(
            f"Cosmos checkpoint shard degree {actual} != configured {expected}"
        )
    return {"data_parallel_shard_degree": actual}


def validate_action_stats(config: TrainConfig) -> dict[str, Any]:
    if config.cosmos.action_stats_path is None:
        raise ConfigError("Cosmos action stats path is missing")
    path = Path(config.cosmos.action_stats_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    required_arrays = ("min", "max", "mean", "std", "q01", "q99")
    for key in required_arrays:
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != config.cosmos.action_dim:
            raise ConfigError(f"action stats {key} must contain {config.cosmos.action_dim} values")
    expected = {
        "schema_version": 1,
        "action_dim": config.cosmos.action_dim,
        "chunk_length": config.cosmos.action_chunk,
        "normalization": "quantile",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ConfigError(f"action stats metadata.{key}={metadata.get(key)!r}, expected {value!r}")
    if float(metadata.get("fps", -1.0)) != config.cosmos.action_fps:
        raise ConfigError("action stats FPS does not match Cosmos action_fps")
    return {"path": str(path), "sha256": _sha256(path), "metadata": metadata}


def _asset_identity(path: Path) -> dict[str, Any]:
    if path.is_file():
        result = {"path": str(path), "kind": "file", "size": path.stat().st_size}
        if path.stat().st_size <= 64 * 1024 * 1024:
            result["sha256"] = _sha256(path)
        return result
    entries = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = str(child.relative_to(path))
        entries.append({"path": relative, "size": child.stat().st_size})
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()
    return {"path": str(path), "kind": "directory", "entries": len(entries), "manifest_sha256": digest}


def validate_expected_asset_hashes(
    config: TrainConfig, identities: dict[str, Any]
) -> dict[str, str]:
    verified: dict[str, str] = {}
    for label, expected in config.assets.expected_sha256.items():
        identity_label = label
        if label.startswith("reward_model_"):
            identity_label = f"reward_model[{label.removeprefix('reward_model_')}]"
        identity = identities.get(identity_label)
        if identity is None:
            raise ConfigError(f"expected_sha256 has no active asset named {label!r}")
        actual = identity.get("sha256") or identity.get("manifest_sha256")
        if actual is None:
            path = Path(str(identity["path"]))
            if not path.is_file():
                raise ConfigError(f"cannot compute a file SHA256 for {label}: {path}")
            actual = _sha256(path)
            identity["sha256"] = actual
        if actual != expected:
            raise ConfigError(f"asset checksum mismatch for {label}: {actual} != {expected}")
        verified[label] = actual
    return verified


def validate_vlm_processor(path: Path) -> None:
    required = (
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ConfigError(
            f"Cosmos VLM processor is missing required files at {path}: "
            + ", ".join(missing)
        )



def _parse_lustre_project_quota(output: str) -> dict[str, int]:
    """Parse the final eight fields from ``lfs quota -q -p`` output."""

    fields = output.split()
    if len(fields) < 9:
        raise ValueError("incomplete Lustre project quota output")
    (
        used_kib,
        soft_kib,
        hard_kib,
        _block_grace,
        used_inodes,
        soft_inodes,
        hard_inodes,
        _inode_grace,
    ) = fields[-8:]

    def integer(value: str, label: str) -> int:
        normalized = value.rstrip("*")
        if not normalized.isdigit():
            raise ValueError(f"invalid Lustre {label}: {value!r}")
        return int(normalized)

    values = {
        "used_kib": integer(used_kib, "used blocks"),
        "soft_kib": integer(soft_kib, "soft block quota"),
        "hard_kib": integer(hard_kib, "hard block quota"),
        "used_inodes": integer(used_inodes, "used inodes"),
        "soft_inodes": integer(soft_inodes, "soft inode quota"),
        "hard_inodes": integer(hard_inodes, "hard inode quota"),
    }
    values["free_bytes"] = (
        max(0, values["hard_kib"] - values["used_kib"]) * 1024
    )
    values["free_inodes"] = max(
        0, values["hard_inodes"] - values["used_inodes"]
    )
    return values


def _lustre_project_hard_headroom(path: Path) -> dict[str, int] | None:
    """Return hard-quota headroom when ``df`` exposes a Lustre soft quota."""

    lfs = shutil.which("lfs")
    if lfs is None:
        return None
    try:
        project = subprocess.run(
            [lfs, "project", "-d", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        project_id = project.stdout.split(maxsplit=1)[0]
        if (
            project.returncode != 0
            or not project_id.isdigit()
            or int(project_id) <= 0
        ):
            return None
        quota = subprocess.run(
            [lfs, "quota", "-q", "-p", project_id, str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if quota.returncode != 0:
            return None
        parsed = _parse_lustre_project_quota(quota.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None
    if parsed["hard_kib"] <= 0 or parsed["hard_inodes"] <= 0:
        return None
    parsed["project_id"] = int(project_id)
    return parsed


def run_preflight(config: TrainConfig, *, run_dir: Path) -> dict[str, Any]:
    """Validate paths, schemas, disk/inodes, and asset hashes."""

    root = repository_root()
    if not config.runtime.require_real_assets:
        # This profile explicitly requests a raw diagnostic run. Keep only the
        # config normalization needed by the native Cosmos loader; skip all
        # real-asset, SHA256, and quota preflight gates.
        normalized = normalize_cosmos_checkpoint_config(config, run_dir)
        report = {
            "schema_version": 1,
            "normalized_checkpoint_config": str(normalized),
            "integrity_checks_skipped": True,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "preflight.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + chr(10),
            encoding="utf-8",
        )
        return report

    required_vendor = (
        root / "third_party" / "rlinf_runtime" / "rlinf",
        root / "third_party" / "cosmos_framework",
        root / "third_party" / "ctrl_world" / "models",
    )
    missing_vendor = [str(path) for path in required_vendor if not path.is_dir()]
    if missing_vendor:
        raise ConfigError(f"vendored runtime sources are missing: {missing_vendor}")

    configured = {
        "cosmos_checkpoint": config.cosmos.checkpoint_path,
        "cosmos_checkpoint_config": config.cosmos.checkpoint_config_path,
        "cosmos_action_stats": config.cosmos.action_stats_path,
        "cosmos_vlm_processor": config.cosmos.vlm_processor_path,
        "dataset": config.assets.dataset_path,
        "ctrl_world_checkpoint": config.assets.ctrl_world_checkpoint,
        "ctrl_world_stats": config.assets.ctrl_world_stats,
        "svd_model": config.assets.svd_model_path,
        "clip_model": config.assets.clip_model_path,
    }
    missing = []
    identities: dict[str, Any] = {}
    for label, raw_path in configured.items():
        if raw_path is None:
            missing.append(f"{label}=null")
            continue
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            missing.append(f"{label}={path}")
        else:
            identities[label] = _asset_identity(path)
    for variant, raw_path in config.assets.reward_models.items():
        if variant not in config.task.active_variants:
            continue
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            missing.append(f"reward_model[{variant}]={path}")
        else:
            identities[f"reward_model[{variant}]"] = _asset_identity(path)
    if missing:
        raise ConfigError("required real assets are missing:\n- " + "\n- ".join(missing))
    # An existing HF snapshot is insufficient if tokenizer files are absent.
    validate_vlm_processor(Path(config.cosmos.vlm_processor_path or ""))
    verified_hashes = validate_expected_asset_hashes(config, identities)

    checkpoint = Path(config.cosmos.checkpoint_path or "")
    if not (checkpoint / ".metadata").is_file():
        raise ConfigError(f"Cosmos DCP checkpoint has no .metadata: {checkpoint}")
    if not list(checkpoint.glob("*.distcp")):
        raise ConfigError(f"Cosmos DCP checkpoint has no .distcp shards: {checkpoint}")

    run_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(run_dir)
    stat = os.statvfs(run_dir)
    statvfs_free_gib = usage.free / 1024**3
    statvfs_free_inodes = int(stat.f_favail)
    lustre_quota = _lustre_project_hard_headroom(run_dir)
    if lustre_quota is None:
        capacity_source = "statvfs"
        free_gib = statvfs_free_gib
        free_inodes = statvfs_free_inodes
    else:
        capacity_source = "lustre_project_hard_quota"
        free_gib = lustre_quota["free_bytes"] / 1024**3
        free_inodes = lustre_quota["free_inodes"]
    if free_gib < config.runtime.minimum_free_disk_gib:
        raise ConfigError(
            f"effective free disk {free_gib:.1f} GiB ({capacity_source}) "
            f"is below required {config.runtime.minimum_free_disk_gib} GiB"
        )
    if free_inodes < config.runtime.minimum_free_inodes:
        raise ConfigError(
            f"effective free inodes {free_inodes} ({capacity_source}) below "
            f"required {config.runtime.minimum_free_inodes}"
        )
    normalized = normalize_cosmos_checkpoint_config(config, run_dir)
    checkpoint_parallelism = validate_checkpoint_parallelism(config, normalized)
    action_stats = validate_action_stats(config)
    report = {
        "schema_version": 1,
        "normalized_checkpoint_config": str(normalized),
        "checkpoint_parallelism": checkpoint_parallelism,
        "action_stats": action_stats,
        "assets": identities,
        "verified_asset_sha256": verified_hashes,
        "free_disk_gib": free_gib,
        "free_inodes": free_inodes,
        "storage_capacity_source": capacity_source,
        "statvfs_free_disk_gib": statvfs_free_gib,
        "statvfs_free_inodes": statvfs_free_inodes,
        "lustre_project_quota": lustre_quota,
    }
    report_path = run_dir / "preflight.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
