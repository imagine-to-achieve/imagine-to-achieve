from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from rlinf_modified.config import ConfigError, load_config
from rlinf_modified.preflight import (
    BASE_MODEL_TARGET,
    _parse_lustre_project_quota,
    normalize_cosmos_checkpoint_config,
    validate_vlm_processor,
)


ROOT = Path(__file__).resolve().parents[1]


def test_lustre_project_quota_parser_uses_hard_headroom_past_soft_limit():
    parsed = _parse_lustre_project_quota(
        "/nobackup/storage/disk\n"
        "6219328764 6291456000 15728640000 - "
        "3634766* 300000 21000000 6w6h21m29s\n"
    )
    assert parsed["used_kib"] == 6_219_328_764
    assert parsed["soft_kib"] == 6_291_456_000
    assert parsed["hard_kib"] == 15_728_640_000
    assert parsed["free_bytes"] == (
        15_728_640_000 - 6_219_328_764
    ) * 1024
    assert parsed["free_inodes"] == 21_000_000 - 3_634_766


def test_validation_only_checkpoint_target_is_normalized(tmp_path):
    source = tmp_path / "config.yaml"
    source.write_text(
        yaml.safe_dump(
            {"model": {"_target_": "cosmos_framework.callbacks.close_desktop_validation.CloseDesktopValidationOmniMoTModel"}}
        )
    )
    config = load_config(ROOT / "configs" / "synthetic.yaml")
    config = dataclasses.replace(
        config,
        cosmos=dataclasses.replace(config.cosmos, checkpoint_config_path=str(source)),
    )
    output = normalize_cosmos_checkpoint_config(config, tmp_path / "run")
    assert yaml.safe_load(output.read_text())["model"]["_target_"] == BASE_MODEL_TARGET


def test_vlm_processor_requires_complete_offline_tokenizer(tmp_path):
    for name in ("preprocessor_config.json", "tokenizer_config.json", "vocab.json"):
        (tmp_path / name).write_text("{}")
    with pytest.raises(ConfigError, match="merges.txt"):
        validate_vlm_processor(tmp_path)
    (tmp_path / "merges.txt").write_text("")
    validate_vlm_processor(tmp_path)


def test_atomic_checkpoint_and_new_trainer_resume(tmp_path):
    torch = pytest.importorskip("torch")
    from rlinf_modified.engine.synthetic import SyntheticTrainer

    base = load_config(ROOT / "configs" / "synthetic.yaml")
    runtime = dataclasses.replace(
        base.runtime,
        output_dir=str(tmp_path / "run"),
        max_updates=2,
        segment_updates=1,
        continuous_stress=False,
    )
    config = dataclasses.replace(base, runtime=runtime)
    assert SyntheticTrainer(config).run() == {"state": "completed", "update": 1}
    manifest = json.loads(
        (tmp_path / "run/checkpoints/update_000001/manifest.json").read_text()
    )
    assert manifest["state"] == "completed"
    # This second run also covers CUDA-to-CPU RNG-state restore.
    assert SyntheticTrainer(config).run() == {"state": "completed", "update": 2}
