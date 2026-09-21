from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from rlinf_modified import analysis


def _record(
    *,
    update: int,
    group_id: int,
    member_id: int,
    color: str | None,
) -> dict:
    success = member_id == 0
    probability = 0.9 if success else 0.1
    combined_reward = -1.0 if success else -2.0
    group_mean = -1.5
    group_std = 0.5
    normalized = (combined_reward - group_mean) / (group_std + 1e-6)
    record = {
        "update": update,
        "logical_round": 0,
        "physical_wave": 0,
        "group_slot": group_id,
        "group_id": group_id,
        "member_id": member_id,
        "rollout_uid": update * 1000 + group_id * 10 + member_id,
        "actor_rank": group_id,
        "source_env_rank": group_id,
        "local_env_id": member_id,
        "node": "test-node",
        "gpu": group_id,
        "reset_episode": group_id,
        "success": success,
        "success_probability_max": probability,
        "success_threshold": 0.5,
        "success_last4_probabilities": [probability] * 4,
        "trajectory_reward": combined_reward * 0.4,
        "trajectory_mse": -combined_reward * 0.04,
        "trajectory_similarity": combined_reward * 0.04,
        "lastframe_reward": combined_reward * 0.6,
        "lastframe_mse": -combined_reward * 0.06,
        "lastframe_similarity": combined_reward * 0.06,
        "combined_reward": combined_reward,
        "chunk_rewards": [combined_reward / 2] * 2,
        "chunk_advantages": [normalized] * 2,
        "old_chunk_logprobs": [-0.1, -0.1],
        "new_chunk_logprobs": [-0.1, -0.1],
        "chunk_raw_log_ratios": [0.0, 0.0],
        "chunk_log_ratios": [0.0, 0.0],
        "chunk_ratios": [1.0, 1.0],
        "chunk_clip_indicators": [False, False],
        "chunk_clip_fractions": [0.0, 0.0],
        "group_mean": group_mean,
        "group_std": group_std,
        "normalized_advantage": normalized,
        "old_logprob": -0.2,
        "new_logprob": -0.2,
        "raw_log_ratio": 0.0,
        "log_ratio": 0.0,
        "ratio": 1.0,
        "action_ratio_mean": 1.0,
        "clip_indicator": False,
        "clip_fraction": 0.0,
        "retry_count": 0,
        "valid": True,
        "validity_status": "valid",
        "exception": None,
        "provenance_mode": "lightweight_analysis",
    }
    if color is not None:
        record["color"] = color
    return record


def _make_run(tmp_path: Path, *, duck: bool) -> Path:
    run_dir = tmp_path / ("duck" if duck else "close")
    record_dir = run_dir / "trajectory_records"
    record_dir.mkdir(parents=True)
    variants = ["red", "white"] if duck else ["close"]
    config = {
        "experiment_name": run_dir.name,
        "runtime": {"output_dir": str(run_dir), "max_updates": 2},
        "batch": {
            "trajectories_per_update": 4,
            "chunks_per_trajectory": 2,
            "global_minibatch_chunks": 4,
            "micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "expected_optimizer_steps": 2,
        },
        "task": {
            "profile": "duck" if duck else "close",
            "active_variants": variants,
        },
        "algorithm": {"group_size": 2},
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "status.json").write_text(
        json.dumps({"state": "running", "completed_updates": 2})
    )
    metric_lines = []
    for update in range(2):
        rows = []
        for group_id in range(2):
            color = variants[group_id] if duck else None
            for member_id in range(2):
                rows.append(
                    _record(
                        update=update,
                        group_id=update * 10 + group_id,
                        member_id=member_id,
                        color=color,
                    )
                )
        (record_dir / f"update_{update:04d}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        metric_lines.append(
            json.dumps(
                {
                    "step": update + 1,
                    "metrics": {
                        "time/step": 60.0,
                        "time/generate_rollouts": 40.0,
                        "time/actor/run_training": 20.0,
                        "train/actor/optimizer_steps_per_update": 2.0,
                    },
                }
            )
        )
    (run_dir / "metrics_history.jsonl").write_text("\n".join(metric_lines) + "\n")
    return run_dir


def _fake_gnuplot(script: str, *, required: bool) -> bool:
    del required
    match = re.search(r'set output "([^"]+)"', script)
    assert match is not None
    path = Path(match.group(1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-png")
    return True


@pytest.mark.parametrize("duck", [False, True])
def test_analysis_generates_full_artifact_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duck: bool
) -> None:
    run_dir = _make_run(tmp_path, duck=duck)
    monkeypatch.setattr(analysis, "_run_gnuplot", _fake_gnuplot)

    summary = analysis.analyze_run(run_dir, require_plots=True)
    output = run_dir / "analysis"

    assert summary["complete_steps"] == [1, 2]
    assert summary["validation"]["complete_step_record_counts"] == {"1": 4, "2": 4}
    assert summary["validation"]["success_definition_mismatches"] == 0
    assert (output / "trajectories.csv").is_file()
    assert (output / "per_step_summary.csv").is_file()
    assert (output / "ppo_per_step_summary.csv").is_file()
    assert (output / "REPORT.md").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "run_overview.svg").is_file()
    assert (output / "success_reward_curves.png").is_file()
    assert (output / "success_similarity_relationship.png").is_file()
    assert (output / "success_correlation_analysis/combined_vs_success.png").is_file()
    assert (
        output
        / "success_probability_correlation/combined_vs_success_probability.png"
    ).is_file()
    if duck:
        assert [row["variant"] for row in summary["per_color"]] == ["red", "white"]
        assert (output / "per_color_per_step_summary.csv").is_file()
        assert (output / "per_color_summary.csv").is_file()
        assert (output / "per_color_success_reward_curves.png").is_file()
    else:
        assert summary["per_color"] == []


def test_analysis_rejects_committed_count_mismatch(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, duck=False)
    path = run_dir / "trajectory_records/update_0000.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")

    with pytest.raises(ValueError, match="non-2 trajectory groups"):
        analysis.analyze_run(run_dir)


def test_cli_requires_an_explicit_run() -> None:
    with pytest.raises(SystemExit):
        analysis._parse_args([])


def test_analysis_recomputes_native_terminal_success_rule() -> None:
    record = {
        "success_threshold": 0.5,
        "success_decision_rule": "terminal_positive_ratio",
        "success_terminal_probabilities": [0.9] * 10 + [0.1],
    }
    assert analysis._success_from_record(record) is False

    record["success_terminal_probabilities"][-1] = 0.9
    assert analysis._success_from_record(record) is True


def test_analysis_keeps_legacy_probability_max_rule() -> None:
    record = {
        "success_threshold": 0.5,
        "success_probability_max": 0.75,
    }
    assert analysis._success_from_record(record) is True
