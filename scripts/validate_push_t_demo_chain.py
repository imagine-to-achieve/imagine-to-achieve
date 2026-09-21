#!/usr/bin/env python3
"""Replay real Push-T demonstrations through the production Ctrl-World adapter.

No policy/checkpoint updates. Three paired cases: measured states converted to
policy actions, the exact SFT action construction, and a stationary control.
Physical view names here are always main=d405, side=d405_1, wrist=d435.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from cosmos_framework.data.vfm.action.pose_utils import build_abs_pose_from_components, pose_abs_to_rel
from rlinf.data.datasets.lerobot_book import decode_video_frames
from rlinf.envs.world_model.eef_pose_adapter import analytic_ur5_rot6d_delta_actions_to_eef_states
from rlinf.envs.world_model.world_model_ctrl_world_env import CtrlWorldEnv
from rlinf.rewards.resnet_reward_model import classify_terminal_probabilities
from rlinf_modified.config import load_config
from rlinf_modified.engine.real import RealTrainer

ROOT = Path(__file__).resolve().parents[1]
VIEWS = {"main": "observation.images.d405_rgb", "side": "observation.images.d405_1_rgb", "wrist": "observation.images.d435_rgb"}
INFO_KEYS = {"main": "ctrl_world_video_chunk_raw", "side": "ctrl_world_video_chunk_raw_extra", "wrist": "ctrl_world_video_chunk_raw_wrist"}


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def event(out, kind, **data):
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": kind, **data}
    line = json.dumps(row, allow_nan=False)
    print(line, flush=True)
    with (out / "events.jsonl").open("a") as f:
        f.write(line + "\n")


def production_config(args, out):
    cfg = load_config(args.config)
    bridge = RealTrainer.__new__(RealTrainer)  # Avoid rewriting original run files.
    bridge.config, bridge.root, bridge.run_dir = cfg, ROOT, out
    checkpoint_config = Path(args.config).parent / "inputs/cosmos/checkpoint_config.normalized.yaml"
    overrides = bridge._overrides(str(checkpoint_config))
    with initialize_config_dir(version_base="1.1", config_dir=str(ROOT / "third_party/rlinf_runtime/examples/embodiment/config")):
        full = compose(config_name=cfg.assets.legacy_config_name, overrides=overrides)
    manifest = json.loads(Path(cfg.task.split_manifest_path).read_text())
    split = "train" if args.episode in manifest["training"]["episodes"] else "eval"
    env_cfg = OmegaConf.create(OmegaConf.to_container(full.env[split], resolve=True))
    OmegaConf.save(env_cfg, out / "production_env.yaml")
    changes = {"total_num_envs": 1, "group_size": 1, "use_fixed_reset_state_ids": False,
               "use_ordered_reset_state_ids": False, "random_reset_state_ids": False,
               "specific_reset_id": None, "auto_reset": False, "ignore_terminations": True,
               "max_episode_steps": args.max_chunks * 32, "max_steps_per_rollout_epoch": args.max_chunks * 32}
    for key, value in changes.items():
        env_cfg[key] = value
    env_cfg.video_cfg.save_video = False
    env_cfg.ctrl_world_cfg.common_random_numbers_within_group = True
    env_cfg.ctrl_world_cfg.common_noise_seed = args.seed
    OmegaConf.save(env_cfg, out / "diagnostic_env.yaml")
    write_json(out / "protocol.json", {
        "source_config": str(Path(args.config).resolve()), "episode": args.episode,
        "split": split, "seed": args.seed, "changes_from_production": changes,
        "cases": args.cases.split(","), "physical_views": VIEWS,
        "case_states": "observation.state at 15 FPS -> backward_framewise rot6d -> production adapter",
        "case_actions": "SFT exact construction: observation.state anchor, future action waypoints -> backward_framewise rot6d",
        "case_hold": "zero translation, identity rot6d, unchanged absolute gripper",
        "noise": "same reset/VAE seed and same advancing production denoise stream per case",
        "horizon": "report original 5 chunks and complete demonstration; hold final waypoint only to fill last chunk",
        "limitations": "4 deliberately fixed episodes are a chain diagnostic, not an unbiased policy success estimate; no Cosmos video generation",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__), ROOT / "third_party/rlinf_runtime/rlinf/envs/world_model/world_model_ctrl_world_env.py",
            ROOT / "third_party/rlinf_runtime/rlinf/envs/world_model/eef_pose_adapter.py",
            ROOT / "third_party/cosmos_framework/data/vfm/action/datasets/ur5_eef_lerobot_dataset.py"]},
    })
    return cfg, env_cfg, full.reward.success_classifier


def relative_actions(anchor, future):
    waypoints = np.concatenate([anchor[None], future], axis=0).astype(np.float32)
    poses = build_abs_pose_from_components(waypoints[:, :3], waypoints[:, 3:6], "axisangle")
    delta = pose_abs_to_rel(poses, rotation_format="rot6d", pose_convention="backward_framewise")
    return np.concatenate([delta, future[:, 6:7]], axis=-1).astype(np.float32)


def pose_errors(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    angles = (Rotation.from_rotvec(a[..., 3:6].reshape(-1, 3)).inv() * Rotation.from_rotvec(b[..., 3:6].reshape(-1, 3))).magnitude()
    return {"position_max_m": float(np.linalg.norm(a[..., :3] - b[..., :3], axis=-1).max()),
            "rotation_max_rad": float(angles.max()), "gripper_max": float(np.abs(a[..., 6] - b[..., 6]).max())}


def decision(probabilities, cfg):
    result = classify_terminal_probabilities(torch.as_tensor(probabilities).float().reshape(1, -1), cfg)
    return {key: (value.detach().cpu().tolist() if torch.is_tensor(value) else value) for key, value in result.items()}


def load_real_views(dataset, episode, rows):
    result = {}
    for view, key in VIEWS.items():
        video = dataset.data_dir / dataset.video_path_template.format(
            video_key=key, chunk_index=int(episode[f"videos/{key}/chunk_index"]), file_index=int(episode[f"videos/{key}/file_index"]))
        frames = []
        # Bound decoder memory on long/shared video files.
        for start in range(0, len(rows), 96):
            timestamps = [float(episode.get(f"videos/{key}/from_timestamp", 0.0)) + float(row["timestamp"]) for row in rows[start:start + 96]]
            raw = decode_video_frames(video, timestamps, tolerance_s=1e-4)
            resized = F.interpolate(raw, size=(192, 320), mode="bilinear", align_corners=False, antialias=True)
            frames.append((resized.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy())
        result[view] = np.concatenate(frames)
    return result


@torch.inference_mode()
def score_real(env, main_frames):
    outputs = []
    for start in range(0, len(main_frames), 64):
        x = torch.from_numpy(main_frames[start:start + 64].copy()).permute(0, 3, 1, 2).to(env.device).float() / 127.5 - 1
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        outputs.append(env.reward_model.predict_rew(x).float().cpu().reshape(-1).numpy())
    return np.concatenate(outputs)


def tile_views(views, index, label):
    canvas = Image.new("RGB", (960, 214), "white")
    draw = ImageDraw.Draw(canvas)
    for col, name in enumerate(VIEWS):
        draw.text((col * 320 + 5, 5), f"{label} | {name}", fill="black")
        canvas.paste(Image.fromarray(views[name][index]), (col * 320, 22))
    return canvas


def contact_sheet(out, real, generated, frame_indices, case):
    canvas = Image.new("RGB", (960, 428 * len(frame_indices)), "white")
    for row, index in enumerate(frame_indices):
        canvas.paste(tile_views(real, min(index, len(real["main"]) - 1), f"real t={index/30:.2f}s"), (0, row * 428))
        canvas.paste(tile_views(generated, min(index, len(generated["main"]) - 1), f"{case} t={index/30:.2f}s"), (0, row * 428 + 214))
    canvas.save(out / f"{case}_comparison.jpg", quality=92)


def save_video(path, real, generated, case):
    import av
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=15)
    stream.width, stream.height, stream.pix_fmt = 960, 428, "yuv420p"
    stream.options = {"crf": "25", "preset": "fast"}
    for index in range(0, len(generated["main"]), 2):
        canvas = Image.new("RGB", (960, 428), "white")
        canvas.paste(tile_views(real, min(index, len(real["main"]) - 1), f"real {index/30:.2f}s"), (0, 0))
        canvas.paste(tile_views(generated, index, f"{case} {index/30:.2f}s"), (0, 214))
        frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", default="states,actions,hold")
    parser.add_argument("--max-chunks", type=int, default=20)
    args = parser.parse_args()
    out = Path(args.output).resolve() / f"episode_{args.episode:03d}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise RuntimeError(f"Refusing to overwrite completed results: {out}")
    torch.set_num_threads(8)
    event(out, "start", episode=args.episode, gpu=torch.cuda.get_device_name())
    config, env_cfg, success_cfg = production_config(args, out)
    env = CtrlWorldEnv(env_cfg, 1, 0, 1, SimpleNamespace(rank=0), record_metrics=True)
    episode = env.dataset.episodes[args.episode]
    assert int(episode["episode_index"]) == args.episode
    rows = env.dataset._load_episode_rows(episode)
    states = np.asarray([row["observation.state"] for row in rows], dtype=np.float32)
    commands = np.asarray([row["action"] for row in rows], dtype=np.float32)
    times = np.asarray([row["timestamp"] for row in rows])
    assert np.allclose(times - times[0], np.arange(len(rows)) / 30, atol=1e-4), "Unexpected source FPS"
    chunks = max(5, math.ceil((len(rows) - 1) / 64))
    if chunks > args.max_chunks:
        raise ValueError(f"Full demonstration needs {chunks} chunks; configured cap is {args.max_chunks}")
    roundtrip = {}
    for field, waypoints in [("states", states), ("actions", commands)]:
        errors = []
        for chunk in range(chunks):
            start = min(chunk * 64, len(rows) - 1)
            indices = np.minimum(chunk * 64 + 2 * np.arange(1, 33), len(rows) - 1)
            action = relative_actions(states[start], waypoints[indices])
            recovered = analytic_ur5_rot6d_delta_actions_to_eef_states(
                torch.from_numpy(states[start:start + 1]), torch.from_numpy(action[None]), translation_gain=1.0)[0].numpy()
            errors.append(pose_errors(recovered, waypoints[indices]))
        roundtrip[field] = {key: max(e[key] for e in errors) for key in errors[0]}
        assert roundtrip[field]["position_max_m"] < 1e-4, roundtrip[field]
        assert roundtrip[field]["rotation_max_rad"] < 1e-3, roundtrip[field]
    write_json(out / "adapter_roundtrip.json", {"roundtrip": roundtrip, "action_vs_observed_state": pose_errors(commands, states), "frames": len(rows), "chunks": chunks})
    event(out, "adapter_verified", roundtrip=roundtrip, frames=len(rows), chunks=chunks, fps_microcondition=env.fps)
    real = load_real_views(env.dataset, episode, rows)
    real_probs = score_real(env, real["main"])
    np.savez_compressed(out / "real_probabilities.npz", probabilities=real_probs)
    endpoints = Image.new("RGB", (960, 642), "white")
    for row, idx in enumerate([0, min(319, len(rows)-1), len(rows)-1]):
        endpoints.paste(tile_views(real, idx, f"real frame={idx} p={real_probs[idx]:.4f}"), (0, row * 214))
    endpoints.save(out / "real_reference.jpg", quality=94)
    summary = {"episode": args.episode, "source_frames": len(rows), "source_duration_seconds": (len(rows)-1)/30,
               "original_horizon_seconds": 320/30, "full_replay_chunks": chunks, "roundtrip": roundtrip,
               "real_at_original_horizon": decision(real_probs[:320], success_cfg), "real_at_end": decision(real_probs, success_cfg), "cases": {}}
    write_json(out / "partial_summary.json", summary)
    event(out, "real_reference_scored", original=summary["real_at_original_horizon"], terminal=summary["real_at_end"])
    for case in args.cases.split(","):
        if case not in {"states", "actions", "hold"}:
            raise ValueError(case)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        env._common_noise_rollout_index = -1
        env.reset(episode_indices=np.array([args.episode]))
        generated, all_probs, chunk_metrics = {view: [] for view in VIEWS}, [], []
        start_time = time.time()
        for chunk in range(chunks):
            indices = np.minimum(chunk * 64 + 2 * np.arange(1, 33), len(rows) - 1)
            anchor = states[min(chunk * 64, len(rows)-1)]
            if case == "hold":
                action = relative_actions(states[0], np.repeat(states[0:1], 32, axis=0))
            else:
                waypoints = states if case == "states" else commands
                action = relative_actions(anchor, waypoints[indices])
            _, _, _, _, info_list = env.chunk_step(torch.from_numpy(action[None]).to(env.device))
            info = info_list[-1]
            probs = info["chunk_raw_rewards"][0].detach().float().cpu().numpy()
            all_probs.append(probs)
            for view, key in INFO_KEYS.items():
                generated[view].append(info[key][0].detach().cpu().numpy())
            expected_endpoint = states[0:1] if case == "hold" else waypoints[indices[-1:]]
            endpoint_error = pose_errors(env.current_states.detach().cpu().numpy()[:, :7], expected_endpoint)
            metric = {"chunk": chunk+1, "duration_seconds": time.time()-start_time, "success": decision(probs, success_cfg), "endpoint_error": endpoint_error}
            chunk_metrics.append(metric)
            event(out, "chunk", case=case, **metric)
            if chunk == 4:
                snapshot = {v: np.concatenate(parts) for v, parts in generated.items()}
                contact_sheet(out, real, snapshot, [63, 191, 319], case + "_horizon5")
                write_json(out / f"{case}_horizon5.json", metric)
        generated = {v: np.concatenate(parts) for v, parts in generated.items()}
        probabilities = np.concatenate(all_probs)
        native_end = min(len(rows)-1, len(probabilities)-1)
        per_view_goal_mse = {}
        for view in VIEWS:
            # Same fixed goal and last 4 policy-aligned frames as terminal reward.
            sampled = generated[view][1::2][-4:].astype(np.float32)/255
            goal = real[view][-1].astype(np.float32)/255
            per_view_goal_mse[view] = float(np.square(sampled-goal).mean())
        weights = {"current": {"wrist": 2/3, "main": 1/6, "side": 1/6},
                   "wrist_010": {"wrist": .1, "main": .45, "side": .45},
                   "wrist_000": {"wrist": 0, "main": .5, "side": .5}}
        result = {"chunks": chunk_metrics, "at_original_horizon": decision(probabilities[:320], success_cfg),
                  "at_demonstration_end": decision(probabilities[:native_end+1], success_cfg),
                  "at_padded_chunk_end": decision(probabilities, success_cfg),
                  "terminal_goal_mse_by_physical_view": per_view_goal_mse,
                  "offline_terminal_reweighting": {name: sum(w[v]*per_view_goal_mse[v] for v in VIEWS) for name,w in weights.items()},
                  "note": "Offline reweighting is terminal MSE only, with independent full-size views; it does not test Cosmos trajectory agreement or retraining."}
        np.savez_compressed(out / f"{case}_probabilities.npz", probabilities=probabilities)
        contact_sheet(out, real, generated, [0, 319, native_end], case)
        save_video(out / f"{case}_comparison.mp4", real, generated, case)
        summary["cases"][case] = result
        write_json(out / "partial_summary.json", summary)
        event(out, "case_complete", case=case, original=result["at_original_horizon"], end=result["at_demonstration_end"], goal_mse=per_view_goal_mse)
    write_json(out / "summary.json", summary)
    event(out, "complete", episode=args.episode)


if __name__ == "__main__":
    main()
