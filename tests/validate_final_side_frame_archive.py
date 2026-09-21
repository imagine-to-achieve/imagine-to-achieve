#!/usr/bin/env python3
"""Strict integrity check for one per-step terminal side-frame NPZ archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CATEGORIES = (
    "success_mse_min",
    "success_mse_max",
    "failure_mse_min",
    "failure_mse_max",
)


def _scalar(data: np.lib.npyio.NpzFile, name: str):
    value = data[name]
    if value.shape != (1,):
        raise AssertionError(f"{name} must have shape (1,), got {value.shape}")
    return value[0].item()


def validate(path: Path, expected_count: int, expected_split: str) -> dict:
    archive_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "global_step",
            "split",
            "camera_key",
            "ctrl_world_view_index",
            "frame_encoding",
            "frame_sha256_scope",
            "frames",
            "frame_sha256",
            "rank",
            "env_id",
            "episode",
            "rollout_uid",
            "group_id",
            "member_id",
            "success",
            "trajectory_mse",
            "reward",
            "extreme_categories",
            "extreme_available",
            "extreme_archive_index",
            "extreme_trajectory_mse",
            "extreme_frames",
            "extreme_frame_sha256",
        }
        missing = required - set(data.files)
        if missing:
            raise AssertionError(f"missing archive fields: {sorted(missing)}")

        if _scalar(data, "schema_version") != 1:
            raise AssertionError("unsupported schema_version")
        step = int(_scalar(data, "global_step"))
        if _scalar(data, "split") != expected_split:
            raise AssertionError("unexpected split")
        if _scalar(data, "camera_key") != "observation.images.d405_1_rgb":
            raise AssertionError("archive is not sourced from the side camera")
        if _scalar(data, "ctrl_world_view_index") != 2:
            raise AssertionError("archive is not mapped to Ctrl-World view index 2")
        if _scalar(data, "frame_encoding") != "lossless_uint8_rgb_hwc":
            raise AssertionError("unexpected frame encoding")
        if _scalar(data, "frame_sha256_scope") != "contiguous_uint8_rgb_bytes":
            raise AssertionError("unexpected frame hash scope")

        frames = data["frames"]
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise AssertionError(
                f"frames must be uint8 [N,H,W,3], got {frames.dtype} {frames.shape}"
            )
        count = int(frames.shape[0])
        if count != expected_count:
            raise AssertionError(f"expected {expected_count} frames, got {count}")

        vector_fields = (
            "frame_sha256",
            "rank",
            "env_id",
            "episode",
            "rollout_uid",
            "group_id",
            "member_id",
            "success",
            "trajectory_mse",
            "reward",
        )
        for name in vector_fields:
            if data[name].shape != (count,):
                raise AssertionError(
                    f"{name} must have shape ({count},), got {data[name].shape}"
                )
        if data["success"].dtype != np.bool_:
            raise AssertionError("success must be bool")
        if not np.isfinite(data["trajectory_mse"]).all():
            raise AssertionError("trajectory_mse contains non-finite values")
        if not np.isfinite(data["reward"]).all():
            raise AssertionError("reward contains non-finite values")

        identities = list(
            zip(
                data["rank"].tolist(),
                data["env_id"].tolist(),
                data["rollout_uid"].tolist(),
                strict=True,
            )
        )
        if len(set(identities)) != count:
            raise AssertionError("duplicate (rank, env_id, rollout_uid) identity")

        computed_hashes = np.asarray(
            [
                hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
                for frame in frames
            ]
        )
        if not np.array_equal(computed_hashes, data["frame_sha256"]):
            bad = np.flatnonzero(computed_hashes != data["frame_sha256"])
            raise AssertionError(
                f"frame SHA-256 mismatch at archive indices {bad.tolist()}"
            )

        categories = tuple(data["extreme_categories"].tolist())
        if categories != CATEGORIES:
            raise AssertionError(f"unexpected extreme categories: {categories}")
        available = data["extreme_available"]
        indices = data["extreme_archive_index"]
        extreme_mse = data["extreme_trajectory_mse"]
        extreme_frames = data["extreme_frames"]
        extreme_hashes = data["extreme_frame_sha256"]
        if available.shape != (4,) or available.dtype != np.bool_:
            raise AssertionError("extreme_available must be bool[4]")
        if (
            indices.shape != (4,)
            or extreme_mse.shape != (4,)
            or extreme_hashes.shape != (4,)
        ):
            raise AssertionError("extreme metadata must contain four entries")
        if (
            extreme_frames.shape != (4, *frames.shape[1:])
            or extreme_frames.dtype != np.uint8
        ):
            raise AssertionError("extreme_frames shape/dtype mismatch")

        success = data["success"]
        mse = data["trajectory_mse"]
        for category_index, category in enumerate(CATEGORIES):
            category_success = category.startswith("success_")
            candidates = np.flatnonzero(success == category_success)
            if not len(candidates):
                if available[category_index] or indices[category_index] != -1:
                    raise AssertionError(
                        f"{category} marked available without candidates"
                    )
                if not np.isnan(extreme_mse[category_index]):
                    raise AssertionError(
                        f"{category} unavailable MSE must be NaN"
                    )
                continue
            if not available[category_index]:
                raise AssertionError(
                    f"{category} missing despite available candidates"
                )
            choose_max = category.endswith("_max")
            candidate_values = mse[candidates]
            expected_index = int(
                candidates[
                    np.argmax(candidate_values)
                    if choose_max
                    else np.argmin(candidate_values)
                ]
            )
            archive_index = int(indices[category_index])
            if archive_index != expected_index:
                raise AssertionError(
                    f"{category} index {archive_index} != expected {expected_index}"
                )
            if extreme_mse[category_index] != mse[archive_index]:
                raise AssertionError(
                    f"{category} MSE does not match the selected rollout"
                )
            if not np.array_equal(
                extreme_frames[category_index], frames[archive_index]
            ):
                raise AssertionError(
                    f"{category} frame does not match the selected rollout"
                )
            if extreme_hashes[category_index] != computed_hashes[archive_index]:
                raise AssertionError(f"{category} frame hash mismatch")

        return {
            "path": str(path.resolve()),
            "archive_sha256": archive_digest,
            "global_step": step,
            "split": expected_split,
            "frame_count": count,
            "frame_shape": list(frames.shape[1:]),
            "success_count": int(success.sum()),
            "failure_count": int((~success).sum()),
            "extreme_categories_saved": int(available.sum()),
            "camera_key": _scalar(data, "camera_key"),
            "ctrl_world_view_index": int(
                _scalar(data, "ctrl_world_view_index")
            ),
            "all_frame_hashes_verified": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", type=Path, nargs="+")
    parser.add_argument("--expected-count", type=int, default=128)
    parser.add_argument("--expected-split", default="training")
    args = parser.parse_args()
    summaries = [
        validate(path, args.expected_count, args.expected_split)
        for path in args.archives
    ]
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
