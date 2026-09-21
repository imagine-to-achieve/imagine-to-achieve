"""Standalone Cosmos/Ctrl-World FPO-GRPO training package."""

from rlinf_modified.config import TrainConfig, load_config
from rlinf_modified.contracts import (
    BatchContract,
    CameraBundle,
    CheckpointState,
    ReplayBatch,
    TaskSpec,
    Trajectory,
)

__all__ = [
    "BatchContract",
    "CameraBundle",
    "CheckpointState",
    "ReplayBatch",
    "TaskSpec",
    "TrainConfig",
    "Trajectory",
    "load_config",
]

__version__ = "0.1.0"

