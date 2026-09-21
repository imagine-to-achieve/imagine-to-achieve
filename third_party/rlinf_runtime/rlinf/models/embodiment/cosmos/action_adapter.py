# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit Cosmos action adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CosmosDirectActionAdapter:
    """Direct adapter for raw Cosmos actions that already match Ctrl-World."""

    action_dim: int
    num_action_chunks: int

    def to_ctrl_world(self, raw_action: torch.Tensor) -> torch.Tensor:
        expected = (int(raw_action.shape[0]), self.num_action_chunks, self.action_dim)
        if tuple(raw_action.shape) != expected:
            raise ValueError(
                "Direct Cosmos action adapter expected raw action shape "
                f"{expected}, got {tuple(raw_action.shape)}."
            )
        return raw_action

    def from_ctrl_world(self, ctrl_world_action: torch.Tensor) -> torch.Tensor:
        return self.to_ctrl_world(ctrl_world_action)


def validate_direct_action_adapter(
    *,
    raw_action_dim: Any,
    action_dim: int,
    action_chunk_size: Any,
    num_action_chunks: int,
) -> CosmosDirectActionAdapter:
    """Validate native-training-scale direct action representation and return the adapter."""

    if raw_action_dim is None:
        raise ValueError("cosmos.raw_action_dim is required for native backend.")
    if action_chunk_size is None:
        raise ValueError("cosmos.action_chunk_size is required for native backend.")
    if int(raw_action_dim) != int(action_dim):
        raise NotImplementedError(
            "Native Cosmos training only accepts direct Ctrl-World action shape. "
            f"Got raw_action_dim={raw_action_dim}, action_dim={action_dim}; "
            "add an explicit tested action adapter before enabling conversion."
        )
    if int(action_chunk_size) != int(num_action_chunks):
        raise ValueError(
            "cosmos.action_chunk_size must match num_action_chunks for native "
            f"training, got {action_chunk_size} and {num_action_chunks}."
        )
    return CosmosDirectActionAdapter(
        action_dim=int(action_dim),
        num_action_chunks=int(num_action_chunks),
    )
