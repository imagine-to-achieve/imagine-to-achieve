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

from omegaconf import DictConfig

from rlinf.models.embodiment.cosmos.training_modes import (
    cfg_get_training_mode,
    validate_cosmos_training_mode,
)


def get_model(cfg: DictConfig, torch_dtype=None):
    cosmos_cfg = cfg.get("cosmos", {})
    validate_cosmos_training_mode(cfg_get_training_mode(cosmos_cfg))
    backend = cosmos_cfg.get("backend", "native")

    if backend == "native":
        from rlinf.models.embodiment.cosmos.cosmos_backend import (
            CosmosNativeInferencePolicy,
        )

        return CosmosNativeInferencePolicy(cfg=cfg, torch_dtype=torch_dtype)

    if backend == "edge4b":
        from rlinf.models.embodiment.cosmos.edge4b_backend import (
            CosmosEdge4BInferencePolicy,
        )

        return CosmosEdge4BInferencePolicy(cfg=cfg, torch_dtype=torch_dtype)

    raise ValueError(
        f"Unsupported Cosmos backend: {backend!r}. Use 'native' or 'edge4b'."
    )
