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

import math
from contextlib import nullcontext
from typing import ContextManager, Union

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Shard
from torch.optim import Optimizer

from rlinf.hybrid_engines.fsdp import FSDPModule
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.hybrid_engines.fsdp.utils import (
    FSDPVersion,
    clip_grad_by_total_norm_,
    get_grad_norm,
)
from rlinf.utils.utils import clear_memory


def _local_gradient_norm_statistic(
    local_gradients: list[torch.Tensor],
    *,
    norm_type: float,
    collective_device: torch.device,
) -> torch.Tensor:
    """Accumulate one local norm statistic without assuming shard residency.

    Native FSDP2 may expose CPU local shards when ``CPUOffloadPolicy`` is
    enabled and CUDA local shards when it is disabled. Accumulate scalars on
    each source device first, then move only one scalar per device to the
    collective device. This avoids both CPU/CUDA in-place-add failures and a
    synchronization for every individual parameter shard.
    """

    infinity_norm = norm_type == float("inf")
    per_device: dict[torch.device, torch.Tensor] = {}
    for gradient in local_gradients:
        if infinity_norm:
            if gradient.numel() == 0:
                continue
            statistic = gradient.abs().max().to(dtype=torch.float32)
        else:
            statistic = (
                gradient.to(dtype=torch.float32)
                .abs()
                .pow(norm_type)
                .sum()
            )
        previous = per_device.get(statistic.device)
        if previous is None:
            per_device[statistic.device] = statistic
        elif infinity_norm:
            per_device[statistic.device] = torch.maximum(previous, statistic)
        else:
            previous.add_(statistic)

    total = torch.zeros((), dtype=torch.float32, device=collective_device)
    for statistic in per_device.values():
        statistic = statistic.to(device=collective_device)
        if infinity_norm:
            total = torch.maximum(total, statistic)
        else:
            total.add_(statistic)
    return total


class NativeParallelStrategy(FSDPStrategyBase):
    """Strategy for models that already parallelize themselves internally.

    Some models (e.g. the Cosmos native backend) apply their own FSDP2
    ``fully_shard()`` sharding to a large frozen backbone before RLinf ever
    sees the module, exposing only a small trainable-parameter subset
    (already DTensor-backed) through the module RLinf receives from
    ``model_provider_func``. Wrapping that module again with RLinf's own
    FSDP1/FSDP2 would try to re-shard already-sharded storage and fails.
    This strategy skips wrapping entirely: it builds the optimizer, clips
    gradients, and offloads directly against whatever ``model.parameters()``
    already exposes, reusing FSDP2Strategy's generic (DTensor-safe)
    offload/grad-clip implementations.
    """

    def wrap_model(self, model: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """
        Return the model unchanged.

        Args:
            - model (nn.Module): The model, already parallelized internally
              by its own construction code.
            - device_mesh (DeviceMesh): Unused; the model manages its own
              device mesh/parallelism.

        Returns:
            - nn.Module: The same model instance, unwrapped.
        """
        return model

    @classmethod
    def get_fsdp_version(cls) -> FSDPVersion:
        return FSDPVersion.NATIVE

    @torch.no_grad()
    def onload_param_and_grad(
        self, model: nn.Module, device: torch.device, onload_grad: bool
    ) -> None:
        """
        Load model parameters and gradients to the specified device.

        Args:
            - model (nn.Module): The model.
            - device (torch.device): The target device.
            - onload_grad (bool): Whether to load gradients or not.
        """
        model.to(device=device)
        if onload_grad:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.to(device)
        clear_memory()

    @torch.no_grad()
    def offload_param_and_grad(self, model: nn.Module, offload_grad: bool) -> None:
        """
        Offload model parameters and gradients to CPU.

        Args:
            - model (nn.Module): The model.
            - offload_grad (bool): Whether to offload gradients or not.
        """
        model.to(device="cpu")

        if offload_grad:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.cpu()
        clear_memory()

    @torch.no_grad()
    def offload_optimizer(self, optimizer: Optimizer) -> None:
        """
        Offload optimizer states to CPU.

        Args:
            - optimizer (Optimizer): The optimizer.
        """
        for st in optimizer.state.values():
            if not isinstance(st, dict):
                continue
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    if v.device.type != "cpu":
                        st[k] = v.detach().to("cpu", non_blocking=True)
                        del v
        clear_memory()

    @torch.no_grad()
    def onload_optimizer(self, optimizer: Optimizer, device: torch.device) -> None:
        """
        Load optimizer states to the specified device.

        Args:
            - optimizer (Optimizer): The optimizer.
            - device (torch.device): The target device.
        """
        for st in optimizer.state.values():
            if not isinstance(st, dict):
                continue
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    if v.device != device:
                        st[k] = v.detach().to(device, non_blocking=True)
                        del v
        clear_memory()

    def clip_grad_norm_(
        self,
        model: nn.Module,
        norm_type: Union[float, int] = 2.0,
    ) -> float:
        """
        Clip the gradients of the model parameters by total norm.

        Args:
            - model (nn.Module): The model.
            - norm_type (float): The type of the used p-norm.

        Returns:
            - float: The total norm of the gradients before clipping.
        """
        parameters = list(model.parameters())
        gradients = [param.grad for param in parameters if param.grad is not None]
        if not gradients:
            return 0.0

        dtensor_gradients = [grad for grad in gradients if isinstance(grad, DTensor)]
        if not dtensor_gradients:
            grad_norm = get_grad_norm(
                parameters,
                dp_group=self._dp_group,
                norm_type=norm_type,
            )
            clip_grad_by_total_norm_(
                parameters,
                max_grad_norm=self.cfg.optim.clip_grad,
                total_norm=grad_norm,
            )
            return grad_norm
        if len(dtensor_gradients) != len(gradients):
            raise RuntimeError(
                "Native gradient clipping does not support a mixture of DTensor "
                "and ordinary gradients: "
                f"dtensor={len(dtensor_gradients)}, total={len(gradients)}"
            )

        # Internal Cosmos FSDP2 exposes HSDP gradients with placements
        # (Replicate, Shard). Each replica owns an identical reduced local
        # shard, so the global norm must reduce only across sharded mesh
        # dimensions. Reducing over replicated dimensions would count the
        # same logical gradient once per node and over-clip it.
        reference = dtensor_gradients[0]
        mesh = reference.device_mesh
        mesh_signature = (
            str(mesh.device_type),
            tuple(int(value) for value in mesh.mesh.shape),
            tuple(int(value) for value in mesh.mesh.reshape(-1).tolist()),
            tuple(mesh.mesh_dim_names or ()),
        )
        shard_mesh_dims = tuple(
            index
            for index, placement in enumerate(reference.placements)
            if isinstance(placement, Shard)
        )
        if any(isinstance(placement, Partial) for placement in reference.placements):
            raise RuntimeError(
                "Native gradient clipping requires reduced DTensor gradients, "
                f"found placements={reference.placements}"
            )

        for gradient in dtensor_gradients[1:]:
            gradient_mesh = gradient.device_mesh
            gradient_mesh_signature = (
                str(gradient_mesh.device_type),
                tuple(int(value) for value in gradient_mesh.mesh.shape),
                tuple(
                    int(value)
                    for value in gradient_mesh.mesh.reshape(-1).tolist()
                ),
                tuple(gradient_mesh.mesh_dim_names or ()),
            )
            gradient_shard_dims = tuple(
                index
                for index, placement in enumerate(gradient.placements)
                if isinstance(placement, Shard)
            )
            if any(
                isinstance(placement, Partial)
                for placement in gradient.placements
            ):
                raise RuntimeError(
                    "Native gradient clipping requires reduced DTensor gradients, "
                    f"found placements={gradient.placements}"
                )
            if (
                gradient_mesh_signature != mesh_signature
                or gradient_shard_dims != shard_mesh_dims
            ):
                raise RuntimeError(
                    "Native DTensor gradients do not share one HSDP shard mesh: "
                    f"reference={(mesh_signature, shard_mesh_dims)!r}, "
                    f"found={(gradient_mesh_signature, gradient_shard_dims)!r}"
                )

        norm_type_float = float(norm_type)
        if norm_type_float <= 0.0:
            raise ValueError(f"Gradient norm type must be positive, got {norm_type}")
        local_gradients = [
            gradient.to_local().detach() for gradient in dtensor_gradients
        ]
        if mesh.device_type == "cuda":
            collective_device = torch.device("cuda", torch.cuda.current_device())
        else:
            collective_device = torch.device(mesh.device_type)
        total = _local_gradient_norm_statistic(
            local_gradients,
            norm_type=norm_type_float,
            collective_device=collective_device,
        )
        reduce_op = (
            torch.distributed.ReduceOp.MAX
            if norm_type_float == float("inf")
            else torch.distributed.ReduceOp.SUM
        )
        for mesh_dim in shard_mesh_dims:
            if int(mesh.mesh.shape[mesh_dim]) > 1:
                torch.distributed.all_reduce(
                    total,
                    op=reduce_op,
                    group=mesh.get_group(mesh_dim),
                )
        total_value = float(total.item())
        grad_norm = (
            total_value
            if norm_type_float == float("inf")
            else total_value ** (1.0 / norm_type_float)
        )

        # Scale the actual local gradient storage. Converting a BF16 gradient
        # to FP32 and multiplying that temporary tensor (as the generic helper
        # does) would leave the optimizer-visible gradient unchanged.
        if math.isfinite(grad_norm):
            clip_coefficient = min(
                1.0,
                float(self.cfg.optim.clip_grad) / (grad_norm + 1.0e-6),
            )
            if clip_coefficient < 1.0:
                for gradient in local_gradients:
                    gradient.mul_(clip_coefficient)
        return grad_norm

    def before_micro_batch(
        self, model: nn.Module, is_last_micro_batch: bool
    ) -> ContextManager:
        """
        Context manager to control gradient synchronization.

        The model is not itself an ``FSDPModule`` root (RLinf never called
        ``fully_shard()`` on it), so ``set_requires_gradient_sync`` is not
        generally available. When present (e.g. the model exposes it on a
        submodule that happens to also be the root), use it; otherwise fall
        back to always syncing every micro-batch, which is correctness-safe
        and only costs a bit of extra communication for the small trainable
        parameter subset this strategy is meant for.

        Args:
            - model (nn.Module): The model.
            - is_last_micro_batch (bool): Whether this is the last micro batch.

        Returns:
            - ContextManager: nullcontext, just for interface consistency.
        """
        if not self.cfg.fsdp_config.enable_gradient_accumulation:
            return nullcontext()
        if isinstance(model, FSDPModule) or hasattr(
            model, "set_requires_gradient_sync"
        ):
            model.set_requires_gradient_sync(is_last_micro_batch)
        return nullcontext()
