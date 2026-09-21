"""Frozen native-Cosmos forward-dynamics inference adapter."""

from __future__ import annotations

import gc
import importlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from rlinf.models.embodiment.cosmos.cosmos_backend import (
    CosmosNativeBackendUnavailable,
    CosmosNativeInferencePolicy,
)


def _merge_single_item_batches(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge official single-item action batches without changing nesting."""
    if not items:
        raise ValueError("At least one forward-dynamics batch item is required.")
    merged: dict[str, Any] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        first = values[0]
        if torch.is_tensor(first):
            merged[key] = torch.cat(values, dim=0)
        elif isinstance(first, list):
            merged[key] = [entry for value in values for entry in value]
        else:
            raise TypeError(
                f"Unsupported Cosmos action-batch field {key!r}: {type(first)}"
            )
    return merged


def _future_frames_from_decoded(
    decoded: torch.Tensor,
    *,
    action_chunk_size: int,
) -> torch.Tensor:
    """Select exactly the future frames, excluding the clean condition frame."""
    if decoded.ndim == 5 and decoded.shape[0] == 1:
        decoded = decoded.squeeze(0)
    if decoded.ndim != 4 or decoded.shape[0] not in (1, 3):
        raise CosmosNativeBackendUnavailable(
            "Forward-dynamics decode must be [C,T,H,W], got "
            f"{tuple(decoded.shape)}."
        )
    if decoded.shape[1] < action_chunk_size + 1:
        raise CosmosNativeBackendUnavailable(
            "Forward-dynamics decode returned fewer frames than condition + "
            f"actions: {decoded.shape[1]} < {action_chunk_size + 1}."
        )
    future = decoded[:, 1 : action_chunk_size + 1]
    if future.shape[1] != action_chunk_size:
        raise CosmosNativeBackendUnavailable(
            f"Expected {action_chunk_size} future frames, got {future.shape[1]}."
        )
    return future


def _decoded_video_to_uint8(decoded: torch.Tensor) -> torch.Tensor:
    """Convert either Cosmos decoder range to uint8 without darkening [0, 1]."""
    value = decoded.float()
    if not value.numel():
        return value.to(torch.uint8)
    if not torch.isfinite(value).all():
        raise CosmosNativeBackendUnavailable(
            "Forward-dynamics decoder returned NaN or Inf pixels."
        )
    minimum = float(value.amin())
    if minimum < 0.0:
        # Cosmos' official inference path clamps raw VAE output before mapping
        # [-1, 1] to [0, 1]. Small decoder overshoots are normal.
        value = (value.clamp(-1.0, 1.0) + 1.0) / 2.0
    return (value.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def _forward_dynamics_sampling_parameters(
    cosmos_cfg: Any,
) -> tuple[int, float, float]:
    """Return validated (num_steps, guidance, shift) replay parameters."""
    try:
        raw_num_steps = float(cosmos_cfg.get("num_steps", 10))
        guidance = float(cosmos_cfg.get("guidance", 3.0))
        shift = float(cosmos_cfg.get("shift", 10.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Forward-dynamics num_steps, guidance, and shift must be numeric."
        ) from exc
    values = {
        "num_steps": raw_num_steps,
        "guidance": guidance,
        "shift": shift,
    }
    invalid = {
        name: value
        for name, value in values.items()
        if not math.isfinite(value) or value <= 0
    }
    if invalid:
        raise ValueError(
            "Forward-dynamics sampling values must be finite and positive, "
            f"got {invalid}."
        )
    if not raw_num_steps.is_integer():
        raise ValueError(
            f"Forward-dynamics num_steps must be an integer, got {raw_num_steps}."
        )
    return int(raw_num_steps), guidance, shift


class CosmosForwardDynamicsBackend:
    """Inference-only wrapper around the existing native Cosmos service."""

    def __init__(self, model_cfg: Any):
        self.action_dim = int(model_cfg.action_dim)
        self.action_chunk_size = int(model_cfg.num_action_chunks)
        self.cosmos_cfg = model_cfg.cosmos
        self.max_action_dim = int(self.cosmos_cfg.max_action_dim)
        self.domain_name = str(self.cosmos_cfg.domain_name)
        self.prompt = str(self.cosmos_cfg.prompt)
        self.fps = int(self.cosmos_cfg.fps)
        self.resolution = str(self.cosmos_cfg.get("resolution_tier", "720"))
        self._action_normalizer = _load_forward_dynamics_action_normalizer(
            self.cosmos_cfg
        )
        self._policy_loader = CosmosNativeInferencePolicy(model_cfg)
        self.last_action_normalization_stats: dict[str, torch.Tensor] = {}
        self._offloaded = False

    @property
    def loaded(self) -> bool:
        return self._policy_loader._service is not None

    def _service(self):
        service = self._policy_loader._get_service()
        model = getattr(service, "model", None)
        if model is None:
            self._policy_loader._service = None
            service = self._policy_loader._get_service()
            model = service.model
        self._configure_sampler(model)
        model.eval()
        if isinstance(model, torch.nn.Module):
            model.requires_grad_(False)
        self._offloaded = False
        return service

    def _configure_sampler(self, model: Any) -> None:
        """Apply the B sampler override exactly like official Cosmos inference."""
        requested = str(self.cosmos_cfg.get("sampler", "unipc")).lower()
        if requested not in {"unipc", "edm"}:
            raise ValueError(
                "Forward-dynamics sampler must be unipc or edm, got "
                f"{requested!r}."
            )
        model_config = getattr(model, "config", None)
        inference_config = getattr(
            model_config, "rectified_flow_inference_config", None
        )
        if inference_config is None:
            raise CosmosNativeBackendUnavailable(
                "Forward-dynamics Cosmos model is missing its inference sampler config."
            )
        current = str(inference_config.scheduler_type).lower()
        if current == requested:
            return
        inference_config.scheduler_type = requested
        setup_sampler = getattr(model, "set_up_scheduler_and_sampler", None)
        if not callable(setup_sampler):
            raise CosmosNativeBackendUnavailable(
                "Forward-dynamics Cosmos model cannot rebuild its inference sampler."
            )
        setup_sampler()

    @torch.no_grad()
    def generate(
        self,
        condition_video: torch.Tensor,
        actions: torch.Tensor,
        *,
        prompts: list[str] | None = None,
        seeds: list[int] | None = None,
    ) -> torch.Tensor:
        """Generate uint8 reference video ``[B,T,H,W,C]`` from actions."""
        if condition_video.ndim != 5 or condition_video.shape[1] not in (1, 3):
            raise ValueError(
                "condition_video must be [B,C,T,H,W], got "
                f"{tuple(condition_video.shape)}."
            )
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (
            self.action_chunk_size,
            self.action_dim,
        ):
            raise ValueError(
                "Forward-dynamics actions must have shape "
                f"[B,{self.action_chunk_size},{self.action_dim}], got "
                f"{tuple(actions.shape)}."
            )
        batch_size = int(actions.shape[0])
        if condition_video.shape[0] != batch_size:
            raise ValueError("Condition-video and action batch sizes must match.")
        prompts = prompts or [self.prompt] * batch_size
        seeds = seeds or [int(self.cosmos_cfg.get("seed", 0))] * batch_size
        if len(prompts) != batch_size or len(seeds) != batch_size:
            raise ValueError("prompts and seeds must contain one item per batch row.")

        num_steps, guidance, shift = _forward_dynamics_sampling_parameters(
            self.cosmos_cfg
        )
        service = self._service()
        normalizer = getattr(self, "_action_normalizer", None) or service
        normalized_actions = _normalize_forward_dynamics_actions(
            normalizer, actions
        )
        reduction_dims = (1, 2)
        self.last_action_normalization_stats = {
            "raw_min": actions.amin(dim=reduction_dims).detach().float().cpu(),
            "raw_max": actions.amax(dim=reduction_dims).detach().float().cpu(),
            "normalized_min": normalized_actions.amin(dim=reduction_dims)
            .detach()
            .float()
            .cpu(),
            "normalized_max": normalized_actions.amax(dim=reduction_dims)
            .detach()
            .float()
            .cpu(),
            "saturation_fraction": (
                normalized_actions.abs() >= 1.0 - 1.0e-6
            )
            .float()
            .mean(dim=reduction_dims)
            .detach()
            .cpu(),
        }
        action_api = importlib.import_module("cosmos_framework.inference.action")
        args_api = importlib.import_module("cosmos_framework.inference.args")
        libero_server = importlib.import_module(
            "cosmos_framework.scripts.action_policy_server_libero"
        )
        input_video_key = service._input_video_key()

        items = []
        for index in range(batch_size):
            padded_actions = torch.zeros(
                (self.action_chunk_size, self.max_action_dim), dtype=torch.float32
            )
            padded_actions[:, : self.action_dim] = (
                normalized_actions[index].detach().cpu()
            )
            items.append(
                action_api.build_action_batch(
                    video=condition_video[index].detach().cpu(),
                    action=padded_actions,
                    raw_action_dim=self.action_dim,
                    prompt=str(prompts[index]),
                    view_point="concat_view",
                    domain_name=self.domain_name,
                    model_mode=args_api.ModelMode.FORWARD_DYNAMICS,
                    action_chunk_size=self.action_chunk_size,
                    fps=self.fps,
                    resolution=self.resolution,
                    input_video_key=input_video_key,
                    batch_size=1,
                    device="cuda",
                )
            )
        batch = _merge_single_item_batches(items)

        with service._lock:
            samples = service.model.generate_samples_from_batch(
                batch,
                # These defaults intentionally match the B checkpoint's
                # official replay recipe. In particular, shift is a sampling
                # call argument; changing only the model sampler config leaves
                # generate_samples_from_batch at its unrelated default (5.0).
                guidance=guidance,
                # Cross-rank rollout seeds are int63; pinned Cosmos creates
                # NumPy RandomState instances and therefore requires uint32.
                seed=[int(seed) & 0xFFFFFFFF for seed in seeds],
                num_steps=num_steps,
                shift=shift,
                has_negative_prompt=False,
            )

        videos = []
        image_sizes = batch.get("image_size")
        for index, latent in enumerate(samples["vision"]):
            decoded = service.model.decode(latent)
            sample_image_size = (
                image_sizes[index : index + 1]
                if torch.is_tensor(image_sizes)
                else image_sizes
            )
            decoded = libero_server.remove_reflection_padding(
                decoded.squeeze(0), sample_image_size
            )
            future = _future_frames_from_decoded(
                decoded, action_chunk_size=self.action_chunk_size
            )
            frames = _decoded_video_to_uint8(future)
            videos.append(frames.permute(1, 2, 3, 0).cpu())
        result = torch.stack(videos, dim=0)
        if result.shape[:2] != (batch_size, self.action_chunk_size):
            raise CosmosNativeBackendUnavailable(
                f"Invalid forward-dynamics output shape {tuple(result.shape)}."
            )
        return result

    def offload(self) -> bool:
        """Drop B completely; unlike A, never build a trainable-state proxy."""
        if not self.loaded:
            return False
        service = self._policy_loader._service
        model = getattr(service, "model", None)
        if service is not None:
            service.model = None
            pipe = getattr(service, "pipe", None)
            if pipe is not None and getattr(pipe, "model", None) is model:
                pipe.model = None
        self._policy_loader._service = None
        object.__setattr__(self._policy_loader, "_native_service_model", None)
        self._policy_loader.native_trainable_proxy = None
        self._policy_loader._native_rollout_service_evicted = False
        self._offloaded = True
        del model, service
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    def onload(self) -> bool:
        """Reload the frozen B checkpoint after a pure inference offload."""
        if not self._offloaded:
            return False
        self._service()
        return True


def _normalize_forward_dynamics_actions(
    service: Any, actions: torch.Tensor
) -> torch.Tensor:
    """Map A's physical actions back to B's training normalization."""
    action_offset = getattr(service, "action_offset", None)
    action_scale = getattr(service, "action_scale", None)
    if action_offset is not None and action_scale is not None:
        action_offset = action_offset.to(device=actions.device, dtype=actions.dtype)
        action_scale = action_scale.to(device=actions.device, dtype=actions.dtype)
        if action_offset.numel() != actions.shape[-1]:
            raise ValueError(
                "Forward-dynamics affine stats width does not match actions: "
                f"{action_offset.numel()} vs {actions.shape[-1]}."
            )
        # Preserve B's exact ActionAffineNormalization arithmetic order.
        # The algebraically equivalent low/range formula differs by ~3e-5
        # in float32 for narrow rotation-6D channels.
        return ((actions - action_offset) / action_scale).clamp(-1.0, 1.0)

    normalization = str(getattr(service, "action_normalization", "none"))
    if normalization == "meanstd":
        mean = getattr(service, "action_mean", None)
        std = getattr(service, "action_std", None)
        if mean is None or std is None:
            return actions
        mean = mean.to(device=actions.device, dtype=actions.dtype)
        std = std.to(device=actions.device, dtype=actions.dtype)
        if mean.numel() != actions.shape[-1]:
            raise ValueError(
                "Forward-dynamics action stats width does not match actions: "
                f"{mean.numel()} vs {actions.shape[-1]}."
            )
        return ((actions - mean) / std).clamp(-1.0, 1.0)

    action_min = getattr(service, "action_min", None)
    action_range = getattr(service, "action_range", None)
    if action_min is None or action_range is None:
        return actions
    action_min = action_min.to(device=actions.device, dtype=actions.dtype)
    action_range = action_range.to(device=actions.device, dtype=actions.dtype)
    if action_min.numel() != actions.shape[-1]:
        raise ValueError(
            "Forward-dynamics action stats width does not match actions: "
            f"{action_min.numel()} vs {actions.shape[-1]}."
        )
    return (2.0 * (actions - action_min) / action_range - 1.0).clamp(-1.0, 1.0)


def _load_forward_dynamics_action_normalizer(cosmos_cfg: Any) -> Any | None:
    """Load B's conditioning stats, including the mixed close-laptop format."""
    stats_path = str(
        cosmos_cfg.get("forward_dynamics_action_stats_path", "") or ""
    )
    if not stats_path:
        config_path = str(cosmos_cfg.get("checkpoint_config_path", "") or "")
        if config_path and Path(config_path).is_file():
            import yaml

            config = yaml.safe_load(Path(config_path).read_text())
            datasets = (
                config.get("dataloader_train", {})
                .get("dataloader", {})
                .get("datasets", {})
            )
            for entry in datasets.values():
                candidate = entry.get("dataset", {}).get("action_stats_path", "")
                if candidate:
                    stats_path = str(candidate)
                    break
    if not stats_path:
        return None
    path = Path(stats_path).expanduser()
    if not path.is_file():
        raise ValueError(f"Forward-dynamics action stats are not readable: {path}")
    stats = json.loads(path.read_text())
    if "effective_low" in stats and "effective_high" in stats:
        low, high = stats["effective_low"], stats["effective_high"]
    else:
        stats = stats.get("global_raw", stats.get("global", stats))
        low = stats.get("q01", stats.get("min"))
        high = stats.get("q99", stats.get("max"))
    if low is None or high is None:
        raise ValueError(f"Unsupported forward-dynamics action stats format: {path}")
    action_min = torch.tensor(low, dtype=torch.float32)
    action_max = torch.tensor(high, dtype=torch.float32)
    expected_dim = int(cosmos_cfg.get("raw_action_dim", action_min.numel()))
    if action_min.ndim != 1 or action_max.ndim != 1 or (
        action_min.numel() != expected_dim or action_max.numel() != expected_dim
    ):
        raise ValueError(
            "Forward-dynamics action stats width does not match raw_action_dim: "
            f"low={tuple(action_min.shape)}, high={tuple(action_max.shape)}, "
            f"raw_action_dim={expected_dim}."
        )
    action_range = action_max - action_min
    action_offset = (action_max + action_min) / 2.0
    action_scale = action_range / 2.0
    if not torch.isfinite(action_min).all() or not torch.isfinite(action_max).all():
        raise ValueError(
            f"Forward-dynamics action stats contain non-finite values: {path}"
        )
    if torch.any(action_range <= 0):
        raise ValueError(
            f"Forward-dynamics action stats contain empty ranges: {path}"
        )
    return type(
        "ForwardDynamicsActionNormalizer",
        (),
        {
            "action_normalization": "mixed",
            "action_offset": action_offset,
            "action_scale": action_scale,
            "action_min": action_min,
            "action_range": action_range,
        },
    )()
