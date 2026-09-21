"""Edge 4B native Cosmos policy backend for Ctrl-World GRPO.

This module keeps the legacy VFM backend untouched.  It loads the Edge
generator/reasoner DCP through its RoboLab service and preserves normalized
actions inside the diffusion replay chain.  Only the environment-facing action
is denormalized through the SFT run's mixed affine statistics.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
from pathlib import Path
import sys
import types
from typing import Any

import torch

from rlinf.models.embodiment.cosmos.cosmos_backend import (
    COSMOS_FRAMEWORK_IMPORTS,
    CosmosNativeBackendProbe,
    CosmosNativeBackendUnavailable,
    CosmosNativeInferencePolicy,
    CosmosNativeActionTraceSampler,
    _NUMPY_RANDOM_STATE_MAX_SEED,
    _cfg_get,
    _expand_path,
    _generate_samples_from_batch_with_grad,
    _ensure_cosmos_dcp_hybrid_process_group,
    _maybe_add_framework_path,
    _matches_any_parameter_pattern,
    _patch_cosmos_single_rank_sync_model_states,
    _require_direct_action_shape,
    _slice_batched_tensors,
    _build_action_training_transform_policy_batch,
    compute_native_recorded_action_logprobs,
    extract_native_action_replay_tensors,
)
from rlinf.utils.logging import get_logger


EDGE_REQUIRED_IMPORTS = (
    "cosmos_framework",
    "cosmos_framework.scripts.action_policy_server_robolab",
    "cosmos_framework.model.generator.omni_mot_model",
)


def _edge_missing_imports() -> tuple[str, ...]:
    missing: list[str] = []
    for module_name in EDGE_REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    return tuple(missing)


def _patch_python313_pathlib_pickle_compat() -> None:
    """Expose Python 3.13 pathlib pickle names to the RL Python 3.11 runtime."""

    if sys.version_info >= (3, 13) or "pathlib._local" in sys.modules:
        return
    local = types.ModuleType("pathlib._local")
    for name in (
        "PurePath",
        "PurePosixPath",
        "PureWindowsPath",
        "Path",
        "PosixPath",
        "WindowsPath",
    ):
        setattr(local, name, getattr(pathlib, name))
    sys.modules["pathlib._local"] = local


def _patch_edge_replay_condition_encoding(model: torch.nn.Module) -> None:
    """Keep frozen vision-tokenizer work outside the action replay graph."""

    original = model._encode_vision_x0_tokens
    if getattr(original, "_rlinf_edge_condition_no_grad", False):
        return

    def encode_vision_x0_tokens(*args, **kwargs):
        if torch.is_grad_enabled():
            with torch.no_grad():
                return original(*args, **kwargs)
        return original(*args, **kwargs)

    encode_vision_x0_tokens._rlinf_edge_condition_no_grad = True
    model._encode_vision_x0_tokens = encode_vision_x0_tokens


def _patch_edge_fsdp_mixed_precision() -> None:
    """Keep trainable Edge shards in FP32 while computing the model in BF16."""

    from torch.distributed.fsdp import MixedPrecisionPolicy

    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    module_names = (
        "cosmos_framework.model.generator.mot.parallelize_unified_mot",
        "cosmos_framework.model.generator.mot.parallelize_vfm_network",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        original = getattr(
            module.fully_shard,
            "_rlinf_edge_original_fully_shard",
            module.fully_shard,
        )

        def fully_shard_with_mixed_precision(
            *args,
            _original=original,
            **kwargs,
        ):
            kwargs.setdefault("mp_policy", policy)
            return _original(*args, **kwargs)

        fully_shard_with_mixed_precision._rlinf_edge_original_fully_shard = (
            original
        )
        module.fully_shard = fully_shard_with_mixed_precision


def _trace_edge_trainable_dtypes(
    model: torch.nn.Module,
    *,
    trainable_patterns: tuple[str, ...],
    blocked_patterns: tuple[str, ...],
    stage: str,
) -> None:
    """Diagnostic-only: log the dtype of every trainable canonical shard.

    Temporary instrumentation for tracing job 604561's "canonical shards
    must be FP32" failure across the three stages where the dtype could be
    silently reset (FP32 cast -> parallelize_vfm_network -> DCP load). Safe
    to remove once the root cause is confirmed and fixed.
    """

    logger = get_logger()
    dtypes: dict[str, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        requested = _matches_any_parameter_pattern(name, trainable_patterns)
        blocked = _matches_any_parameter_pattern(name, blocked_patterns)
        if requested and not blocked:
            dtypes.setdefault(str(parameter.dtype), []).append(name)
    summary = {dtype: len(names) for dtype, names in dtypes.items()}
    logger.info(
        "[EDGE4B-DTYPE-TRACE] stage=%s trainable_param_dtypes=%s",
        stage,
        summary,
    )
    if len(dtypes) > 1 or (dtypes and "torch.float32" not in dtypes):
        sample = next(iter(dtypes.values()))[:3]
        logger.warning(
            "[EDGE4B-DTYPE-TRACE] stage=%s unexpected dtype mix; sample_params=%s",
            stage,
            sample,
        )


def _recast_edge_trainable_params_to_fp32(
    model: torch.nn.Module,
    *,
    trainable_patterns: tuple[str, ...],
    blocked_patterns: tuple[str, ...],
) -> None:
    """Re-apply the FP32 cast after Edge's own model construction resets it.

    ``OmniMoTModel``'s network builder does ``net = net.to(dtype=dtype)``
    immediately after ``parallelize_vfm_network`` returns (meta-device
    materialization: cast, ``to_empty``, ``init_weights``), unconditionally
    downcasting every parameter -- including the ones this backend already
    cast to FP32 pre-fully_shard -- to the model's BF16 compute dtype. The
    DCP load that follows preserves whatever dtype the destination tensor
    has at that point (``target_tensor.copy_(tensor)`` in
    ``torch.distributed.checkpoint``), so the trained values arrive as BF16.
    Re-casting here after ``Service(args)`` returns only widens the storage
    dtype for the already-loaded values; it does not need to run before DCP
    load since copy_ is dtype-preserving in either direction.

    A raw ``parameter.data = ...`` assignment does not work for a DTensor:
    DTensor is a wrapper-subclass tensor that keeps its real state
    (``_local_tensor``, ``_spec``) in Python-level ``__dict__`` attributes,
    not in the base TensorImpl that ``.data =`` swaps. So a ``.data =``
    reassignment flips what ``parameter.dtype`` *reports* (read off the
    swapped-in TensorImpl) while leaving the parameter's actual
    ``_local_tensor`` -- what FSDP2's compute/gradient/optimizer paths
    actually read -- silently stuck on the old (BF16) storage: exactly the
    "reports FP32, computes as BF16" split that later surfaced as
    ``RuntimeError: Tensors of the same index must be on the same device and
    the same dtype`` out of ``AdamW.step()``.  ``nn.Module._apply`` (which is
    what the framework's own working ``net.to(dtype=bf16)`` goes through)
    knows this and uses ``torch.utils.swap_tensors`` instead of ``.data =``
    whenever the parameter is a traceable wrapper subclass (DTensor
    qualifies) -- swap_tensors exchanges the *entire* tensor state,
    ``__dict__`` included, while preserving the Python object identity that
    ``model.named_parameters()`` and the optimizer's param references rely
    on. This mirrors that exact mechanism instead of reinventing it, then
    calls ``FSDPParam.reset_sharded_param()`` (also normally invoked by
    ``FSDPModule._apply``) so FSDP2's own bookkeeping re-syncs to the swapped
    storage on every FSDPModule in the tree.
    """

    from torch.distributed.fsdp import FSDPModule

    for name, parameter in model.named_parameters(remove_duplicate=False):
        requested = _matches_any_parameter_pattern(name, trainable_patterns)
        blocked = _matches_any_parameter_pattern(name, blocked_patterns)
        if requested and not blocked and parameter.dtype != torch.float32:
            with torch.no_grad():
                fp32_param = torch.nn.Parameter(
                    parameter.to(dtype=torch.float32),
                    requires_grad=parameter.requires_grad,
                )
                if parameter.grad is not None:
                    parameter.grad = None
                torch.utils.swap_tensors(parameter, fp32_param)

    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, FSDPModule):
                continue
            fsdp_param_group = module._get_fsdp_state()._fsdp_param_group
            if fsdp_param_group is None:
                continue
            for fsdp_param in fsdp_param_group.fsdp_params:
                fsdp_param.reset_sharded_param()


def _patch_edge_pre_fsdp_freeze(
    *,
    trainable_patterns: tuple[str, ...],
    blocked_patterns: tuple[str, ...],
    activation_checkpointing_mode: str | None,
) -> None:
    """Select the action path before Edge FSDP turns parameters into DTensors."""

    omni = importlib.import_module(
        "cosmos_framework.model.generator.omni_mot_model"
    )
    original = getattr(
        omni.parallelize_vfm_network,
        "_rlinf_edge_original_parallelize_vfm_network",
        omni.parallelize_vfm_network,
    )

    def wrapped(model, *args, **kwargs):
        for name, parameter in model.named_parameters(remove_duplicate=False):
            requested = _matches_any_parameter_pattern(name, trainable_patterns)
            blocked = _matches_any_parameter_pattern(name, blocked_patterns)
            should_train = requested and not blocked
            parameter.requires_grad_(should_train)
            if should_train and parameter.dtype != torch.float32:
                parameter.data = parameter.data.to(dtype=torch.float32)
        _trace_edge_trainable_dtypes(
            model,
            trainable_patterns=trainable_patterns,
            blocked_patterns=blocked_patterns,
            stage="1_after_fp32_cast_before_parallelize",
        )
        if activation_checkpointing_mode is not None:
            ac_config = kwargs.get("ac_config")
            if ac_config is not None:
                ac_config.mode = activation_checkpointing_mode
        result = original(model, *args, **kwargs)
        _trace_edge_trainable_dtypes(
            result,
            trainable_patterns=trainable_patterns,
            blocked_patterns=blocked_patterns,
            stage="2_after_parallelize_vfm_network",
        )
        return result

    wrapped._rlinf_edge_original_parallelize_vfm_network = original
    omni.parallelize_vfm_network = wrapped


class _EdgeUR5PolicyService:
    """Thin wrapper around Edge RoboLab service with exact UR5 preprocessing."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        stats_path: Path,
        output_dir: Path,
        chunk_length: int,
        fps: int,
        guidance: float,
        num_steps: int,
        shift: float,
        view_layout: str,
        internal_fsdp_shard: bool,
    ) -> None:
        robolab = importlib.import_module(
            "cosmos_framework.scripts.action_policy_server_robolab"
        )
        action_processing = importlib.import_module(
            "cosmos_framework.data.generator.action.action_processing"
        )
        payload = json.loads(stats_path.read_text())
        low = torch.tensor(payload["effective_low"], dtype=torch.float32)
        high = torch.tensor(payload["effective_high"], dtype=torch.float32)
        methods = payload.get("method_by_dim")
        if low.shape != (10,) or high.shape != (10,) or methods != ["quantile"] * 9 + ["minmax"]:
            raise ValueError(f"Invalid UR5 Edge mixed stats: {stats_path}")
        if not torch.all(high > low):
            raise ValueError(f"Non-increasing UR5 Edge mixed stats: {stats_path}")
        self._normalizer = action_processing.ActionAffineNormalization(
            offset=(high + low) / 2.0,
            scale=(high - low) / 2.0,
            forward_clamp=(-1.0, 1.0),
        )
        transforms = importlib.import_module(
            "cosmos_framework.data.generator.action.transforms"
        )
        self._robolab = robolab
        cosmos_package = importlib.import_module("cosmos_framework")
        framework_root = Path(cosmos_package.__file__).resolve().parent.parent
        nemotron_config = (
            framework_root
            / "cosmos_framework"
            / "model"
            / "generator"
            / "reasoner"
            / "nemotron_3_dense_vl"
            / "configs"
            / "Nemotron-2B-Dense-VL.json"
        )
        if not nemotron_config.is_file():
            raise FileNotFoundError(
                f"Edge Nemotron model config is missing: {nemotron_config}"
            )

        class Service(robolab.RobolabPolicyService):
            def _build_setup_args(service_self, args):
                setup = super(Service, service_self)._build_setup_args(args)
                return setup.model_copy(
                    update={
                        "guardrails": False,
                        "use_torch_compile": False,
                        "dp_replicate_size": 1,
                        "dp_shard_size": dp_shard,
                        "cp_size": 1,
                        "cfgp_size": 1,
                    }
                )

            def _build_transform(service_self, training_config, args):
                if training_config is None:
                    raise RuntimeError(
                        "Edge RL requires the original UR5 Edge training config."
                    )
                transform = transforms.ActionTransformPipeline(
                    tokenizer_config=(
                        training_config.dataloader_train.dataloader.datasets
                        .ur5_edge.dataset.tokenizer_config
                    ),
                    cfg_dropout_rate=0.0,
                    max_action_dim=64,
                    append_viewpoint_info=True,
                    append_duration_fps_timestamps=True,
                    append_resolution_info=True,
                    append_idle_frames=False,
                    format_prompt_as_json=True,
                )
                return transform, {
                    "action_chunk_size": chunk_length,
                    "conditioning_fps": float(fps),
                    "resolution": "480",
                }

        dp_shard = 4 if internal_fsdp_shard else 1
        args = robolab.RobolabServerArgs(
            checkpoint_path=checkpoint_path,
            allow_dcp_checkpoint=True,
            experiment="action_policy_ur5_eef_edge",
            experiment_overrides=[
                f"model.config.parallelism.data_parallel_shard_degree={dp_shard}",
                "model.config.parallelism.data_parallel_replicate_degree=1",
                "model.config.parallelism.context_parallel_shard_degree=1",
                "model.config.parallelism.cfg_parallel_shard_degree=1",
                f"model.config.tokenizer.encode_exact_durations=[{chunk_length + 1}]",
                "model.config.vlm_config.model_instance.config.base_config.json_file="
                f"{nemotron_config}",
                (
                    "dataloader_train.dataloader.datasets.ur5_edge.dataset."
                    f"chunk_length={chunk_length}"
                ),
                f"dataloader_train.dataloader.datasets.ur5_edge.dataset.view_layout={view_layout}",
                "checkpoint.hf_export.enabled=false",
            ],
            output_dir=output_dir,
            sampler="unipc",
            seed=0,
            deterministic_seed=True,
            guidance=guidance,
            num_steps=num_steps,
            shift=shift,
            domain_name="robomind-ur",
            decode_video=False,
            resolution="480",
            conditioning_fps=float(fps),
            action_chunk_size=chunk_length,
            action_dim=10,
            image_height=720,
            image_width=640,
            action_space="midtrain",
            use_state=False,
            history_length=0,
            format_prompt_as_json=True,
        )
        self.service = Service(args)
        self.model = self.service.model
        _patch_edge_replay_condition_encoding(self.model)
        # Expose the inner inference pipe so the inherited eviction lifecycle
        # clears every model alias between rollout and GRPO replay.
        self.pipe = self.service.pipe
        self.cfg = self.service.cfg
        self.raw_action_dim = 10
        self._remove_padding = transforms.remove_reflection_padding
        self._lock = self.service._lock

    def build_batch(self, image, prompt: str, domain_id: int) -> dict[str, Any]:
        return _build_action_training_transform_policy_batch(
            transform=self.service._transform,
            batch_builder=self._robolab._build_data_batch_from_sample,
            image=image,
            prompt=prompt,
            domain_id=domain_id,
            action_chunk_size=self.cfg.action_chunk_size,
            raw_action_dim=10,
            conditioning_fps=int(self.cfg.conditioning_fps),
            resolution_tier="480",
            condition_state=None,
            action_state_rows=0,
        )

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        return self._normalizer.denormalize_action(action.clamp(-1.0, 1.0))

    def decoded_content(self, samples: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor:
        decoded = self.model.decode(samples["vision"][0]).squeeze(0)
        return self._remove_padding(decoded, batch["image_size"])


class CosmosEdge4BInferencePolicy(CosmosNativeInferencePolicy):
    """Native 4B Edge policy with 10D normalized UR5 relative EEF actions."""

    def __init__(self, cfg: Any, torch_dtype=None):
        # eager_load_trainable_state makes super().__init__() eagerly call
        # ensure_native_trainable_state() -> _get_service() -> the full
        # Service(args) construction *inside itself*, before this method's own
        # body ever runs -- so this sys.path insertion must happen BEFORE
        # super().__init__(), not after. The base class's own
        # probe_cosmos_native_backend(cfg) call (which inserts
        # cosmos.framework_path the same way) only runs partway through that
        # same super().__init__(), which is early enough; ours has to run
        # even earlier since we don't own that call site. Config-carried
        # (not inherited-env-carried) is what makes this survive Ray actor
        # construction, where OS env vars like PYTHONPATH are not reliably
        # propagated (see rlinf.scheduler.cluster.Cluster's curated
        # runtime_env allowlist). Edge4B additionally needs the collaborator2-owned
        # third_party/lerobot shim on sys.path for the same reason; nothing
        # else adds it, so add it the same way here.
        _maybe_add_framework_path(_cfg_get(cfg.get("cosmos", {}), "third_party_path"))
        # Same "must run before super().__init__()" constraint as the sys.path
        # insertion above: eager_load_trainable_state makes super().__init__()
        # eagerly call dcp.load() (via ensure_native_trainable_state ->
        # _get_service()) before this method's own body resumes. dcp.load()'s
        # load-plan rendezvous (gather_object) needs torch.distributed
        # initialized; rollout workers (unlike the FSDP actor, which already
        # initializes this in FSDPModelManager.__init__ before model
        # construction) have no earlier call site that does this, so without
        # this, PyTorch's own implicit init would pick a plain "nccl" backend.
        # That is fine for GPU tensor collectives but fragile for
        # gather_object's CPU-side pickled-bytes collective once ranks span
        # multiple physical nodes -- see fsdp_model_manager.py's matching fix
        # for the full "NCCL Error 2: unhandled system error" root cause.
        _ensure_cosmos_dcp_hybrid_process_group()
        super().__init__(cfg=cfg, torch_dtype=torch_dtype)
        if self.action_representation != "ur5_eef_relative_10d_edge4b":
            raise ValueError(
                "Edge 4B backend requires action_representation="
                "'ur5_eef_relative_10d_edge4b'."
            )
        if self.action_state_rows:
            raise ValueError("Edge 4B UR5 policy must run without observation.state.")
        self.probe = self._edge_probe()

    def _edge_probe(self) -> CosmosNativeBackendProbe:
        checkpoint_path = _cfg_get(self.cosmos_cfg, "checkpoint_path", None)
        if checkpoint_path is None:
            checkpoint_path = _cfg_get(self.cfg, "model_path", None)
        path = _expand_path(checkpoint_path)
        return CosmosNativeBackendProbe(
            in_slurm_allocation=bool(os.environ.get("SLURM_JOB_ID")),
            cuda_available=torch.cuda.is_available(),
            cuda_device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            cuda_error=None,
            missing_imports=_edge_missing_imports(),
            checkpoint_path=str(path) if path else None,
            checkpoint_path_exists=bool(path and path.exists()),
            checkpoint_name=None,
            hf_token_available=False,
        )

    def _build_joint_action_training_batch(
        self,
        *,
        service: _EdgeUR5PolicyService,
        image: torch.Tensor,
        prompt: str,
        domain_name: str,
        condition_state: torch.Tensor | None,
        resolution_tier: str | int | None = None,
    ) -> dict[str, Any]:
        del condition_state, resolution_tier
        if domain_name != "robomind-ur":
            raise ValueError(f"Unexpected Edge domain {domain_name!r}.")
        return service.build_batch(image, prompt, domain_id=13)

    def _sample_joint_policy(
        self,
        *,
        service: _EdgeUR5PolicyService,
        image: torch.Tensor,
        prompt: str,
        domain_name: str,
        condition_state: torch.Tensor | None,
        chain_logprob: bool,
        vision_seed: int,
        action_seed: int,
    ):
        del condition_state
        if vision_seed != action_seed:
            raise ValueError(
                "Edge 4B currently requires shared vision/action sampling seeds."
            )
        # Cross-rank grouping derives seeds via a 63-bit stable hash
        # (rlinf.algorithms.cross_rank_grpo.derive_stable_seed), but the
        # sampler's own RNG (numpy SeedSequence) requires 0..2**32-1. Mask
        # rather than re-hash so vision_seed stays exactly equal to
        # action_seed after this line.
        vision_seed = int(vision_seed) & _NUMPY_RANDOM_STATE_MAX_SEED
        action_seed = vision_seed
        batch = self._build_joint_action_training_batch(
            service=service,
            image=image,
            prompt=prompt,
            domain_name=domain_name,
            condition_state=None,
        )
        recorder = None
        original = service.model.generate_samples_from_batch
        if chain_logprob:
            recorder = CosmosNativeActionTraceSampler(
                getattr(service.model, "sampler", None),
                chain_sigma=float(_cfg_get(self.cosmos_cfg, "chain_sigma", 0.2)),
                sigma_min=float(_cfg_get(self.cosmos_cfg, "chain_sigma_min", 1e-4)),
            )

            def generate_with_recorder(data_batch, *args, **kwargs):
                kwargs["sampler"] = recorder
                return original(data_batch, *args, **kwargs)

            service.model.generate_samples_from_batch = generate_with_recorder
        try:
            with service._lock, torch.no_grad():
                samples = service.model.generate_samples_from_batch(
                    batch,
                    guidance=float(_cfg_get(self.cosmos_cfg, "guidance", 3.0)),
                    seed=[int(vision_seed)],
                    num_steps=int(_cfg_get(self.cosmos_cfg, "num_steps", 10)),
                    shift=float(_cfg_get(self.cosmos_cfg, "shift", 5.0)),
                    has_negative_prompt=False,
                )
        finally:
            service.model.generate_samples_from_batch = original

        normalized = samples["action"][0].float().squeeze(0)
        if normalized.ndim != 2 or normalized.shape[0] < self.num_action_chunks:
            raise CosmosNativeBackendUnavailable(
                f"Edge action shape is invalid: {tuple(normalized.shape)}."
            )
        actions = service.denormalize(
            normalized[: self.num_action_chunks, : self.action_dim]
        ).unsqueeze(0).to(image.device)
        decoded = service.decoded_content(samples, batch)
        video = ((decoded.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
        video = video.permute(1, 2, 3, 0)[1 : 1 + self.num_action_chunks]
        if video.shape[0] != self.num_action_chunks:
            raise CosmosNativeBackendUnavailable(
                "Edge response did not contain one future frame per action."
            )
        replay_tensors = None
        prev_logprobs = None
        if recorder is not None:
            if recorder.record is None:
                raise CosmosNativeBackendUnavailable(
                    "Edge sampler did not record its action denoise chain."
                )
            replay_tensors = extract_native_action_replay_tensors(
                recorder.record,
                num_action_chunks=self.num_action_chunks,
                raw_action_dim=self.action_dim,
                max_action_dim=int(self.max_action_dim),
            )
            prev_logprobs = compute_native_recorded_action_logprobs(
                replay_tensors,
                sigma_min=float(_cfg_get(self.cosmos_cfg, "chain_sigma_min", 1e-4)),
                logprob_mode=str(
                    _cfg_get(self.cosmos_cfg, "chain_logprob_mode", "joint_mean")
                ),
            )
        return actions, video.unsqueeze(0).to(image.device), replay_tensors, prev_logprobs

    def _get_service(self):
        if self._service is not None:
            return self._service
        # Edge invokes VAE state synchronization even for a one-rank smoke.
        # Reuse the native backend guard so local preflight avoids NCCL collectives.
        _patch_cosmos_single_rank_sync_model_states()
        _patch_python313_pathlib_pickle_compat()
        _patch_edge_fsdp_mixed_precision()
        activation_checkpointing_mode = str(
            _cfg_get(self.cosmos_cfg, "activation_checkpointing", "full")
        ).lower()
        if activation_checkpointing_mode == "none":
            activation_checkpointing_mode = None
        _patch_edge_pre_fsdp_freeze(
            trainable_patterns=self.trainable_param_patterns,
            blocked_patterns=self.blocked_trainable_param_patterns,
            activation_checkpointing_mode=activation_checkpointing_mode,
        )
        checkpoint_path = _expand_path(
            _cfg_get(self.cosmos_cfg, "checkpoint_path", None)
        )
        stats_path = _expand_path(_cfg_get(self.cosmos_cfg, "action_stats_path", None))
        if checkpoint_path is None or not (checkpoint_path / ".metadata").is_file():
            raise CosmosNativeBackendUnavailable(
                "Edge 4B requires a direct DCP model directory with .metadata."
            )
        if stats_path is None or not stats_path.is_file():
            raise CosmosNativeBackendUnavailable(
                "Edge 4B requires its mixed action-normalization statistics."
            )
        output_dir = _expand_path(_cfg_get(self.cosmos_cfg, "output_dir", None))
        if output_dir is None:
            output_dir = Path.cwd() / "logs" / "cosmos_edge4b"
        output_dir.mkdir(parents=True, exist_ok=True)
        self._service = _EdgeUR5PolicyService(
            checkpoint_path=str(checkpoint_path),
            stats_path=stats_path,
            output_dir=output_dir,
            chunk_length=self.num_action_chunks,
            fps=int(_cfg_get(self.cosmos_cfg, "fps", 15)),
            guidance=float(_cfg_get(self.cosmos_cfg, "guidance", 3.0)),
            num_steps=int(_cfg_get(self.cosmos_cfg, "num_steps", 10)),
            shift=float(_cfg_get(self.cosmos_cfg, "shift", 5.0)),
            view_layout=str(_cfg_get(self.cosmos_cfg, "view_layout", "edge_droid_full")),
            internal_fsdp_shard=bool(
                _cfg_get(self.cosmos_cfg, "internal_fsdp_shard", True)
            ),
        )
        _trace_edge_trainable_dtypes(
            self._service.model,
            trainable_patterns=self.trainable_param_patterns,
            blocked_patterns=self.blocked_trainable_param_patterns,
            stage="3_after_service_dcp_load",
        )
        _recast_edge_trainable_params_to_fp32(
            self._service.model,
            trainable_patterns=self.trainable_param_patterns,
            blocked_patterns=self.blocked_trainable_param_patterns,
        )
        _trace_edge_trainable_dtypes(
            self._service.model,
            trainable_patterns=self.trainable_param_patterns,
            blocked_patterns=self.blocked_trainable_param_patterns,
            stage="4_after_post_load_fp32_recast",
        )
        torch.set_grad_enabled(True)
        self._native_rollout_service_evicted = False
        get_logger().info(
            "Loaded Edge 4B DCP policy: chunk=%s fps=%s guidance=%s steps=%s",
            self.num_action_chunks,
            _cfg_get(self.cosmos_cfg, "fps", 15),
            _cfg_get(self.cosmos_cfg, "guidance", 3.0),
            _cfg_get(self.cosmos_cfg, "num_steps", 10),
        )
        return self._service
