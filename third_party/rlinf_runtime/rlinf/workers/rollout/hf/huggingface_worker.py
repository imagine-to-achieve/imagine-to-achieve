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

import copy
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal, TextIO

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from tqdm.auto import tqdm

from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import (
    RolloutResult,
)
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.logging import format_duration, format_hardware_snapshot, get_logger
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import clear_memory


class _TeeProgressStream:
    """Mirror tqdm control bytes to Ray stderr and a clean progress file."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(stream.isatty() for stream in self.streams)


def _open_eval_progress_stream(
    rank: int,
) -> tuple[TextIO | _TeeProgressStream, TextIO | None]:
    """Open the optional tqdm-only log on the single emitting rank."""
    if rank != 0:
        return sys.stderr, None
    progress_path = os.environ.get("RLINF_TQDM_LOG_FILE")
    if not progress_path:
        return sys.stderr, None
    path = Path(progress_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    progress_file = path.open("a", buffering=1)
    return _TeeProgressStream(sys.stderr, progress_file), progress_file


def _load_full_state_dict(model: torch.nn.Module, state_dict: dict) -> None:
    """Load a full (non-sharded) state dict into ``model``.

    Plain ``nn.Module.load_state_dict`` has no DTensor awareness and
    hard-crashes on any parameter the model shards internally (e.g. a
    Cosmos native backbone parallelized via ``internal_fsdp_shard``).
    ``set_model_state_dict`` is the same DCP API used for actor checkpoint
    resume and transparently handles both plain and DTensor-backed params.
    """
    set_model_state_dict(
        model,
        model_state_dict=state_dict,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
    )


def _get_sparse_eval_batch_size(
    total_num_eval_envs: int,
    eval_mapping_world_size: int,
    rank: int,
    num_pipeline_stages: int,
) -> int:
    """Return the rank-local eval batch without activating unused ranks."""
    if total_num_eval_envs == 0 or rank >= eval_mapping_world_size:
        return 0
    eval_parallel_size = eval_mapping_world_size * num_pipeline_stages
    if total_num_eval_envs % eval_parallel_size == 0:
        return total_num_eval_envs // eval_parallel_size
    if num_pipeline_stages != 1:
        raise ValueError("Sparse evaluation currently requires pipeline_stage_num=1.")
    return CommMapper.get_rank_batch_size(
        total_num_eval_envs,
        eval_mapping_world_size,
        rank,
    )


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = self.torch_platform.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size
        self.rollout_epoch = cfg.algorithm.get("rollout_epoch", 1)
        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.expert_model = None

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas,
            accel_min_ctas=min_ctas,
            disable_same_device_ipc=bool(
                cfg.rollout.get("sync_weight_disable_same_device_ipc", False)
            ),
        )
        self.total_num_train_envs = cfg.env.train.total_num_envs
        self.total_num_eval_envs = cfg.env.eval.total_num_envs
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num

        eval_collective_group_size = int(
            cfg.rollout.get("sparse_eval_collective_group_size", 1)
        )
        self.eval_mapping_world_size = (
            CommMapper.get_collective_aligned_world_size(
                int(self.total_num_eval_envs),
                self._world_size,
                eval_collective_group_size,
            )
            if int(self.total_num_eval_envs) > 0
            else 0
        )
        self.train_batch_size = (
            self.total_num_train_envs // self._world_size // self.num_pipeline_stages
        )
        self.eval_batch_size = _get_sparse_eval_batch_size(
            int(self.total_num_eval_envs),
            self.eval_mapping_world_size,
            self._rank,
            self.num_pipeline_stages,
        )
        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)
        self.enable_eval = cfg.runner.val_check_interval > 0 or cfg.runner.only_eval

        self.n_train_chunk_steps = (
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            cfg.env.eval.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.version = 0
        self.finished_episodes = None
        self._pre_eval_rng_state: dict[str, Any] | None = None
        self._post_update_eval_context: dict[str, Any] | None = None

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model: BasePolicy = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            _load_full_state_dict(self.hf_model, model_dict)

        if self.cfg.rollout.get("expert_model", None):
            expert_model_config = copy.deepcopy(self.cfg.actor.model)
            with open_dict(expert_model_config):
                expert_model_config.precision = self.cfg.rollout.expert_model.precision
                expert_model_config.model_path = (
                    self.cfg.rollout.expert_model.model_path
                )
            self.expert_model = get_model(expert_model_config)

            if self.cfg.runner.get("expert_ckpt_path", None):
                expert_model_dict = torch.load(self.cfg.runner.expert_ckpt_path)
                _load_full_state_dict(self.expert_model, expert_model_dict)

        self.hf_model.eval()
        if self.expert_model is not None:
            self.expert_model.eval()

        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            self.hf_model.enable_torch_compile(mode=mode)
        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=max(self.eval_batch_size, 1),
            )

        self.dst_ranks = {
            "train": self._setup_dst_ranks(
                self.total_num_train_envs // self.num_pipeline_stages
            ),
        }
        self.src_ranks = {
            "train": self._setup_src_ranks(
                self.total_num_train_envs // self.num_pipeline_stages
            ),
        }
        if self.enable_eval:
            self.dst_ranks["eval"] = self._setup_dst_ranks(
                self.total_num_eval_envs // self.num_pipeline_stages,
                active_world_size=self.eval_mapping_world_size,
            )
            self.src_ranks["eval"] = self._setup_src_ranks(
                self.total_num_eval_envs // self.num_pipeline_stages,
                active_world_size=self.eval_mapping_world_size,
            )

        self.log_info(f"Rollout worker initialized with dst_ranks: {self.dst_ranks}")
        self.log_info(f"Rollout worker initialized with src_ranks: {self.src_ranks}")
        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"]
            if self._sampling_params["do_sample"]
            else 1.0,
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        self._eval_sampling_params = {
            "do_sample": True
            if self._sampling_params.get("temperature_eval", -1) > 0
            else False,
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        if self.expert_model is not None:
            self._dagger_sampling_params = {
                "beta": self.cfg.algorithm.get("dagger", {}).get("init_beta", 0.5),
                "beta_schedule": self.cfg.algorithm.get("dagger", {}).get(
                    "beta_schedule", "exponential"
                ),
                "beta_min": self.cfg.algorithm.get("dagger", {}).get("beta_min", 0.05),
                "beta_decay": self.cfg.algorithm.get("dagger", {}).get(
                    "beta_decay", 0.99
                ),
            }

    def update_dagger_beta(self):
        if self.expert_model is None:
            return

        if self._dagger_sampling_params["beta_schedule"] == "exponential":
            self._dagger_sampling_params["beta"] = max(
                self._dagger_sampling_params["beta_min"],
                self._dagger_sampling_params["beta"]
                * self._dagger_sampling_params["beta_decay"],
            )
        else:
            raise NotImplementedError(
                f"Beta schedule {self._dagger_sampling_params['beta_schedule']} is not implemented"
            )

    def _setup_dst_ranks(
        self, batch_size: int, *, active_world_size: int | None = None
    ) -> list[tuple[int, int]]:
        """Compute env peer ranks for this rollout worker.

        This mapping supports both one-to-many and many-to-one env/rollout layouts.
        The returned ranks are used as communication counterparts for receiving env
        outputs and sending action chunks.

        Args:
            batch_size: Total env batch size per pipeline stage across all workers.

        Returns:
            Ordered ``(env_rank, batch_size)`` tuples this rollout worker should
            send action chunks to.
        """
        env_world_size = self.placement.get_world_size("env")
        rollout_world_size = self.placement.get_world_size("rollout")
        return CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=rollout_world_size,
            dst_world_size=env_world_size,
            src_rank=self._rank,
            src_active_world_size=active_world_size,
            dst_active_world_size=active_world_size,
        )

    def _setup_src_ranks(
        self, batch_size: int, *, active_world_size: int | None = None
    ) -> list[tuple[int, int]]:
        """Compute env source ranks and sizes for receiving env outputs."""
        env_world_size = self.placement.get_world_size("env")
        rollout_world_size = self.placement.get_world_size("rollout")
        return CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=rollout_world_size,
            dst_rank=self._rank,
            src_active_world_size=active_world_size,
            dst_active_world_size=active_world_size,
        )

    @Worker.timer("predict")
    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "train"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Sparse duck validation may assign two episodes to the first eight
        # rollout ranks when 40 episodes are mapped over 32 ranks.  The native
        # Cosmos backend consumes per-sample ``*_noise_seed`` fields, so attach
        # the rank-local seeds to this eval batch before slicing each sample.
        # Training observations are left untouched and the context is cleared
        # by the normal evaluation RNG restore path.
        if mode == "eval":
            eval_context = getattr(self, "_post_update_eval_context", None)
            if eval_context is not None and eval_context.get("active", False):
                local_seeds = tuple(
                    int(seed) for seed in eval_context.get("local_seeds", ())
                )
                batch_size = int(env_obs["main_images"].shape[0])
                if len(local_seeds) != batch_size:
                    raise RuntimeError(
                        "post-update eval seed batch does not match observations: "
                        f"{len(local_seeds)} != {batch_size}"
                    )
                env_obs = dict(env_obs)
                seed_tensor = torch.as_tensor(
                    local_seeds,
                    dtype=torch.int64,
                    device=env_obs["main_images"].device,
                )
                env_obs["vision_noise_seed"] = seed_tensor
                env_obs["action_noise_seed"] = seed_tensor.clone()
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ]:
            if self.cfg.algorithm.loss_type == "embodied_dagger":
                kwargs = {"mode": "eval"}
            else:
                kwargs = {"mode": mode}

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.CNN_POLICY,
            SupportedModel.FLOW_POLICY,
            SupportedModel.MLP_POLICY,
        ]:
            kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        only_save_expert = self.cfg.algorithm.get("dagger", {}).get(
            "only_save_expert", True
        )

        if mode == "train" and self.expert_model is not None:
            # training with expert model. Beta-probability acting.
            use_expert = torch.rand(1).item() < self._dagger_sampling_params["beta"]
        else:
            use_expert = False

        with torch.no_grad():
            expert_label_flag = False
            # Decide which model to act via use_expert
            if use_expert:
                actions, result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                expert_label_flag = True
            else:
                actions, result = self.hf_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )

            # Decide re-label or not
            if (
                not only_save_expert  # only re-label in classic dagger mode
                and not use_expert  # only re-label if not using expert
                and self.expert_model is not None  # only re-label if expert exists
                and mode == "train"  # only re-label in train mode
            ):
                _, expert_result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                expert_forward_inputs = expert_result["forward_inputs"]
                expert_target = expert_forward_inputs.get(
                    "model_action", expert_forward_inputs.get("action")
                )
                if expert_target is not None:
                    result["forward_inputs"]["model_action"] = expert_target
                expert_label_flag = True

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        result["expert_label_flag"] = bool(expert_label_flag)
        return actions, result

    def _has_bootstrap_value_head(self) -> bool:
        """Return whether a final-state policy forward can produce a value."""
        return hasattr(self.hf_model, "value_head") or hasattr(
            self.hf_model, "q_head"
        )

    def get_bootstrap_values(
        self, final_obs: dict[str, Any] | None
    ) -> torch.Tensor | None:
        if final_obs is None:
            return None
        if not self._has_bootstrap_value_head():
            return None
        with torch.no_grad():
            actions, result = self.predict(final_obs)
            if "prev_values" in result and result["prev_values"] is not None:
                final_values = result["prev_values"]
            else:
                final_values = torch.zeros_like(actions[:, :1], dtype=torch.float32)
        return final_values[:, :1].cpu().contiguous()

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = await self.recv(
            self.actor_group_name,
            src_rank=self.actor_weight_src_rank,
            async_op=True,
            options=self._sync_weight_comm_options,
        ).async_wait()
        _load_full_state_dict(self.hf_model, param_state_dict)

        del param_state_dict
        clear_memory(sync=True, trim_cpu=True, collect_ipc=True)

    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        self.update_dagger_beta()
        n_chunks = self.n_train_chunk_steps
        logger = get_logger() if self._rank == 0 else None
        epoch_start = time.monotonic()
        total_steps = int(self.cfg.runner.get("max_steps", -1))
        if total_steps <= 0:
            total_steps = int(self.cfg.runner.get("max_epochs", 1))
        total_steps = max(total_steps, 1)
        step_index = min(int(self.version) + 1, total_steps)
        if logger is not None:
            n_steps = n_chunks * self.cfg.actor.model.num_action_chunks
            logger.info(
                f"[Rollout][train] step {step_index}/{total_steps} starting: "
                f"{n_chunks} chunks ({n_steps} env steps/env)"
            )
        for chunk_idx in range(n_chunks):
            for _ in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                actions, result = self.predict(env_output["obs"])

                save_flags = None
                if result.get("expert_label_flag", False):
                    save_flags = torch.full(
                        (actions.shape[0], self.cfg.actor.model.num_action_chunks),
                        True,
                        dtype=torch.bool,
                        device=actions.device,
                    )
                rollout_result = RolloutResult(
                    actions=actions,
                    prev_logprobs=result["prev_logprobs"]
                    if self.collect_prev_infos
                    else None,
                    prev_values=result.get("prev_values")
                    if self.collect_prev_infos
                    else None,
                    bootstrap_values=self.get_bootstrap_values(
                        env_output.get("final_obs", None)
                    ),
                    save_flags=save_flags,
                    forward_inputs=result["forward_inputs"],
                    versions=torch.full_like(
                        result["prev_logprobs"],
                        float(self.version),
                        dtype=torch.float32,
                    ),
                )
                self.send_rollout_result(output_channel, rollout_result, mode="train")
            if logger is not None:
                done = chunk_idx + 1
                elapsed = time.monotonic() - epoch_start
                avg_per_chunk = elapsed / done
                eta = avg_per_chunk * (n_chunks - done)
                total_units = total_steps * n_chunks
                done_units = min(int(self.version) * n_chunks + done, total_units)
                hardware = format_hardware_snapshot()
                hardware_suffix = f" | {hardware}" if hardware else ""
                logger.info(
                    f"[Rollout][train] step {step_index}/{total_steps} "
                    f"| task {done_units}/{total_units} "
                    f"({done_units / total_units:.1%}) "
                    f"| chunk {done}/{n_chunks} ({done / n_chunks:.0%}) "
                    f"| elapsed {format_duration(elapsed)} | ETA {format_duration(eta)} "
                    f"({avg_per_chunk:.1f}s/chunk)"
                    f"{hardware_suffix}"
                )
        for _ in range(self.num_pipeline_stages):
            env_output = await self.recv_env_output(input_channel)
            prev_values = None
            if self._has_bootstrap_value_head():
                _, result = self.predict(env_output["obs"])
                if self.collect_prev_infos:
                    prev_values = result.get("prev_values")

            # The terminal message intentionally has no actions: critic-free
            # Cosmos must not run a sixth policy sample just to acknowledge the
            # final state. It still needs one batch-shaped protocol field so
            # env workers can validate and merge mapped shards. ``versions``
            # is metadata only at this boundary and is ignored by the trailing
            # ChunkStepResult on the env side.
            terminal_batch_size = self._infer_env_batch_size(env_output["obs"])

            rollout_result = RolloutResult(
                prev_values=prev_values,
                bootstrap_values=self.get_bootstrap_values(
                    env_output.get("final_obs", None)
                ),
                versions=torch.full(
                    (terminal_batch_size, 1),
                    float(self.version),
                    dtype=torch.float32,
                ),
            )
            self.send_rollout_result(output_channel, rollout_result, mode="train")
        if logger is not None:
            logger.info(
                f"[Rollout][train] rollout epoch done in "
                f"{format_duration(time.monotonic() - epoch_start)}"
            )

    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ):
        if self.enable_offload:
            self.reload_model()

        logger = get_logger() if self._rank == 0 else None
        for epoch_idx in range(self.rollout_epoch):
            if logger is not None and self.rollout_epoch > 1:
                logger.info(
                    f"[Rollout][train] === rollout epoch {epoch_idx + 1}/{self.rollout_epoch} ==="
                )
            await self.generate_one_epoch(input_channel, output_channel)

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        # Cosmos HSDP service reconstruction is SPMD across the full rollout
        # world. Even ranks without a held-out episode must reload before they
        # wait, then offload only after the active shard groups finish.
        if self.enable_offload:
            self.reload_model()
        if self.eval_batch_size == 0:
            try:
                return {}
            finally:
                # The runner waits for every eval handle; a
                # global barrier here times out while active HSDP shards run.
                if self.enable_offload:
                    self.offload_model()
                self._restore_post_update_eval_rng_state()
        eval_rollout_epochs = self.cfg.algorithm.eval_rollout_epoch
        total_steps = (
            eval_rollout_epochs
            * self.n_eval_chunk_steps
            * self.num_pipeline_stages
        )
        progress_stream, progress_file = _open_eval_progress_stream(self._rank)
        progress_bar = tqdm(
            total=total_steps,
            desc="Steps",
            disable=self._rank != 0,
            mininterval=1.0,
            dynamic_ncols=True,
            file=progress_stream,
        )
        try:
            for _ in range(eval_rollout_epochs):
                for _ in range(self.n_eval_chunk_steps):
                    for _ in range(self.num_pipeline_stages):
                        env_output = await self.recv_env_output(
                            input_channel, mode="eval"
                        )
                        actions, result = self.predict(env_output["obs"], mode="eval")
                        self.send_chunk_actions(output_channel, actions, mode="eval")
                        if self._needs_eval_imagined_video():
                            imagined_video_chunk = result.get(
                                "forward_inputs", {}
                            ).get("imagined_video_chunk")
                            if not isinstance(imagined_video_chunk, torch.Tensor):
                                raise RuntimeError(
                                    "env.eval.video_cfg.save_cosmos_comparison=true "
                                    "requires the rollout model to return "
                                    "forward_inputs['imagined_video_chunk']."
                                )
                            self.send_eval_imagined_video(
                                output_channel, imagined_video_chunk
                            )
                        progress_bar.update(1)
        finally:
            progress_bar.close()
            if progress_file is not None:
                progress_file.close()
            # Shard-aligned workers may finish at different
            # times; runner-level Handle.wait() is the required synchronization.
            if self.enable_offload:
                self.offload_model()
            self._restore_post_update_eval_rng_state()

    def set_post_update_eval_context(
        self,
        global_step: int,
        episode_ids: list[int],
        base_seed: int = 42,
        execution_episode_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Select rank-local deterministic seeds for held-out evaluation.

        ``global_step`` is one-based.  The first completed update therefore
        uses zero-based evaluation index ``N=0`` and seeds 42--49 for eight
        sorted episodes.  RNG state is restored when :meth:`evaluate` exits so
        evaluation cannot perturb the next training rollout.
        """
        official_ids = tuple(sorted(int(episode_id) for episode_id in episode_ids))
        if not official_ids or len(set(official_ids)) != len(official_ids):
            raise ValueError(
                "Official post-update eval episode IDs must be non-empty and unique."
            )
        if execution_episode_ids is None:
            execution_ids = official_ids
        else:
            execution_ids = tuple(int(value) for value in execution_episode_ids)
        padding = (-len(official_ids)) % int(
            self.cfg.rollout.sparse_eval_collective_group_size
        )
        expected_execution = official_ids + tuple(
            official_ids[index % len(official_ids)] for index in range(padding)
        )
        # Never let only part of a 4-rank Cosmos HSDP shard
        # enter eval forwards; only the deterministic suffix may be padding.
        if execution_ids != expected_execution:
            raise ValueError("Post-update eval execution IDs violate shard padding")
        if len(execution_ids) != int(self.total_num_eval_envs):
            raise ValueError("Execution episode count disagrees with initialized eval envs")
        if self._rank < self.eval_mapping_world_size:
            start, stop = CommMapper.get_rank_batch_range(
                len(execution_ids), self.eval_mapping_world_size, self._rank
            )
        else:
            start = stop = len(execution_ids)
        local_ids = execution_ids[start:stop]
        local_padding = tuple(
            index >= len(official_ids) for index in range(start, stop)
        )
        if any(local_padding) and not all(local_padding):
            raise RuntimeError(
                "Official and padding eval episodes may not share one worker batch"
            )
        if len(local_ids) != self.eval_batch_size:
            raise RuntimeError(
                "Post-update evaluation assignment does not match configured "
                f"batch: rank={self._rank}, ids={local_ids}, "
                f"eval_batch_size={self.eval_batch_size}."
            )
        if self._pre_eval_rng_state is not None:
            raise RuntimeError(
                "Previous post-update evaluation RNG state was not restored."
            )

        cosmos_cfg = getattr(self.hf_model, "cosmos_cfg", None)
        if cosmos_cfg is None:
            raise RuntimeError("The rollout policy does not expose a Cosmos config.")
        previous_cosmos_seed = int(
            cosmos_cfg.get("seed", self.cfg.actor.model.cosmos.seed)
        )
        self._pre_eval_rng_state = {
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "cfg_seed": int(self.cfg.actor.model.cosmos.seed),
            "model_seed": previous_cosmos_seed,
        }
        if not local_ids:
            self._post_update_eval_context = {
                "active": False,
                "episode": None,
                "seed": None,
                "eval_index": None,
                "local_episode_ids": (),
                "local_seeds": (),
                "official_episode_ids": official_ids,
                "execution_episode_ids": execution_ids,
                "local_padding": local_padding,
            }
            return {
                "active": False,
                "episode": None,
                "seed": None,
                "eval_index": None,
            }
        eval_index = start
        local_seeds = tuple(
            int(base_seed)
            + len(official_ids) * (int(global_step) - 1)
            + (index % len(official_ids))
            for index in range(start, stop)
        )
        # Set process-level streams to the first local seed for APIs that
        # still consult global RNG state. Native Cosmos receives the complete
        # per-sample seed vector through ``predict`` above.
        self.set_cosmos_sampling_seed(local_seeds[0])
        self._post_update_eval_context = {
            "active": True,
            "episode": local_ids[0],
            "seed": local_seeds[0],
            "eval_index": eval_index,
            "local_episode_ids": tuple(local_ids),
            "local_seeds": local_seeds,
            "official_episode_ids": official_ids,
            "execution_episode_ids": execution_ids,
            "local_padding": local_padding,
        }
        return {
            "active": True,
            "episode": local_ids[0],
            "seed": local_seeds[0],
            "eval_index": eval_index,
            "local_episode_ids": tuple(local_ids),
            "local_seeds": local_seeds,
            "local_padding": local_padding,
        }

    def _restore_post_update_eval_rng_state(self) -> None:
        """Restore training RNG streams captured before held-out evaluation."""
        state = self._pre_eval_rng_state
        if state is None:
            return
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
        with open_dict(self.cfg.actor.model.cosmos):
            self.cfg.actor.model.cosmos.seed = int(state["cfg_seed"])
        cosmos_cfg = getattr(self.hf_model, "cosmos_cfg", None)
        if isinstance(cosmos_cfg, DictConfig):
            with open_dict(cosmos_cfg):
                cosmos_cfg.seed = int(state["model_seed"])
        elif cosmos_cfg is not None:
            cosmos_cfg["seed"] = int(state["model_seed"])
        self._pre_eval_rng_state = None
        self._post_update_eval_context = None

    def set_cosmos_sampling_seed(self, seed: int) -> dict[str, int]:
        """Set the live native-Cosmos sampler seed for reproducible evaluation."""
        seed = int(seed)
        with open_dict(self.cfg.actor.model.cosmos):
            self.cfg.actor.model.cosmos.seed = seed
        cosmos_cfg = getattr(self.hf_model, "cosmos_cfg", None)
        if cosmos_cfg is None:
            raise RuntimeError("The rollout policy does not expose a Cosmos config.")
        if isinstance(cosmos_cfg, DictConfig):
            with open_dict(cosmos_cfg):
                cosmos_cfg.seed = seed
        else:
            cosmos_cfg["seed"] = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return {"cosmos_sampling_seed": seed}

    def _needs_eval_imagined_video(self) -> bool:
        return self._save_eval_cosmos_comparison() or bool(
            self.cfg.get("reward", {}).get("video_similarity", {}).get("enabled", False)
        )

    def _save_eval_cosmos_comparison(self) -> bool:
        """Whether eval must forward Cosmos's predicted frames to EnvWorker.

        This is opt-in: ordinary evaluation communicates only action chunks,
        exactly as before. Native Cosmos evaluation can opt in to reproduce
        the established 10D comparison video without changing control.
        """
        return bool(
            self.cfg.env.eval.env_type == "cosmos_self_ctrl_world"
            or self.cfg.env.eval.video_cfg.get("save_cosmos_comparison", False)
        )

    def send_eval_imagined_video(
        self, output_channel: Channel, imagined_video_chunk: torch.Tensor
    ) -> None:
        """Send predicted video shards alongside eval actions.

        The split matches the action split, so every EnvWorker receives the
        video for exactly the environments whose actions it executes.
        """
        dst_ranks_and_sizes = self.dst_ranks["eval"]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        video_shards = self._split_actions(imagined_video_chunk, split_sizes)
        for (dst_rank, _), video_shard in zip(dst_ranks_and_sizes, video_shards):
            output_channel.put(
                video_shard.detach().cpu().contiguous(),
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra="eval_imagined_video"
                ),
                async_op=True,
            )

    def offload_model(self):
        self._log_cuda_memory("before native rollout offload")
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        release_native = getattr(self.hf_model, "release_rollout_resources", None)
        if callable(release_native):
            released = release_native()
            self.log_info(
                f"[Offload][rollout rank={self._rank}] "
                f"native_service_evicted={released}"
            )
        self.hf_model.to("cpu")
        # Repeated full-model GPU->CPU->GPU transitions otherwise leave the
        # released CPU storage resident in glibc arenas. With four rollout
        # workers per node that stale RSS is large enough to hit Slurm's host
        # memory cgroup during a later rollout epoch.
        clear_memory(sync=True, trim_cpu=True)
        self._log_cuda_memory("after native rollout offload")

    def reload_model(self):
        self._log_cuda_memory("before native rollout reload")
        self.hf_model.to(self.device)
        restore_native = getattr(self.hf_model, "restore_rollout_resources", None)
        if callable(restore_native):
            restored = restore_native()
            self.log_info(
                f"[Offload][rollout rank={self._rank}] "
                f"native_service_restored={restored}"
            )
        # The CPU parameter storage was replaced by device storage above.
        # Return those now-free heap pages before restoring/generating the next
        # update instead of carrying the allocator high-water mark forever.
        clear_memory(sync=True, trim_cpu=True)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=max(self.eval_batch_size, 1),
            )
        self._log_cuda_memory("after native rollout reload")

    def _log_cuda_memory(self, phase: str) -> None:
        """Log per-rank allocator and device memory around phase switches."""
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize(self.device)
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            gib = 1024**3
            self.log_info(
                f"[Offload][rollout rank={self._rank}] {phase}: "
                f"allocated={allocated / gib:.2f}GiB "
                f"reserved={reserved / gib:.2f}GiB "
                f"free={free_bytes / gib:.2f}/{total_bytes / gib:.2f}GiB"
            )
        except Exception as exc:
            self.log_info(
                f"[Offload][rollout rank={self._rank}] {phase}: "
                f"memory audit unavailable ({exc!r})"
            )

    async def recv_env_output(
        self, input_channel: Channel, mode: Literal["train", "eval"] = "train"
    ) -> dict[str, Any]:
        """Receive env outputs from mapped env ranks and merge if needed.

        Args:
            input_channel: Channel carrying env->rollout outputs.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.

        Returns:
            A single env output dict. When multiple env ranks are mapped to this
            rollout worker, outputs are merged on batch dimension.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        obs_batches = []
        for src_rank, expected_size in src_ranks_and_sizes:
            obs_batch = await input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_obs"
                ),
                async_op=True,
            ).async_wait()
            actual_size = self._infer_env_batch_size(obs_batch)
            assert actual_size == expected_size, (
                f"Expected env output batch size {expected_size} from env rank {src_rank}, "
                f"got {actual_size}."
            )
            obs_batches.append(obs_batch)
        return self._merge_obs_batches(obs_batches)

    def _split_actions(
        self, actions: torch.Tensor | np.ndarray, sizes: list[int]
    ) -> list[torch.Tensor | np.ndarray]:
        """Split rollout actions into size-specified shards along dim-0.

        Args:
            actions: Model-predicted action chunk batch (tensor or ndarray).
            sizes: Batch sizes for each destination env rank.

        Returns:
            A list of action shards aligned with destination rank order.
        """
        assert sum(sizes) == actions.shape[0], (
            f"Number of actions ({actions.shape[0]}) must equal split sizes sum ({sum(sizes)})."
        )
        if isinstance(actions, np.ndarray):
            split_indices = np.cumsum(sizes[:-1]).tolist()
            return list(np.split(actions, split_indices, axis=0))
        return list(torch.split(actions, sizes, dim=0))

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [
            obs_batch["obs"] if "obs" in obs_batch else obs_batch
            for obs_batch in obs_batches
        ]
        final_obs_list = [obs_batch.get("final_obs", None) for obs_batch in obs_batches]

        def _merge_obs_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for key in dicts[0].keys():
                values = [obs_dict[key] for obs_dict in dicts]
                first_non_none = next(
                    (value for value in values if value is not None), None
                )
                if first_non_none is None:
                    merged[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(first_non_none, list):
                    merged[key] = [item for sublist in values for item in sublist]
                else:
                    merged[key] = values
            return merged

        merged_obs = _merge_obs_dicts(obs_dicts)
        merged_final_obs = None
        if any(final_obs is not None for final_obs in final_obs_list):
            final_obs_or_obs = [
                final_obs if final_obs is not None else obs_dict
                for obs_dict, final_obs in zip(obs_dicts, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        return {"obs": merged_obs, "final_obs": merged_final_obs}

    def send_chunk_actions(
        self,
        output_channel: Channel,
        chunk_actions: torch.Tensor | np.ndarray,
        mode: Literal["train", "eval"] = "train",
    ):
        """Send action shards to mapped env ranks.

        Args:
            output_channel: Channel carrying rollout->env action chunks.
            chunk_actions: Predicted action chunk batch (tensor or ndarray).
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        chunk_actions_split = self._split_actions(chunk_actions, split_sizes)
        for (dst_rank, _), chunk_action_i in zip(
            dst_ranks_and_sizes, chunk_actions_split
        ):
            if isinstance(chunk_action_i, torch.Tensor):
                chunk_action_i = (
                    chunk_action_i.detach().cpu().contiguous()
                )  # for evaluation
            output_channel.put(
                chunk_action_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"{mode}_actions"
                ),
                async_op=True,
            )

    def _split_rollout_result(
        self, rollout_result: RolloutResult, sizes: list[int]
    ) -> list[RolloutResult]:
        def _split_optional_tensor(
            tensor: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            if tensor is None:
                return tuple(None for _ in sizes)
            return tuple(torch.split(tensor, sizes, dim=0))

        split_actions = _split_optional_tensor(rollout_result.actions)
        split_prev_logprobs = _split_optional_tensor(rollout_result.prev_logprobs)
        split_prev_values = _split_optional_tensor(rollout_result.prev_values)
        split_bootstrap_values = _split_optional_tensor(rollout_result.bootstrap_values)
        split_save_flags = _split_optional_tensor(rollout_result.save_flags)
        split_versions = _split_optional_tensor(rollout_result.versions)
        split_forward_inputs = (
            [{} for _ in sizes]
            if not rollout_result.forward_inputs
            else [
                {
                    key: torch.split(value, sizes, dim=0)[idx]
                    for key, value in rollout_result.forward_inputs.items()
                }
                for idx in range(len(sizes))
            ]
        )

        return [
            RolloutResult(
                actions=split_actions[idx],
                prev_logprobs=split_prev_logprobs[idx],
                prev_values=split_prev_values[idx],
                bootstrap_values=split_bootstrap_values[idx],
                save_flags=split_save_flags[idx],
                forward_inputs=split_forward_inputs[idx],
                versions=split_versions[idx],
            )
            for idx in range(len(sizes))
        ]

    def send_rollout_result(
        self,
        output_channel: Channel,
        rollout_result: RolloutResult,
        mode: Literal["train", "eval"] = "train",
    ):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        split_rollout_results = self._split_rollout_result(rollout_result, split_sizes)
        for (dst_rank, _), rollout_result_i in zip(
            dst_ranks_and_sizes, split_rollout_results
        ):
            output_channel.put(
                rollout_result_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"{mode}_rollout_results"
                ),
                async_op=True,
            )

    def set_global_step(self, global_step: int):
        self.version = global_step
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version * self.total_num_train_envs * self.rollout_epoch
            )
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
