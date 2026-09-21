# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure helpers for restart-safe cross-rank GRPO grouping."""

from __future__ import annotations

import hashlib
import struct
from collections import defaultdict
from functools import lru_cache
from collections.abc import Mapping, Sequence
from typing import Any

import torch


_PERSONALIZATION = b"rlinf-grpo-v1"
_SEED_MASK = (1 << 63) - 1
_UINT32_MASK = (1 << 32) - 1
SEED_DERIVATION_VERSION = "blake2b-int63-v2"


def _normalize_reset_episode_pools(
    reset_episode_ids_by_color: Mapping[str, Sequence[int]],
    color_order: Sequence[str] | None,
) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    """Validate and normalize named reset-episode pools."""
    order = tuple(
        str(color) for color in (color_order or reset_episode_ids_by_color.keys())
    )
    if not order or len(set(order)) != len(order):
        raise ValueError("reset episode color order must be non-empty and unique")
    if set(order) != {str(color) for color in reset_episode_ids_by_color}:
        raise ValueError(
            "reset episode color order must contain every configured color exactly once"
        )
    pools: dict[str, tuple[int, ...]] = {}
    all_episode_ids: list[int] = []
    for color in order:
        values = tuple(int(value) for value in reset_episode_ids_by_color[color])
        if not values:
            raise ValueError(f"reset episode pool {color!r} must not be empty")
        if len(set(values)) != len(values):
            raise ValueError(f"reset episode pool {color!r} contains duplicates")
        if any(value < 0 for value in values):
            raise ValueError(f"reset episode pool {color!r} contains a negative id")
        pools[color] = values
        all_episode_ids.extend(values)
    if len(set(all_episode_ids)) != len(all_episode_ids):
        raise ValueError("reset episode ids must be disjoint across color pools")
    return order, pools


def select_color_balanced_reset_episode(
    *,
    reset_episode_ids_by_color: Mapping[str, Sequence[int]],
    base_seed: int,
    update_id: int,
    logical_round_id: int,
    group_slot: int,
    logical_rounds_per_update: int,
    group_slots_per_round: int,
    color_order: Sequence[str] | None = None,
    groups_per_color: int = 2,
) -> tuple[int, str, int]:
    """Select one episode from deterministic, without-replacement color cycles.

    Group slots are interleaved by color_order. For four colors and two groups
    per color, slots 0..3 are the first draw of each color and slots 4..7 are
    the second draw. Every exhaustion cycle owns a stable BLAKE2 permutation.
    """
    order, pools = _normalize_reset_episode_pools(
        reset_episode_ids_by_color, color_order
    )
    groups_per_color = int(groups_per_color)
    expected_slots = len(order) * groups_per_color
    if groups_per_color <= 0:
        raise ValueError("groups_per_color must be positive")
    if int(group_slots_per_round) != expected_slots:
        raise ValueError(
            "color-balanced reset scheduling requires group_slots_per_round == "
            f"len(color_order) * groups_per_color ({expected_slots}), got "
            f"{group_slots_per_round}"
        )
    if not 0 <= int(group_slot) < expected_slots:
        raise ValueError(
            f"group_slot must be in [0, {expected_slots}), got {group_slot}"
        )
    if not 0 <= int(logical_round_id) < int(logical_rounds_per_update):
        raise ValueError("logical_round_id is outside logical_rounds_per_update")

    color_index = int(group_slot) % len(order)
    draw_in_round = int(group_slot) // len(order)
    color = order[color_index]
    pool = pools[color]
    draw_index = (
        (int(update_id) * int(logical_rounds_per_update) + int(logical_round_id))
        * groups_per_color
        + draw_in_round
    )
    cycle, offset = divmod(draw_index, len(pool))
    shuffled: list[int] = []
    shuffle_seed = -1
    for cycle_id in range(cycle + 1):
        shuffle_seed = derive_stable_seed(
            "reset_episode_shuffle", int(base_seed), color_index, cycle_id
        )
        next_order = sorted(
            pool,
            key=lambda episode_id: (
                derive_stable_seed(
                    "reset_episode_order", shuffle_seed, int(episode_id)
                ),
                int(episode_id),
            ),
        )
        if shuffled and len(next_order) > 1 and next_order[0] == shuffled[-1]:
            next_order = next_order[1:] + next_order[:1]
        shuffled = next_order
    return int(shuffled[offset]), color, int(shuffle_seed)


def select_cross_rank_collective_device(
    backend: object,
    *,
    accelerator_device_type: str,
    local_rank: int,
) -> torch.device:
    """Choose tensors accepted by explicit Gloo and CUDA-only WORLD groups."""
    backend_name = str(backend).lower()
    if "gloo" in backend_name:
        return torch.device("cpu")
    if accelerator_device_type == "cuda":
        return torch.device("cuda", int(local_rank))
    raise RuntimeError(
        "cross-rank GRPO cannot select a collective device for backend "
        f"{backend!r} and accelerator {accelerator_device_type!r}"
    )


def derive_stable_seed(namespace: str, *components: int) -> int:
    """Derive a stable non-negative int63 from an explicit namespace and integers."""
    if not namespace:
        raise ValueError("seed namespace must be non-empty")
    encoded_namespace = namespace.encode("utf-8")
    payload = bytearray(struct.pack(">I", len(encoded_namespace)))
    payload.extend(encoded_namespace)
    payload.extend(struct.pack(">I", len(components)))
    for component in components:
        payload.extend(struct.pack(">q", int(component)))
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=_PERSONALIZATION,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & _SEED_MASK


def seed_int63_to_uint32(seed: int) -> int:
    """Map a semantic int63 seed to the uint32 range accepted by Cosmos."""
    return int(seed) & _UINT32_MASK


def map_semantic_seed_to_uint32(namespace: str, seed: int) -> int:
    """Apply the exact stable uint32 adapter used at Cosmos noise creation."""
    return derive_stable_seed(namespace, int(seed)) & _UINT32_MASK


@lru_cache(maxsize=128)
def _semantic_joint_seed_table(
    base_seed: int,
    update_id: int,
    logical_rounds: int,
    group_slots_per_round: int,
    group_size: int,
    chunks_per_trajectory: int,
) -> dict[tuple[int, int, int, int], tuple[int, int, int]]:
    """Build one update's topology-independent, uint32-collision-free seeds."""
    dimensions = (
        logical_rounds,
        group_slots_per_round,
        group_size,
        chunks_per_trajectory,
    )
    if any(int(value) <= 0 for value in dimensions):
        raise ValueError(f"semantic seed dimensions must be positive: {dimensions}")

    used_uint32: set[int] = set()
    table: dict[tuple[int, int, int, int], tuple[int, int, int]] = {}
    for logical_round_id in range(logical_rounds):
        for group_slot in range(group_slots_per_round):
            for member_id in range(group_size):
                for chunk_id in range(chunks_per_trajectory):
                    key = (logical_round_id, group_slot, member_id, chunk_id)
                    nonce = 0
                    while True:
                        seed = derive_stable_seed(
                            "cosmos_joint",
                            base_seed,
                            update_id,
                            logical_round_id,
                            group_slot,
                            member_id,
                            chunk_id,
                            nonce,
                        )
                        seed32 = map_semantic_seed_to_uint32(
                            "cosmos_joint_initial_noise", seed
                        )
                        if seed32 not in used_uint32:
                            used_uint32.add(seed32)
                            table[key] = (seed, seed32, nonce)
                            break
                        nonce += 1
    return table


def build_update_seed_manifest(
    *,
    base_seed: int,
    update_id: int,
    logical_rounds: int = 4,
    group_slots_per_round: int = 8,
    group_size: int = 16,
    chunks_per_trajectory: int = 5,
    reset_episode_min: int = 5,
    reset_episode_max: int = 39,
    reset_episode_ids: Sequence[int] | None = None,
    reset_episode_ids_by_color: Mapping[str, Sequence[int]] | None = None,
    reset_episode_color_order: Sequence[str] | None = None,
    groups_per_color: int = 2,
    ctrl_seed: int = 42,
) -> dict[str, Any]:
    """Return the complete auditable semantic seed manifest for one update."""
    if reset_episode_max < reset_episode_min:
        raise ValueError("reset_episode_max must be >= reset_episode_min")
    explicit_reset_ids = (
        tuple(int(value) for value in reset_episode_ids)
        if reset_episode_ids is not None
        else None
    )
    if explicit_reset_ids is not None and (
        not explicit_reset_ids
        or len(set(explicit_reset_ids)) != len(explicit_reset_ids)
        or any(value < 0 for value in explicit_reset_ids)
    ):
        raise ValueError(
            "reset_episode_ids must be non-empty, unique, and non-negative"
        )
    color_pools = reset_episode_ids_by_color
    color_order = None
    if color_pools is not None:
        color_order, color_pools = _normalize_reset_episode_pools(
            color_pools, reset_episode_color_order
        )
        flattened_ids = {
            episode_id
            for color in color_order
            for episode_id in color_pools[color]
        }
        if explicit_reset_ids is not None and flattened_ids != set(explicit_reset_ids):
            raise ValueError(
                "reset_episode_ids must equal the union of reset_episode_ids_by_color"
            )
    table = _semantic_joint_seed_table(
        int(base_seed),
        int(update_id),
        int(logical_rounds),
        int(group_slots_per_round),
        int(group_size),
        int(chunks_per_trajectory),
    )
    groups = []
    for logical_round_id in range(logical_rounds):
        for group_slot in range(group_slots_per_round):
            reset_seed = derive_stable_seed(
                "reset",
                base_seed,
                update_id,
                logical_round_id,
                group_slot,
            )
            if color_pools is not None:
                reset_episode, reset_color, reset_shuffle_seed = (
                    select_color_balanced_reset_episode(
                        reset_episode_ids_by_color=color_pools,
                        color_order=color_order,
                        groups_per_color=groups_per_color,
                        base_seed=base_seed,
                        update_id=update_id,
                        logical_round_id=logical_round_id,
                        group_slot=group_slot,
                        logical_rounds_per_update=logical_rounds,
                        group_slots_per_round=group_slots_per_round,
                    )
                )
            elif explicit_reset_ids is not None:
                reset_episode = explicit_reset_ids[
                    int(reset_seed) % len(explicit_reset_ids)
                ]
                reset_color = None
                reset_shuffle_seed = None
            else:
                episode_span = reset_episode_max - reset_episode_min + 1
                reset_episode = reset_episode_min + reset_seed % episode_span
                reset_color = None
                reset_shuffle_seed = None
            members = []
            for member_id in range(group_size):
                chunks = []
                for chunk_id in range(chunks_per_trajectory):
                    seed, seed32, nonce = table[
                        (logical_round_id, group_slot, member_id, chunk_id)
                    ]
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "cosmos_joint_seed_int63": seed,
                            "cosmos_joint_seed_uint32": seed32,
                            "collision_nonce": nonce,
                        }
                    )
                members.append({"member_id": member_id, "chunks": chunks})
            group = {
                "logical_round_id": logical_round_id,
                "group_slot": group_slot,
                "reset_seed": reset_seed,
                "reset_episode": int(reset_episode),
                "ctrl_seed": int(ctrl_seed),
                "members": members,
            }
            if reset_color is not None:
                group["reset_color"] = reset_color
                group["reset_shuffle_seed"] = int(reset_shuffle_seed)
            groups.append(group)
    digest_payload = repr(
        [(key, value) for key, value in sorted(table.items())]
    ).encode("utf-8")
    return {
        "seed_derivation_version": SEED_DERIVATION_VERSION,
        "base_seed": int(base_seed),
        "update_id": int(update_id),
        "logical_rounds": int(logical_rounds),
        "group_slots_per_round": int(group_slots_per_round),
        "group_size": int(group_size),
        "chunks_per_trajectory": int(chunks_per_trajectory),
        "ctrl_seed": int(ctrl_seed),
        "uint32_collision_count": int(
            sum(value[2] > 0 for value in table.values())
        ),
        "table_blake2b": hashlib.blake2b(
            digest_payload, digest_size=32
        ).hexdigest(),
        "groups": groups,
    }


def validate_cross_rank_group_configuration(
    *,
    runner_mode: str,
    task_type: str,
    adv_type: str,
    actor_backend: str,
    num_nodes: int,
    pipeline_stage_num: int,
    rollout_epoch: int,
    group_size: int,
    members_per_rank: int,
    rank_layout: str,
    gather_backend: str,
    env_world_size: int,
    actor_world_size: int,
    total_num_envs: int,
    local_env_batch: int,
    local_group_size: int,
) -> None:
    """Reject unsupported first-release layouts before workers are launched."""
    if runner_mode != "synchronous":
        raise ValueError("cross-rank groups currently require a synchronous runner")
    if task_type != "embodied":
        raise ValueError("cross-rank GRPO is only supported for embodied runners")
    if adv_type not in ("grpo", "grpo_action_suffix"):
        raise ValueError(
            "cross-rank groups require algorithm.adv_type=grpo or "
            "grpo_action_suffix"
        )
    if actor_backend != "fsdp":
        raise ValueError("cross-rank groups currently require the FSDP actor backend")
    del num_nodes  # kept in the signature for logging/back-compat call sites
    # The original first release rejected num_nodes != 1 out of caution, not
    # because of a known cross-node correctness issue: the actual gather
    # (fsdp_actor_worker.py's _compute_cross_rank_grpo_advantages) is a plain
    # torch.distributed.all_gather over the actor's WORLD process group with
    # tiny scalar payloads (rewards/seeds/checksums, not video tensors), and
    # select_cross_rank_collective_device already picks the CUDA device via
    # LOCAL_RANK (not global rank) -- the standard, node-agnostic pattern.
    # build_cross_rank_rollout_metadata's group/member assignment is likewise
    # pure integer arithmetic on the global actor rank, with no node-topology
    # assumption. Multi-node is therefore allowed here; the broader
    # multi-node FSDP+Ray+Ctrl-World path this unblocks (checkpoint DCP
    # saves, per-node Ctrl-World replica construction, cross-node NCCL) has
    # its own first real validation via the smoke test, not via this check.
    if pipeline_stage_num != 1:
        raise ValueError(
            "cross-rank groups currently require rollout.pipeline_stage_num=1"
        )
    if rollout_epoch <= 0:
        raise ValueError("algorithm.rollout_epoch must be positive")
    if rank_layout != "contiguous":
        raise ValueError("cross-rank groups currently require rank_layout=contiguous")
    if gather_backend != "actor_world":
        raise ValueError(
            "cross-rank groups currently require gather_backend=actor_world"
        )
    if members_per_rank <= 0:
        raise ValueError("cross_rank_group.members_per_rank must be positive")
    if group_size <= 1 or group_size % members_per_rank != 0:
        raise ValueError(
            "algorithm.group_size must be greater than one and divisible by "
            "cross_rank_group.members_per_rank"
        )
    ranks_per_group = group_size // members_per_rank
    if env_world_size % ranks_per_group != 0:
        raise ValueError(
            "env world size must contain an integer number of cross-rank groups"
        )
    if actor_world_size != env_world_size:
        raise ValueError(
            "cross-rank groups require equal actor and env world sizes: "
            f"{actor_world_size} != {env_world_size}"
        )
    if local_env_batch != members_per_rank:
        raise ValueError(
            "local env batch must equal cross_rank_group.members_per_rank: "
            f"{local_env_batch} != {members_per_rank}"
        )
    if local_group_size != members_per_rank:
        raise ValueError(
            "env.train.group_size must equal cross_rank_group.members_per_rank"
        )
    if total_num_envs % group_size != 0:
        raise ValueError(
            "env.train.total_num_envs must be divisible by global algorithm.group_size"
        )
    if total_num_envs % actor_world_size != 0:
        raise ValueError(
            "env.train.total_num_envs must be divisible by actor world size"
        )


def build_cross_rank_rollout_metadata(
    *,
    base_seed: int,
    global_step: int,
    rollout_phase: int,
    source_env_rank: int,
    local_batch_size: int,
    members_per_rank: int,
    group_size: int,
    chunk_index: int,
    per_member_vision_seed: bool = False,
    logical_round_id: int | None = None,
    physical_wave_id: int = 0,
    group_slot_offset: int = 0,
    logical_rounds_per_update: int = 4,
    group_slots_per_round: int = 8,
    chunks_per_trajectory: int = 5,
    reset_episode_min: int = 5,
    reset_episode_max: int = 39,
    reset_episode_ids: Sequence[int] | None = None,
    reset_episode_ids_by_color: Mapping[str, Sequence[int]] | None = None,
    reset_episode_color_order: Sequence[str] | None = None,
    groups_per_color: int = 2,
    fixed_ctrl_seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build one chunk's metadata for a contiguous env-rank layout.

    ``vision_noise_seed`` is derived group-wide (shared across every member,
    varying only by ``group_index``) by default: this is the validated
    common-random-numbers behavior backends that sample vision and action
    independently rely on for fair within-group reward comparison.
    ``per_member_vision_seed=True`` instead derives it the same way as
    ``action_noise_seed`` (varying by ``member_id`` too), for backends whose
    policy does one joint vision+action diffusion pass and therefore requires
    ``vision_noise_seed == action_noise_seed`` per member (e.g. Cosmos Edge's
    ``_sample_joint_policy``, which asserts this). This only affects the
    policy's own internal sampling noise -- the actual simulated rollout
    context stays shared across the group via ``ctrl_world_noise_seed``,
    which this flag does not touch, so the reward-comparison fairness that
    matters (same Ctrl-World trajectory context per group) is unaffected.
    """
    if members_per_rank <= 0:
        raise ValueError("members_per_rank must be positive")
    if group_size <= 0 or group_size % members_per_rank != 0:
        raise ValueError("group_size must be divisible by members_per_rank")
    if local_batch_size != members_per_rank:
        raise ValueError(
            "cross-rank GRPO requires local env batch to equal members_per_rank: "
            f"{local_batch_size} != {members_per_rank}"
        )

    semantic_v2 = logical_round_id is not None
    if reset_episode_max < reset_episode_min:
        raise ValueError("reset_episode_max must be >= reset_episode_min")
    explicit_reset_ids = (
        tuple(int(value) for value in reset_episode_ids)
        if reset_episode_ids is not None
        else None
    )
    if explicit_reset_ids is not None:
        if not explicit_reset_ids:
            raise ValueError("reset_episode_ids must not be empty")
        if len(set(explicit_reset_ids)) != len(explicit_reset_ids):
            raise ValueError("reset_episode_ids must not contain duplicates")
        if any(value < 0 for value in explicit_reset_ids):
            raise ValueError("reset_episode_ids must be non-negative")
    color_pools = reset_episode_ids_by_color
    if color_pools is not None:
        color_order, normalized_pools = _normalize_reset_episode_pools(
            color_pools, reset_episode_color_order
        )
        color_pools = normalized_pools
        flattened_color_ids = tuple(
            episode_id
            for color in color_order
            for episode_id in normalized_pools[color]
        )
        if explicit_reset_ids is not None and set(flattened_color_ids) != set(
            explicit_reset_ids
        ):
            raise ValueError(
                "reset_episode_ids must equal the union of reset_episode_ids_by_color"
            )
    training_seed = derive_stable_seed("training", base_seed)
    rollout_order_seed = derive_stable_seed(
        "rollout_order",
        training_seed,
        global_step,
        rollout_phase,
    )
    fields: dict[str, list[int]] = {
        "rollout_uid": [],
        "global_group_id": [],
        "group_member_id": [],
        "source_env_rank": [],
        "local_env_id": [],
        "reset_seed": [],
        "vision_noise_seed": [],
        "action_noise_seed": [],
        "ctrl_world_noise_seed": [],
        "update_id": [],
        "logical_round_id": [],
        "physical_wave_id": [],
        "group_slot": [],
        "chunk_id": [],
        "reset_episode": [],
        "seed_nonce": [],
        "shuffle_seed": [],
    }
    for local_env_id in range(local_batch_size):
        global_member_offset = (
            int(source_env_rank) * members_per_rank + local_env_id
        )
        group_index = global_member_offset // group_size
        member_id = global_member_offset % group_size
        group_slot = int(group_slot_offset) + group_index
        if semantic_v2:
            group_id = derive_stable_seed(
                "global_group_id",
                base_seed,
                global_step,
                int(logical_round_id),
                group_slot,
            )
            rollout_uid = derive_stable_seed(
                "rollout_uid",
                base_seed,
                global_step,
                int(logical_round_id),
                group_slot,
                member_id,
            )
            reset_seed = derive_stable_seed(
                "reset",
                base_seed,
                global_step,
                int(logical_round_id),
                group_slot,
            )
            table = _semantic_joint_seed_table(
                int(base_seed),
                int(global_step),
                int(logical_rounds_per_update),
                int(group_slots_per_round),
                int(group_size),
                int(chunks_per_trajectory),
            )
            action_noise_seed, _, seed_nonce = table[
                (int(logical_round_id), group_slot, member_id, int(chunk_index))
            ]
            vision_noise_seed = action_noise_seed
            ctrl_world_seed = (
                int(fixed_ctrl_seed)
                if fixed_ctrl_seed is not None
                else derive_stable_seed(
                    "ctrl_world_noise",
                    base_seed,
                    global_step,
                    int(logical_round_id),
                    group_slot,
                    chunk_index,
                )
            )
        else:
            group_id = derive_stable_seed(
                "global_group_id",
                rollout_order_seed,
                group_index,
            )
            rollout_uid = derive_stable_seed(
                "rollout_uid",
                rollout_order_seed,
                source_env_rank,
                local_env_id,
            )
            reset_seed = derive_stable_seed(
                "reset",
                rollout_order_seed,
                group_index,
            )
            action_noise_seed = derive_stable_seed(
                "action_noise",
                rollout_order_seed,
                group_index,
                member_id,
                chunk_index,
            )
            if per_member_vision_seed:
                vision_noise_seed = action_noise_seed
            else:
                vision_noise_seed = derive_stable_seed(
                    "vision_noise",
                    rollout_order_seed,
                    group_index,
                    chunk_index,
                )
            ctrl_world_seed = derive_stable_seed(
                "ctrl_world_noise",
                rollout_order_seed,
                group_index,
                chunk_index,
            )
            seed_nonce = 0
        fields["rollout_uid"].append(rollout_uid)
        fields["global_group_id"].append(group_id)
        fields["group_member_id"].append(member_id)
        fields["source_env_rank"].append(source_env_rank)
        fields["local_env_id"].append(local_env_id)
        fields["reset_seed"].append(reset_seed)
        fields["vision_noise_seed"].append(vision_noise_seed)
        fields["action_noise_seed"].append(action_noise_seed)
        fields["ctrl_world_noise_seed"].append(ctrl_world_seed)
        fields["update_id"].append(int(global_step))
        fields["logical_round_id"].append(
            int(logical_round_id)
            if logical_round_id is not None
            else int(rollout_phase)
        )
        fields["physical_wave_id"].append(int(physical_wave_id))
        fields["group_slot"].append(int(group_slot))
        fields["chunk_id"].append(int(chunk_index))
        if color_pools is not None:
            reset_episode, _, _ = select_color_balanced_reset_episode(
                reset_episode_ids_by_color=color_pools,
                color_order=color_order,
                groups_per_color=groups_per_color,
                base_seed=base_seed,
                update_id=global_step,
                logical_round_id=int(logical_round_id),
                group_slot=group_slot,
                logical_rounds_per_update=logical_rounds_per_update,
                group_slots_per_round=group_slots_per_round,
            )
        elif explicit_reset_ids is not None:
            reset_episode = explicit_reset_ids[int(reset_seed) % len(explicit_reset_ids)]
        else:
            reset_episode = (
                int(reset_episode_min)
                + int(reset_seed)
                % (int(reset_episode_max) - int(reset_episode_min) + 1)
            )
        fields["reset_episode"].append(int(reset_episode))
        fields["seed_nonce"].append(int(seed_nonce))
        fields["shuffle_seed"].append(
            derive_stable_seed("actor_shuffle", base_seed, global_step)
        )
    return {
        name: torch.tensor(values, dtype=torch.int64).unsqueeze(-1)
        for name, values in fields.items()
    }


def validate_and_compute_group_advantages(
    *,
    rollout_uid: torch.Tensor,
    global_group_id: torch.Tensor,
    group_member_id: torch.Tensor,
    scores: torch.Tensor,
    group_size: int,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Validate explicit group records and compute sample-std GRPO advantages.

    Returned advantages follow the input row order.
    """
    tensors = {
        "rollout_uid": rollout_uid.reshape(-1),
        "global_group_id": global_group_id.reshape(-1),
        "group_member_id": group_member_id.reshape(-1),
        "scores": scores.reshape(-1),
    }
    row_count = tensors["scores"].numel()
    if any(value.numel() != row_count for value in tensors.values()):
        shapes = {name: tuple(value.shape) for name, value in tensors.items()}
        raise ValueError(f"cross-rank GRPO metadata length mismatch: {shapes}")
    if group_size <= 1:
        raise ValueError("cross-rank GRPO group_size must be greater than one")

    uid_values = [int(value) for value in tensors["rollout_uid"].tolist()]
    if len(uid_values) != len(set(uid_values)):
        raise ValueError("duplicate rollout_uid in cross-rank GRPO batch")

    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, (group_id, member_id) in enumerate(
        zip(
            tensors["global_group_id"].tolist(),
            tensors["group_member_id"].tolist(),
            strict=True,
        )
    ):
        groups[int(group_id)].append((int(member_id), row))

    if not groups:
        raise ValueError("cross-rank GRPO received no rollout metadata")

    output = torch.empty_like(tensors["scores"])
    reward_stds = []
    adv_mean_abs = []
    adv_std_errors = []
    completeness = []
    expected_members = list(range(group_size))
    for group_id, entries in sorted(groups.items()):
        entries.sort(key=lambda item: item[0])
        member_ids = [member_id for member_id, _ in entries]
        completeness.append(len(set(member_ids)) / group_size)
        if member_ids != expected_members:
            raise ValueError(
                "incomplete or duplicate cross-rank GRPO members for "
                f"group {group_id}: got {member_ids}, expected {expected_members}"
            )
        rows = torch.tensor(
            [row for _, row in entries],
            dtype=torch.long,
            device=tensors["scores"].device,
        )
        group_scores = tensors["scores"].index_select(0, rows)
        mean = group_scores.mean()
        std = group_scores.std(unbiased=True)
        group_advantages = (group_scores - mean) / (std + epsilon)
        output.index_copy_(0, rows, group_advantages)
        reward_stds.append(float(std.detach().cpu()))
        adv_mean_abs.append(float(group_advantages.mean().abs().detach().cpu()))
        expected_std = 0.0 if float(std.detach().cpu()) == 0.0 else 1.0
        adv_std_errors.append(
            abs(float(group_advantages.std(unbiased=True).detach().cpu()) - expected_std)
        )

    metrics = {
        "cross_rank_group_size": float(group_size),
        "cross_rank_group_count": float(len(groups)),
        "group_member_completeness_min": float(min(completeness)),
        "group_reward_std_mean": float(sum(reward_stds) / len(reward_stds)),
        "group_adv_mean_abs_max": float(max(adv_mean_abs)),
        "group_adv_std_error_max": float(max(adv_std_errors)),
    }
    return output, metrics


def validate_and_compute_group_suffix_advantages(
    *,
    rollout_uid: torch.Tensor,
    global_group_id: torch.Tensor,
    group_member_id: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    loss_mask: torch.Tensor | None,
    group_size: int,
    gamma: float = 0.95,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute per-step suffix GRPO using explicit groups across actor ranks."""
    rollout_uid = rollout_uid.reshape(-1)
    global_group_id = global_group_id.reshape(-1)
    group_member_id = group_member_id.reshape(-1)
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be [T,N], got {tuple(rewards.shape)}")
    n_steps, row_count = rewards.shape
    if dones.shape != (n_steps + 1, row_count):
        raise ValueError(
            "dones must be [T+1,N], got "
            f"{tuple(dones.shape)} for rewards {tuple(rewards.shape)}"
        )
    if loss_mask is None:
        loss_mask = torch.ones_like(rewards, dtype=torch.bool)
    elif loss_mask.shape != rewards.shape:
        raise ValueError(
            f"loss_mask must match rewards: {loss_mask.shape} != {rewards.shape}"
        )
    if any(
        value.numel() != row_count
        for value in (rollout_uid, global_group_id, group_member_id)
    ):
        raise ValueError("cross-rank suffix metadata length mismatch")
    uid_values = [int(value) for value in rollout_uid.tolist()]
    if len(uid_values) != len(set(uid_values)):
        raise ValueError("duplicate rollout_uid in cross-rank suffix GRPO batch")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"gamma must be in [0,1], got {gamma}")

    running = torch.zeros(
        row_count, dtype=rewards.dtype, device=rewards.device
    )
    suffix_returns = torch.empty_like(rewards)
    for step in reversed(range(n_steps)):
        running = rewards[step] + float(gamma) * running * (
            ~dones[step + 1].to(torch.bool)
        ).to(rewards.dtype)
        suffix_returns[step] = running

    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, (group_id, member_id) in enumerate(
        zip(global_group_id.tolist(), group_member_id.tolist(), strict=True)
    ):
        groups[int(group_id)].append((int(member_id), row))
    if not groups:
        raise ValueError("cross-rank suffix GRPO received no groups")

    output = torch.empty_like(rewards)
    expected_members = list(range(group_size))
    std_values = []
    for group_id, entries in sorted(groups.items()):
        entries.sort(key=lambda item: item[0])
        member_ids = [member_id for member_id, _ in entries]
        if member_ids != expected_members:
            raise ValueError(
                "incomplete or duplicate cross-rank suffix members for "
                f"group {group_id}: got {member_ids}, expected {expected_members}"
            )
        rows = torch.tensor(
            [row for _, row in entries],
            dtype=torch.long,
            device=rewards.device,
        )
        group_returns = suffix_returns.index_select(1, rows)
        group_mean = group_returns.mean(dim=1, keepdim=True)
        group_std = group_returns.std(dim=1, keepdim=True, unbiased=True)
        group_advantages = (group_returns - group_mean) / (group_std + epsilon)
        output.index_copy_(1, rows, group_advantages)
        std_values.append(float(group_std.mean().detach().cpu()))
    output *= loss_mask.to(output.dtype)
    return output, {
        "cross_rank_group_size": float(group_size),
        "cross_rank_group_count": float(len(groups)),
        "group_member_completeness_min": 1.0,
        "group_suffix_return_std_mean": float(
            sum(std_values) / len(std_values)
        ),
        "group_suffix_gamma": float(gamma),
    }


def validate_cross_rank_runtime_contract(
    *,
    metadata: dict[str, torch.Tensor],
    validity: torch.Tensor,
    action_checksums: torch.Tensor,
    group_size: int,
    members_per_rank: int,
    per_member_vision_seed: bool = False,
) -> dict[str, float]:
    """Validate gathered scalar metadata identically on every actor rank.

    ``per_member_vision_seed`` must match what
    :func:`build_cross_rank_rollout_metadata` was called with: when True,
    ``vision_noise_seed`` is expected to vary per member (equal to
    ``action_noise_seed``) rather than be shared group-wide.
    """
    required = (
        "rollout_uid",
        "global_group_id",
        "group_member_id",
        "reset_seed",
        "vision_noise_seed",
        "action_noise_seed",
        "ctrl_world_noise_seed",
        "source_env_rank",
        "local_env_id",
    )
    missing = [name for name in required if name not in metadata]
    if missing:
        raise ValueError(f"missing cross-rank runtime metadata: {missing}")
    if group_size <= 1 or members_per_rank <= 0:
        raise ValueError("invalid cross-rank group_size or members_per_rank")

    flattened = {name: value.reshape(-1) for name, value in metadata.items()}
    row_count = flattened["rollout_uid"].numel()
    lengths = {name: value.numel() for name, value in flattened.items()}
    lengths["validity"] = validity.numel()
    lengths["action_checksums"] = action_checksums.numel()
    if any(length != row_count for length in lengths.values()):
        raise ValueError(f"cross-rank runtime metadata length mismatch: {lengths}")

    validity = validity.reshape(-1)
    action_checksums = action_checksums.reshape(-1)
    invalid_count = int(
        ((~torch.isfinite(validity)) | (validity != 1)).sum().item()
    )
    nonfinite_checksum_count = int(
        (~torch.isfinite(action_checksums)).sum().item()
    )
    negative_metadata_count = sum(
        int((flattened[name] < 0).sum().item()) for name in flattened
    )

    source_rank = flattened["source_env_rank"]
    local_env_id = flattened["local_env_id"]
    member_id = flattened["group_member_id"]
    expected_member_id = (
        source_rank * int(members_per_rank) + local_env_id
    ).remainder(int(group_size))
    source_layout_mismatch_count = int(
        (
            (local_env_id >= int(members_per_rank))
            | (member_id != expected_member_id)
        )
        .sum()
        .item()
    )
    source_identity = [source_rank, local_env_id]
    if "physical_wave_id" in flattened:
        source_identity.append(flattened["physical_wave_id"])
    source_pairs = torch.stack(source_identity, dim=1)
    duplicate_source_env_count = int(
        row_count - torch.unique(source_pairs, dim=0).shape[0]
    )

    duplicate_action_seed_count = 0
    duplicate_action_checksum_count = 0
    action_checksum_pairwise_abs = []
    action_checksum_stds = []
    reset_mismatch_count = 0
    namespace_collision_count = 0
    vision_action_mismatch_count = 0
    group_ids = flattened["global_group_id"]
    for group_id in group_ids.unique().tolist():
        rows = group_ids == int(group_id)
        group_rows = {name: value[rows] for name, value in flattened.items()}
        duplicate_action_seed_count += int(
            group_rows["action_noise_seed"].numel()
            - group_rows["action_noise_seed"].unique().numel()
        )
        duplicate_action_checksum_count += int(
            action_checksums[rows].numel()
            - action_checksums[rows].unique().numel()
        )
        group_checksums = action_checksums[rows].to(torch.float64)
        if group_checksums.numel() > 1:
            action_checksum_pairwise_abs.append(
                float(torch.pdist(group_checksums[:, None], p=1).mean().item())
            )
            action_checksum_stds.append(float(group_checksums.std().item()))
        shared_names = ["reset_seed", "ctrl_world_noise_seed"]
        if not per_member_vision_seed:
            shared_names.append("vision_noise_seed")
        if "reset_state_ids" in group_rows:
            shared_names.append("reset_state_ids")
        if any(group_rows[name].unique().numel() != 1 for name in shared_names):
            reset_mismatch_count += 1
        if per_member_vision_seed:
            vision_action_mismatch_count += int(
                (
                    group_rows["vision_noise_seed"] != group_rows["action_noise_seed"]
                )
                .sum()
                .item()
            )
            seed_rows = torch.stack(
                [
                    group_rows["reset_seed"],
                    group_rows["action_noise_seed"],
                    group_rows["ctrl_world_noise_seed"],
                ],
                dim=1,
            )
        else:
            seed_rows = torch.stack(
                [
                    group_rows["reset_seed"],
                    group_rows["vision_noise_seed"],
                    group_rows["action_noise_seed"],
                    group_rows["ctrl_world_noise_seed"],
                ],
                dim=1,
            )
        namespace_collision_count += sum(
            int(row.unique().numel() != row.numel()) for row in seed_rows
        )

    metrics = {
        "group_invalid_trajectory_count": float(invalid_count),
        "group_nonfinite_action_checksum_count": float(
            nonfinite_checksum_count
        ),
        "group_negative_metadata_count": float(negative_metadata_count),
        "group_source_layout_mismatch_count": float(
            source_layout_mismatch_count
        ),
        "group_duplicate_source_env_count": float(duplicate_source_env_count),
        "group_duplicate_action_seed_count": float(
            duplicate_action_seed_count
        ),
        "group_duplicate_action_checksum_count": float(
            duplicate_action_checksum_count
        ),
        "group_action_checksum_pairwise_abs_mean": float(
            sum(action_checksum_pairwise_abs) / len(action_checksum_pairwise_abs)
            if action_checksum_pairwise_abs
            else 0.0
        ),
        "group_action_checksum_std_mean": float(
            sum(action_checksum_stds) / len(action_checksum_stds)
            if action_checksum_stds
            else 0.0
        ),
        "group_seed_namespace_collision_count": float(
            namespace_collision_count
        ),
        "group_reset_mismatch_count": float(reset_mismatch_count),
        "group_vision_action_mismatch_count": float(vision_action_mismatch_count),
    }
    diagnostic_metrics = {
        "group_duplicate_action_checksum_count",
        "group_action_checksum_pairwise_abs_mean",
        "group_action_checksum_std_mean",
    }
    fatal = {
        key: value
        for key, value in metrics.items()
        if value and key not in diagnostic_metrics
    }
    if fatal:
        raise ValueError(f"invalid cross-rank runtime metadata contract: {fatal}")
    return metrics
