# Copyright 2026 The RLinf Authors.
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


class CommMapper:
    """Communication mapping helpers with batch sharding among two worker groups that require fixed rank pairing in communications.

    For example, env and rollout should always use the same rank pair for communications.
    """

    @staticmethod
    def build_channel_key(src_rank: int, dst_rank: int, extra: str) -> str:
        """Build a canonical point-to-point channel key."""
        return f"{src_rank}_{dst_rank}_{extra}"

    @staticmethod
    def get_rank_batch_range(
        batch_size: int, world_size: int, rank: int
    ) -> tuple[int, int]:
        """Return the balanced half-open batch range assigned to one rank.

        This supports batches smaller than, or not divisible by, the worker
        count. Earlier ranks own one extra item when there is a remainder;
        ranks beyond the batch size receive an empty range. Previously
        divisible mappings remain unchanged.
        """
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank ({rank}) must be in [0, {world_size}).")
        quotient, remainder = divmod(batch_size, world_size)
        start = rank * quotient + min(rank, remainder)
        size = quotient + int(rank < remainder)
        return start, start + size

    @staticmethod
    def get_rank_batch_size(batch_size: int, world_size: int, rank: int) -> int:
        """Return the number of global batch elements assigned to ``rank``."""
        start, stop = CommMapper.get_rank_batch_range(batch_size, world_size, rank)
        return stop - start

    @staticmethod
    def get_collective_aligned_world_size(
        batch_size: int, world_size: int, collective_group_size: int
    ) -> int:
        """Choose active ranks without splitting an internal collective group."""
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if collective_group_size <= 0:
            raise ValueError("collective_group_size must be positive")
        if world_size % collective_group_size != 0:
            raise ValueError(
                "world_size must be divisible by collective_group_size: "
                f"{world_size} % {collective_group_size} != 0"
            )
        if batch_size == 0:
            return 0
        if batch_size >= world_size:
            return world_size
        aligned = (batch_size // collective_group_size) * collective_group_size
        if aligned == 0:
            raise ValueError(
                "sparse batch is smaller than one collective group: "
                f"batch_size={batch_size}, collective_group_size={collective_group_size}"
            )
        return aligned

    @staticmethod
    def get_active_rank_batch_range(
        batch_size: int, world_size: int, active_world_size: int, rank: int
    ) -> tuple[int, int]:
        """Return a balanced range over an explicitly active rank prefix."""
        if not 0 <= active_world_size <= world_size:
            raise ValueError(
                f"active_world_size ({active_world_size}) must be in [0, {world_size}]"
            )
        if not 0 <= rank < world_size:
            raise ValueError(f"rank ({rank}) must be in [0, {world_size}).")
        if active_world_size == 0 or rank >= active_world_size:
            return batch_size, batch_size
        return CommMapper.get_rank_batch_range(batch_size, active_world_size, rank)

    @staticmethod
    def get_dst_ranks(
        batch_size: int,
        src_world_size: int,
        dst_world_size: int,
        src_rank: int,
        *,
        src_active_world_size: int | None = None,
        dst_active_world_size: int | None = None,
    ) -> list[tuple[int, int]]:
        """Compute destination ranks and transfer sizes for one source rank."""
        src_active = (
            src_world_size if src_active_world_size is None else src_active_world_size
        )
        dst_active = (
            dst_world_size if dst_active_world_size is None else dst_active_world_size
        )
        src_begin, src_end = CommMapper.get_active_rank_batch_range(
            batch_size, src_world_size, src_active, src_rank
        )
        dst_ranks_and_sizes: list[tuple[int, int]] = []
        for dst_rank in range(dst_world_size):
            dst_begin, dst_end = CommMapper.get_active_rank_batch_range(
                batch_size, dst_world_size, dst_active, dst_rank
            )
            overlap = min(src_end, dst_end) - max(src_begin, dst_begin)
            if overlap > 0:
                dst_ranks_and_sizes.append((dst_rank, overlap))
        return dst_ranks_and_sizes

    @staticmethod
    def get_src_ranks(
        batch_size: int,
        src_world_size: int,
        dst_world_size: int,
        dst_rank: int,
        *,
        src_active_world_size: int | None = None,
        dst_active_world_size: int | None = None,
    ) -> list[tuple[int, int]]:
        """Compute source ranks/sizes for one destination rank."""
        src_active = (
            src_world_size if src_active_world_size is None else src_active_world_size
        )
        dst_active = (
            dst_world_size if dst_active_world_size is None else dst_active_world_size
        )
        CommMapper.get_active_rank_batch_range(
            batch_size, dst_world_size, dst_active, dst_rank
        )

        src_ranks_and_sizes: list[tuple[int, int]] = []
        for src_rank in range(src_world_size):
            dst_ranks_and_sizes = CommMapper.get_dst_ranks(
                batch_size=batch_size,
                src_world_size=src_world_size,
                dst_world_size=dst_world_size,
                src_rank=src_rank,
                src_active_world_size=src_active,
                dst_active_world_size=dst_active,
            )
            for mapped_dst_rank, size in dst_ranks_and_sizes:
                if mapped_dst_rank == dst_rank:
                    src_ranks_and_sizes.append((src_rank, size))

        expected_begin, expected_end = CommMapper.get_active_rank_batch_range(
            batch_size, dst_world_size, dst_active, dst_rank
        )
        expected_size = expected_end - expected_begin
        actual_size = sum(size for _, size in src_ranks_and_sizes)
        assert actual_size == expected_size, (
            f"Expected receive size {expected_size} for destination rank {dst_rank}, "
            f"got {actual_size} from mappings {src_ranks_and_sizes}."
        )
        return src_ranks_and_sizes
