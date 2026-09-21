"""Compute-node regression test for collective rendezvous and queue failures."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from rlinf.scheduler.collective.async_work import AsyncFuncWork
from rlinf.scheduler.collective.collective_group import CollectiveWorkQueue
from rlinf.scheduler.collective.multi_channel_pg import MultiChannelProcessGroup
from rlinf.scheduler.hardware import AcceleratorType
from rlinf.scheduler.worker import Worker


TIMEOUT = timedelta(seconds=90)
STRESS_GROUPS = 64


def _wait_for_mapping(mapping, key: str, timeout_seconds: float = 90.0):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if key in mapping:
            return mapping[key]
        time.sleep(0.001)
    raise TimeoutError(f"timed out waiting for {key}")


def _stress_rank(rank: int, ports, results) -> None:
    stores: list[dist.Store | None] = [None] * STRESS_GROUPS

    if rank == 0:

        def run_one(index: int) -> int:
            store = dist.TCPStore(
                host_name="127.0.0.1",
                port=0,
                world_size=2,
                is_master=True,
                timeout=TIMEOUT,
                wait_for_workers=False,
            )
            stores[index] = store
            ports[f"port_{index}"] = store.port
            store.wait(["client_ready"], TIMEOUT)
            store.set("server_ready", "1")
            return store.port

    else:

        def run_one(index: int) -> int:
            port = int(_wait_for_mapping(ports, f"port_{index}"))
            store = dist.TCPStore(
                host_name="127.0.0.1",
                port=port,
                world_size=2,
                is_master=False,
                timeout=TIMEOUT,
                wait_for_workers=False,
            )
            stores[index] = store
            store.set("client_ready", "1")
            store.wait(["server_ready"], TIMEOUT)
            return port

    with ThreadPoolExecutor(max_workers=STRESS_GROUPS) as pool:
        observed = list(pool.map(run_one, range(STRESS_GROUPS)))
    results[f"stress_{rank}"] = observed


def _process_group_rank(rank: int, shared, results) -> None:
    if rank == 0:
        raw_store = dist.TCPStore(
            host_name="127.0.0.1",
            port=0,
            world_size=2,
            is_master=True,
            timeout=TIMEOUT,
            wait_for_workers=False,
        )
        shared["pg_port"] = raw_store.port
    else:
        port = int(_wait_for_mapping(shared, "pg_port"))
        raw_store = dist.TCPStore(
            host_name="127.0.0.1",
            port=port,
            world_size=2,
            is_master=False,
            timeout=TIMEOUT,
            wait_for_workers=False,
        )

    group_info = SimpleNamespace(
        workers=[
            SimpleNamespace(
                accelerator_type=AcceleratorType.NO_ACCEL,
                accelerator_model="cpu",
            )
            for _ in range(2)
        ]
    )
    multi_channel_group = MultiChannelProcessGroup(
        cur_rank=rank,
        num_channels=1,
        group_info=group_info,
        logger=logging.getLogger(f"process-group-rank-{rank}"),
    )
    multi_channel_group.init(
        init_method=None,
        world_size=2,
        rank=rank,
        group_name="regression",
        store=raw_store,
    )
    group = multi_channel_group._collective_gloo_process_groups[0]
    value = torch.tensor([rank + 1], dtype=torch.int64)
    dist.all_reduce(value, group=group)
    results[f"pg_{rank}"] = int(value.item())


class _CpuOnlyWorker:
    has_accelerator = False


def _queue_failure_regression() -> None:
    Worker.current_worker = _CpuOnlyWorker()
    queue = CollectiveWorkQueue(
        CollectiveWorkQueue.SEND,
        logging.getLogger("collective-port-regression"),
    )

    def fail_once() -> None:
        raise RuntimeError("intentional queue failure")

    failed = AsyncFuncWork(fail_once)
    queue.enqueue(failed, comm_id=0)
    try:
        failed.wait()
    except RuntimeError as exc:
        assert "intentional queue failure" in str(exc)
    else:
        raise AssertionError("failed work did not surface its exception")

    assert queue._thread.is_alive(), "queue thread died after one failed operation"
    succeeded = AsyncFuncWork(lambda: 123)
    queue.enqueue(succeeded, comm_id=1)
    assert succeeded.wait() == 123
    assert queue._thread.is_alive(), "queue thread died before processing later work"


def _run_pair(target, shared, results) -> None:
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(target=target, args=(rank, shared, results)) for rank in (0, 1)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=180)
        if process.is_alive():
            process.terminate()
            raise TimeoutError(f"child process {process.pid} did not finish")
        if process.exitcode != 0:
            raise RuntimeError(
                f"child process {process.pid} failed with exit code {process.exitcode}"
            )


def main() -> None:
    if os.uname().machine != "aarch64":
        raise RuntimeError("this regression must use the aarch64 environment")
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        shared = manager.dict()
        results = manager.dict()
        _run_pair(_stress_rank, shared, results)
        rank0_ports = list(results["stress_0"])
        rank1_ports = list(results["stress_1"])
        assert rank0_ports == rank1_ports
        assert len(set(rank0_ports)) == STRESS_GROUPS

        shared.clear()
        _run_pair(_process_group_rank, shared, results)
        assert results["pg_0"] == 3
        assert results["pg_1"] == 3

    _queue_failure_regression()
    print(
        f"PASS: {STRESS_GROUPS} concurrent bound stores, "
        "two-rank Gloo process group, and queue exception recovery"
    )


if __name__ == "__main__":
    main()
