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

import json
import math
import os
import shutil
import subprocess
import time
from typing import Optional

import torch
import torch.distributed


def append_metrics_history(
    log_path: str, step: int, total_steps: int, elapsed_time: float, metrics: dict
) -> None:
    """Append one line of the current step's full (untruncated) metrics to
    `{log_path}/metrics_history.jsonl`, so a post-training report generator
    (`scripts/generate_training_report.py`) has a reliable machine-readable
    source instead of having to screen-scrape the fixed-width printed table.
    """
    record = {
        "step": step + 1,
        "total_steps": total_steps,
        "elapsed_time": elapsed_time,
        "metrics": {
            k: (v.item() if torch.is_tensor(v) else v) for k, v in metrics.items()
        },
    }
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, "metrics_history.jsonl"), "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

from rlinf.scheduler import Worker


def _collect_hardware_stats() -> dict:
    """Best-effort GPU/CPU utilization snapshot for the current process.

    Never raises: a missing optional dependency (pynvml/psutil) or the
    absence of a CUDA device simply means fewer fields are reported.
    """
    stats = {}
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        stats["gpu_mem_alloc_gb"] = torch.cuda.memory_allocated(device) / (1024**3)
        stats["gpu_mem_alloc_peak_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
        stats["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024**3)
        stats["gpu_mem_reserved_peak_gb"] = torch.cuda.max_memory_reserved(device) / (1024**3)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            stats["gpu_mem_free_gb"] = free_bytes / (1024**3)
            stats["gpu_mem_used_gb"] = (total_bytes - free_bytes) / (1024**3)
            stats["gpu_mem_total_gb"] = total_bytes / (1024**3)
            stats["gpu_mem_used_pct"] = (
                (total_bytes - free_bytes) / total_bytes * 100 if total_bytes else 0
            )
        except Exception:
            pass
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats["gpu_util_pct"] = float(util.gpu)
        except Exception:
            pass
    if os.environ.get("RLINF_NVIDIA_SMI_SNAPSHOT", "1") not in ("0", "false", "False"):
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            gpu_used = []
            gpu_total = []
            gpu_util = []
            for line in output.strip().splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 4:
                    continue
                _, util_pct, used_mib, total_mib = parts
                gpu_util.append(float(util_pct))
                gpu_used.append(float(used_mib) / 1024)
                gpu_total.append(float(total_mib) / 1024)
            if gpu_used and gpu_total:
                max_idx = max(
                    range(len(gpu_used)),
                    key=lambda idx: gpu_used[idx] / gpu_total[idx],
                )
                stats["node_gpu_count"] = len(gpu_used)
                stats["node_gpu_mem_used_max_gb"] = gpu_used[max_idx]
                stats["node_gpu_mem_total_max_gb"] = gpu_total[max_idx]
                stats["node_gpu_mem_used_max_pct"] = (
                    gpu_used[max_idx] / gpu_total[max_idx] * 100
                    if gpu_total[max_idx]
                    else 0
                )
                stats["node_gpu_util_max_pct"] = max(gpu_util) if gpu_util else 0
        except Exception:
            pass
    try:
        import psutil

        stats["cpu_util_pct"] = psutil.cpu_percent(interval=None)
        virtual_memory = psutil.virtual_memory()
        stats["ram_used_pct"] = virtual_memory.percent
        stats["ram_used_gb"] = virtual_memory.used / (1024**3)
        stats["ram_total_gb"] = virtual_memory.total / (1024**3)
        stats["ram_available_gb"] = virtual_memory.available / (1024**3)
    except Exception:
        pass
    for label, path in (
        ("log", os.environ.get("LOG_DIR")),
        ("tmp", os.environ.get("RAY_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"),
    ):
        if not path:
            continue
        try:
            usage = shutil.disk_usage(path)
            stats[f"disk_{label}_free_gb"] = usage.free / (1024**3)
            stats[f"disk_{label}_used_pct"] = usage.used / usage.total * 100
        except Exception:
            pass
    return stats


def _format_metric_value(value) -> str:
    if isinstance(value, float):
        if abs(value) < 0.001 and value != 0:
            return f"{value:.2e}"
        if abs(value) < 0.01:
            return f"{value:.4f}"
        if abs(value) > 10000:
            return f"{value:.2e}"
        if abs(value) > 100:
            return f"{value:.1f}"
        return f"{value:.3f}"
    return str(value)


def _first_metric(metrics: dict, *keys: str):
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def _format_duration_short(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_progress_summary(
    step: int,
    total_steps: int,
    elapsed_time: float,
    eta_seconds: float,
    metrics: dict,
    hardware_stats: dict,
) -> str:
    progress = (step + 1) / total_steps * 100
    parts = [
        f"[Train] step {step + 1}/{total_steps} ({progress:.1f}%)",
        f"elapsed={_format_duration_short(elapsed_time)}",
        f"eta={_format_duration_short(eta_seconds)}",
    ]
    for label, keys in (
        ("reward", ("rollout/rewards", "env/return")),
        ("loss", ("train/actor/total_loss", "train/total_loss")),
        ("ratio", ("train/actor/ratio", "train/action/ratio_mean")),
    ):
        value = _first_metric(metrics, *keys)
        if value is not None:
            parts.append(f"{label}={_format_metric_value(float(value))}")

    gpu_used = _first_metric(
        hardware_stats,
        "node_gpu_mem_used_max_gb",
        "gpu_mem_used_gb",
        "gpu_mem_reserved_gb",
    )
    gpu_total = _first_metric(
        hardware_stats, "node_gpu_mem_total_max_gb", "gpu_mem_total_gb"
    )
    gpu_pct = _first_metric(
        hardware_stats, "node_gpu_mem_used_max_pct", "gpu_mem_used_pct"
    )
    gpu_util = _first_metric(hardware_stats, "node_gpu_util_max_pct", "gpu_util_pct")
    if gpu_used is not None and gpu_total is not None:
        gpu_text = f"gpu_mem={float(gpu_used):.1f}/{float(gpu_total):.1f}GB"
        if gpu_pct is not None:
            gpu_text += f"({float(gpu_pct):.0f}%)"
        if gpu_util is not None:
            gpu_text += f" util={float(gpu_util):.0f}%"
        parts.append(gpu_text)

    ram_used_pct = hardware_stats.get("ram_used_pct")
    ram_available_gb = hardware_stats.get("ram_available_gb")
    if ram_used_pct is not None and ram_available_gb is not None:
        parts.append(
            f"ram={float(ram_used_pct):.0f}% free={float(ram_available_gb):.1f}GB"
        )
    cpu_util = hardware_stats.get("cpu_util_pct")
    if cpu_util is not None:
        parts.append(f"cpu={float(cpu_util):.0f}%")
    for label in ("log", "tmp"):
        disk_free = hardware_stats.get(f"disk_{label}_free_gb")
        disk_used_pct = hardware_stats.get(f"disk_{label}_used_pct")
        if disk_free is not None and disk_used_pct is not None:
            parts.append(
                f"disk_{label}=free {float(disk_free):.1f}GB used {float(disk_used_pct):.0f}%"
            )
    return " | ".join(parts)


def compute_split_num(num, split_num):
    return math.lcm(num, split_num) // split_num


def count_trajectories(metrics_dict):
    """
    Count the total number of trajectories from metrics dictionary.

    Args:
        metrics_dict: Dictionary of metrics where each value is a tensor after concatenation.
                     Each tensor's first dimension represents the number of trajectories.

    Returns:
        int: Total number of trajectories. If metrics_dict is empty, returns 0.
    """
    if not metrics_dict:
        return 0

    # Use the first metric tensor to get the trajectory count
    # All metrics should have the same first dimension (number of trajectories)
    first_key = next(iter(metrics_dict.keys()))
    first_tensor = metrics_dict[first_key]

    if isinstance(first_tensor, torch.Tensor):
        return first_tensor.shape[0]
    elif isinstance(first_tensor, list):
        # If it's a list of tensors, sum up all trajectory counts
        return sum(
            t.shape[0] if isinstance(t, torch.Tensor) else len(t) for t in first_tensor
        )
    else:
        raise TypeError(f"Unsupported tensor type: {type(first_tensor)}")


def compute_evaluate_metrics(eval_metrics_list):
    """
    List of evaluate metrics, list length stands for rollout process

    Returns:
        dict: Aggregated metrics with mean values and trajectory count
    """
    all_eval_metrics = {}
    env_info_keys = eval_metrics_list[0].keys()

    # Count trajectories from each process
    # If num_trajectories is already in the metrics, use it; otherwise count from tensor shape
    trajectory_counts = []
    for eval_metrics in eval_metrics_list:
        count = count_trajectories(eval_metrics)
        trajectory_counts.append(count)

    for env_info_key in env_info_keys:
        all_eval_metrics[env_info_key] = [
            eval_metrics[env_info_key] for eval_metrics in eval_metrics_list
        ]

    for key in all_eval_metrics:
        all_eval_metrics[key] = (
            torch.concat(all_eval_metrics[key]).float().mean().numpy()
        )

    # Add total trajectory count to metrics
    all_eval_metrics["num_trajectories"] = sum(trajectory_counts)

    return all_eval_metrics


def compute_rollout_metrics(
    data_buffer: dict, chunk_reward_log: Optional[dict] = None
) -> dict:
    rollout_metrics = {}

    if "rewards" in data_buffer:
        rewards = data_buffer["rewards"].clone()
        mean_rewards = torch.mean(rewards).to(Worker.torch_platform.current_device())
        torch.distributed.all_reduce(mean_rewards, op=torch.distributed.ReduceOp.AVG)

        rewards_metrics = {
            "rewards": mean_rewards.item(),
        }
        rollout_metrics.update(rewards_metrics)

    if "advantages" in data_buffer:
        advantages = data_buffer["advantages"]
        mean_adv = torch.mean(advantages).to(Worker.torch_platform.current_device())
        torch.distributed.all_reduce(mean_adv, op=torch.distributed.ReduceOp.AVG)
        max_adv = torch.max(advantages).detach().item()
        min_adv = torch.min(advantages).detach().item()
        reduce_adv_tensor = torch.as_tensor(
            [-min_adv, max_adv],
            device=Worker.torch_platform.current_device(),
            dtype=torch.float32,
        )
        torch.distributed.all_reduce(
            reduce_adv_tensor, op=torch.distributed.ReduceOp.MAX
        )
        min_adv, max_adv = reduce_adv_tensor.tolist()

        advantages_metrics = {
            "advantages_mean": mean_adv.item(),
            "advantages_max": max_adv,
            "advantages_min": -min_adv,
        }
        rollout_metrics.update(advantages_metrics)

    if data_buffer.get("returns", None) is not None:
        returns = data_buffer["returns"]
        mean_ret = torch.mean(returns).to(Worker.torch_platform.current_device())
        torch.distributed.all_reduce(mean_ret, op=torch.distributed.ReduceOp.AVG)
        max_ret = torch.max(returns).detach().item()
        min_ret = torch.min(returns).detach().item()
        reduce_ret_tensor = torch.as_tensor(
            [-min_ret, max_ret],
            device=Worker.torch_platform.current_device(),
            dtype=torch.float32,
        )
        torch.distributed.all_reduce(
            reduce_ret_tensor, op=torch.distributed.ReduceOp.MAX
        )
        min_ret, max_ret = reduce_ret_tensor.tolist()

        returns_metrics = {
            "returns_mean": mean_ret.item(),
            "returns_max": max_ret,
            "returns_min": -min_ret,
        }
        rollout_metrics.update(returns_metrics)

    chunk_reward_log = chunk_reward_log or {}
    grouped_std = chunk_reward_log.get("grouped_std", None)
    if grouped_std is not None:
        std_mean = torch.mean(grouped_std).to(Worker.torch_platform.current_device())
        torch.distributed.all_reduce(std_mean, op=torch.distributed.ReduceOp.AVG)
        neg_std_min = torch.as_tensor(
            [-torch.min(grouped_std).detach().item()],
            device=Worker.torch_platform.current_device(),
            dtype=torch.float32,
        )
        torch.distributed.all_reduce(neg_std_min, op=torch.distributed.ReduceOp.MAX)
        rollout_metrics["grouped_std_mean"] = std_mean.item()
        rollout_metrics["grouped_std_min"] = -neg_std_min.item()

    chunk_rewards_raw = chunk_reward_log.get("chunk_rewards_raw", None)
    chunk_rewards_gamma = chunk_reward_log.get("chunk_rewards_gamma", None)
    if chunk_rewards_raw is not None and chunk_rewards_gamma is not None:
        # Both are [n_steps, bsz] (with reward_type=chunk_level, n_steps ==
        # number of action chunks), so row i is chunk i's reward before vs.
        # after gamma discounting. Reported per-chunk so a training run can
        # be checked for "early chunks already good, later chunks lagging".
        n_chunks = chunk_rewards_raw.shape[0]
        for i in range(n_chunks):
            raw_mean = torch.mean(chunk_rewards_raw[i]).to(
                Worker.torch_platform.current_device()
            )
            torch.distributed.all_reduce(raw_mean, op=torch.distributed.ReduceOp.AVG)
            gamma_mean = torch.mean(chunk_rewards_gamma[i]).to(
                Worker.torch_platform.current_device()
            )
            torch.distributed.all_reduce(gamma_mean, op=torch.distributed.ReduceOp.AVG)
            rollout_metrics[f"chunk_{i}_reward_raw_mean"] = raw_mean.item()
            rollout_metrics[f"chunk_{i}_reward_gamma_mean"] = gamma_mean.item()

    return rollout_metrics


def append_to_dict(data, new_data):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


def compute_loss_mask(dones):
    _, actual_bsz, num_action_chunks = dones.shape
    n_chunk_step = dones.shape[0] - 1
    flattened_dones = dones.transpose(1, 2).reshape(
        -1, actual_bsz
    )  # [(n_chunk_step + 1) * num_action_chunks, rollout_epoch x bsz]
    flattened_dones = flattened_dones[
        -(n_chunk_step * num_action_chunks + 1) :
    ]  # [n_steps+1, actual-bsz]
    flattened_loss_mask = (flattened_dones.cumsum(dim=0) == 0)[
        :-1
    ]  # [n_steps, actual-bsz]

    loss_mask = flattened_loss_mask.reshape(n_chunk_step, num_action_chunks, actual_bsz)
    loss_mask = loss_mask.transpose(
        1, 2
    )  # [n_chunk_step, actual_bsz, num_action_chunks]

    loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True)  # [1, bsz, 1]
    loss_mask_sum = loss_mask_sum.expand_as(loss_mask)

    return loss_mask, loss_mask_sum


def print_metrics_table(
    step: int, total_steps: int, start_time: float, metrics: dict, start_step: int = 0
):
    """Print training metrics in a simple, fast formatted table."""
    # Calculate progress info
    progress = (step + 1) / total_steps * 100
    elapsed_time = time.time() - start_time
    steps_done = step + 1 - start_step
    eta_seconds = (
        elapsed_time / steps_done * (total_steps - step - 1) if steps_done > 0 else 0
    )

    def format_time(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    # Format elapsed time and ETA
    elapsed_str = format_time(elapsed_time)
    eta_str = format_time(eta_seconds)
    hardware_stats = _collect_hardware_stats()

    print(
        _format_progress_summary(
            step, total_steps, elapsed_time, eta_seconds, metrics, hardware_stats
        )
    )

    # Create progress bar
    bar_width = 40
    filled = int(bar_width * progress / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Print header with progress
    total_width = 120

    def _fit_line(text: str, width: int) -> str:
        if len(text) <= width:
            return text + (" " * (width - len(text)))
        if width <= 1:
            return text[:width]
        return text[: width - 1] + "…"

    def _fit_cell(text: str, width: int) -> str:
        return _fit_line(text, width)

    def _print_section_title(title: str) -> None:
        title_text = f" {title} "
        padding = total_width - 2 - len(title_text)
        left = padding // 2
        right = padding - left
        print(f"├{'─' * left}{title_text}{'─' * right}┤")

    print(f"\n╭{'─' * (total_width - 2)}╮")
    _print_section_title("Metric Table")

    # First line: Global Step and Progress
    step_str = f"Global Step: {step + 1:4d}/{total_steps}"
    progress_str = f"Progress: {bar} │ {progress:5.1f}%"
    line1 = f"│ {step_str} │ {progress_str}"
    line1 = _fit_line(line1, total_width - 2)
    print(f"{line1} │")

    # Second line: Time information
    elapsed_str_formatted = f"Elapsed: {elapsed_str}"
    eta_str_formatted = f"ETA: {eta_str}"
    step_time_str = f"Step Time: {elapsed_time / steps_done:.3f}s"
    line2 = f"│ {elapsed_str_formatted} │ {eta_str_formatted} │ {step_time_str}"
    line2 = _fit_line(line2, total_width - 2)
    print(f"{line2} │")

    # Group metrics by category
    categories = {
        "Time": {},
        "Environment": {},
        "Rollout": {},
        "Evaluation": {},
        "Replay Buffer": {},
        "Training/Actor": {},
        "Training/Critic": {},
        "Training/Other": {},
        "Hardware": hardware_stats,
    }

    for key, value in metrics.items():
        if "/" in key:
            category, metric_name = key.split("/", 1)
            category_map = {
                "time": "Time",
                "env": "Environment",
                "rollout": "Rollout",
                "eval": "Evaluation",
                "replay_buffer": "Replay Buffer",
            }
            if category in category_map:
                categories[category_map[category]][metric_name] = value
            elif category == "train":
                if metric_name.startswith("actor/"):
                    categories["Training/Actor"][metric_name] = value
                elif metric_name.startswith("critic/"):
                    categories["Training/Critic"][metric_name] = value
                elif metric_name.startswith("replay_buffer/"):
                    categories["Replay Buffer"][
                        metric_name.replace("replay_buffer/", "")
                    ] = value
                else:
                    categories["Training/Other"][metric_name] = value

    # Print metrics by category - 3 metrics per row
    table_width = total_width  # Match header width
    base_col_width = (table_width - 4) // 3
    remainder = (table_width - 4) - (base_col_width * 3)
    col_widths = [
        base_col_width + (1 if remainder > 0 else 0),
        base_col_width + (1 if remainder > 1 else 0),
        base_col_width,
    ]

    for category_name, category_metrics in categories.items():
        if category_metrics:
            _print_section_title(category_name)

            # Sort metrics for consistent output
            sorted_metrics = sorted(category_metrics.items())

            # Print in 3-column layout
            for i in range(0, len(sorted_metrics), 3):
                # Get up to 3 metrics for this row
                row_metrics = []
                for j in range(3):
                    if i + j < len(sorted_metrics):
                        metric_name, metric_value = sorted_metrics[i + j]

                        formatted_value = _format_metric_value(metric_value)

                        display = f"{metric_name}={formatted_value}"
                        row_metrics.append(display)
                    else:
                        row_metrics.append("")

                # Create the line with exactly 3 columns
                line = (
                    f"│{_fit_cell(row_metrics[0], col_widths[0])}"
                    f"│{_fit_cell(row_metrics[1], col_widths[1])}"
                    f"│{_fit_cell(row_metrics[2], col_widths[2])}│"
                )
                print(line)

    # Long metric names get truncated ("…") in the fixed-width table columns
    # above. Print each name in full the first time it's ever seen so the
    # complete key still appears verbatim somewhere in the log (acceptance
    # checks and manual `grep` both rely on the untruncated name existing) —
    # without repeating the same static list on every single step.
    new_metric_names = sorted(set(metrics) - print_metrics_table.seen_metric_names)
    if new_metric_names:
        print(f"New Metrics: {', '.join(new_metric_names)}")
        print_metrics_table.seen_metric_names.update(new_metric_names)

    # Bottom border
    print(f"╰{'─' * (table_width - 2)}╯")

    print()


print_metrics_table.seen_metric_names = set()
