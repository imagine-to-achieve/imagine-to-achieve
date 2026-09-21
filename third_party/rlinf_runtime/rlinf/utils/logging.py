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


def get_logger():
    """Get the logger instance of the current worker."""
    from rlinf.scheduler.worker import Worker

    return Worker.logger


def format_duration(seconds: float) -> str:
    """Render a duration in seconds as e.g. "1h 03m 12s" / "04m 12s" / "12s"."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_hardware_snapshot() -> str:
    """Best-effort node hardware snapshot for compact progress logs."""
    import os
    import shutil
    import subprocess

    parts = []
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
        gpu_rows = []
        for line in output.strip().splitlines():
            cols = [col.strip() for col in line.split(",")]
            if len(cols) != 4:
                continue
            idx, util_pct, used_mib, total_mib = cols
            used_gb = float(used_mib) / 1024
            total_gb = float(total_mib) / 1024
            used_pct = used_gb / total_gb * 100 if total_gb else 0
            gpu_rows.append((int(idx), float(util_pct), used_gb, total_gb, used_pct))
        if gpu_rows:
            idx, util_pct, used_gb, total_gb, used_pct = max(
                gpu_rows, key=lambda row: row[4]
            )
            parts.append(
                f"gpu{idx}_mem={used_gb:.1f}/{total_gb:.1f}GB({used_pct:.0f}%) "
                f"util={util_pct:.0f}%"
            )
    except Exception:
        pass

    try:
        import psutil

        vm = psutil.virtual_memory()
        parts.append(f"ram={vm.percent:.0f}% free={vm.available / (1024**3):.1f}GB")
        parts.append(f"cpu={psutil.cpu_percent(interval=None):.0f}%")
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
            parts.append(
                f"disk_{label}=free {usage.free / (1024**3):.1f}GB "
                f"used {usage.used / usage.total * 100:.0f}%"
            )
        except Exception:
            pass
    return " | ".join(parts)


def quiet_third_party_progress_bars() -> None:
    """Disable raw tqdm bars from vendored inference code (e.g. cosmos-framework's
    diffusion samplers, diffusers pipelines) so they don't flood file-redirected
    training logs with one "X%|..." line per refresh. Each import site that pulls
    in such a library should call this once at import time. Opt out with
    COSMOS3_QUIET_PROGRESS_BARS=0 for step-by-step debugging.
    """
    import os

    if os.environ.get("COSMOS3_QUIET_PROGRESS_BARS", "1") in ("0", "false", "False"):
        return
    import tqdm as tqdm_module

    if getattr(tqdm_module.tqdm, "_rlinf_quieted", False):
        return

    original_init = tqdm_module.tqdm.__init__

    def _disabled_init(self, *args, **kwargs):
        kwargs.setdefault("disable", True)
        original_init(self, *args, **kwargs)

    tqdm_module.tqdm.__init__ = _disabled_init
    tqdm_module.tqdm._rlinf_quieted = True
