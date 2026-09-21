"""Opt-in Stage 1.5 runtime hook; inert unless RLINF_STAGE15_MODE is set."""

from __future__ import annotations

import os


if os.environ.get("RLINF_STAGE15_MODE", "").strip():
    try:
        from stage15_hook import install

        install()
    except Exception as exc:  # fail closed before expensive allocations start
        raise RuntimeError("failed to install the Stage 1.5 runtime plugin") from exc
