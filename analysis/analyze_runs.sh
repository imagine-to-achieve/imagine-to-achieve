#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# With no arguments, analyze the three formal runs requested for this campaign.
# Pass normal rlinf-analyze arguments to analyze any other run.
if (( $# == 0 )); then
  analysis_args=(
    --config "$REPO_ROOT/configs/close_b128_mb32_u25.yaml"
    --config "$REPO_ROOT/configs/duck4_b512_mb128_u10.yaml"
    --config "$REPO_ROOT/configs/duck_red_b128_mb32_u10.yaml"
    --require-plots
  )
else
  analysis_args=("$@")
fi

exec /usr/bin/python3 -m rlinf_modified.analysis "${analysis_args[@]}"
