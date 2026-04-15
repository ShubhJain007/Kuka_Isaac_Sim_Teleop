#!/usr/bin/env bash
# Source BEFORE launching teleop_med7_vr.py when using NVIDIA CloudXR + Vision Pro.
#
#   source scripts/cloudxr_env.sh
#   python scripts/teleop_med7_vr.py --experience "$(pwd)/apps/isaaclab.python.headless.cloudxr.kit"
#
# XDG_RUNTIME_DIR must be the *same* openxr/run directory your CloudXR Docker container
# bind-mounts to /openxr. If your docker run uses:
#   --mount type=bind,src=/path/to/openxr,dst=/openxr
# then OPENXR_ROOT on the host is /path/to/openxr (not necessarily this repo's openxr/).
#
# Override manually:
#   export CLOUDXR_OPENXR_ROOT=/home/you/IsaacLab/openxr
#   source scripts/cloudxr_env.sh
#
# We try to auto-detect from a running container named cloudxr-runtime (docker inspect).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OPENXR_ROOT="${CLOUDXR_OPENXR_ROOT:-}"
if [[ -z "$OPENXR_ROOT" ]] && command -v docker &>/dev/null; then
  OPENXR_ROOT="$(
    docker inspect cloudxr-runtime --format '{{range .Mounts}}{{if eq .Destination "/openxr"}}{{.Source}}{{end}}{{end}}' 2>/dev/null
  )"
fi
if [[ -z "$OPENXR_ROOT" ]]; then
  OPENXR_ROOT="$PROJECT_ROOT/openxr"
fi

export EXTERNAL_RENDERER=cloudxr
export XDG_RUNTIME_DIR="$OPENXR_ROOT/run"
export XR_RUNTIME_JSON="$OPENXR_ROOT/share/openxr/1/openxr_cloudxr.json"
export LD_LIBRARY_PATH="$OPENXR_ROOT/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$XDG_RUNTIME_DIR"

echo "[cloudxr_env] OPENXR_ROOT=$OPENXR_ROOT"
echo "[cloudxr_env] EXTERNAL_RENDERER=$EXTERNAL_RENDERER"
echo "[cloudxr_env] XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
echo "[cloudxr_env] XR_RUNTIME_JSON=$XR_RUNTIME_JSON"
if [[ -S "$XDG_RUNTIME_DIR/ipc_cloudxr" ]]; then
  echo "[cloudxr_env] ipc_cloudxr socket OK"
else
  echo "[cloudxr_env] WARNING: no ipc_cloudxr yet — start CloudXR Docker first, or set CLOUDXR_OPENXR_ROOT to your bind-mounted openxr/"
fi
