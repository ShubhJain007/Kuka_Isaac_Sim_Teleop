#!/usr/bin/env bash
# Run after CloudXR Docker is up. Prefer: source scripts/cloudxr_env.sh first.

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

RUN_DIR="${XDG_RUNTIME_DIR:-$OPENXR_ROOT/run}"
IPC="$RUN_DIR/ipc_cloudxr"

echo "=== CloudXR quick check ==="
echo "Project: $PROJECT_ROOT"
echo "Detected OPENXR_ROOT (host path bound to /openxr): $OPENXR_ROOT"
echo ""

if [[ "${EXTERNAL_RENDERER:-}" != "cloudxr" ]]; then
  echo "[!] EXTERNAL_RENDERER is not 'cloudxr' (got: ${EXTERNAL_RENDERER:-empty})"
  echo "    Run:  source $SCRIPT_DIR/cloudxr_env.sh"
else
  echo "[OK] EXTERNAL_RENDERER=cloudxr"
fi

echo "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-<unset>}"
echo "XR_RUNTIME_JSON=${XR_RUNTIME_JSON:-<unset>}"
echo ""

if [[ -S "$IPC" ]]; then
  echo "[OK] IPC socket present: $IPC"
elif [[ -e "$IPC" ]]; then
  echo "[?] Path exists but is not a socket: $IPC"
else
  echo "[!] No IPC at: $IPC"
  echo "    Your Docker bind-mount for /openxr must match OPENXR_ROOT above."
  echo "    Fix: use the same host path in docker run --mount src=... as in cloudxr_env.sh."
  echo "    Listing $RUN_DIR:"
  ls -la "$RUN_DIR" 2>/dev/null || echo "    (missing)"
fi

echo ""
echo "Docker (cloudxr) mount for /openxr:"
docker inspect cloudxr-runtime --format '{{range .Mounts}}{{if eq .Destination "/openxr"}}host: {{.Source}} -> /openxr{{end}}{{end}}' 2>/dev/null || true

echo ""
echo "Docker (cloudxr):"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null | head -20 || true
