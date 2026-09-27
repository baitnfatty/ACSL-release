#!/usr/bin/env bash
# Launch the ACSL server in a detached container on the pinned ROCm image.
# Usage:  ./launch_server.sh            (default profile: v3-qwen3, 1.7B)
#         ACSL_PROFILE=v3-qwen3-4b ./launch_server.sh
# Idempotent: replaces any existing acsl-server container.
set -euo pipefail
cd "$(dirname "$0")"

NAME=acsl-server
IMAGE="${ACSL_IMAGE:-rig:2.9.1}"
PROFILE="${ACSL_PROFILE:-v3-qwen3}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "$HOME/Projects:/workspace" \
  -v hf_cache:/root/.cache/huggingface \
  -p 5000:5000 \
  -e ACSL_PROFILE="$PROFILE" \
  -w /workspace/ACSL "$IMAGE" \
  bash -c "bash scripts/setup_container.sh && python server.py"

echo "[launch] container '$NAME' started (profile: $PROFILE)."
echo "[launch] waiting for model load ..."
for i in $(seq 1 120); do
    status=$(curl -s -m 2 http://127.0.0.1:5000/api/status 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || true)
    if [ "$status" = "loaded" ]; then
        echo "[launch] server ready: http://127.0.0.1:5000"
        exit 0
    fi
    if [ "$status" = "error" ]; then
        echo "[launch] FATAL: server reported load error. Logs:" >&2
        docker logs "$NAME" 2>&1 | tail -20 >&2
        exit 1
    fi
    # container died during setup?
    if ! docker ps -q -f name="$NAME" | grep -q .; then
        echo "[launch] FATAL: container exited. Logs:" >&2
        docker logs "$NAME" 2>&1 | tail -20 >&2
        exit 1
    fi
    sleep 2
done
echo "[launch] WARNING: still loading after 240s — check: docker logs -f $NAME" >&2
exit 1
