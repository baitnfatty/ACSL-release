#!/usr/bin/env bash
# Stop and remove the ACSL server container.
set -euo pipefail

NAME=acsl-server

if docker ps -a -q -f name="^${NAME}$" | grep -q .; then
    docker rm -f "$NAME" >/dev/null
    echo "[kill] container '$NAME' stopped and removed."
else
    echo "[kill] no '$NAME' container found — nothing to do."
fi
