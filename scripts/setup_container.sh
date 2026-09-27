#!/usr/bin/env bash
# Session-start environment setup INSIDE the dev container (rig:2.9.1 /
# memprobe:dev). The container is ephemeral (--rm), so run this once per
# session from the repo root:  bash scripts/setup_container.sh
#
# Installs the acsl package + server deps under a constraints file that pins
# the ROCm torch stack: if pip ever tries to replace torch/torchvision/
# torchaudio/numpy, it fails loudly instead of silently breaking the GPU path.
set -euo pipefail
cd "$(dirname "$0")/.."

PRE=$(python -c "import torch; print(torch.__version__)")

PIP_CONSTRAINT="$PWD/constraints.txt" pip install -q -e . fastapi uvicorn

POST=$(python -c "import torch; print(torch.__version__)")
if [ "$PRE" != "$POST" ]; then
    echo "FATAL: torch changed during install: $PRE -> $POST" >&2
    exit 1
fi
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch.cuda (ROCm) not available — GPU path broken"
print(f"[setup] torch {torch.__version__} intact, GPU: {torch.cuda.get_device_name(0)}")
EOF
echo "[setup] done. Start the server with: python server.py"
