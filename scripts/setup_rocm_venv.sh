#!/usr/bin/env bash
# Build a ROCm PyTorch venv for Genesis, without touching the CUDA venv the project normally uses.
#
# Measured 2026-11-10 on this host (Radeon RX 7800 XT, gfx1101, ROCm 7.2.4):
#   * a ROCm build of torch makes `torch.version.hip` non-null, which is what Genesis checks before it will use
#     `gs.amdgpu` (without it, `gs.gpu` silently starts on the CPU: "backend gs.cpu");
#   * the 4.6 GB wheel must be fetched to real disk - /tmp here is a 6 GB tmpfs and the download dies at 2.4 GB
#     with "Failure writing output to destination";
#   * `triton-rocm` lives only on the pytorch ROCm index, so that index has to be passed to the resolver.
set -euo pipefail
ROOT=${1:-/home/yuu/Documents/Sim-Agent-nav}
VENV="$ROOT/.venv-rocm"
UV=${UV:-/home/yuu/.local/bin/uv}
CHANNEL=${CHANNEL:-rocm7.0}
TORCH=${TORCH:-2.10.0}

"$UV" venv --python 3.13 "$VENV"
VIRTUAL_ENV="$VENV" "$UV" pip install --python "$VENV/bin/python" \
  --index "https://download.pytorch.org/whl/$CHANNEL" --default-index https://pypi.org/simple "torch==$TORCH"
"$UV" export --frozen --no-hashes --no-dev -o /tmp/req.txt
grep -viE '^(torch|triton|pytorch-triton|nvidia-)' /tmp/req.txt > /tmp/req-no-torch.txt
VIRTUAL_ENV="$VENV" "$UV" pip install --python "$VENV/bin/python" -r /tmp/req-no-torch.txt
# No HSA_OVERRIDE_GFX_VERSION: gfx1101 (RX 7800 XT) is supported natively by ROCm 7, and the override makes the
# Ryzen iGPU (really gfx1036) masquerade as gfx1101 - a second, useless torch device.
"$VENV/bin/python" - <<'PY'
import torch
print('torch', torch.__version__, '| hip', torch.version.hip, '| avail', torch.cuda.is_available(),
      '| device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')
PY
