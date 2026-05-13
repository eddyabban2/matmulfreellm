#!/usr/bin/env bash
# Build bitsandbytes CUDA backend for Jetson (Orin sm_87) and install the shared
# library into the active virtualenv. PyTorch must be the JetPack NVIDIA wheel
# (CUDA 12.x); bitsandbytes picks libbitsandbytes_cuda<XY>.so from torch.version.cuda.
#
# Usage:
#   source /path/to/venv/bin/activate
#   ./scripts/install_bitsandbytes_jetson.sh
#
# Optional: BNB_SRC=/path/to/bitsandbytes BNB_TAG=0.49.2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: activate a venv first (VIRTUAL_ENV is empty)." >&2
  exit 1
fi

PYTHON="${VIRTUAL_ENV}/bin/python"
PIP="${VIRTUAL_ENV}/bin/pip"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: not found: $PYTHON" >&2
  exit 1
fi

BNB_TAG="${BNB_TAG:-0.49.2}"
BNB_SRC="${BNB_SRC:-${HOME}/bitsandbytes-src}"
SITE_PACKAGES="$("$PYTHON" -c "import sysconfig; print(sysconfig.get_path('purelib'))")"
INSTALL_DIR="${SITE_PACKAGES}/bitsandbytes"

TORCH_CUDA="$("$PYTHON" -c "import torch; p=torch.version.cuda.split('.'); print(f'{int(p[0])}{int(p[1])}')")"
SO_NAME="libbitsandbytes_cuda${TORCH_CUDA}.so"
DEST="${INSTALL_DIR}/${SO_NAME}"

if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
  echo "ERROR: PyTorch in this venv does not see CUDA. Install the JetPack torch wheel first." >&2
  exit 1
fi

echo "torch.version.cuda -> matching library suffix: cuda${TORCH_CUDA} (${SO_NAME})"

if [[ ! -d "$BNB_SRC/.git" ]] && [[ ! -f "$BNB_SRC/CMakeLists.txt" ]]; then
  echo "Cloning bitsandbytes ${BNB_TAG} -> ${BNB_SRC}"
  git clone --depth 1 --branch "$BNB_TAG" https://github.com/bitsandbytes-foundation/bitsandbytes.git "$BNB_SRC"
fi

cd "$BNB_SRC"
git checkout "$BNB_TAG"

"$PIP" show bitsandbytes >/dev/null 2>&1 || "$PIP" install --no-deps "bitsandbytes==${BNB_TAG}"

# scikit-build / cmake build (pip install . alone may ship CPU-only on Jetson)
if ! command -v cmake &>/dev/null; then
  echo "Installing cmake + ninja into venv..."
  "$PIP" install -q 'cmake>=3.22' ninja scikit-build-core setuptools
fi

rm -rf build
cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=87 -S . -B build
cmake --build build -j"$(nproc)"

BUILT="bitsandbytes/${SO_NAME}"
if [[ ! -f "$BUILT" ]]; then
  echo "ERROR: expected artifact missing: $BNB_SRC/$BUILT" >&2
  echo "       (build may have named the .so differently; check bitsandbytes/)" >&2
  ls -la bitsandbytes/*.so 2>/dev/null || true
  exit 1
fi

mkdir -p "$INSTALL_DIR"
cp -v "$BUILT" "$DEST"

# JetPack PyTorch builds target NumPy 1.x ABI
"$PIP" install -q 'numpy>=1.26,<2' || true

"$PYTHON" -c "
import bitsandbytes as bnb
import torch
from bitsandbytes.nn import Linear8bitLt
m = Linear8bitLt(16, 8, bias=False, has_fp16_weights=False).cuda()
x = torch.randn(1, 16, device='cuda', dtype=torch.float16)
y = m(x)
assert y.is_cuda, 'expected CUDA output'
print('bitsandbytes OK:', getattr(bnb, '__version__', '?'), '->', y.shape)
"

echo ""
echo "Installed ${DEST}"
echo "Do not: pip install bitsandbytes from Jetson indices without --no-deps (can pull CUDA 13 torch)."
echo "Optional: ${REPO_ROOT}/scripts/link_jetpack_tensorrt_venv.sh"
