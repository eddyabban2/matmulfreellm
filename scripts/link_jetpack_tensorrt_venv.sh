#!/usr/bin/env bash
# Make JetPack's system TensorRT Python bindings visible inside the active venv.
# Avoids pip tensorrt_cu13* wheels (wrong CUDA major) on Jetson.
#
# Usage:
#   source /path/to/venv/bin/activate
#   ./scripts/link_jetpack_tensorrt_venv.sh
#
# Creates: \$VIRTUAL_ENV/lib/python3.x/site-packages/jetpack_tensorrt.pth
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: activate a venv first (VIRTUAL_ENV is empty)." >&2
  exit 1
fi

PYTHON="${VIRTUAL_ENV}/bin/python"
PYVER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
SYS_DIST="/usr/lib/python${PYVER}/dist-packages"
PTH="$("$PYTHON" -c "import sysconfig; print(sysconfig.get_path('purelib'))")/jetpack_tensorrt.pth"

if [[ ! -f "${SYS_DIST}/tensorrt/__init__.py" ]]; then
  echo "ERROR: system TensorRT not found at ${SYS_DIST}/tensorrt/" >&2
  exit 1
fi

printf '%s\n' "$SYS_DIST" >"$PTH"
echo "Wrote $PTH -> $SYS_DIST"

"$PYTHON" -c "import tensorrt as trt; print('tensorrt', trt.__version__)"
