#!/usr/bin/env bash
# Rebuild the self-contained local toolchain (uv + Python 3.12 + deps).
#
# Everything lands under tools/ and .venv312/, both of which are git-ignored, so
# nothing here touches the system Python. Skip this entirely if you already have
# Python >= 3.10 — just use `python3 -m venv .venv` and pip instead.
#
#   ./tools_setup.sh
#
# Why a local toolchain: `pip install pin` needs Python >= 3.10 (the coal ->
# cmeel-assimp dependency chain has no 3.9 wheels). If your system python is
# older, this script gives you a working one without touching the system.
set -euo pipefail

cd "$(dirname "$0")"
UV_DIR="$PWD/tools/bin"

if [ ! -x "$UV_DIR/uv" ]; then
  echo "==> installing uv into tools/bin"
  mkdir -p "$UV_DIR"
  curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
  UV_INSTALL_DIR="$UV_DIR" UV_NO_MODIFY_PATH=1 sh /tmp/uv-install.sh
fi

# Keep the interpreter and cache inside the project so `rm -rf tools` is a
# complete undo.
export UV_PYTHON_INSTALL_DIR="$PWD/tools/pythons"
export UV_CACHE_DIR="$PWD/tools/uvcache"
export UV_NO_MODIFY_PATH=1

echo "==> installing Python 3.12 (into tools/pythons)"
"$UV_DIR/uv" python install 3.12

echo "==> creating .venv312"
"$UV_DIR/uv" venv --python 3.12 .venv312

echo "==> installing dependencies"
"$UV_DIR/uv" pip install --python .venv312/bin/python -r requirements.txt

echo
echo "==> verifying (IPOPT must be available, it is bundled in the casadi wheel)"
./.venv312/bin/python - <<'PY'
import numpy, yaml
print("numpy", numpy.__version__)
try:
    import pinocchio, casadi
    print("pinocchio", pinocchio.__version__, "| casadi", casadi.__version__)
    o = casadi.Opti(); x = o.variable(); o.minimize((x - 3) ** 2)
    o.solver("ipopt")
    print("IPOPT OK ->", float(o.solve()))
except ImportError as exc:
    print("MISSING:", exc)
    raise SystemExit(1)
try:
    import zmq
    print("pyzmq", zmq.__version__, "(VLA/ZMQ 6002 path available)")
except ImportError:
    print("pyzmq not installed -- only needed for the ZMQ 6002 path")
PY

echo
echo "done. next:  ./.venv312/bin/python run_all_checks.py"
