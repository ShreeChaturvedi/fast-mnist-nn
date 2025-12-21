#!/usr/bin/env bash
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install it from https://brew.sh/"
  exit 1
fi

brew install cmake libomp

if python3 - <<'PY' >/dev/null 2>&1; then
import tqdm  # noqa: F401
PY
  exit 0
fi

python3 -m pip install --user --break-system-packages \
  --no-warn-script-location tqdm || {
  echo "tqdm install failed; prepare_mnist.py will use basic progress."
}
