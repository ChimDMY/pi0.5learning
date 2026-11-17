#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_DIR="$REPO_DIR/.runtime/openpi"
PIN=215abfb217dbac7d5f1273282331b9b1866c0479
command -v git >/dev/null
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
if [[ -e "$OPENPI_DIR" ]]; then
  echo "Existing $OPENPI_DIR left untouched. See DEPLOY.md for continuing an interrupted install."
  exit 1
fi
mkdir -p "$REPO_DIR/.runtime"
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
git -C "$OPENPI_DIR" checkout --detach "$PIN"
GIT_LFS_SKIP_SMUDGE=1 git -C "$OPENPI_DIR" submodule update --init --recursive
git -C "$OPENPI_DIR" apply --check "$REPO_DIR/patches/openpi-inference.patch"
git -C "$OPENPI_DIR" apply "$REPO_DIR/patches/openpi-inference.patch"
cd "$OPENPI_DIR"
# Dedicated environment; do not install into the working robot SDK environment.
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11
cd "$REPO_DIR"
"$OPENPI_DIR/.venv/bin/python" verify_runtime.py
"$OPENPI_DIR/.venv/bin/python" -m unittest discover -s tests -v
echo 'Source and CPU interface checks passed. Next: transfer and verify the checkpoint.'
