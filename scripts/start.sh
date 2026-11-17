#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
exec "$REPO_DIR/.runtime/openpi/.venv/bin/python" serve.py "$@"
