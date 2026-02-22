#!/usr/bin/env sh
# One-time setup: create venv (Python 3.12), install deps, optional aligners. Then run ./run.sh
set -e
cd "$(dirname "$0")"
PYTHON=python3.12
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python3
fi
echo "Using $PYTHON: $($PYTHON --version 2>&1)"
if [ ! -d venv ]; then
  echo "Creating venv..."
  "$PYTHON" -m venv venv
fi
echo "Installing requirements..."
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -r requirements.txt
echo "Setup done. Run ./run.sh to start the server."
