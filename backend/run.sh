#!/usr/bin/env sh
# Run the backend.
set -e
cd "$(dirname "$0")"
if [ -z "${KARAOKE_RUST_UTILITY:-}" ]; then
  cargo build --locked --manifest-path ../frontend/src-tauri/Cargo.toml --bin app
fi
exec ./venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000 --reload-exclude venv
