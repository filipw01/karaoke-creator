#!/usr/bin/env sh
# Run the Rust browser API or the standalone alignment worker.
set -e
cd "$(dirname "$0")"
if [ -z "${KARAOKE_RUST_UTILITY:-}" ]; then
  cargo build --locked --manifest-path ../frontend/src-tauri/Cargo.toml --bin app
fi
executable="${KARAOKE_RUST_UTILITY:-../frontend/src-tauri/target/debug/app}"
exec "$executable" "${1:---serve-api}" "$PWD" "${2:-0.0.0.0:8000}"
