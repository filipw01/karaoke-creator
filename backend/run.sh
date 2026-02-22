#!/usr/bin/env sh
# Run the backend. Optional aligners: set env vars below or in .env.optional (not committed).
# Example .env.optional:
#   LYRICS_ALIGNER_ROOT=/path/to/lyrics-aligner
#   LYRICS_ALIGNER_PYTHON=python
#   AUTOLYRIXALIGN_ROOT=/path/to/autolyrixalign

cd "$(dirname "$0")"
if [ -f .env.optional ]; then
  set -a
  . ./.env.optional
  set +a
fi
exec ./venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000 --reload-exclude venv
