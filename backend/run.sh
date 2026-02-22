#!/usr/bin/env sh
# Run the backend.
cd "$(dirname "$0")"
exec ./venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000 --reload-exclude venv
