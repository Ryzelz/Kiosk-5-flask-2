#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-5000}"
APP_DEBUG="${APP_DEBUG:-1}"

if [[ -x "$PROJECT_ROOT/.venv/Scripts/python.exe" ]]; then
    PYTHON_CMD=("$PROJECT_ROOT/.venv/Scripts/python.exe")
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_CMD=("$PROJECT_ROOT/.venv/bin/python")
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=("python3")
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=("python")
else
    echo "Python was not found. Install Python or create a .venv first." >&2
    exit 1
fi

echo "Starting Wideye local host..."
echo "Host: $APP_HOST"
echo "Port: $APP_PORT"
echo "Debug: $APP_DEBUG"

export APP_HOST APP_PORT APP_DEBUG
exec "${PYTHON_CMD[@]}" "$PROJECT_ROOT/main.py"
