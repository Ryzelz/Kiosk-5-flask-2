#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-5000}"
APP_DEBUG="${APP_DEBUG:-1}"

LINUX_VENV_DIR="${PROJECT_ROOT}/.venv-linux"
WINDOWS_VENV_PYTHON="${PROJECT_ROOT}/.venv/Scripts/python.exe"
LINUX_VENV_PYTHON="${LINUX_VENV_DIR}/bin/python"
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 was not found. Install Python 3 first." >&2
    exit 1
fi

if [[ -f "$WINDOWS_VENV_PYTHON" && ! -x "$WINDOWS_VENV_PYTHON" ]]; then
    echo "Detected a Windows virtual environment at .venv."
    echo "Linux cannot use that interpreter, so run.sh will use ${LINUX_VENV_DIR##*/} instead."
fi

if [[ ! -x "$LINUX_VENV_PYTHON" ]]; then
    echo "Creating Linux virtual environment at ${LINUX_VENV_DIR##*/}..."
    python3 -m venv "$LINUX_VENV_DIR"
fi

echo "Installing or updating Python packages for Linux..."
"$LINUX_VENV_PYTHON" -m pip install --upgrade pip
"$LINUX_VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"

echo "Starting Wideye local host..."
echo "Host: $APP_HOST"
echo "Port: $APP_PORT"
echo "Debug: $APP_DEBUG"

export APP_HOST APP_PORT APP_DEBUG
exec "$LINUX_VENV_PYTHON" "$PROJECT_ROOT/main.py"
