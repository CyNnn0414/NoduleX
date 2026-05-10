#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="artifacts/logs"
LOG_FILE="$LOG_DIR/nodulex-launch.log"
mkdir -p "$LOG_DIR" ".cache/matplotlib" ".cache/ultralytics"
rm -f "$LOG_FILE"
touch "$LOG_FILE"

export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"

on_error() {
  local exit_code=$?
  echo
  echo "NoduleX could not start."
  echo "See the launcher log at: $LOG_FILE"
  echo "Press Enter to close this window."
  read -r _
  exit "$exit_code"
}

trap on_error ERR

log() {
  printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

materialize_icloud_files() {
  local placeholders
  placeholders=$(find ".venv" -type f -name "*.icloud" 2>/dev/null || true)
  if [ -z "$placeholders" ]; then
    return 0
  fi

  log "Found iCloud-offloaded files inside the local Python environment. Requesting local download..."
  while IFS= read -r placeholder; do
    [ -n "$placeholder" ] || continue
    if command -v brctl >/dev/null 2>&1; then
      brctl download "$placeholder" >>"$LOG_FILE" 2>&1 || true
    fi
    if command -v fileproviderctl >/dev/null 2>&1; then
      fileproviderctl materialize "$placeholder" >>"$LOG_FILE" 2>&1 || true
    fi
  done <<< "$placeholders"

  sleep 2
}

verify_python_stack() {
  ./.venv/bin/python - <<'PY' >/dev/null
import numpy  # noqa: F401
import cv2  # noqa: F401
PY
}

log "Preparing NoduleX in $PWD"

if [ ! -x ".venv/bin/python" ]; then
  log "Creating local Python environment..."
  python3 -m venv .venv
fi

if [ ! -x ".venv/bin/uvicorn" ]; then
  log "Installing local dependencies..."
  ./.venv/bin/python -m pip install -r backend/requirements.txt 2>&1 | tee -a "$LOG_FILE"
else
  log "Using existing local Python environment."
fi

materialize_icloud_files

if ! verify_python_stack 2>>"$LOG_FILE"; then
  log ""
  log "The local Python environment is still missing required binary files."
  log "This usually means iCloud offloaded part of .venv."
  log "In Finder, open this folder and download the .venv directory or the whole project for offline use:"
  log "$PWD"
  log "Then run ./start_nodulex.command again."
  exit 1
fi

log ""
log "Starting NoduleX at http://127.0.0.1:8000"
log "Keep this window open while using the website."
log "If the page still says it is offline, click Reconnect in the browser."
log ""

exec ./.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 2>&1 | tee -a "$LOG_FILE"
