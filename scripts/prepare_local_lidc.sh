#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/yuxinzhang/Desktop/BC senior 2nd/Biomedical Image Analysis/Final_Project/Dataset"
OUT="${1:-./artifacts/lidc}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" ml/scripts/prepare_lidc_dataset.py \
  --lidc-root "$ROOT" \
  --output-root "$OUT"
