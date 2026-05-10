#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="/Users/yuxinzhang/Desktop/BC senior 2nd/Biomedical Image Analysis/Final_Project/Dataset"
COUNT_XLSX="${COUNT_XLSX:-$ROOT/lidc-idri-nodule-counts-6-23-2015.xlsx}"
DEFAULT_MANIFEST="./artifacts/lidc/manifest.json"
DEFAULT_WEIGHTS="./artifacts/runs/lidc_yolo/weights/best.pt"
OUTPUT_JSON="${3:-./artifacts/experiments/lidc_count_eval/count_metrics.json}"
OUTPUT_CSV="${4:-./artifacts/experiments/lidc_count_eval/count_predictions.csv}"

if [ -x ".venv/bin/python" ]; then
  DEFAULT_PYTHON="./.venv/bin/python"
else
  DEFAULT_PYTHON="python3"
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"

mkdir -p ".cache/matplotlib" ".cache/ultralytics"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"

pick_first_existing() {
  local candidate
  for candidate in "$@"; do
    if [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if [ $# -ge 1 ]; then
  MANIFEST="$1"
else
  MANIFEST="$(pick_first_existing \
    "$DEFAULT_MANIFEST" \
    "./artifacts/experiments/lidc_first200/dataset/manifest.json" \
    "./artifacts/experiments/size_probe/dataset/manifest.json" \
    "./artifacts/experiments/smoke_first5_prepared3/dataset/manifest.json" \
    "./artifacts/experiments/smoke_first5_prepared2/dataset/manifest.json" \
    "./artifacts/experiments/smoke_first5/dataset/manifest.json")"
fi

if [ $# -ge 2 ]; then
  WEIGHTS="$2"
else
  WEIGHTS="$(pick_first_existing \
    "$DEFAULT_WEIGHTS" \
    "./artifacts/runs/nodulex_yolo_partial/weights/best.pt" \
    "./artifacts/experiments/lidc_first200/runs/first200_yolo/weights/best.pt" \
    "./artifacts/experiments/smoke_first5_prepared3/runs/first200_yolo/weights/best.pt" \
    "./artifacts/experiments/smoke_first5_prepared2/runs/first200_yolo/weights/best.pt" \
    "./artifacts/experiments/smoke_first5/runs/first200_yolo/weights/best.pt")"
fi

if [ ! -f "$COUNT_XLSX" ]; then
  echo "Count spreadsheet not found: $COUNT_XLSX" >&2
  exit 1
fi

if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
  echo "Weights not found: $WEIGHTS" >&2
  exit 1
fi

"$PYTHON_BIN" ml/scripts/evaluate_count_accuracy.py \
  --manifest "$MANIFEST" \
  --count-xlsx "$COUNT_XLSX" \
  --weights "$WEIGHTS" \
  --output-json "$OUTPUT_JSON" \
  --output-csv "$OUTPUT_CSV"
