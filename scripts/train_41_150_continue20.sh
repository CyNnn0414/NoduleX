#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p ".cache/matplotlib" ".cache/ultralytics"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"

DATASET_YAML="${DATASET_YAML:-artifacts/experiments/patients41_150_positive_fast_transfer/dataset/dataset.yaml}"
START_WEIGHTS="${START_WEIGHTS:-artifacts/weights/trained_41_150_positive.pt}"
PROJECT_ROOT="${PROJECT_ROOT:-artifacts/experiments/patients41_150_positive_continue20}"
RUN_NAME="${RUN_NAME:-continue20_yolo}"
FINAL_WEIGHTS="${FINAL_WEIGHTS:-artifacts/weights/trained_41_150_positive.pt}"
DEVICE="${DEVICE:-cpu}"
BATCH="${BATCH:-4}"
EPOCHS="${EPOCHS:-20}"
IMGSZ="${IMGSZ:-640}"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"

"$PYTHON_BIN" ml/scripts/train_yolo.py \
  --dataset-yaml "$DATASET_YAML" \
  --model "$START_WEIGHTS" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --project "$PROJECT_ROOT/runs" \
  --name "$RUN_NAME" \
  --device "$DEVICE" \
  --no-val \
  --save-trained-copy "$FINAL_WEIGHTS"
