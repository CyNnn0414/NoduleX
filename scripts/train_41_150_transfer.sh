#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p ".cache/matplotlib" ".cache/ultralytics"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"

LIDC_ROOT="${LIDC_ROOT:-/Users/yuxinzhang/Desktop/BC senior 2nd/Biomedical Image Analysis/Final_Project/Dataset}"
COUNT_XLSX="${COUNT_XLSX:-$LIDC_ROOT/lidc-idri-nodule-counts-6-23-2015.xlsx}"
PREPARED_ROOT="${PREPARED_ROOT:-artifacts/lidc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/experiments/patients41_150_positive_fast_transfer}"
START_WEIGHTS="${START_WEIGHTS:-artifacts/weights/trained_21_40_positive.pt}"
FINAL_WEIGHTS="${FINAL_WEIGHTS:-artifacts/weights/trained_41_150_positive.pt}"
DEVICE="${DEVICE:-cpu}"
BATCH="${BATCH:-4}"
EPOCHS="${EPOCHS:-10}"
PATIENT_COUNT="${PATIENT_COUNT:-110}"
PATIENT_OFFSET="${PATIENT_OFFSET:-40}"
POSITIVE_TRAIN_ONLY="${POSITIVE_TRAIN_ONLY:-1}"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"

POSITIVE_ARGS=()
if [[ "$POSITIVE_TRAIN_ONLY" == "1" || "$POSITIVE_TRAIN_ONLY" == "true" ]]; then
  POSITIVE_ARGS+=(--positive-train-only)
fi

"$PYTHON_BIN" ml/scripts/run_first200_experiment.py \
  --lidc-root "$LIDC_ROOT" \
  --count-xlsx "$COUNT_XLSX" \
  --prepared-root "$PREPARED_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --patient-count "$PATIENT_COUNT" \
  --patient-offset "$PATIENT_OFFSET" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --model "$START_WEIGHTS" \
  "${POSITIVE_ARGS[@]}" \
  --no-train-val \
  --skip-extra-eval \
  --save-trained-copy "$FINAL_WEIGHTS"
