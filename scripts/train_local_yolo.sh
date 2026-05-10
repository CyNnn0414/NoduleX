#!/usr/bin/env bash
set -euo pipefail

DATASET_YAML="${1:-./artifacts/lidc/dataset.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" ml/scripts/train_yolo.py \
  --dataset-yaml "$DATASET_YAML"
