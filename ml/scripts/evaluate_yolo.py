from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model on the LIDC-IDRI test split.")
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save-json", type=Path, default=Path("artifacts/eval_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    metrics = model.val(data=str(args.dataset_yaml), imgsz=args.imgsz, split="test")

    result = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
