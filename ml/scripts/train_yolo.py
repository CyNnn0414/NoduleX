from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on prepared LIDC-IDRI slices.")
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--model", type=str, default="yolov8n.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--name", type=str, default="lidc_yolo")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--no-val", action="store_true", help="Disable YOLO validation during training.")
    parser.add_argument("--save-trained-copy", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    result = model.train(
        data=str(args.dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        device=args.device,
        workers=0,
        val=not args.no_val,
        plots=True,
    )

    if args.save_trained_copy is not None:
        best_path = Path(result.save_dir) / "weights" / "best.pt"
        last_path = Path(result.save_dir) / "weights" / "last.pt"
        source = last_path if args.no_val else best_path if best_path.exists() else last_path
        if not source.exists():
            raise FileNotFoundError(f"No trained weights found in {source.parent}")
        args.save_trained_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, args.save_trained_copy)


if __name__ == "__main__":
    main()
