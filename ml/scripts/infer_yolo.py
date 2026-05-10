from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from ml.lumenai_ml.constants import YOLO_CLASSES
from ml.lumenai_ml.preprocessing import preprocess_study, save_slice_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a DICOM study.")
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/inference/findings.json"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = preprocess_study(args.study_dir, image_size=args.image_size)
    temp_images_dir = args.output_json.parent / "slices"
    image_paths = save_slice_images(processed, temp_images_dir, prefix=args.study_dir.name)

    model = YOLO(str(args.weights))
    results = model.predict([str(path) for path in image_paths], imgsz=args.image_size, conf=args.conf, verbose=False)

    findings = []
    for slice_index, result in enumerate(results):
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            width = x2 - x1
            height = y2 - y1
            findings.append(
                {
                    "slice_index": slice_index,
                    "class_name": names.get(class_id, YOLO_CLASSES[class_id]),
                    "confidence": float(box.conf.item()),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, width, height],
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(findings, indent=2))
    print(json.dumps({"finding_count": len(findings), "output_json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
