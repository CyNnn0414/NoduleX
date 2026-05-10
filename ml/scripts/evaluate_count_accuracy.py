from __future__ import annotations

import argparse
import csv
import itertools
import json
from math import sqrt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from backend.app.config import Settings
from backend.app.services.inference import InferenceService, SliceCandidate
from ml.lumenai_ml.constants import YOLO_CLASSES
from ml.lumenai_ml.preprocessing import preprocess_study, save_slice_images
from ml.lumenai_ml.xlsx import read_first_sheet_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate patient-level nodule count accuracy against the LIDC spreadsheet.")
    parser.add_argument("--manifest", type=Path, required=True, help="Prepared dataset manifest.json")
    parser.add_argument("--count-xlsx", type=Path, required=True, help="Path to lidc-idri-nodule-counts-6-23-2015.xlsx")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--tune-split", type=str, default="val")
    parser.add_argument("--report-split", type=str, default="test")
    parser.add_argument("--conf-values", type=str, default="0.10,0.15,0.20,0.25,0.30,0.35")
    parser.add_argument("--singleton-values", type=str, default="0.70,0.76,0.82,0.88")
    parser.add_argument("--centroid-values", type=str, default="0.10,0.12,0.14")
    parser.add_argument("--predict-batch-size", type=int, default=32)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/experiments/lidc_first200/count_metrics.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("artifacts/experiments/lidc_first200/count_predictions.csv"))
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def load_count_map(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in read_first_sheet_rows(path)[1:]:
        patient_id = row[0]
        count_ge_3mm = row[2]
        if patient_id:
            counts[str(patient_id).strip()] = int(count_ge_3mm or 0)
    return counts


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def unique_patients_for_split(manifest: list[dict], split: str) -> list[dict]:
    seen: set[str] = set()
    selected: list[dict] = []
    for item in manifest:
        if item.get("split") != split:
            continue
        patient_id = item["patient_id"]
        if patient_id in seen:
            continue
        seen.add(patient_id)
        selected.append(item)
    return selected


def build_raw_candidates(
    service: InferenceService,
    model: YOLO,
    series_dir: Path,
    image_size: int,
    min_conf: float,
    temp_root: Path,
    predict_batch_size: int,
    image_paths: list[str] | None = None,
) -> tuple[list[SliceCandidate], object]:
    processed = preprocess_study(series_dir, image_size=image_size)
    if image_paths:
        resolved_image_paths = [Path(path) for path in image_paths]
    else:
        image_dir = temp_root / series_dir.name
        resolved_image_paths = save_slice_images(processed, image_dir, prefix=series_dir.name)
    preview_urls = [str(path) for path in resolved_image_paths]
    candidates: list[SliceCandidate] = []
    batch_size = max(1, predict_batch_size)
    for start_index in range(0, len(resolved_image_paths), batch_size):
        batch_paths = resolved_image_paths[start_index : start_index + batch_size]
        results = model.predict(
            [str(path) for path in batch_paths],
            imgsz=image_size,
            conf=min_conf,
            verbose=False,
            device="cpu",
        )
        for offset, result in enumerate(results):
            slice_index = start_index + offset
            geometry = processed.slice_geometries[slice_index]
            names = result.names
            for box in result.boxes:
                class_name = names.get(int(box.cls.item()), YOLO_CLASSES[int(box.cls.item())])
                mapped = service._map_yolo_class(class_name)
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                x1, y1, x2, y2 = service._clamp_bbox(x1, y1, x2, y2, geometry.resized_width, geometry.resized_height)
                width_mm, height_mm = service._bbox_size_mm((x1, y1, x2, y2), geometry)
                candidates.append(
                    SliceCandidate(
                        slice_index=slice_index,
                        bbox_px=(x1, y1, x2, y2),
                        bbox_norm=(
                            x1 / geometry.resized_width,
                            y1 / geometry.resized_height,
                            (x2 - x1) / geometry.resized_width,
                            (y2 - y1) / geometry.resized_height,
                        ),
                        confidence=float(box.conf.item()),
                        classification=mapped,
                        width_mm=width_mm,
                        height_mm=height_mm,
                        preview_image_url=preview_urls[slice_index],
                        centroid=((x1 + x2) / (2 * geometry.resized_width), (y1 + y2) / (2 * geometry.resized_height)),
                    )
                )
    return candidates, processed


def score_predictions(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"exact_match_accuracy": 0.0, "within_one_accuracy": 0.0, "mae": 0.0, "rmse": 0.0, "mean_bias": 0.0}
    errors = [row["predicted_count"] - row["ground_truth_count"] for row in rows]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = sqrt(sum((error * error) for error in errors) / len(errors))
    exact = sum(error == 0 for error in errors) / len(errors)
    within_one = sum(abs(error) <= 1 for error in errors) / len(errors)
    mean_bias = sum(errors) / len(errors)
    return {
        "exact_match_accuracy": exact,
        "within_one_accuracy": within_one,
        "mae": mae,
        "rmse": rmse,
        "mean_bias": mean_bias,
    }


def evaluate_split(
    entries: list[dict],
    count_map: dict[str, int],
    raw_cache: dict[str, tuple[list[SliceCandidate], object]],
    service: InferenceService,
    conf_threshold: float,
    singleton_confidence_threshold: float,
    centroid_distance_threshold: float,
) -> tuple[list[dict], dict[str, float]]:
    rows: list[dict] = []
    for entry in entries:
        patient_id = entry["patient_id"]
        raw_candidates, processed = raw_cache[patient_id]
        filtered = [candidate for candidate in raw_candidates if candidate.confidence >= conf_threshold]
        findings = service._group_candidates(
            filtered,
            processed,
            centroid_distance_threshold=centroid_distance_threshold,
            singleton_confidence_threshold=singleton_confidence_threshold,
        )
        rows.append(
            {
                "patient_id": patient_id,
                "series_uid": entry["series_uid"],
                "split": entry["split"],
                "ground_truth_count": int(count_map.get(patient_id, 0)),
                "predicted_count": len(findings),
            }
        )
    return rows, score_predictions(rows)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    count_map = load_count_map(args.count_xlsx)

    tune_entries = unique_patients_for_split(manifest, args.tune_split)
    report_entries = unique_patients_for_split(manifest, args.report_split)
    if not report_entries:
        raise ValueError(f"No patients were found for split={args.report_split!r}.")

    settings = Settings(yolo_weights_path=args.weights, default_slice_size=args.image_size, demo_mode=False)
    service = InferenceService(settings)
    model = YOLO(str(args.weights))

    conf_values = parse_float_list(args.conf_values)
    singleton_values = parse_float_list(args.singleton_values)
    centroid_values = parse_float_list(args.centroid_values)
    min_conf = min(conf_values)

    raw_cache: dict[str, tuple[list[SliceCandidate], object]] = {}
    temp_root = args.output_json.parent / "eval_slices"
    for entry in {item["patient_id"]: item for item in tune_entries + report_entries}.values():
        raw_cache[entry["patient_id"]] = build_raw_candidates(
            service=service,
            model=model,
            series_dir=Path(entry["series_dir"]),
            image_size=args.image_size,
            min_conf=min_conf,
            temp_root=temp_root,
            predict_batch_size=args.predict_batch_size,
            image_paths=entry.get("image_paths"),
        )

    parameter_grid = list(itertools.product(conf_values, singleton_values, centroid_values))
    tuning_results = []
    for conf_threshold, singleton_threshold, centroid_threshold in parameter_grid:
        tune_rows, tune_metrics = evaluate_split(
            tune_entries if tune_entries else report_entries,
            count_map,
            raw_cache,
            service,
            conf_threshold=conf_threshold,
            singleton_confidence_threshold=singleton_threshold,
            centroid_distance_threshold=centroid_threshold,
        )
        tuning_results.append(
            {
                "conf_threshold": conf_threshold,
                "singleton_confidence_threshold": singleton_threshold,
                "centroid_distance_threshold": centroid_threshold,
                "metrics": tune_metrics,
                "cases": len(tune_rows),
            }
        )

    best = min(
        tuning_results,
        key=lambda item: (
            item["metrics"]["mae"],
            -item["metrics"]["exact_match_accuracy"],
            -item["metrics"]["within_one_accuracy"],
        ),
    )

    report_rows, report_metrics = evaluate_split(
        report_entries,
        count_map,
        raw_cache,
        service,
        conf_threshold=best["conf_threshold"],
        singleton_confidence_threshold=best["singleton_confidence_threshold"],
        centroid_distance_threshold=best["centroid_distance_threshold"],
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "series_uid", "split", "ground_truth_count", "predicted_count"])
        writer.writeheader()
        writer.writerows(report_rows)

    payload = {
        "tune_split": args.tune_split,
        "report_split": args.report_split,
        "best_parameters": {
            "conf_threshold": best["conf_threshold"],
            "singleton_confidence_threshold": best["singleton_confidence_threshold"],
            "centroid_distance_threshold": best["centroid_distance_threshold"],
        },
        "tuning_metrics": best["metrics"],
        "report_metrics": report_metrics,
        "report_case_count": len(report_rows),
        "output_csv": str(args.output_csv),
    }
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
