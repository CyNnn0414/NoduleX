from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.lumenai_ml.lidc import find_annotation_xmls, index_series_by_uid, parse_lidc_xml
from ml.lumenai_ml.xlsx import read_first_sheet_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a first-N LIDC training and count-accuracy experiment.")
    parser.add_argument("--lidc-root", type=Path, required=True)
    parser.add_argument("--count-xlsx", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("artifacts/lidc"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments/lidc_first200"))
    parser.add_argument("--patient-count", type=int, default=200)
    parser.add_argument("--patient-offset", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", type=str, default="yolov8n.yaml")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--positive-train-only",
        action="store_true",
        help="Use only slices with non-empty YOLO label files for the training split.",
    )
    parser.add_argument(
        "--skip-extra-eval",
        action="store_true",
        help="Stop after YOLO training instead of running the slower mAP and count evaluation scripts.",
    )
    parser.add_argument("--no-train-val", action="store_true", help="Disable YOLO validation inside the training loop.")
    parser.add_argument("--save-trained-copy", type=Path, default=None)
    return parser.parse_args()


def load_patient_window(count_xlsx: Path, patient_count: int, patient_offset: int = 0) -> list[str]:
    patient_ids: list[str] = []
    seen = 0
    for row in read_first_sheet_rows(count_xlsx)[1:]:
        if not row or not row[0]:
            continue
        if seen < patient_offset:
            seen += 1
            continue
        patient_ids.append(str(row[0]).strip())
        if len(patient_ids) >= patient_count:
            break
    return patient_ids


def run_command(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def split_patients(patient_ids: list[str], seed: int) -> dict[str, str]:
    import random

    shuffled = patient_ids[:]
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if total <= 2:
        buckets = {"train": shuffled[:1], "test": shuffled[1:]}
    else:
        val_count = max(1, int(round(total * 0.15)))
        test_count = max(1, int(round(total * 0.15)))
        train_count = max(1, total - val_count - test_count)
        while train_count + val_count + test_count > total:
            if train_count >= max(val_count, test_count) and train_count > 1:
                train_count -= 1
            elif val_count >= test_count and val_count > 1:
                val_count -= 1
            else:
                test_count -= 1
        while train_count + val_count + test_count < total:
            train_count += 1
        buckets = {
            "train": shuffled[:train_count],
            "val": shuffled[train_count : train_count + val_count],
            "test": shuffled[train_count + val_count : train_count + val_count + test_count],
        }
    assignments: dict[str, str] = {}
    for split, ids in buckets.items():
        for patient_id in ids:
            assignments[patient_id] = split
    return assignments


def patient_id_from_series_dir(series_dir: Path) -> str:
    for part in reversed(series_dir.parts):
        if part.startswith("LIDC-IDRI-"):
            return part
    return series_dir.name


def label_path_for_image(prepared_root: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(prepared_root / "images")
    return prepared_root / "labels" / relative.with_suffix(".txt")


def has_positive_label(prepared_root: Path, image_path: Path) -> bool:
    label_path = label_path_for_image(prepared_root, image_path)
    if not label_path.exists():
        return False
    return bool(label_path.read_text().strip())


def sanitized_yolo_labels(prepared_root: Path, image_path: Path) -> list[str]:
    label_path = label_path_for_image(prepared_root, image_path)
    if not label_path.exists():
        return []

    sanitized: list[str] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError:
            continue

        x1 = max(0.0, x_center - width / 2)
        y1 = max(0.0, y_center - height / 2)
        x2 = min(1.0, x_center + width / 2)
        y2 = min(1.0, y_center + height / 2)
        clipped_width = x2 - x1
        clipped_height = y2 - y1
        if clipped_width <= 0 or clipped_height <= 0:
            continue

        clipped_x_center = x1 + clipped_width / 2
        clipped_y_center = y1 + clipped_height / 2
        sanitized.append(
            f"{class_id} {clipped_x_center:.6f} {clipped_y_center:.6f} "
            f"{clipped_width:.6f} {clipped_height:.6f}"
        )
    return sanitized


def materialize_subset_slice(
    *,
    prepared_root: Path,
    output_root: Path,
    image_path: Path,
    split: str,
    labels: list[str],
) -> Path:
    target_image = output_root / "images" / split / image_path.name
    target_label = output_root / "labels" / split / image_path.with_suffix(".txt").name
    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_label.parent.mkdir(parents=True, exist_ok=True)

    if target_image.exists() or target_image.is_symlink():
        target_image.unlink()
    shutil.copy2(image_path, target_image)

    target_label.write_text("\n".join(labels) + ("\n" if labels else ""))
    return target_image


def build_subset_from_prepared(
    *,
    lidc_root: Path,
    prepared_root: Path,
    output_root: Path,
    patient_ids: list[str],
    seed: int,
    positive_train_only: bool = False,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    selected = set(patient_ids)
    patient_splits = split_patients(patient_ids, seed=seed)

    series_index = index_series_by_uid(lidc_root, patient_ids=selected)
    series_to_patient: dict[str, str] = {}
    manifest: list[dict] = []
    train_lines: list[str] = []
    val_lines: list[str] = []
    test_lines: list[str] = []

    for xml_path in find_annotation_xmls(lidc_root):
        annotation = parse_lidc_xml(xml_path)
        series_uid = annotation.series_uid
        series_dir = series_index.get(series_uid)
        if series_dir is None:
            continue
        patient_id = annotation.patient_id or patient_id_from_series_dir(series_dir)
        if patient_id not in selected:
            continue
        series_key = series_uid.replace(".", "_")
        if series_key in series_to_patient:
            continue
        series_to_patient[series_key] = patient_id

        image_paths: list[Path] = []
        for existing_split in ("train", "val", "test"):
            image_paths.extend(sorted((prepared_root / "images" / existing_split).glob(f"{series_key}_*.png")))
        if not image_paths:
            continue

        assigned_split = patient_splits[patient_id]
        assigned_split = patient_splits[patient_id]
        selected_slices: list[tuple[Path, list[str]]] = []
        for image_path in image_paths:
            labels = sanitized_yolo_labels(prepared_root, image_path)
            if assigned_split == "train" and positive_train_only and not labels:
                continue
            selected_slices.append((image_path, labels))

        if assigned_split == "train" and positive_train_only and not selected_slices:
            continue

        subset_image_paths = [
            materialize_subset_slice(
                prepared_root=prepared_root,
                output_root=output_root,
                image_path=image_path,
                split=assigned_split,
                labels=labels,
            )
            for image_path, labels in selected_slices
        ]
        image_strings = [str(path.resolve()) for path in subset_image_paths]
        if assigned_split == "train":
            train_lines.extend(image_strings)
        elif assigned_split == "val":
            val_lines.extend(image_strings)
        else:
            test_lines.extend(image_strings)

        manifest.append(
            {
                "patient_id": patient_id,
                "series_uid": series_uid,
                "split": assigned_split,
                "series_dir": str(series_dir),
                "image_paths": image_strings,
                "slice_count": len(image_strings),
                "source_slice_count": len(image_paths),
                "positive_train_only": bool(assigned_split == "train" and positive_train_only),
            }
        )

    if not manifest:
        raise ValueError("No prepared image slices were found for the selected patient subset.")

    (output_root / "train.txt").write_text("\n".join(train_lines) + ("\n" if train_lines else ""))
    (output_root / "val.txt").write_text("\n".join(val_lines) + ("\n" if val_lines else ""))
    (output_root / "test.txt").write_text("\n".join(test_lines) + ("\n" if test_lines else ""))
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_root / "patient_splits.json").write_text(json.dumps(patient_splits, indent=2, sort_keys=True))

    dataset_yaml = {
        "path": str(output_root.resolve()),
        "train": str((output_root / "train.txt").resolve()),
        "val": str((output_root / "val.txt").resolve()),
        "test": str((output_root / "test.txt").resolve()),
        "names": {0: "solid", 1: "part-solid", 2: "ground-glass"},
    }
    dataset_yaml_path = output_root / "dataset.yaml"
    dataset_yaml_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False))
    return dataset_yaml_path


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    patient_ids = load_patient_window(args.count_xlsx, args.patient_count, args.patient_offset)
    patient_list_path = args.output_root / "first200_patients.txt"
    patient_list_path.write_text("\n".join(patient_ids) + "\n")

    dataset_root = args.output_root / "dataset"
    project_root = args.output_root / "runs"
    run_name = "first200_yolo"
    weights_root = project_root / run_name / "weights"
    best_weights = weights_root / "best.pt"
    last_weights = weights_root / "last.pt"

    python = sys.executable

    if args.prepared_root.exists():
        dataset_yaml = build_subset_from_prepared(
            lidc_root=args.lidc_root,
            prepared_root=args.prepared_root,
            output_root=dataset_root,
            patient_ids=patient_ids,
            seed=args.seed,
            positive_train_only=args.positive_train_only,
        )
    else:
        run_command(
            [
                python,
                "ml/scripts/prepare_lidc_dataset.py",
                "--lidc-root",
                str(args.lidc_root),
                "--output-root",
                str(dataset_root),
                "--patient-list",
                str(patient_list_path),
                "--image-size",
                str(args.image_size),
                "--seed",
                str(args.seed),
            ]
        )
        dataset_yaml = dataset_root / "dataset.yaml"

    run_command(
        [
            python,
            "ml/scripts/train_yolo.py",
            "--dataset-yaml",
            str(dataset_yaml),
            "--model",
            args.model,
            "--epochs",
            str(args.epochs),
            "--imgsz",
            str(args.image_size),
            "--batch",
            str(args.batch),
            "--project",
            str(project_root),
            "--name",
            run_name,
            "--device",
            args.device,
            *(["--no-val"] if args.no_train_val else []),
            *(
                ["--save-trained-copy", str(args.save_trained_copy)]
                if args.save_trained_copy is not None
                else []
            ),
        ]
    )

    weights_path = last_weights if args.no_train_val else best_weights if best_weights.exists() else last_weights
    if not weights_path.exists():
        raise FileNotFoundError(f"Training finished but no weights were found under {weights_root}")

    map_json = args.output_root / "map_metrics.json"
    count_json = args.output_root / "count_metrics.json"
    count_csv = args.output_root / "count_predictions.csv"
    if not args.skip_extra_eval:
        run_command(
            [
                python,
                "ml/scripts/evaluate_yolo.py",
                "--dataset-yaml",
                str(dataset_yaml),
                "--weights",
                str(weights_path),
                "--imgsz",
                str(args.image_size),
                "--save-json",
                str(map_json),
            ]
        )

        run_command(
            [
                python,
                "ml/scripts/evaluate_count_accuracy.py",
                "--manifest",
                str(dataset_root / "manifest.json"),
                "--count-xlsx",
                str(args.count_xlsx),
                "--weights",
                str(weights_path),
                "--image-size",
                str(args.image_size),
                "--output-json",
                str(count_json),
                "--output-csv",
                str(count_csv),
            ]
        )

    summary = {
        "patient_count": len(patient_ids),
        "dataset_yaml": str(dataset_yaml),
        "weights": str(weights_path),
        "map_metrics_json": str(map_json) if map_json.exists() else None,
        "count_metrics_json": str(count_json) if count_json.exists() else None,
        "count_predictions_csv": str(count_csv) if count_csv.exists() else None,
        "extra_eval_skipped": args.skip_extra_eval,
    }
    (args.output_root / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
