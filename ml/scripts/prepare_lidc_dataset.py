from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from ml.lumenai_ml.constants import DEFAULT_IMAGE_SIZE, YOLO_CLASSES
from ml.lumenai_ml.lidc import (
    find_annotation_xmls,
    index_series_by_uid,
    map_annotations_to_processed_slices,
    parse_lidc_xml,
    write_yolo_label_file,
)
from ml.lumenai_ml.preprocessing import preprocess_study, save_slice_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LIDC-IDRI slices and YOLO labels.")
    parser.add_argument("--lidc-root", type=Path, required=True, help="Path to the local LIDC-IDRI dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for YOLO-ready images and labels.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patient-list", type=Path, help="Optional text or JSON file listing patient IDs to include.")
    parser.add_argument("--patient-limit", type=int, help="Optional cap after patient selection order is resolved.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def patient_id_from_series_dir(series_dir: Path) -> str:
    parts = list(series_dir.parts)
    for part in reversed(parts):
        if part.startswith("LIDC-IDRI-"):
            return part
    return series_dir.parents[1].name if len(series_dir.parents) > 1 else series_dir.name


def load_patient_list(path: Path) -> list[str]:
    text = path.read_text().strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            loaded = loaded.get("patient_ids", [])
        return [str(value).strip() for value in loaded if str(value).strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def split_patients(patient_ids: Iterable[str], seed: int, train_ratio: float, val_ratio: float) -> dict[str, str]:
    ordered = list(patient_ids)
    shuffled = ordered[:]
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_cut = int(total * train_ratio)
    val_cut = train_cut + int(total * val_ratio)
    assignments: dict[str, str] = {}
    for index, patient_id in enumerate(shuffled):
        if index < train_cut:
            split = "train"
        elif index < val_cut:
            split = "val"
        else:
            split = "test"
        assignments[patient_id] = split
    return assignments


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    if args.train_ratio <= 0 or args.val_ratio < 0 or (args.train_ratio + args.val_ratio) >= 1:
        raise ValueError("train-ratio must be > 0, val-ratio must be >= 0, and train-ratio + val-ratio must be < 1.")
    requested_patient_ids = load_patient_list(args.patient_list) if args.patient_list else None

    annotation_xmls = find_annotation_xmls(args.lidc_root)
    if not annotation_xmls:
        raise FileNotFoundError("No XML annotations found under the provided LIDC root.")

    series_index = index_series_by_uid(args.lidc_root, patient_ids=set(requested_patient_ids) if requested_patient_ids else None)
    studies = []
    for xml_path in annotation_xmls:
        annotation = parse_lidc_xml(xml_path)
        series_dir = series_index.get(annotation.series_uid)
        if series_dir is None:
            continue
        patient_id = annotation.patient_id or patient_id_from_series_dir(series_dir)
        studies.append((patient_id, annotation, series_dir))

    available_patient_ids = sorted({patient_id for patient_id, _, _ in studies})
    if args.patient_list:
        available_patient_set = set(available_patient_ids)
        patient_ids = [patient_id for patient_id in requested_patient_ids if patient_id in available_patient_set]
    else:
        patient_ids = available_patient_ids
    if args.patient_limit is not None:
        patient_ids = patient_ids[: args.patient_limit]
    selected_patients = set(patient_ids)
    if not selected_patients:
        raise ValueError("No patients were selected for dataset preparation.")

    patient_splits = split_patients(patient_ids, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    output_root: Path = args.output_root
    manifest = []

    selected_studies = [(patient_id, annotation, series_dir) for patient_id, annotation, series_dir in studies if patient_id in selected_patients]
    total_studies = len(selected_studies)
    for index, (patient_id, annotation, series_dir) in enumerate(selected_studies, start=1):
        split = patient_splits[patient_id]
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split

        processed = preprocess_study(series_dir, image_size=args.image_size)
        image_paths = save_slice_images(processed, image_dir, prefix=annotation.series_uid.replace(".", "_"))
        labels_by_slice = map_annotations_to_processed_slices(annotation, processed)

        for slice_index, image_path in enumerate(image_paths):
            label_path = label_dir / f"{image_path.stem}.txt"
            write_yolo_label_file(label_path, labels_by_slice.get(slice_index, []))

        manifest.append(
            {
                "patient_id": patient_id,
                "series_uid": annotation.series_uid,
                "split": split,
                "slice_count": len(image_paths),
                "series_dir": str(series_dir),
            }
        )
        if index % 25 == 0 or index == total_studies:
            print(f"[{index}/{total_studies}] prepared {patient_id} -> {split}")

    dataset_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(YOLO_CLASSES)},
    }
    (output_root / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False))
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_root / "patient_splits.json").write_text(json.dumps(patient_splits, indent=2, sort_keys=True))
    (output_root / "selected_patients.json").write_text(json.dumps({"patient_ids": patient_ids}, indent=2))
    print(f"Prepared {len(manifest)} annotated studies at {output_root}")


if __name__ == "__main__":
    main()
