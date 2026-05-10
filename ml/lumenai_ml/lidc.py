from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from lxml import etree
import pydicom

from .constants import TEXTURE_TO_CLASS, YOLO_CLASSES
from .preprocessing import ProcessedStudy


XML_NS = {"nih": "http://www.nih.gov"}


@dataclass
class SliceAnnotation:
    slice_index: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    diameter_mm: float
    malignancy: int | None
    z_position_mm: float


@dataclass
class AnnotationStudy:
    patient_id: str
    series_uid: str
    slice_annotations: list[SliceAnnotation]


def _text(node: etree._Element | None, default: str = "") -> str:
    return node.text.strip() if node is not None and node.text else default


def _findall(node: etree._Element, xpath: str) -> list[etree._Element]:
    return node.xpath(xpath, namespaces=XML_NS)


def _findone(node: etree._Element, xpath: str) -> etree._Element | None:
    found = _findall(node, xpath)
    return found[0] if found else None


def _class_from_texture(texture_value: int | None) -> str:
    if texture_value is None:
        return "solid"
    return TEXTURE_TO_CLASS.get(texture_value, "solid")


def parse_lidc_xml(xml_path: Path) -> AnnotationStudy:
    root = etree.fromstring(xml_path.read_bytes())

    patient_id = _text(_findone(root, ".//nih:ResponseHeader/nih:PatientId"))
    series_uid = _text(_findone(root, ".//nih:ResponseHeader/nih:SeriesInstanceUid"))

    annotations: list[SliceAnnotation] = []

    for nodule in _findall(root, ".//nih:readingSession/nih:unblindedReadNodule"):
        texture_node = _findone(nodule, ".//nih:characteristics/nih:texture")
        malignancy_node = _findone(nodule, ".//nih:characteristics/nih:malignancy")
        texture = int(_text(texture_node, "4")) if texture_node is not None else 4
        malignancy = int(_text(malignancy_node, "0")) if malignancy_node is not None else None
        class_name = _class_from_texture(texture)

        for roi in _findall(nodule, ".//nih:roi"):
            inclusion = _text(_findone(roi, "./nih:inclusion"), "TRUE").upper()
            if inclusion == "FALSE":
                continue

            z_position_mm = float(_text(_findone(roi, "./nih:imageZposition"), "0"))
            points = [
                (
                    float(_text(_findone(edge_map, "./nih:xCoord"), "0")),
                    float(_text(_findone(edge_map, "./nih:yCoord"), "0")),
                )
                for edge_map in _findall(roi, "./nih:edgeMap")
            ]
            if len(points) < 3:
                continue

            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            bbox = (min(xs), min(ys), max(xs), max(ys))

            annotations.append(
                SliceAnnotation(
                    slice_index=-1,
                    class_name=class_name,
                    bbox_xyxy=bbox,
                    diameter_mm=0.0,
                    malignancy=malignancy,
                    z_position_mm=z_position_mm,
                )
            )

    return AnnotationStudy(patient_id=patient_id, series_uid=series_uid, slice_annotations=annotations)


def index_series_by_uid(lidc_root: Path, patient_ids: set[str] | None = None) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    if patient_ids:
        search_roots = []
        for patient_id in sorted(patient_ids):
            candidate = lidc_root / "TCIA_LIDC-IDRI_20200921" / "lidc_idri" / patient_id
            if candidate.exists():
                search_roots.append(candidate)
        if not search_roots:
            search_roots = [lidc_root]
    else:
        search_roots = [lidc_root]

    for root in search_roots:
        for dicom_path in root.rglob("*.dcm"):
            try:
                dataset = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
            except Exception:
                continue
            series_uid = getattr(dataset, "SeriesInstanceUID", None)
            if series_uid and series_uid not in indexed:
                indexed[series_uid] = dicom_path.parent
    return indexed


def find_annotation_xmls(lidc_root: Path) -> list[Path]:
    return sorted(path for path in lidc_root.rglob("*.xml") if path.is_file())


def map_annotations_to_processed_slices(annotation_study: AnnotationStudy, processed: ProcessedStudy, min_diameter_mm: float = 3.0) -> dict[int, list[tuple[int, float, float, float, float]]]:
    labels_by_slice: dict[int, list[tuple[int, float, float, float, float]]] = {}

    for annotation in annotation_study.slice_annotations:
        slice_index = int(np.argmin(np.abs(np.array(processed.target_z_positions) - annotation.z_position_mm)))
        geometry = processed.slice_geometries[slice_index]

        x_min, y_min, x_max, y_max = annotation.bbox_xyxy
        x_scale = geometry.resized_width / geometry.original_width
        y_scale = geometry.resized_height / geometry.original_height
        x_min *= x_scale
        x_max *= x_scale
        y_min *= y_scale
        y_max *= y_scale

        x_min = min(max(x_min, 0.0), float(geometry.resized_width))
        x_max = min(max(x_max, 0.0), float(geometry.resized_width))
        y_min = min(max(y_min, 0.0), float(geometry.resized_height))
        y_max = min(max(y_max, 0.0), float(geometry.resized_height))

        bbox_width_px = max(x_max - x_min, 1.0)
        bbox_height_px = max(y_max - y_min, 1.0)
        bbox_width_mm = bbox_width_px / geometry.resized_width * geometry.original_width * geometry.pixel_spacing_xy[1]
        bbox_height_mm = bbox_height_px / geometry.resized_height * geometry.original_height * geometry.pixel_spacing_xy[0]
        diameter_mm = max(bbox_width_mm, bbox_height_mm)
        if diameter_mm < min_diameter_mm:
            continue

        class_id = YOLO_CLASSES.index(annotation.class_name)
        x_center = ((x_min + x_max) / 2.0) / geometry.resized_width
        y_center = ((y_min + y_max) / 2.0) / geometry.resized_height
        width = bbox_width_px / geometry.resized_width
        height = bbox_height_px / geometry.resized_height
        labels_by_slice.setdefault(slice_index, []).append((class_id, x_center, y_center, width, height))

    return labels_by_slice


def write_yolo_label_file(label_path: Path, labels: Iterable[tuple[int, float, float, float, float]]) -> None:
    lines = [f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}" for class_id, x_center, y_center, width, height in labels]
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines))
