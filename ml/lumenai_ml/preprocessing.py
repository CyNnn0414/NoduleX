from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pydicom
from pydicom.dataset import FileDataset
from scipy.ndimage import zoom

from .constants import DEFAULT_IMAGE_SIZE, ISOTROPIC_SPACING_MM, WINDOW_MAX_HU, WINDOW_MIN_HU


@dataclass
class SliceGeometry:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    z_position_mm: float
    pixel_spacing_xy: tuple[float, float]


@dataclass
class ProcessedStudy:
    volume_hu: np.ndarray
    resampled_volume: np.ndarray
    original_spacing: tuple[float, float, float]
    target_spacing: tuple[float, float, float]
    original_z_positions: list[float]
    target_z_positions: list[float]
    original_slice_images_rgb: list[np.ndarray]
    original_slice_geometries: list[SliceGeometry]
    slice_images_rgb: list[np.ndarray]
    slice_geometries: list[SliceGeometry]


@dataclass
class StudyDicomContext:
    patient_name: str | None = None
    patient_id: str | None = None
    patient_sex: str | None = None
    patient_age: int | None = None
    accession_number: str | None = None
    study_description: str | None = None
    modality: str | None = None
    collected_at: datetime | None = None


def discover_dicom_files(series_dir: Path) -> list[Path]:
    return sorted(path for path in series_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".dcm", ""})


def _slice_sort_key(dataset: FileDataset) -> tuple[float, int]:
    image_position = getattr(dataset, "ImagePositionPatient", None)
    z_pos = float(image_position[2]) if image_position is not None else 0.0
    instance_number = int(getattr(dataset, "InstanceNumber", 0))
    return z_pos, instance_number


def _clean_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_patient_sex(value: object | None) -> str | None:
    text = (_clean_text(value) or "").upper()
    if text.startswith("M"):
        return "male"
    if text.startswith("F"):
        return "female"
    if text.startswith("O"):
        return "other"
    return None


def _parse_patient_age(value: object | None) -> int | None:
    text = _clean_text(value)
    if not text:
        return None
    digits = "".join(character for character in text if character.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _parse_study_datetime(dataset: FileDataset) -> datetime | None:
    date_text = _clean_text(getattr(dataset, "StudyDate", None))
    time_text = _clean_text(getattr(dataset, "StudyTime", None)) or "000000"
    if not date_text or len(date_text) != 8:
        return None
    time_digits = "".join(character for character in time_text if character.isdigit())
    time_digits = (time_digits + "000000")[:6]
    try:
        return datetime.strptime(f"{date_text}{time_digits}", "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _read_datasets(series_dir: Path, *, stop_before_pixels: bool) -> list[FileDataset]:
    datasets: list[FileDataset] = []
    for dicom_path in discover_dicom_files(series_dir):
        try:
            datasets.append(pydicom.dcmread(str(dicom_path), stop_before_pixels=stop_before_pixels, force=True))
        except Exception:
            continue
    return datasets


def load_primary_datasets(series_dir: Path, *, stop_before_pixels: bool = False) -> list[FileDataset]:
    datasets = _read_datasets(series_dir, stop_before_pixels=stop_before_pixels)
    if not datasets:
        raise FileNotFoundError(f"No readable DICOM files found in {series_dir}")

    valid = [
        dataset
        for dataset in datasets
        if int(getattr(dataset, "Rows", 0) or 0) > 0 and int(getattr(dataset, "Columns", 0) or 0) > 0
    ]
    if not valid:
        valid = datasets

    grouped: dict[str, list[FileDataset]] = {}
    for dataset in valid:
        series_uid = _clean_text(getattr(dataset, "SeriesInstanceUID", None))
        study_uid = _clean_text(getattr(dataset, "StudyInstanceUID", None))
        key = series_uid or study_uid or "default"
        grouped.setdefault(key, []).append(dataset)

    def _group_score(group: list[FileDataset]) -> tuple[int, int, int]:
        shape_count = Counter((int(getattr(item, "Rows", 0) or 0), int(getattr(item, "Columns", 0) or 0)) for item in group)
        dominant_shape_count = shape_count.most_common(1)[0][1] if shape_count else 0
        non_localizer = sum("LOCALIZER" not in {str(value).upper() for value in getattr(item, "ImageType", [])} for item in group)
        return (len(group), dominant_shape_count, non_localizer)

    primary_group = max(grouped.values(), key=_group_score)
    shape_count = Counter((int(getattr(item, "Rows", 0) or 0), int(getattr(item, "Columns", 0) or 0)) for item in primary_group)
    if shape_count:
        dominant_shape = shape_count.most_common(1)[0][0]
        primary_group = [
            item
            for item in primary_group
            if (int(getattr(item, "Rows", 0) or 0), int(getattr(item, "Columns", 0) or 0)) == dominant_shape
        ]

    filtered = [
        item
        for item in primary_group
        if "LOCALIZER" not in {str(value).upper() for value in getattr(item, "ImageType", [])}
    ] or primary_group
    filtered.sort(key=_slice_sort_key)
    return filtered


def extract_study_context(series_dir: Path) -> StudyDicomContext:
    datasets = load_primary_datasets(series_dir, stop_before_pixels=True)
    reference = datasets[len(datasets) // 2]
    return StudyDicomContext(
        patient_name=_clean_text(getattr(reference, "PatientName", None)),
        patient_id=_clean_text(getattr(reference, "PatientID", None)),
        patient_sex=_normalize_patient_sex(getattr(reference, "PatientSex", None)),
        patient_age=_parse_patient_age(getattr(reference, "PatientAge", None)),
        accession_number=_clean_text(getattr(reference, "AccessionNumber", None)),
        study_description=_clean_text(getattr(reference, "StudyDescription", None))
        or _clean_text(getattr(reference, "SeriesDescription", None)),
        modality=_clean_text(getattr(reference, "Modality", None)),
        collected_at=_parse_study_datetime(reference),
    )


def load_dicom_volume(series_dir: Path) -> tuple[np.ndarray, list[FileDataset], tuple[float, float, float], list[float]]:
    datasets = load_primary_datasets(series_dir, stop_before_pixels=False)
    slices_hu: list[np.ndarray] = []
    for dataset in datasets:
        slice_pixels = dataset.pixel_array.astype(np.float32)
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        slices_hu.append((slice_pixels * slope) + intercept)
    volume_hu = np.stack(slices_hu, axis=0)

    pixel_spacing = getattr(datasets[0], "PixelSpacing", [1.0, 1.0])
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])

    if len(datasets) > 1:
        z_positions = [float(getattr(dataset, "ImagePositionPatient", [0.0, 0.0, index])[2]) for index, dataset in enumerate(datasets)]
        slice_thickness = abs(z_positions[1] - z_positions[0]) or float(getattr(datasets[0], "SliceThickness", 1.0))
    else:
        z_positions = [float(getattr(datasets[0], "ImagePositionPatient", [0.0, 0.0, 0.0])[2])]
        slice_thickness = float(getattr(datasets[0], "SliceThickness", 1.0))

    return volume_hu, datasets, (slice_thickness, row_spacing, col_spacing), z_positions


def resample_isotropic(volume: np.ndarray, spacing: tuple[float, float, float], target_spacing: tuple[float, float, float]) -> np.ndarray:
    zoom_factors = tuple(source / target for source, target in zip(spacing, target_spacing))
    return zoom(volume, zoom=zoom_factors, order=1)


def window_lung(volume_hu: np.ndarray) -> np.ndarray:
    clipped = np.clip(volume_hu, WINDOW_MIN_HU, WINDOW_MAX_HU)
    return (clipped - WINDOW_MIN_HU) / (WINDOW_MAX_HU - WINDOW_MIN_HU)


def resize_for_yolo(slice_image: np.ndarray, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    return cv2.resize(slice_image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def to_rgb(slice_image: np.ndarray) -> np.ndarray:
    slice_uint8 = np.clip(slice_image * 255.0, 0, 255).astype(np.uint8)
    return np.stack([slice_uint8] * 3, axis=-1)


def preprocess_study(series_dir: Path, image_size: int = DEFAULT_IMAGE_SIZE) -> ProcessedStudy:
    volume_hu, datasets, original_spacing, original_z_positions = load_dicom_volume(series_dir)
    resampled_volume = resample_isotropic(volume_hu, original_spacing, ISOTROPIC_SPACING_MM)
    original_normalized = window_lung(volume_hu)
    normalized = window_lung(resampled_volume)

    z_start = min(original_z_positions)
    z_direction = 1 if original_z_positions[-1] >= original_z_positions[0] else -1
    target_z_positions = [z_start + z_direction * index * ISOTROPIC_SPACING_MM[0] for index in range(normalized.shape[0])]

    original_slice_images_rgb: list[np.ndarray] = []
    original_geometries: list[SliceGeometry] = []
    slice_images_rgb: list[np.ndarray] = []
    geometries: list[SliceGeometry] = []

    for slice_index, slice_image in enumerate(original_normalized):
        resized = resize_for_yolo(slice_image, image_size=image_size)
        rgb = to_rgb(resized)
        original_slice_images_rgb.append(rgb)
        original_geometries.append(
            SliceGeometry(
                original_width=int(slice_image.shape[1]),
                original_height=int(slice_image.shape[0]),
                resized_width=image_size,
                resized_height=image_size,
                z_position_mm=original_z_positions[slice_index],
                pixel_spacing_xy=(original_spacing[1], original_spacing[2]),
            )
        )

    scale_y = normalized.shape[1] / volume_hu.shape[1]
    scale_x = normalized.shape[2] / volume_hu.shape[2]
    pixel_spacing_xy = (
        original_spacing[1] / scale_y if scale_y else original_spacing[1],
        original_spacing[2] / scale_x if scale_x else original_spacing[2],
    )

    for slice_index, slice_image in enumerate(normalized):
        resized = resize_for_yolo(slice_image, image_size=image_size)
        rgb = to_rgb(resized)
        slice_images_rgb.append(rgb)
        geometries.append(
            SliceGeometry(
                original_width=int(slice_image.shape[1]),
                original_height=int(slice_image.shape[0]),
                resized_width=image_size,
                resized_height=image_size,
                z_position_mm=target_z_positions[slice_index],
                pixel_spacing_xy=pixel_spacing_xy,
            )
        )

    return ProcessedStudy(
        volume_hu=volume_hu,
        resampled_volume=resampled_volume,
        original_spacing=original_spacing,
        target_spacing=ISOTROPIC_SPACING_MM,
        original_z_positions=original_z_positions,
        target_z_positions=target_z_positions,
        original_slice_images_rgb=original_slice_images_rgb,
        original_slice_geometries=original_geometries,
        slice_images_rgb=slice_images_rgb,
        slice_geometries=geometries,
    )


def save_rgb_images(images: list[np.ndarray], output_dir: Path, prefix: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for index, image in enumerate(images):
        path = output_dir / f"{prefix}_{index:04d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image_paths.append(path)
    return image_paths


def save_slice_images(processed_study: ProcessedStudy, output_dir: Path, prefix: str) -> list[Path]:
    return save_rgb_images(processed_study.slice_images_rgb, output_dir, prefix)
