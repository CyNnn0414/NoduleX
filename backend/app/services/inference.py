from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Literal

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.lumenai_ml.preprocessing import ProcessedStudy, preprocess_study, save_rgb_images, save_slice_images

from ..config import Settings
from ..models import BoundingBox, NoduleFinding, NoduleMeasurement, StudyRecord

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional at runtime
    YOLO = None


@dataclass
class SliceCandidate:
    slice_index: int
    bbox_px: tuple[float, float, float, float]
    bbox_norm: tuple[float, float, float, float]
    confidence: float
    classification: Literal["solid", "part-solid", "ground-glass"]
    width_mm: float
    height_mm: float
    preview_image_url: str
    centroid: tuple[float, float]


@dataclass
class DetectionCluster:
    candidates: list[SliceCandidate] = field(default_factory=list)

    @property
    def last(self) -> SliceCandidate:
        return self.candidates[-1]

    def add(self, candidate: SliceCandidate) -> None:
        self.candidates.append(candidate)


class InferenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._yolo_model = None
        self._predict_batch_size = 32

    def analyze(self, study: StudyRecord) -> StudyRecord:
        if not study.source_path:
            study.findings = self._demo_findings()
            study.basic_diagnosis = self._build_basic_diagnosis(study.findings, "demo")
            study.metadata.analysis_method = "demo"
            study.status = "ready"
            return study

        series_dir = Path(study.source_path)
        processed = preprocess_study(series_dir, image_size=self.settings.default_slice_size)
        viewer_preview_dir = series_dir / "previews"
        viewer_preview_paths = save_rgb_images(processed.original_slice_images_rgb, viewer_preview_dir, prefix="slice")
        viewer_preview_urls = [self._data_url(path) for path in viewer_preview_paths]
        analysis_preview_dir = series_dir / "analysis_previews"
        analysis_preview_paths = save_slice_images(processed, analysis_preview_dir, prefix="analysis")

        study.metadata.source_slice_count = len(processed.original_z_positions)
        study.metadata.analysis_slice_count = len(processed.target_z_positions)
        study.metadata.slice_count = len(viewer_preview_paths)
        study.metadata.voxel_spacing_mm = tuple(float(value) for value in processed.target_spacing)
        study.metadata.slice_image_urls = viewer_preview_urls
        if viewer_preview_urls:
            study.metadata.preview_image_url = viewer_preview_urls[len(viewer_preview_urls) // 2]

        if Path(self.settings.yolo_weights_path).exists() and YOLO is not None:
            try:
                candidates = self._detect_with_yolo_adaptive(analysis_preview_paths, viewer_preview_urls, processed)
                study.metadata.analysis_method = "yolo"
            except Exception:
                candidates = self._detect_with_heuristics(processed, viewer_preview_urls)
                study.metadata.analysis_method = "heuristic"
        elif self.settings.demo_mode:
            study.findings = self._demo_findings()
            study.basic_diagnosis = self._build_basic_diagnosis(study.findings, "demo")
            study.metadata.analysis_method = "demo"
            study.status = "ready"
            return study
        else:
            candidates = self._detect_with_heuristics(processed, viewer_preview_urls)
            study.metadata.analysis_method = "heuristic"

        study.findings = self._group_candidates(candidates, processed)
        if not study.findings and candidates:
            study.findings = self._single_slice_findings(
                candidates,
                processed,
                min_confidence=0.001,
                min_longest_mm=2.0,
                max_longest_mm=45.0,
                max_findings=8,
            )
        if not study.findings and study.metadata.analysis_method == "yolo":
            heuristic_candidates = self._detect_with_heuristics(processed, viewer_preview_urls)
            study.findings = self._group_candidates(
                heuristic_candidates,
                processed,
                centroid_distance_threshold=0.16,
                singleton_confidence_threshold=0.35,
                min_longest_mm=2.0,
                max_longest_mm=45.0,
                min_shape_ratio=0.08,
                max_findings=16,
            )
            if not study.findings and heuristic_candidates:
                study.findings = self._single_slice_findings(
                    heuristic_candidates,
                    processed,
                    min_confidence=0.35,
                    min_longest_mm=2.0,
                    max_longest_mm=45.0,
                    max_findings=10,
                )
            if study.findings:
                study.metadata.analysis_method = "yolo+heuristic"
        study.basic_diagnosis = self._build_basic_diagnosis(study.findings, study.metadata.analysis_method or "heuristic")
        study.status = "ready"
        return study

    def _detect_with_yolo_adaptive(
        self,
        preview_paths: list[Path],
        viewer_preview_urls: list[str],
        processed: ProcessedStudy,
    ) -> list[SliceCandidate]:
        model = self._load_yolo_model()
        if model is None:
            return self._detect_with_heuristics(processed, viewer_preview_urls)

        for conf in (0.20, 0.05, 0.01, 0.001):
            candidates = self._detect_with_yolo(
                preview_paths,
                viewer_preview_urls,
                processed,
                conf=conf,
                model=model,
            )
            if candidates:
                return candidates
        return []

    def _detect_with_yolo(
        self,
        preview_paths: list[Path],
        viewer_preview_urls: list[str],
        processed: ProcessedStudy,
        *,
        conf: float = 0.20,
        model=None,
    ) -> list[SliceCandidate]:
        model = model or self._load_yolo_model()
        if model is None:
            return self._detect_with_heuristics(processed, viewer_preview_urls)

        candidates: list[SliceCandidate] = []
        batch_size = max(1, self._predict_batch_size)
        for start_index in range(0, len(preview_paths), batch_size):
            batch_paths = preview_paths[start_index : start_index + batch_size]
            results = model.predict(
                [str(path) for path in batch_paths],
                imgsz=self.settings.default_slice_size,
                conf=conf,
                verbose=False,
                device="cpu",
            )
            for offset, result in enumerate(results):
                slice_index = start_index + offset
                geometry = processed.slice_geometries[slice_index]
                names = result.names
                for box in result.boxes:
                    class_name = names.get(int(box.cls.item()), "solid")
                    mapped = self._map_yolo_class(class_name)
                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                    x1, y1, x2, y2 = self._clamp_bbox(x1, y1, x2, y2, geometry.resized_width, geometry.resized_height)
                    width_mm, height_mm = self._bbox_size_mm((x1, y1, x2, y2), geometry)
                    viewer_slice_index = self._viewer_slice_index(processed, slice_index)
                    candidates.append(
                        SliceCandidate(
                            slice_index=slice_index,
                            bbox_px=(x1, y1, x2, y2),
                            bbox_norm=(x1 / geometry.resized_width, y1 / geometry.resized_height, (x2 - x1) / geometry.resized_width, (y2 - y1) / geometry.resized_height),
                            confidence=float(box.conf.item()),
                            classification=mapped,
                            width_mm=width_mm,
                            height_mm=height_mm,
                            preview_image_url=viewer_preview_urls[viewer_slice_index],
                            centroid=((x1 + x2) / (2 * geometry.resized_width), (y1 + y2) / (2 * geometry.resized_height)),
                        )
                    )
        return candidates

    def _detect_with_heuristics(self, processed: ProcessedStudy, viewer_preview_urls: list[str]) -> list[SliceCandidate]:
        candidates: list[SliceCandidate] = []
        kernel_small = np.ones((3, 3), dtype=np.uint8)
        kernel_large = np.ones((7, 7), dtype=np.uint8)

        for slice_index, rgb in enumerate(processed.slice_images_rgb):
            geometry = processed.slice_geometries[slice_index]
            gray = rgb[:, :, 0].astype(np.float32) / 255.0
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            lung_seed = (blurred < 0.55).astype(np.uint8) * 255
            lung_seed = cv2.morphologyEx(lung_seed, cv2.MORPH_OPEN, kernel_large)
            lung_seed = cv2.morphologyEx(lung_seed, cv2.MORPH_CLOSE, kernel_large)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lung_seed, connectivity=8)
            candidate_components: list[np.ndarray] = []
            for label_index in range(1, num_labels):
                area = stats[label_index, cv2.CC_STAT_AREA]
                if area < 1000:
                    continue
                candidate_components.append((labels == label_index).astype(np.uint8))  # type: ignore[arg-type]
            if not candidate_components:
                continue

            candidate_components.sort(key=lambda component: int(component.sum()), reverse=True)
            lung_mask = np.clip(sum(candidate_components[:2]), 0, 1).astype(np.uint8)
            lung_mask = cv2.dilate(lung_mask, kernel_large, iterations=1)

            bright = (blurred > 0.56).astype(np.uint8)
            candidate_mask = cv2.bitwise_and(bright, bright, mask=lung_mask)
            candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel_small)
            candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel_small)

            num_candidates, candidate_labels, candidate_stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
            for label_index in range(1, num_candidates):
                x = int(candidate_stats[label_index, cv2.CC_STAT_LEFT])
                y = int(candidate_stats[label_index, cv2.CC_STAT_TOP])
                w = int(candidate_stats[label_index, cv2.CC_STAT_WIDTH])
                h = int(candidate_stats[label_index, cv2.CC_STAT_HEIGHT])
                area = int(candidate_stats[label_index, cv2.CC_STAT_AREA])

                if area < 6 or area > 1400:
                    continue
                if x <= 0 or y <= 0 or x + w >= geometry.resized_width or y + h >= geometry.resized_height:
                    continue
                if max(w / max(h, 1), h / max(w, 1)) > 3.4:
                    continue

                component_mask = candidate_labels == label_index
                mean_intensity = float(blurred[component_mask].mean())
                max_intensity = float(blurred[component_mask].max())
                if mean_intensity < 0.54:
                    continue

                x1, y1, x2, y2 = self._clamp_bbox(float(x), float(y), float(x + w), float(y + h), geometry.resized_width, geometry.resized_height)
                width_mm, height_mm = self._bbox_size_mm((x1, y1, x2, y2), geometry)
                if max(width_mm, height_mm) < 1.5 or max(width_mm, height_mm) > 45.0:
                    continue

                classification: Literal["solid", "part-solid", "ground-glass"]
                if mean_intensity < 0.69:
                    classification = "ground-glass"
                elif mean_intensity < 0.80:
                    classification = "part-solid"
                else:
                    classification = "solid"

                confidence = min(0.97, 0.38 + (mean_intensity * 0.45) + min(area / 250.0, 0.15) + min(max_intensity * 0.08, 0.08))
                viewer_slice_index = self._viewer_slice_index(processed, slice_index)
                candidates.append(
                    SliceCandidate(
                        slice_index=slice_index,
                        bbox_px=(x1, y1, x2, y2),
                        bbox_norm=(x1 / geometry.resized_width, y1 / geometry.resized_height, (x2 - x1) / geometry.resized_width, (y2 - y1) / geometry.resized_height),
                        confidence=round(confidence, 3),
                        classification=classification,
                        width_mm=width_mm,
                        height_mm=height_mm,
                        preview_image_url=viewer_preview_urls[viewer_slice_index],
                        centroid=((x1 + x2) / (2 * geometry.resized_width), (y1 + y2) / (2 * geometry.resized_height)),
                    )
                )

        return candidates

    def _group_candidates(
        self,
        candidates: list[SliceCandidate],
        processed: ProcessedStudy,
        *,
        centroid_distance_threshold: float = 0.12,
        singleton_confidence_threshold: float = 0.82,
        min_longest_mm: float = 3.0,
        max_longest_mm: float = 30.0,
        min_shape_ratio: float = 0.22,
        max_findings: int = 12,
    ) -> list[NoduleFinding]:
        if not candidates:
            return []

        clusters: list[DetectionCluster] = []
        ordered = sorted(candidates, key=lambda item: (item.slice_index, -item.confidence))
        for candidate in ordered:
            attached = False
            for cluster in reversed(clusters[-12:]):
                if candidate.slice_index - cluster.last.slice_index > 4:
                    continue
                if self._centroid_distance(candidate, cluster.last) > centroid_distance_threshold:
                    continue
                cluster.add(candidate)
                attached = True
                break
            if not attached:
                clusters.append(DetectionCluster(candidates=[candidate]))

        findings: list[NoduleFinding] = []
        slice_spacing_mm = processed.target_spacing[0]
        for cluster in clusters:
            if len(cluster.candidates) == 1 and cluster.candidates[0].confidence < singleton_confidence_threshold:
                continue

            representative = max(cluster.candidates, key=lambda item: item.confidence)
            viewer_slice_index = self._viewer_slice_index(processed, representative.slice_index)
            longest_mm = max(max(item.width_mm, item.height_mm) for item in cluster.candidates)
            shortest_mm = max(1.0, min(min(item.width_mm, item.height_mm) for item in cluster.candidates))
            if longest_mm < min_longest_mm or longest_mm > max_longest_mm:
                continue
            if (shortest_mm / max(longest_mm, 1.0)) < min_shape_ratio:
                continue
            depth_mm = max(1.0, (max(item.slice_index for item in cluster.candidates) - min(item.slice_index for item in cluster.candidates) + 1) * slice_spacing_mm)
            volume_mm3 = (math.pi / 6.0) * longest_mm * shortest_mm * depth_mm
            classification = self._majority_class(cluster.candidates)
            risk = self._risk_level(longest_mm, classification)

            findings.append(
                NoduleFinding(
                    slice_index=viewer_slice_index,
                    bbox=BoundingBox(
                        x=representative.bbox_norm[0],
                        y=representative.bbox_norm[1],
                        width=representative.bbox_norm[2],
                        height=representative.bbox_norm[3],
                    ),
                    confidence=round(max(item.confidence for item in cluster.candidates), 3),
                    classification=classification,
                    malignancy_risk=risk,
                    measurement=NoduleMeasurement(
                        longest_diameter_mm=round(longest_mm, 1),
                        shortest_diameter_mm=round(shortest_mm, 1),
                        estimated_volume_mm3=round(volume_mm3, 1),
                    ),
                    reasoning=self._reasoning_text(classification, risk, longest_mm, len(cluster.candidates)),
                    patient_summary=self._patient_summary(classification, longest_mm, risk),
                    preview_image_url=representative.preview_image_url,
                    accepted=None,
                )
            )

        findings.sort(key=lambda item: ({"high": 0, "intermediate": 1, "low": 2}[item.malignancy_risk], -item.measurement.longest_diameter_mm, -item.confidence))
        return findings[:max_findings]

    def _single_slice_findings(
        self,
        candidates: list[SliceCandidate],
        processed: ProcessedStudy,
        *,
        min_confidence: float,
        min_longest_mm: float,
        max_longest_mm: float,
        max_findings: int,
    ) -> list[NoduleFinding]:
        findings: list[NoduleFinding] = []
        used_keys: set[tuple[int, int, int]] = set()
        depth_mm = max(1.0, processed.target_spacing[0])

        for candidate in sorted(candidates, key=lambda item: (-item.confidence, -max(item.width_mm, item.height_mm))):
            if candidate.confidence < min_confidence:
                continue

            longest_mm = max(candidate.width_mm, candidate.height_mm)
            shortest_mm = max(1.0, min(candidate.width_mm, candidate.height_mm))
            if longest_mm < min_longest_mm or longest_mm > max_longest_mm:
                continue

            viewer_slice_index = self._viewer_slice_index(processed, candidate.slice_index)
            bucket_key = (
                viewer_slice_index,
                round(candidate.centroid[0] * 20),
                round(candidate.centroid[1] * 20),
            )
            if bucket_key in used_keys:
                continue
            used_keys.add(bucket_key)

            volume_mm3 = (math.pi / 6.0) * longest_mm * shortest_mm * depth_mm
            risk = self._risk_level(longest_mm, candidate.classification)
            findings.append(
                NoduleFinding(
                    slice_index=viewer_slice_index,
                    bbox=BoundingBox(
                        x=candidate.bbox_norm[0],
                        y=candidate.bbox_norm[1],
                        width=candidate.bbox_norm[2],
                        height=candidate.bbox_norm[3],
                    ),
                    confidence=round(candidate.confidence, 3),
                    classification=candidate.classification,
                    malignancy_risk=risk,
                    measurement=NoduleMeasurement(
                        longest_diameter_mm=round(longest_mm, 1),
                        shortest_diameter_mm=round(shortest_mm, 1),
                        estimated_volume_mm3=round(volume_mm3, 1),
                    ),
                    reasoning=(
                        f"Local {self.settings.app_name} surfaced a single-slice candidate for faster review. "
                        f"It was classified as {candidate.classification} with an estimated longest diameter of {longest_mm:.1f} mm."
                    ),
                    patient_summary=self._patient_summary(candidate.classification, longest_mm, risk),
                    preview_image_url=candidate.preview_image_url,
                    accepted=None,
                )
            )
            if len(findings) >= max_findings:
                break

        findings.sort(key=lambda item: ({"high": 0, "intermediate": 1, "low": 2}[item.malignancy_risk], -item.measurement.longest_diameter_mm, -item.confidence))
        return findings

    def _reasoning_text(self, classification: str, risk: str, longest_mm: float, slices: int) -> str:
        return (
            f"Local {self.settings.app_name} screening grouped this candidate across {slices} slice(s), "
            f"classified it as {classification}, and estimated a longest diameter of {longest_mm:.1f} mm. "
            f"Overall screening risk was marked {risk} based on size and density pattern."
        )

    def _patient_summary(self, classification: str, longest_mm: float, risk: str) -> str:
        return (
            f"NoduleX found a {classification} lung nodule measuring about {longest_mm:.1f} mm. "
            f"This local screening marked it as {risk} risk, but your doctor should confirm what it means for you."
        )

    def _build_basic_diagnosis(self, findings: list[NoduleFinding], method: str) -> str:
        if not findings:
            return (
                "No high-confidence lung nodules were detected in this local screening pass. "
                "This computer result is a preliminary aid and does not replace a clinician review."
            )

        count = len(findings)
        largest = max(findings, key=lambda item: item.measurement.longest_diameter_mm)
        highest = findings[0].malignancy_risk
        if method == "yolo":
            engine = "YOLO-based"
            verb = "identified"
        elif method == "yolo+heuristic":
            engine = "permissive screening"
            verb = "surfaced"
        else:
            engine = "local image-analysis"
            verb = "flagged"
        return (
            f"{engine} screening {verb} {count} candidate lung nodule{'s' if count != 1 else ''}. "
            f"The largest estimated nodule measures {largest.measurement.longest_diameter_mm:.1f} mm and is described as {largest.classification}. "
            f"Overall screening impression: {highest} risk. This is a basic AI screening summary, not a final diagnosis."
        )

    def _map_yolo_class(self, class_name: str) -> Literal["solid", "part-solid", "ground-glass"]:
        normalized = class_name.lower().strip()
        if "ground" in normalized:
            return "ground-glass"
        if "part" in normalized:
            return "part-solid"
        return "solid"

    def _load_yolo_model(self):
        if self._yolo_model is None and YOLO is not None:
            self._yolo_model = YOLO(str(self.settings.yolo_weights_path))
        return self._yolo_model

    def _risk_level(self, longest_mm: float, classification: str) -> Literal["low", "intermediate", "high"]:
        if longest_mm >= 15 or (classification == "solid" and longest_mm >= 10):
            return "high"
        if longest_mm >= 6 or classification in {"solid", "part-solid"}:
            return "intermediate"
        return "low"

    def _majority_class(self, candidates: list[SliceCandidate]) -> Literal["solid", "part-solid", "ground-glass"]:
        counts = {"solid": 0, "part-solid": 0, "ground-glass": 0}
        for candidate in candidates:
            counts[candidate.classification] += 1
        return max(counts, key=counts.get)  # type: ignore[return-value]

    def _bbox_size_mm(self, bbox_xyxy: tuple[float, float, float, float], geometry) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox_xyxy
        width_px = max(1.0, x2 - x1)
        height_px = max(1.0, y2 - y1)
        width_mm = (width_px / geometry.resized_width) * geometry.original_width * geometry.pixel_spacing_xy[1]
        height_mm = (height_px / geometry.resized_height) * geometry.original_height * geometry.pixel_spacing_xy[0]
        return width_mm, height_mm

    def _clamp_bbox(self, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
        x1 = min(max(x1, 0.0), float(width - 1))
        x2 = min(max(x2, x1 + 1.0), float(width))
        y1 = min(max(y1, 0.0), float(height - 1))
        y2 = min(max(y2, y1 + 1.0), float(height))
        return x1, y1, x2, y2

    def _centroid_distance(self, left: SliceCandidate, right: SliceCandidate) -> float:
        return math.sqrt(((left.centroid[0] - right.centroid[0]) ** 2) + ((left.centroid[1] - right.centroid[1]) ** 2))

    def _viewer_slice_index(self, processed: ProcessedStudy, resampled_slice_index: int) -> int:
        if not processed.original_z_positions:
            return resampled_slice_index
        target_z = processed.target_z_positions[resampled_slice_index]
        return min(
            range(len(processed.original_z_positions)),
            key=lambda index: abs(processed.original_z_positions[index] - target_z),
        )

    def _data_url(self, path: Path) -> str:
        relative_path = path.relative_to(self.settings.data_dir).as_posix()
        return f"/data/{relative_path}"

    def _demo_findings(self) -> list[NoduleFinding]:
        return [
            NoduleFinding(
                slice_index=126,
                bbox=BoundingBox(x=0.42, y=0.31, width=0.08, height=0.08),
                confidence=0.94,
                classification="solid",
                malignancy_risk="intermediate",
                measurement=NoduleMeasurement(
                    longest_diameter_mm=7.8,
                    shortest_diameter_mm=5.9,
                    estimated_volume_mm3=176.0,
                ),
                reasoning="Local demo data shows a solid nodule candidate with moderate AI confidence.",
                patient_summary="A small solid lung nodule was identified and should be reviewed with your clinician.",
                accepted=None,
                preview_image_url=None,
            ),
            NoduleFinding(
                slice_index=133,
                bbox=BoundingBox(x=0.58, y=0.47, width=0.1, height=0.1),
                confidence=0.88,
                classification="ground-glass",
                malignancy_risk="low",
                measurement=NoduleMeasurement(
                    longest_diameter_mm=4.2,
                    shortest_diameter_mm=3.7,
                    estimated_volume_mm3=41.0,
                ),
                reasoning="Local demo data shows a small ground-glass candidate.",
                patient_summary="A very small faint nodule was seen. These can have many causes and need clinician interpretation.",
                accepted=None,
                preview_image_url=None,
            ),
        ]
