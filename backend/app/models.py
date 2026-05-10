from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class NoduleMeasurement(BaseModel):
    longest_diameter_mm: float
    shortest_diameter_mm: float
    estimated_volume_mm3: float


class NoduleFinding(BaseModel):
    id: str = Field(default_factory=lambda: f"finding_{uuid4().hex[:12]}")
    slice_index: int
    bbox: BoundingBox
    confidence: float
    classification: Literal["solid", "part-solid", "ground-glass", "calcified", "benign-pattern", "suspicious"]
    malignancy_risk: Literal["low", "intermediate", "high"]
    measurement: NoduleMeasurement
    reasoning: str
    accepted: bool | None = None
    clinician_note: str | None = None
    patient_summary: str
    preview_image_url: str | None = None


class PatientProfile(BaseModel):
    patient_id: str
    name: str
    age: int
    sex: Literal["female", "male", "other"]
    smoking_history: str


class StudyMetadata(BaseModel):
    accession_number: str
    modality: str = "CT"
    study_description: str = "Chest CT"
    collected_at: datetime = Field(default_factory=utcnow)
    voxel_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    uploaded_file_count: int = 0
    source_slice_count: int = 0
    analysis_slice_count: int = 0
    slice_count: int = 0
    preview_image_url: str | None = None
    slice_image_urls: list[str] = Field(default_factory=list)
    analysis_method: Literal["demo", "heuristic", "yolo", "yolo+heuristic"] | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=utcnow)


class StudyRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"study_{uuid4().hex[:12]}")
    patient: PatientProfile
    metadata: StudyMetadata
    status: Literal["uploaded", "processing", "ready", "error"] = "uploaded"
    findings: list[NoduleFinding] = Field(default_factory=list)
    patient_chat_history: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    source_path: str | None = None
    basic_diagnosis: str | None = None

    def approved_findings(self) -> list[NoduleFinding]:
        return [finding for finding in self.findings if finding.accepted is not False]

    def patient_visible_findings(self) -> list[NoduleFinding]:
        return [finding for finding in self.findings if finding.accepted is True]
