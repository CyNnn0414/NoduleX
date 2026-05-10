from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .models import StudyRecord


class StudyCreateResponse(BaseModel):
    study: StudyRecord


class StudyListResponse(BaseModel):
    studies: list[StudyRecord]


class AnalyzeStudyResponse(BaseModel):
    study: StudyRecord
    nodule_count: int


class FindingReviewRequest(BaseModel):
    accepted: bool
    clinician_note: str | None = None


class PatientChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class PatientChatResponse(BaseModel):
    answer: str
    citations: list[str]
    safety_note: str


class StudySummaryResponse(BaseModel):
    study_id: str
    patient_name: str
    nodule_count: int
    classifications: dict[str, int]
    highest_risk: Literal["low", "intermediate", "high"]
    basic_diagnosis: str | None = None
