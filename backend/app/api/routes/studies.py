from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ...dependencies import get_study_service
from ...schemas import AnalyzeStudyResponse, FindingReviewRequest, StudyCreateResponse, StudyListResponse, StudySummaryResponse
from ...services.studies import StudyService

router = APIRouter(prefix="/studies", tags=["studies"])


@router.get("", response_model=StudyListResponse)
def list_studies(study_service: StudyService = Depends(get_study_service)) -> StudyListResponse:
    return StudyListResponse(studies=study_service.repository.list())


@router.post("", response_model=StudyCreateResponse)
async def create_study(
    files: list[UploadFile] = File(...),
    patient_name: str = Form(...),
    patient_age: int = Form(...),
    patient_sex: str = Form(...),
    smoking_history: str = Form(...),
    study_service: StudyService = Depends(get_study_service),
) -> StudyCreateResponse:
    study = await study_service.create_study(
        files=files,
        patient_name=patient_name,
        patient_age=patient_age,
        patient_sex=patient_sex,
        smoking_history=smoking_history,
    )
    return StudyCreateResponse(study=study)


@router.post("/{study_id}/analyze", response_model=AnalyzeStudyResponse)
def analyze_study(study_id: str, study_service: StudyService = Depends(get_study_service)) -> AnalyzeStudyResponse:
    study = study_service.analyze_study(study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found")
    return AnalyzeStudyResponse(study=study, nodule_count=len(study.findings))


@router.get("/{study_id}", response_model=StudyCreateResponse)
def get_study(study_id: str, study_service: StudyService = Depends(get_study_service)) -> StudyCreateResponse:
    study = study_service.repository.get(study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found")
    return StudyCreateResponse(study=study)


@router.patch("/{study_id}/findings/{finding_id}", response_model=StudyCreateResponse)
def review_finding(
    study_id: str,
    finding_id: str,
    payload: FindingReviewRequest,
    study_service: StudyService = Depends(get_study_service),
) -> StudyCreateResponse:
    study = study_service.update_finding(
        study_id=study_id,
        finding_id=finding_id,
        accepted=payload.accepted,
        clinician_note=payload.clinician_note,
    )
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found")
    return StudyCreateResponse(study=study)


@router.get("/{study_id}/summary", response_model=StudySummaryResponse)
def study_summary(study_id: str, study_service: StudyService = Depends(get_study_service)) -> StudySummaryResponse:
    summary = study_service.get_summary(study_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Study not found")
    return StudySummaryResponse(**summary)
