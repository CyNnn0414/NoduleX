from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...dependencies import get_chat_service, get_repository
from ...schemas import PatientChatRequest, PatientChatResponse
from ...services.chat import ChatService
from ...services.repository import StudyRepository

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/patient/{study_id}", response_model=PatientChatResponse)
async def patient_chat(
    study_id: str,
    payload: PatientChatRequest,
    repository: StudyRepository = Depends(get_repository),
    chat_service: ChatService = Depends(get_chat_service),
) -> PatientChatResponse:
    study = repository.get(study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found")

    try:
        answer, citations = await chat_service.answer_patient_question(study, payload.question)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    repository.save(study)
    return PatientChatResponse(
        answer=answer,
        citations=citations,
        safety_note="This assistant is for educational support and does not replace your clinician.",
    )
