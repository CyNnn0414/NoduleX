from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from ml.lumenai_ml.preprocessing import extract_study_context

from ..models import PatientProfile, StudyMetadata, StudyRecord
from .inference import InferenceService
from .repository import StudyRepository


class StudyService:
    def __init__(self, repository: StudyRepository, inference_service: InferenceService, upload_dir: Path) -> None:
        self.repository = repository
        self.inference_service = inference_service
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _refresh_study_from_dicom(
        self,
        study: StudyRecord,
        *,
        fallback_name: str,
        fallback_age: int,
        fallback_sex: str,
        fallback_smoking_history: str,
    ) -> StudyRecord:
        if not study.source_path:
            return study
        try:
            context = extract_study_context(Path(study.source_path))
        except Exception:
            return study

        resolved_name = context.patient_name or context.patient_id or fallback_name
        resolved_age = context.patient_age or fallback_age
        resolved_sex = context.patient_sex or fallback_sex
        resolved_patient_id = context.patient_id or f"patient_{resolved_name.lower().replace(' ', '_')}"
        study.patient = PatientProfile(
            patient_id=resolved_patient_id,
            name=resolved_name,
            age=resolved_age,
            sex=resolved_sex,
            smoking_history=fallback_smoking_history,
        )
        if context.accession_number:
            study.metadata.accession_number = context.accession_number
        if context.modality:
            study.metadata.modality = context.modality
        if context.study_description:
            study.metadata.study_description = context.study_description
        if context.collected_at:
            study.metadata.collected_at = context.collected_at
        return study

    async def create_study(
        self,
        files: list[UploadFile],
        patient_name: str,
        patient_age: int,
        patient_sex: str,
        smoking_history: str,
    ) -> StudyRecord:
        study = StudyRecord(
            patient=PatientProfile(
                patient_id=f"patient_{patient_name.lower().replace(' ', '_')}",
                name=patient_name,
                age=patient_age,
                sex=patient_sex,
                smoking_history=smoking_history,
            ),
            metadata=StudyMetadata(
                accession_number=f"ACC-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                uploaded_file_count=len(files),
                source_slice_count=len(files),
                slice_count=len(files),
            ),
            status="uploaded",
        )

        study_dir = self.upload_dir / study.id
        study_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            relative_name = (file.filename or "slice.dcm").replace("\\", "/")
            safe_parts = [part for part in Path(relative_name).parts if part not in {"", ".", ".."}]
            destination = study_dir.joinpath(*safe_parts) if safe_parts else study_dir / "slice.dcm"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(await file.read())

        study.source_path = str(study_dir)
        study = self._refresh_study_from_dicom(
            study,
            fallback_name=patient_name,
            fallback_age=patient_age,
            fallback_sex=patient_sex,
            fallback_smoking_history=smoking_history,
        )
        self.repository.save(study)
        return study

    def analyze_study(self, study_id: str) -> StudyRecord | None:
        study = self.repository.get(study_id)
        if study is None:
            return None
        study = self._refresh_study_from_dicom(
            study,
            fallback_name=study.patient.name,
            fallback_age=study.patient.age,
            fallback_sex=study.patient.sex,
            fallback_smoking_history=study.patient.smoking_history,
        )
        study.status = "processing"
        self.repository.save(study)
        try:
            analyzed = self.inference_service.analyze(study)
        except Exception as exc:
            study.status = "error"
            study.basic_diagnosis = f"Local analysis failed: {exc}"
            return self.repository.save(study)
        analyzed.updated_at = datetime.now(timezone.utc)
        return self.repository.save(analyzed)

    def update_finding(self, study_id: str, finding_id: str, accepted: bool, clinician_note: str | None) -> StudyRecord | None:
        study = self.repository.get(study_id)
        if study is None:
            return None
        for finding in study.findings:
            if finding.id == finding_id:
                finding.accepted = accepted
                finding.clinician_note = clinician_note
                break
        study.updated_at = datetime.now(timezone.utc)
        return self.repository.save(study)

    def get_summary(self, study_id: str) -> dict | None:
        study = self.repository.get(study_id)
        if study is None:
            return None
        approved = study.approved_findings()
        classifications: dict[str, int] = {}
        for finding in approved:
            classifications[finding.classification] = classifications.get(finding.classification, 0) + 1
        highest_risk = "low"
        if any(f.malignancy_risk == "high" for f in approved):
            highest_risk = "high"
        elif any(f.malignancy_risk == "intermediate" for f in approved):
            highest_risk = "intermediate"
        return {
            "study_id": study.id,
            "patient_name": study.patient.name,
            "nodule_count": len(approved),
            "classifications": classifications,
            "highest_risk": highest_risk,
            "basic_diagnosis": study.basic_diagnosis,
        }
