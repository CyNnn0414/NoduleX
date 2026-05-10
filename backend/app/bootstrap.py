from __future__ import annotations

from .models import BoundingBox, NoduleFinding, NoduleMeasurement, PatientProfile, StudyMetadata, StudyRecord
from .services.repository import StudyRepository


def ensure_demo_study(repository: StudyRepository) -> None:
    if repository.list():
        return

    study = StudyRecord(
        patient=PatientProfile(
            patient_id="patient_demo_jane_doe",
            name="Jane Doe",
            age=63,
            sex="female",
            smoking_history="Former smoker, quit 8 years ago",
        ),
        metadata=StudyMetadata(
            accession_number="ACC-DEMO-0001",
            study_description="Chest CT - Demo",
            slice_count=248,
        ),
        status="ready",
        findings=[
            NoduleFinding(
                id="finding_demo_01",
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
                reasoning="Solid nodule candidate with strong slice-level confidence and consistent contour across adjacent slices.",
                patient_summary="A small solid lung nodule was identified. Your care team can explain what this means in the context of your full scan.",
                accepted=True,
            ),
            NoduleFinding(
                id="finding_demo_02",
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
                reasoning="Ground-glass pattern with lower risk features and smaller measured diameter.",
                patient_summary="A very small faint nodule was seen. These findings can have many causes and need clinician interpretation.",
                accepted=True,
            ),
        ],
    )
    repository.save(study)
