# Architecture Notes

## Clinical workflow mapping

- Study ingestion: DICOM series upload from radiologist or PACS bridge
- AI analysis: CT preprocessing, YOLO slice inference, lesion aggregation, measurement extraction
- Doctor review: overlay toggle, confidence-ranked findings, accept/reject loop
- Patient education: view the approved findings only, with plain-language explanations and guided chat

## Key backend services

- `StudyService`: ingestion, metadata extraction, analysis coordination
- `InferenceService`: model loading, slice inference, 3D grouping, measurement extraction
- `ChatService`: patient-safe Q&A grounded in approved findings and study metadata
- `ReviewService`: finding acceptance state and doctor notes

## Safety boundaries

- Patient chat is restricted to this patient's approved findings and general lung nodule education
- The chat response includes a medical safety footer and avoids definitive treatment advice
- Doctor-side findings remain editable, with separate patient-visible and clinician-visible notes
