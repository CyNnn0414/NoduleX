# Requirements Mapping

## Functional coverage

- DICOM ingestion: `backend/app/services/studies.py`
- YOLO preprocessing and slice export: `ml/lumenai_ml/preprocessing.py`
- XML to YOLO labels: `ml/lumenai_ml/lidc.py`
- 70/30 LIDC-IDRI split: `ml/scripts/prepare_lidc_dataset.py`
- Nodule count, size, and class display: `frontend/src/App.tsx`
- Doctor accept/reject workflow: `backend/app/api/routes/studies.py`, `frontend/src/App.tsx`
- Patient-specific chat: `backend/app/services/chat.py`, `frontend/src/App.tsx`

## Non-functional alignment

- Inference under 30 seconds:
  - Intended path is slice preprocessing plus YOLO inference with cached weights
  - Benchmarking still needs to be run on the target machine after dependencies and model weights are installed
- HIPAA-minded design:
  - App separates patient-facing summaries from clinician notes
  - Real deployment still needs encryption, access control, audit logging, and secure hosting

## Integration notes

- PACS / FHIR / HL7 are documented as future integration points and are not fully implemented in this starter.
- Automatic signed-report generation is intentionally left out, matching the MVP scope in the requirements document.

## Local dataset detected

On this machine, a usable LIDC-IDRI copy was found at:

`/Users/yuxinzhang/Desktop/BC senior 2nd/Biomedical Image Analysis/Final_Project/Dataset`

This path contains both the extracted DICOM studies and the extracted XML annotations, so it can be used directly with `ml/scripts/prepare_lidc_dataset.py`.
