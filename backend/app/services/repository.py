from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..models import StudyRecord


class StudyRepository:
    def __init__(self, data_dir: Path) -> None:
        self._root = data_dir / "studies"
        self._root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[StudyRecord]:
        studies = []
        for path in sorted(self._root.glob("*.json")):
            studies.append(StudyRecord.model_validate_json(path.read_text()))
        return studies

    def get(self, study_id: str) -> StudyRecord | None:
        path = self._root / f"{study_id}.json"
        if not path.exists():
            return None
        return StudyRecord.model_validate_json(path.read_text())

    def save(self, study: StudyRecord) -> StudyRecord:
        path = self._root / f"{study.id}.json"
        path.write_text(json.dumps(study.model_dump(mode="json"), indent=2))
        return study

    def bulk_save(self, studies: Iterable[StudyRecord]) -> None:
        for study in studies:
            self.save(study)
