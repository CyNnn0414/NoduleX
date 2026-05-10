from functools import lru_cache

from .config import get_settings
from .services.chat import ChatService
from .services.inference import InferenceService
from .services.repository import StudyRepository
from .services.studies import StudyService


@lru_cache(maxsize=1)
def get_repository() -> StudyRepository:
    settings = get_settings()
    return StudyRepository(settings.data_dir)


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    return InferenceService(get_settings())


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    return ChatService(get_settings())


@lru_cache(maxsize=1)
def get_study_service() -> StudyService:
    settings = get_settings()
    return StudyService(
        repository=get_repository(),
        inference_service=get_inference_service(),
        upload_dir=settings.upload_dir,
    )
