from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "NoduleX Lung Nodule Workbench"
    api_prefix: str = "/api"
    data_dir: Path = Field(default=Path("backend/data"))
    upload_dir: Path = Field(default=Path("backend/data/uploads"))
    patient_chat_model: str = "gpt-4.1-mini"
    patient_chat_temperature: float = 0.2
    patient_chat_history_limit: int = 6
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    default_slice_size: int = 640
    yolo_weights_path: Path = Field(default=Path("artifacts/weights/last.pt"))
    demo_mode: bool = False

    model_config = SettingsConfigDict(
        env_prefix="LUMENAI_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    if not settings.yolo_weights_path.exists():
        fallback_weights = [
            Path("artifacts/weights/best.pt")
        ]
        for candidate in fallback_weights:
            if candidate.exists():
                settings.yolo_weights_path = candidate
                break
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
