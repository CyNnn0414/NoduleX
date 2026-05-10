from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.routes.chat import router as chat_router
from .api.routes.health import router as health_router
from .api.routes.studies import router as studies_router
from .bootstrap import ensure_demo_study
from .config import get_settings
from .dependencies import get_repository

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(studies_router, prefix=settings.api_prefix)
app.include_router(chat_router, prefix=settings.api_prefix)
app.mount("/data", StaticFiles(directory=settings.data_dir), name="study-data")
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")


@app.on_event("startup")
async def startup_event() -> None:
    ensure_demo_study(get_repository())
