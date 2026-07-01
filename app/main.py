"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI

from app.chat.router import router as chat_router
from app.config import get_settings
from app.extract.router import router as extract_router
from app.forecasting.router import router as forecasting_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.3.0",
    description=(
        "GastronomIA core-ai microservice — "
        "forecasting + Text-to-SQL chat + Smart Onboarding document extraction."
    ),
)

app.include_router(forecasting_router)
app.include_router(chat_router)
app.include_router(extract_router)  # E11 Smart Onboarding


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
