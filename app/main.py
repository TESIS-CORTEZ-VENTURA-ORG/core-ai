"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI

from app.chat.router import router as chat_router
from app.config import get_settings
from app.forecasting.router import router as forecasting_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    description="GastronomIA core-ai microservice — forecasting + Text-to-SQL chat.",
)

app.include_router(forecasting_router)
app.include_router(chat_router)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
