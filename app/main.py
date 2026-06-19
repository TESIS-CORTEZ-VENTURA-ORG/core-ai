"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI

from app.config import get_settings
from app.forecasting.router import router as forecasting_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="GastronomIA demand forecasting microservice.",
)

app.include_router(forecasting_router)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
