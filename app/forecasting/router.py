"""Forecasting API router."""

from __future__ import annotations

from fastapi import APIRouter

from app.forecasting.schemas import ForecastRequest, ForecastResponse
from app.forecasting.service import forecast

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.post("/run", response_model=ForecastResponse)
def run_forecast_endpoint(request: ForecastRequest) -> ForecastResponse:
    """
    Run a demand forecast.

    Accepts a time series history and returns *horizon* future predictions
    with prediction intervals and optional backtest metrics.
    """
    return forecast(request)
