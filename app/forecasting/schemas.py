"""Pydantic v2 schemas for the forecasting API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class HistoryPoint(BaseModel):
    ds: date
    y: float


class ForecastRequest(BaseModel):
    series_id: str
    frequency: Literal["D", "W"]
    horizon: int = Field(gt=0)
    history: list[HistoryPoint] = Field(min_length=2)
    season_length: int | None = None


class ForecastPoint(BaseModel):
    target_date: date
    yhat: float
    yhat_lo: float
    yhat_hi: float


class BacktestMetrics(BaseModel):
    holdout_size: int
    model_smape: float
    baseline_smape: float
    improvement_pct: float


class ForecastResponse(BaseModel):
    series_id: str
    model: str
    baseline: str
    frequency: str
    points: list[ForecastPoint]
    backtest: BacktestMetrics | None
