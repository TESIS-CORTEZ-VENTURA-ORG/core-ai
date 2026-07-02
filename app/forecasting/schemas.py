"""Pydantic v2 schemas for the forecasting API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

# Default restaurant location (Lima, Peru) applied when `use_context=True`
# and the caller doesn't provide an explicit `location`. Part of the public
# request contract (not just an internal default), so it lives here next to
# `Location` rather than buried in `features/weather.py`.
DEFAULT_LATITUDE = -12.046
DEFAULT_LONGITUDE = -77.043


class HistoryPoint(BaseModel):
    ds: date
    y: float


class Location(BaseModel):
    """Geographic point used to fetch weather covariates for `use_context`."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class ForecastRequest(BaseModel):
    series_id: str
    frequency: Literal["D", "W"]
    horizon: int = Field(gt=0)
    history: list[HistoryPoint] = Field(min_length=2)
    season_length: int | None = None
    # Engine to run: "auto" (best available, default), "statsforecast",
    # "seasonalnaive", "timesfm", "chronos" or "ml". None -> service default.
    engine: str | None = None
    # --- HU-08-07: exogenous context (Peruvian calendar + weather) ---
    # Opt-in and additive: omitting both fields reproduces the exact response
    # this API returned before this increment (drivers=[], context_status=
    # "off", no calendar/weather computation — zero behavior change).
    use_context: bool = False
    location: Location | None = None


class ForecastPoint(BaseModel):
    target_date: date
    yhat: float
    yhat_lo: float
    yhat_hi: float


class Driver(BaseModel):
    """A context factor inside the forecast horizon the UI can narrate, e.g.
    "Fiestas Patrias en 12 días: +35% demanda proyectada"."""

    date: date
    kind: Literal["holiday", "gastro_event", "weather", "weekend", "payday"]
    label: str
    # Historical uplift for this factor vs. equivalent non-event days, ONLY
    # when the submitted history actually has evidence for it. Omitted (None)
    # rather than guessed when the event never occurred in `history`.
    impact_pct: float | None = None


class BacktestMetrics(BaseModel):
    holdout_size: int
    model_smape: float
    baseline_smape: float
    improvement_pct: float
    # Only populated when use_context=True and the resolved engine is "ml":
    # the same holdout re-run WITHOUT context features, so the response can
    # show "univariate vs +exogenous" — the thesis' core comparison.
    model_smape_no_context: float | None = None


class ForecastResponse(BaseModel):
    series_id: str
    engine: str
    model: str
    baseline: str
    frequency: str
    points: list[ForecastPoint]
    backtest: BacktestMetrics | None
    drivers: list[Driver] = Field(default_factory=list)
    # "off" = use_context=False (default; no calendar/weather computation
    #   attempted at all).
    # "full" = use_context=True and weather fetched successfully.
    # "calendar_only" = use_context=True but Open-Meteo degraded — calendar
    #   features (always-available, no network) still power drivers/ML input.
    context_status: Literal["full", "calendar_only", "off"] = "off"
