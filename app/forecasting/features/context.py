"""Shared exogenous-context DTO passed from the service to engines.

Kept as its own tiny module (rather than living inside `engines/base.py` or
`schemas.py`) so the engine layer depends on a plain data holder — not on the
calendar/weather fetching logic — and `features/drivers.py` can depend on the
same type without reaching into `engines/*`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from app.forecasting.features.calendar import DateFeatures
from app.forecasting.features.weather import WeatherPoint


@dataclass(frozen=True)
class ForecastContext:
    """Precomputed exogenous context, built once per request by the service.

    Carrying already-fetched data (not raw flags like `use_context`/`location`)
    avoids duplicate Open-Meteo calls: both the service (for the `drivers`
    narration) and a context-aware engine (e.g. `MLEngine`, for feature
    columns) reuse this exact same snapshot instead of independently
    re-fetching. `weather_by_date` is an empty dict when weather degraded —
    engines and `drivers` must treat that as "no weather signal", not an error.
    """

    date_features: dict[date, DateFeatures]
    weather_by_date: dict[date, WeatherPoint]
    context_status: Literal["full", "calendar_only"]
