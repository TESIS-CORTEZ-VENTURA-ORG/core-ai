"""Pluggable forecasting engines.

Each engine implements the same :class:`ForecastEngine` contract so the model
behind the forecast can be swapped (statsforecast baseline, TimesFM, Chronos)
without changing any caller. Engine selection is resolved at request time by the
:mod:`app.forecasting.engines.registry`.
"""

from app.forecasting.engines.base import EngineNotAvailableError, ForecastEngine
from app.forecasting.engines.registry import (
    available_engines,
    list_engine_keys,
    resolve_engine,
)
from app.forecasting.features.context import ForecastContext

__all__ = [
    "EngineNotAvailableError",
    "ForecastContext",
    "ForecastEngine",
    "available_engines",
    "list_engine_keys",
    "resolve_engine",
]
