"""Statsforecast engine — AutoETS primary with a numpy/pandas fallback.

This wraps the proven baseline implementation in :mod:`app.forecasting.engine`
so the existing, tested logic is reused verbatim behind the engine contract.
It is always available (the numpy fallback has no heavy dependencies).
"""

from __future__ import annotations

from typing import ClassVar

from app.forecasting import engine as _baseline
from app.forecasting.engines.base import ForecastEngine
from app.forecasting.features.context import ForecastContext
from app.forecasting.schemas import ForecastPoint, HistoryPoint


class StatsforecastEngine(ForecastEngine):
    key: ClassVar[str] = "statsforecast"

    def __init__(self) -> None:
        # Set during forecast() to the model actually used for the last series
        # (AutoETS for long series, SeasonalNaive for short ones). A fresh engine
        # is created per request, so this is request-scoped.
        self._model_used: str | None = None

    @classmethod
    def is_available(cls) -> bool:
        # The numpy/pandas fallback guarantees this engine always runs.
        return True

    def model_name(self) -> str:
        return self._model_used or _baseline.model_name()

    def forecast(
        self,
        history: list[HistoryPoint],
        frequency: str,
        horizon: int,
        season_length: int | None,
        context: ForecastContext | None = None,
    ) -> list[ForecastPoint]:
        # Context (calendar/weather) is not consumed by this engine — AutoETS
        # doesn't take exogenous regressors in this increment. Accepted here
        # only to satisfy the shared ForecastEngine contract.
        self._model_used = _baseline.resolve_model_name(
            history, frequency, season_length
        )
        return _baseline.run_forecast(history, frequency, horizon, season_length)
