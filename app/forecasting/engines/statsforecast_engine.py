"""Statsforecast engine — AutoETS primary with a numpy/pandas fallback.

This wraps the proven baseline implementation in :mod:`app.forecasting.engine`
so the existing, tested logic is reused verbatim behind the engine contract.
It is always available (the numpy fallback has no heavy dependencies).
"""

from __future__ import annotations

from typing import ClassVar

from app.forecasting import engine as _baseline
from app.forecasting.engines.base import ForecastEngine
from app.forecasting.schemas import ForecastPoint, HistoryPoint


class StatsforecastEngine(ForecastEngine):
    key: ClassVar[str] = "statsforecast"

    @classmethod
    def is_available(cls) -> bool:
        # The numpy/pandas fallback guarantees this engine always runs.
        return True

    def model_name(self) -> str:
        return _baseline.model_name()

    def forecast(
        self,
        history: list[HistoryPoint],
        frequency: str,
        horizon: int,
        season_length: int | None,
    ) -> list[ForecastPoint]:
        return _baseline.run_forecast(history, frequency, horizon, season_length)
