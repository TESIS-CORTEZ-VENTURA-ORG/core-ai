"""SeasonalNaive baseline engine.

The explicit thesis baseline (HU-08-08): a deterministic seasonal-naive forecast
with empirical prediction bands. Always available, never auto-selected — it is the
yardstick the real model must beat, exposed here so it can be requested directly.
"""

from __future__ import annotations

from typing import ClassVar

from app.forecasting import engine as _baseline
from app.forecasting.engines.base import ForecastEngine
from app.forecasting.features.context import ForecastContext
from app.forecasting.schemas import ForecastPoint, HistoryPoint


class SeasonalNaiveEngine(ForecastEngine):
    key: ClassVar[str] = "seasonalnaive"
    auto_selectable: ClassVar[bool] = False

    @classmethod
    def is_available(cls) -> bool:
        return True

    def model_name(self) -> str:
        return _baseline.baseline_name()

    def forecast(
        self,
        history: list[HistoryPoint],
        frequency: str,
        horizon: int,
        season_length: int | None,
        context: ForecastContext | None = None,
    ) -> list[ForecastPoint]:
        # This is the deterministic backtest baseline — it intentionally never
        # uses exogenous context, so the comparison stays apples-to-apples.
        return _baseline.run_seasonal_naive(history, frequency, horizon, season_length)
