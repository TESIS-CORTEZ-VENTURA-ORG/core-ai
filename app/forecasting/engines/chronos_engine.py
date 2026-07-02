"""Chronos-2 engine — wired, model integration pending.

The README names Chronos-2 as the eventual replacement model. Like the TimesFM
engine, the contract is in place behind the shared :class:`ForecastEngine`
interface; only ``forecast`` needs filling in when the dependency is added.
"""

from __future__ import annotations

from typing import ClassVar

from app.forecasting.engines.base import EngineNotAvailableError, ForecastEngine
from app.forecasting.features.context import ForecastContext
from app.forecasting.schemas import ForecastPoint, HistoryPoint

_INSTALL_HINT = (
    "Chronos engine selected but the 'chronos-forecasting' package is not "
    "installed. Install the model extra (see README) or use engine='statsforecast'."
)


class ChronosEngine(ForecastEngine):
    key: ClassVar[str] = "chronos"
    auto_selectable: ClassVar[bool] = False

    @classmethod
    def is_available(cls) -> bool:
        try:
            import chronos  # noqa: F401
        except Exception:
            return False
        return True

    def model_name(self) -> str:
        return "Chronos-2"

    def forecast(
        self,
        history: list[HistoryPoint],
        frequency: str,
        horizon: int,
        season_length: int | None,
        context: ForecastContext | None = None,
    ) -> list[ForecastPoint]:
        if not self.is_available():
            raise EngineNotAvailableError(_INSTALL_HINT)
        raise NotImplementedError(
            "Chronos is installed but the inference adapter is not implemented "
            "yet. Map the Chronos quantile output to "
            "ForecastPoint(target_date, yhat=q50, yhat_lo=q10, yhat_hi=q90) here."
        )
