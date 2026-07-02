"""TimesFM 2.5 engine — wired, model integration pending.

The backlog (E08) specifies TimesFM 2.5 + exogenous covariates as the target
model. The contract is in place: the day the model dependency lands, only
``_run`` below is filled in — callers, request/response shapes and the registry
do not change.

Until then ``is_available()`` reflects whether the ``timesfm`` package can be
imported, and ``forecast()`` raises a clear, actionable error.
"""

from __future__ import annotations

from typing import ClassVar

from app.forecasting.engines.base import EngineNotAvailableError, ForecastEngine
from app.forecasting.features.context import ForecastContext
from app.forecasting.schemas import ForecastPoint, HistoryPoint

_INSTALL_HINT = (
    "TimesFM engine selected but the 'timesfm' package is not installed. "
    "Install the model extra (see README) or use engine='statsforecast'."
)


class TimesFMEngine(ForecastEngine):
    key: ClassVar[str] = "timesfm"
    # Not auto-selected until the integration below is implemented, so a default
    # "auto" request keeps using the working baseline instead of failing.
    auto_selectable: ClassVar[bool] = False

    @classmethod
    def is_available(cls) -> bool:
        try:
            import timesfm  # noqa: F401
        except Exception:
            return False
        return True

    def model_name(self) -> str:
        return "TimesFM-2.5"

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
        # Model dependency is present but the inference adapter is not wired yet.
        raise NotImplementedError(
            "TimesFM is installed but the inference adapter is not implemented "
            "yet (pending E08 TimesFM increment). Map the TimesFM output to "
            "ForecastPoint(target_date, yhat=q50, yhat_lo=q10, yhat_hi=q90) here."
        )
