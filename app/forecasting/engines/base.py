"""Common contract for all forecasting engines.

A forecasting engine is a swappable strategy that turns a historical series into
*horizon* future :class:`ForecastPoint` values. The whole point of this seam is
that the model can change (statsforecast -> TimesFM -> Chronos) while the rest of
the service — request validation, backtest, response shape — stays identical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from app.forecasting.schemas import ForecastPoint, HistoryPoint


class EngineNotAvailableError(RuntimeError):
    """Raised when a selected engine's model dependencies are not installed.

    The service maps this to an HTTP 501 with the engine's install hint, so a
    caller asking for an engine that isn't wired yet gets a clear, actionable
    error instead of an opaque import traceback.
    """


class ForecastEngine(ABC):
    """Strategy contract every forecasting model must satisfy."""

    #: Stable identifier used to select the engine (``request.engine`` / settings).
    key: ClassVar[str]

    #: Whether ``"auto"`` selection may pick this engine. Pure baselines opt out
    #: so they are only ever used on explicit request or as the backtest baseline.
    auto_selectable: ClassVar[bool] = True

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True when the engine's model dependencies can be imported."""

    @abstractmethod
    def model_name(self) -> str:
        """Human-facing model name surfaced in the response (e.g. ``"AutoETS"``)."""

    @abstractmethod
    def forecast(
        self,
        history: list[HistoryPoint],
        frequency: str,
        horizon: int,
        season_length: int | None,
    ) -> list[ForecastPoint]:
        """Produce exactly *horizon* forecast points with ``yhat``/``lo``/``hi``."""
