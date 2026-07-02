"""Common contract for all forecasting engines.

A forecasting engine is a swappable strategy that turns a historical series into
*horizon* future :class:`ForecastPoint` values. The whole point of this seam is
that the model can change (statsforecast -> TimesFM -> Chronos) while the rest of
the service — request validation, backtest, response shape — stays identical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from app.forecasting.features.context import ForecastContext
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

    @classmethod
    def auto_selectable_for(
        cls,
        history_length: int,
        frequency: str,
        season_length: int | None,
        use_context: bool = False,
    ) -> bool:
        """Extra ``"auto"`` eligibility guard consulted beyond ``is_available()``.

        Most engines don't need per-request context to decide eligibility —
        that's what ``is_available()`` covers. Engines whose usefulness
        depends on data volume or on the caller opting into exogenous context
        (e.g. :class:`~app.forecasting.engines.ml_engine.MLEngine`, which
        needs several full seasons of history AND ``use_context=True`` to
        earn its place over the proven statsforecast baseline) override this
        so ``"auto"`` skips them instead of picking them and then failing
        with :class:`EngineNotAvailableError`.
        """
        return True

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
        context: ForecastContext | None = None,
    ) -> list[ForecastPoint]:
        """Produce exactly *horizon* forecast points with ``yhat``/``lo``/``hi``.

        ``context`` (HU-08-07) carries precomputed calendar/weather features
        when the caller opted in via ``use_context``. It is an *additive*
        capability — engines that don't use exogenous features accept and
        ignore it, so pre-existing engines keep working unchanged.
        """
