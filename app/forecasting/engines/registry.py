"""Engine registry and selection.

Maps a stable engine key to its implementation and resolves the engine to use
for a given request. ``"auto"`` walks a preference order and picks the best
*available* engine, always degrading to the working statsforecast baseline so a
default request never fails because a heavy model is missing.
"""

from __future__ import annotations

from app.forecasting.engines.base import ForecastEngine
from app.forecasting.engines.chronos_engine import ChronosEngine
from app.forecasting.engines.seasonalnaive_engine import SeasonalNaiveEngine
from app.forecasting.engines.statsforecast_engine import StatsforecastEngine
from app.forecasting.engines.timesfm_engine import TimesFMEngine

AUTO = "auto"

#: All registered engines, keyed by their stable identifier.
_ENGINES: dict[str, type[ForecastEngine]] = {
    StatsforecastEngine.key: StatsforecastEngine,
    SeasonalNaiveEngine.key: SeasonalNaiveEngine,
    TimesFMEngine.key: TimesFMEngine,
    ChronosEngine.key: ChronosEngine,
}

#: Preference order used by ``"auto"`` — best model first, baseline last.
_AUTO_ORDER: list[str] = [
    TimesFMEngine.key,
    ChronosEngine.key,
    StatsforecastEngine.key,
]


class UnknownEngineError(ValueError):
    """Raised when a request asks for an engine key that is not registered."""


def list_engine_keys() -> list[str]:
    """Return all registered engine keys (plus ``"auto"``)."""
    return [AUTO, *_ENGINES.keys()]


def available_engines() -> dict[str, bool]:
    """Map each engine key to whether its dependencies are importable."""
    return {key: cls.is_available() for key, cls in _ENGINES.items()}


def resolve_engine(requested: str | None, default: str) -> ForecastEngine:
    """Resolve a concrete engine instance from a request value and a default.

    Resolution order:
    1. Use *requested* when provided, else *default*.
    2. ``"auto"`` -> first auto-selectable, available engine in preference order,
       degrading to the statsforecast baseline.
    3. An explicit key -> that engine (instantiated even if its model is missing,
       so ``forecast`` can raise a clear install hint).
    """
    key = (requested or default or AUTO).strip().lower()

    if key == AUTO:
        for candidate in _AUTO_ORDER:
            cls = _ENGINES[candidate]
            if cls.auto_selectable and cls.is_available():
                return cls()
        return StatsforecastEngine()

    cls = _ENGINES.get(key)
    if cls is None:
        valid = ", ".join(list_engine_keys())
        raise UnknownEngineError(f"Unknown engine '{key}'. Valid options: {valid}.")
    return cls()
