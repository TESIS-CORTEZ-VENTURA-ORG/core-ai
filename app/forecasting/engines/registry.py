"""Engine registry and selection.

Maps a stable engine key to its implementation and resolves the engine to use
for a given request. ``"auto"`` walks a preference order and picks the best
*available* engine, always degrading to the working statsforecast baseline so a
default request never fails because a heavy model is missing.
"""

from __future__ import annotations

from app.forecasting.engines.base import ForecastEngine
from app.forecasting.engines.chronos_engine import ChronosEngine
from app.forecasting.engines.ml_engine import MLEngine
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
    MLEngine.key: MLEngine,
}

#: Preference order used by ``"auto"`` — best model first, baseline last.
#: MLEngine leads because it's the only engine that can use exogenous context,
#: but `MLEngine.auto_selectable_for` gates it to `use_context=True` requests
#: with enough history, so a plain "auto" request is completely unaffected
#: (falls through to the pre-existing order below it).
_AUTO_ORDER: list[str] = [
    MLEngine.key,
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


def resolve_engine(
    requested: str | None,
    default: str,
    *,
    history_length: int = 0,
    frequency: str = "D",
    season_length: int | None = None,
    use_context: bool = False,
) -> ForecastEngine:
    """Resolve a concrete engine instance from a request value and a default.

    Resolution order:
    1. Use *requested* when provided, else *default*.
    2. ``"auto"`` -> first auto-selectable, available engine in preference order
       whose ``auto_selectable_for(...)`` guard passes, degrading to the
       statsforecast baseline. ``history_length``/``frequency``/
       ``season_length``/``use_context`` are only consulted by that guard
       (currently only :class:`MLEngine` overrides it) — every other engine
       ignores them, and the defaults reproduce the pre-HU-08-07 behavior
       exactly for callers that don't pass them.
    3. An explicit key -> that engine (instantiated even if its model is missing,
       so ``forecast`` can raise a clear install hint).
    """
    key = (requested or default or AUTO).strip().lower()

    if key == AUTO:
        for candidate in _AUTO_ORDER:
            cls = _ENGINES[candidate]
            if (
                cls.auto_selectable
                and cls.is_available()
                and cls.auto_selectable_for(
                    history_length, frequency, season_length, use_context
                )
            ):
                return cls()
        return StatsforecastEngine()

    cls = _ENGINES.get(key)
    if cls is None:
        valid = ", ".join(list_engine_keys())
        raise UnknownEngineError(f"Unknown engine '{key}'. Valid options: {valid}.")
    return cls()
