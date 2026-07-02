"""Builds the `drivers` narration list from a precomputed `ForecastContext`.

`drivers` translates raw calendar/weather context into UI-narratable entries
("Fiestas Patrias en 12 días: +35% demanda proyectada"). Kept separate from
`service.py` so the "how do we compute impact_pct honestly" logic (historical
uplift vs. equivalent days) is unit-testable in isolation, and so it works the
same way regardless of which forecasting engine actually ran.
"""

from __future__ import annotations

from datetime import date, timedelta

from app.forecasting.features.calendar import DateFeatures
from app.forecasting.features.context import ForecastContext
from app.forecasting.schemas import Driver, ForecastPoint, HistoryPoint

# Precipitation threshold (mm/day) considered "notable rain" worth narrating.
# WMO's "moderate rain" band starts around 2.5 mm/h sustained; for a daily sum
# we pick a conservative 5mm so light drizzle doesn't spam the driver list.
_RAIN_THRESHOLD_MM = 5.0

# Window (days, each side) used to build the "equivalent days" baseline when
# estimating an event's historical uplift.
_BASELINE_WINDOW_DAYS = 14


def build_drivers(
    history: list[HistoryPoint],
    forecast_points: list[ForecastPoint],
    context: ForecastContext,
) -> list[Driver]:
    """Return one driver per forecast-horizon date that has a context signal.

    At most one calendar-derived driver per date (gastro event / holiday takes
    priority over a plain weekend), plus an independent weather driver when
    precipitation is notable. `impact_pct` is only ever set when *history*
    actually contains evidence for it — never fabricated (see the ticket's
    "si no hay evidencia, omitilo").
    """
    history_by_date = {p.ds: p.y for p in history}
    drivers: list[Driver] = []

    for point in forecast_points:
        d = point.target_date
        feat = context.date_features.get(d)

        if (
            feat is not None
            and feat.event_name is not None
            and feat.event_kind is not None
        ):
            impact = _historical_event_uplift(
                history_by_date, context.date_features, feat.event_name
            )
            drivers.append(
                Driver(
                    date=d,
                    kind=feat.event_kind,
                    label=feat.event_name,
                    impact_pct=impact,
                )
            )
        elif feat is not None and feat.is_weekend:
            impact = _historical_weekend_uplift(history_by_date, context.date_features)
            drivers.append(
                Driver(date=d, kind="weekend", label="Fin de semana", impact_pct=impact)
            )

        weather = context.weather_by_date.get(d)
        if (
            weather is not None
            and weather.precip_mm is not None
            and weather.precip_mm >= _RAIN_THRESHOLD_MM
        ):
            drivers.append(
                Driver(
                    date=d,
                    kind="weather",
                    label=f"Lluvia esperada ({weather.precip_mm:.1f} mm)",
                    impact_pct=None,  # no reliable historical rain-uplift evidence — never invented.
                )
            )

    return drivers


def _historical_event_uplift(
    history_by_date: dict[date, float],
    date_features: dict[date, DateFeatures],
    event_name: str,
) -> float | None:
    """Average % uplift of *event_name* vs. its equivalent non-event days.

    Uses only occurrences of the SAME event actually present in *history*
    (matched by name, since movable events fall on a different date each
    year). Returns None — never a guess — when the event never occurred
    within the given history.
    """
    uplifts: list[float] = []
    for d, feat in date_features.items():
        if feat.event_name != event_name:
            continue
        event_value = history_by_date.get(d)
        if event_value is None:
            continue
        baseline = _equivalent_days_average(history_by_date, date_features, d)
        if baseline is None or baseline == 0:
            continue
        uplifts.append((event_value - baseline) / baseline * 100.0)

    if not uplifts:
        return None
    return round(sum(uplifts) / len(uplifts), 2)


def _historical_weekend_uplift(
    history_by_date: dict[date, float],
    date_features: dict[date, DateFeatures],
) -> float | None:
    """Average % difference between weekend and (non-event) weekday history."""
    weekend_values = [
        history_by_date[d]
        for d, f in date_features.items()
        if f.is_weekend and d in history_by_date
    ]
    weekday_values = [
        history_by_date[d]
        for d, f in date_features.items()
        if not f.is_weekend and f.event_name is None and d in history_by_date
    ]
    if not weekend_values or not weekday_values:
        return None
    weekend_avg = sum(weekend_values) / len(weekend_values)
    weekday_avg = sum(weekday_values) / len(weekday_values)
    if weekday_avg == 0:
        return None
    return round((weekend_avg - weekday_avg) / weekday_avg * 100.0, 2)


def _equivalent_days_average(
    history_by_date: dict[date, float],
    date_features: dict[date, DateFeatures],
    event_date: date,
) -> float | None:
    """Average `y` over non-event days within +/- `_BASELINE_WINDOW_DAYS` of
    *event_date* — the "equivalent day" baseline used for the event's uplift.
    """
    window_values: list[float] = []
    for offset in range(-_BASELINE_WINDOW_DAYS, _BASELINE_WINDOW_DAYS + 1):
        if offset == 0:
            continue
        d = event_date + timedelta(days=offset)
        feat = date_features.get(d)
        value = history_by_date.get(d)
        if feat is None or value is None or feat.event_name is not None:
            continue
        window_values.append(value)
    if not window_values:
        return None
    return sum(window_values) / len(window_values)
