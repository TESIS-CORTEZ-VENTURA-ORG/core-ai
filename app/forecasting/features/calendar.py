"""Peruvian calendar context: official holidays + curated gastro-demand events.

Two layers of "special day" signal feed the forecast:

1. **Official non-working holidays** (`holidays.PE`, the `vacanza/holidays`
   package) — legally binding days off. They shift footfall broadly (e.g.
   long weekends) but most don't specifically drive a *restaurant's* demand.
2. **A curated gastronomic calendar** — dates that specifically move demand
   in a Lima restobar even though most of them are NOT official holidays
   (San Valentín, Día de la Madre, Día del Ceviche...). These are sourced
   from the product/thesis brief, not from any external feed, so they are
   kept here as an explicit, reviewable list instead of hidden magic dates.

Both layers merge into a single per-date :class:`DateFeatures` record, which
is (a) fed to the ML engine's feature matrix and (b) used by
`app/forecasting/features/drivers.py` to build the `drivers` narration in the
API response ("Fiestas Patrias en 12 días: +35% demanda proyectada").

Purely computational — no network calls, so this is always available and
cheap enough to run for every request that opts into `use_context`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import holidays

EventKind = Literal["holiday", "gastro_event"]

# How far past the latest requested date we look to resolve `days_to_next_event`
# for dates near the end of the range. The widest real gap between two curated
# events (Fiestas Patrias -> Halloween/Canción Criolla) is ~94 days, so 120 is a
# safety margin rather than a tight bound.
_LOOKAHEAD_HORIZON_DAYS = 120

# datetime.date.weekday(): Monday=0 ... Sunday=6.
_SATURDAY = 5
_SUNDAY = 6


@dataclass(frozen=True)
class DateFeatures:
    """Calendar signal for a single date, consumed by the ML engine and drivers."""

    ds: date
    is_holiday: bool
    event_name: str | None
    event_kind: EventKind | None
    days_to_next_event: int | None
    is_weekend: bool


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return the date of the *n*-th occurrence of *weekday* in *year-month*.

    `weekday` follows `datetime.date.weekday()` (Monday=0 ... Sunday=6); `n`
    is 1-based (n=1 -> first occurrence, n=2 -> second, ...).
    """
    first_of_month = date(year, month, 1)
    days_until_weekday = (weekday - first_of_month.weekday()) % 7
    first_occurrence = first_of_month + timedelta(days=days_until_weekday)
    return first_occurrence + timedelta(weeks=n - 1)


def _gastro_events_for_year(year: int) -> dict[date, str]:
    """Curated demand-moving dates for a Motif Restobar-style restaurant in Lima.

    Not sourced from any calendar API — these are business-curated per the
    thesis brief. Fixed dates are literals; movable ones use the "n-th
    weekday of month" rule that actually defines them (e.g. "2nd Sunday of
    May" for Mother's Day in Peru).
    """
    return {
        _nth_weekday_of_month(year, 2, _SATURDAY, 1): "Día del Pisco Sour",
        date(year, 2, 14): "San Valentín",
        _nth_weekday_of_month(year, 5, _SUNDAY, 2): "Día de la Madre",
        _nth_weekday_of_month(year, 6, _SUNDAY, 3): "Día del Padre",
        date(year, 6, 28): "Día del Ceviche",
        date(year, 7, 28): "Fiestas Patrias",
        date(year, 7, 29): "Fiestas Patrias",
        date(year, 10, 31): "Halloween / Día de la Canción Criolla",
        date(year, 12, 24): "Nochebuena",
        date(year, 12, 25): "Navidad",
        date(year, 12, 31): "Nochevieja",
        date(year, 1, 1): "Año Nuevo",
    }


def _official_holidays_for_years(years: set[int]) -> dict[date, str]:
    pe_holidays = holidays.PE(years=sorted(years))
    return dict(pe_holidays.items())


def _gastro_events_for_years(years: set[int]) -> dict[date, str]:
    merged: dict[date, str] = {}
    for year in years:
        merged.update(_gastro_events_for_year(year))
    return merged


def build_date_features(dates: list[date]) -> dict[date, DateFeatures]:
    """Compute calendar features for every date in *dates*.

    Looks ahead up to `_LOOKAHEAD_HORIZON_DAYS` past the latest requested date
    to resolve `days_to_next_event` for dates near the end of the range.
    Gastro events take precedence over the official holiday label on the same
    date (e.g. Jul 28 is legally "Día de la Independencia" but for a
    restaurant's demand story "Fiestas Patrias" is the meaningful label);
    `is_holiday` still reflects the legal status independently of the label.
    """
    if not dates:
        return {}

    start, end = min(dates), max(dates)
    lookahead_end = end + timedelta(days=_LOOKAHEAD_HORIZON_DAYS)
    years = set(range(start.year, lookahead_end.year + 1))

    official = _official_holidays_for_years(years)
    gastro = _gastro_events_for_years(years)
    all_event_dates = sorted(set(official) | set(gastro))

    features: dict[date, DateFeatures] = {}
    for d in dates:
        is_holiday = d in official
        gastro_name = gastro.get(d)
        if gastro_name is not None:
            event_name: str | None = gastro_name
            event_kind: EventKind | None = "gastro_event"
        elif is_holiday:
            event_name = official[d]
            event_kind = "holiday"
        else:
            event_name = None
            event_kind = None

        next_event_date = next((ed for ed in all_event_dates if ed >= d), None)
        days_to_next = (
            (next_event_date - d).days if next_event_date is not None else None
        )

        features[d] = DateFeatures(
            ds=d,
            is_holiday=is_holiday,
            event_name=event_name,
            event_kind=event_kind,
            days_to_next_event=days_to_next,
            is_weekend=d.weekday() in (_SATURDAY, _SUNDAY),
        )

    return features
