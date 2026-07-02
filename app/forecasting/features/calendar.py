"""Peruvian calendar context: official holidays + curated gastro-demand events.

Three layers of "special day" signal feed the forecast:

1. **Official non-working holidays** (`holidays.PE`, the `vacanza/holidays`
   package) — legally binding days off. They shift footfall broadly (e.g.
   long weekends) but most don't specifically drive a *restaurant's* demand.
2. **A curated gastronomic calendar** — dates that specifically move demand
   in a Lima restobar even though most of them are NOT official holidays
   (San Valentín, Día de la Madre, Día del Ceviche...). These are sourced
   from the product/thesis brief, not from any external feed, so they are
   kept here as an explicit, reviewable list instead of hidden magic dates.
3. **Payday windows** — QUINCENA (15th) and FIN DE MES (last calendar day of
   the month), the two paydays that drive Peruvian household spending.
   Unlike the events above (single day), a payday's effect on restaurant
   demand spreads across the days immediately around it, so this layer is
   a +-1 day *window* rather than a single flagged date.

All layers merge into a single per-date :class:`DateFeatures` record, which
is (a) fed to the ML engine's feature matrix and (b) used by
`app/forecasting/features/drivers.py` to build the `drivers` narration in the
API response ("Fiestas Patrias en 12 días: +35% demanda proyectada").

Purely computational — no network calls, so this is always available and
cheap enough to run for every request that opts into `use_context`.
"""

from __future__ import annotations

import calendar as std_calendar
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import holidays

EventKind = Literal["holiday", "gastro_event"]
PaydayLabel = Literal["Quincena", "Fin de mes"]

# How far past the latest requested date we look to resolve `days_to_next_event`
# for dates near the end of the range. The widest real gap between two curated
# events (Fiestas Patrias -> Halloween/Canción Criolla) is ~94 days, so 120 is a
# safety margin rather than a tight bound.
_LOOKAHEAD_HORIZON_DAYS = 120

# datetime.date.weekday(): Monday=0 ... Sunday=6.
_SATURDAY = 5
_SUNDAY = 6

# Number of days on each side of a payday anchor (the 15th / last day of the
# month) whose demand is still considered attributable to that payday — the
# spend triggered by a paycheck isn't confined to the exact day it lands.
_PAYDAY_WINDOW_RADIUS_DAYS = 1

_QUINCENA_DAY = 15
_QUINCENA_LABEL: PaydayLabel = "Quincena"
_FIN_DE_MES_LABEL: PaydayLabel = "Fin de mes"


@dataclass(frozen=True)
class DateFeatures:
    """Calendar signal for a single date, consumed by the ML engine and drivers."""

    ds: date
    is_holiday: bool
    event_name: str | None
    event_kind: EventKind | None
    days_to_next_event: int | None
    is_weekend: bool
    # Payday signal (QUINCENA/FIN DE MES +-1 day). `is_payday_window` is set
    # for every day in the window (used as a plain numeric ML feature);
    # `payday_anchor`/`payday_label` identify WHICH payday the window belongs
    # to and are only meaningful when `is_payday_window` is True — they carry
    # the *same* anchor/label for all 3 days of a given window, which is what
    # lets `drivers.py` dedupe a window down to a single narrated driver.
    is_payday_window: bool
    payday_anchor: date | None
    payday_label: PaydayLabel | None


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


def _merge_payday_window(
    windows: dict[date, tuple[date, PaydayLabel]], anchor: date, label: PaydayLabel
) -> None:
    """Register *anchor* +- `_PAYDAY_WINDOW_RADIUS_DAYS` in *windows* (in place).

    Every day in the window maps to the SAME `(anchor, label)` pair, which is
    what lets a consumer (e.g. `drivers.py`) dedupe "3 days, 1 payday" down to
    a single narrated event by keying on `anchor`. Quincena (day 14-16) and
    fin-de-mes windows never overlap in the same month (every month has >= 28
    days, leaving a double-digit day gap between the two windows), so plain
    assignment is safe — there is no ambiguous day claimed by two anchors.
    """
    for offset in range(-_PAYDAY_WINDOW_RADIUS_DAYS, _PAYDAY_WINDOW_RADIUS_DAYS + 1):
        windows[anchor + timedelta(days=offset)] = (anchor, label)


def _payday_windows_for_years(years: set[int]) -> dict[date, tuple[date, PaydayLabel]]:
    """Map every date within a payday window to its `(anchor, label)`.

    KNOWN SIMPLIFICATION (documented per the ticket, not a bug): Peruvian
    paydays are legally "the 15th" and "the last calendar day of the month",
    full stop — no adjustment when the 15th falls on a Sunday (in practice
    employers often move the payment to the preceding business day, but
    modeling that would require a full business-day/bank-holiday calendar for
    marginal accuracy gain). We intentionally use the literal calendar dates.

    A fin-de-mes window can spill into day 1 of the NEXT month (e.g. Dec 31 ->
    Jan 1), possibly of the next YEAR, so windows are generated for every
    month of every requested year — `date + timedelta` naturally rolls over
    month/year boundaries. We also generate one year before the earliest
    requested year so that a Jan 1 in `years` still resolves the Dec 31
    window from the year before it (which itself falls outside `years`).
    """
    windows: dict[date, tuple[date, PaydayLabel]] = {}
    extended_years = years | {min(years) - 1}
    for year in extended_years:
        for month in range(1, 13):
            _merge_payday_window(
                windows, date(year, month, _QUINCENA_DAY), _QUINCENA_LABEL
            )
            last_day = std_calendar.monthrange(year, month)[1]
            _merge_payday_window(
                windows, date(year, month, last_day), _FIN_DE_MES_LABEL
            )
    return windows


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
    payday_windows = _payday_windows_for_years(years)

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

        payday = payday_windows.get(d)
        payday_anchor, payday_label = payday if payday is not None else (None, None)

        features[d] = DateFeatures(
            ds=d,
            is_holiday=is_holiday,
            event_name=event_name,
            event_kind=event_kind,
            days_to_next_event=days_to_next,
            is_weekend=d.weekday() in (_SATURDAY, _SUNDAY),
            is_payday_window=payday is not None,
            payday_anchor=payday_anchor,
            payday_label=payday_label,
        )

    return features
