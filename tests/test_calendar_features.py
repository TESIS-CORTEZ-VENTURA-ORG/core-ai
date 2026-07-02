"""Tests for the Peruvian calendar context (app/forecasting/features/calendar.py)."""

from __future__ import annotations

import calendar as std_calendar
from datetime import date, timedelta

from app.forecasting.features.calendar import build_date_features


class TestOfficialHolidays:
    def test_fiestas_patrias_is_official_holiday(self):
        feats = build_date_features([date(2026, 7, 28)])
        assert feats[date(2026, 7, 28)].is_holiday is True

    def test_new_year_is_official_holiday(self):
        feats = build_date_features([date(2026, 1, 1)])
        assert feats[date(2026, 1, 1)].is_holiday is True

    def test_ordinary_day_is_not_holiday(self):
        feats = build_date_features([date(2026, 3, 10)])
        assert feats[date(2026, 3, 10)].is_holiday is False


class TestGastroEvents:
    def test_dia_del_ceviche_present_and_not_official_holiday(self):
        feats = build_date_features([date(2026, 6, 28)])
        feat = feats[date(2026, 6, 28)]
        assert feat.event_name == "Día del Ceviche"
        assert feat.event_kind == "gastro_event"
        assert feat.is_holiday is False

    def test_fiestas_patrias_gastro_label_overrides_official_label(self):
        # Jul 28 is legally "Día de la Independencia" but the gastro label
        # ("Fiestas Patrias") is what the response should narrate.
        feats = build_date_features([date(2026, 7, 28)])
        feat = feats[date(2026, 7, 28)]
        assert feat.event_name == "Fiestas Patrias"
        assert feat.event_kind == "gastro_event"
        assert (
            feat.is_holiday is True
        )  # legal status preserved independently of the label

    def test_valentines_day(self):
        feats = build_date_features([date(2026, 2, 14)])
        assert feats[date(2026, 2, 14)].event_name == "San Valentín"

    def test_mothers_day_is_second_sunday_of_may(self):
        # Independently verify "2nd Sunday of May 2026" using stdlib calendar,
        # rather than re-deriving our own `_nth_weekday_of_month` logic.
        cal = std_calendar.Calendar()
        sundays = [
            d for d in cal.itermonthdates(2026, 5) if d.month == 5 and d.weekday() == 6
        ]
        expected = sundays[1]

        feats = build_date_features([expected])
        assert feats[expected].event_name == "Día de la Madre"

    def test_fathers_day_is_third_sunday_of_june(self):
        cal = std_calendar.Calendar()
        sundays = [
            d for d in cal.itermonthdates(2026, 6) if d.month == 6 and d.weekday() == 6
        ]
        expected = sundays[2]

        feats = build_date_features([expected])
        assert feats[expected].event_name == "Día del Padre"

    def test_pisco_sour_is_first_saturday_of_february(self):
        cal = std_calendar.Calendar()
        saturdays = [
            d for d in cal.itermonthdates(2026, 2) if d.month == 2 and d.weekday() == 5
        ]
        expected = saturdays[0]

        feats = build_date_features([expected])
        assert feats[expected].event_name == "Día del Pisco Sour"


class TestWeekendAndLookahead:
    def test_saturday_and_sunday_flagged_weekend(self):
        saturday = date(2026, 3, 7)
        sunday = date(2026, 3, 8)
        monday = date(2026, 3, 9)
        feats = build_date_features([saturday, sunday, monday])
        assert feats[saturday].is_weekend is True
        assert feats[sunday].is_weekend is True
        assert feats[monday].is_weekend is False

    def test_days_to_next_event_counts_down_to_fiestas_patrias(self):
        d = date(2026, 7, 26)  # 2 days before Jul 28
        feats = build_date_features([d])
        assert feats[d].days_to_next_event == 2

    def test_days_to_next_event_is_zero_on_the_event_itself(self):
        d = date(2026, 6, 28)
        feats = build_date_features([d])
        assert feats[d].days_to_next_event == 0

    def test_empty_input_returns_empty_dict(self):
        assert build_date_features([]) == {}

    def test_every_date_in_a_long_range_has_a_days_to_next_event_value(self):
        # Regression guard for the "0 is falsy" bug class: every date in a
        # full year must resolve to a non-None days_to_next_event given how
        # densely the curated calendar covers the year (widest real gap is
        # ~94 days, well under the 120-day lookahead window).
        start = date(2026, 1, 1)
        dates = [start + timedelta(days=i) for i in range(365)]
        feats = build_date_features(dates)
        assert all(f.days_to_next_event is not None for f in feats.values())
