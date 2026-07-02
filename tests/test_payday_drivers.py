"""Tests for the payday (quincena/fin-de-mes) driver
(app/forecasting/features/drivers.py + calendar.py's payday window).

Unit-level tests call `build_drivers` directly with a hand-built
`ForecastContext` (same style as `test_calendar_features.py`); HTTP-level
tests go through the full `/forecast/run` endpoint (same style as
`test_context_contract.py`), always via the `mock_weather_success` /
`mock_weather_failure` fixtures — never the real Open-Meteo network.
"""

from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.forecasting.features.calendar import build_date_features
from app.forecasting.features.context import ForecastContext
from app.forecasting.features.drivers import build_drivers
from app.forecasting.schemas import ForecastPoint, HistoryPoint
from app.main import app

client = TestClient(app)


def _context_for(dates: list[date]) -> ForecastContext:
    """Calendar-only context (no weather) — sufficient for driver unit tests."""
    return ForecastContext(
        date_features=build_date_features(dates),
        weather_by_date={},
        context_status="calendar_only",
    )


def _points(dates: list[date]) -> list[ForecastPoint]:
    return [
        ForecastPoint(target_date=d, yhat=0.0, yhat_lo=0.0, yhat_hi=0.0) for d in dates
    ]


class TestPaydayDriverEmission:
    def test_quincena_driver_emitted_on_the_anchor_date(self):
        anchor = date(2026, 3, 15)
        window = [anchor - timedelta(days=1), anchor, anchor + timedelta(days=1)]
        context = _context_for(window)

        drivers = build_drivers([], _points(window), context)

        payday_drivers = [d for d in drivers if d.kind == "payday"]
        assert len(payday_drivers) == 1
        assert payday_drivers[0].date == anchor
        assert payday_drivers[0].label == "Quincena"

    def test_fin_de_mes_driver_emitted_on_the_anchor_date(self):
        anchor = date(2026, 3, 31)
        window = [anchor - timedelta(days=1), anchor, anchor + timedelta(days=1)]
        context = _context_for(window)

        drivers = build_drivers([], _points(window), context)

        payday_drivers = [d for d in drivers if d.kind == "payday"]
        assert len(payday_drivers) == 1
        assert payday_drivers[0].date == anchor
        assert payday_drivers[0].label == "Fin de mes"

    def test_full_window_in_horizon_dedups_to_a_single_driver(self):
        # Without dedup, a naive per-date loop over a +-1 window would emit 3
        # "payday" chips for the same payday — this is the ticket's explicit
        # "no 3 chips (+-1)" requirement.
        anchor = date(2026, 6, 15)
        # Horizon covers the full window PLUS a couple of unrelated days.
        window = [anchor + timedelta(days=i) for i in range(-3, 4)]
        context = _context_for(window)

        drivers = build_drivers([], _points(window), context)

        payday_drivers = [d for d in drivers if d.kind == "payday"]
        assert len(payday_drivers) == 1

    def test_no_payday_driver_outside_any_window(self):
        ordinary_days = [date(2026, 3, 5), date(2026, 3, 6), date(2026, 3, 20)]
        context = _context_for(ordinary_days)

        drivers = build_drivers([], _points(ordinary_days), context)

        assert not [d for d in drivers if d.kind == "payday"]

    def test_payday_coexists_with_a_gastro_event_on_the_same_date(self):
        # Dec 31 is both "Nochevieja" (gastro_event) and fin-de-mes: both
        # drivers must be present, independently.
        d = date(2026, 12, 31)
        context = _context_for([d])

        drivers = build_drivers([], _points([d]), context)

        kinds = {driver.kind for driver in drivers}
        assert "gastro_event" in kinds
        assert "payday" in kinds
        gastro = next(driver for driver in drivers if driver.kind == "gastro_event")
        payday = next(driver for driver in drivers if driver.kind == "payday")
        assert gastro.label == "Nochevieja"
        assert payday.label == "Fin de mes"


class TestPaydayImpactHonesty:
    def test_impact_pct_present_when_history_shows_a_genuine_uplift(self):
        # 2 years of daily history with a deterministic spike on every
        # payday-window day -> the driver for the next quincena should carry
        # a positive impact_pct.
        start, end = date(2024, 1, 1), date(2026, 3, 20)
        n_days = (end - start).days + 1
        dates = [start + timedelta(days=i) for i in range(n_days)]
        feats = build_date_features(dates)

        history = [
            HistoryPoint(ds=d, y=100.0 * 1.5 if feats[d].is_payday_window else 100.0)
            for d in dates
        ]

        forecast_window = [date(2026, 3, 14), date(2026, 3, 15), date(2026, 3, 16)]
        context = _context_for(dates + forecast_window)

        drivers = build_drivers(history, _points(forecast_window), context)

        payday_drivers = [d for d in drivers if d.kind == "payday"]
        assert payday_drivers
        assert all(d.impact_pct is not None for d in payday_drivers)
        assert all(d.impact_pct > 0 for d in payday_drivers)

    def test_impact_pct_omitted_when_history_has_no_evidence(self):
        # History deliberately confined to days 2-11 of the month -> it never
        # actually touches a payday window (14-16, month-end, or the Jan 1
        # spillover from Dec 31's fin-de-mes window), so there is zero
        # evidence to compute an uplift from -> impact_pct must stay None
        # rather than a fabricated number.
        start, end = date(2026, 1, 2), date(2026, 1, 11)
        dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        history = [HistoryPoint(ds=d, y=100.0) for d in dates]

        forecast_window = [date(2026, 3, 14), date(2026, 3, 15), date(2026, 3, 16)]
        context = _context_for(dates + forecast_window)

        drivers = build_drivers(history, _points(forecast_window), context)

        payday_drivers = [d for d in drivers if d.kind == "payday"]
        assert payday_drivers
        assert all(d.impact_pct is None for d in payday_drivers)


_WEEKLY_SEASON = [10.0, 12.0, 15.0, 20.0, 25.0, 22.0, 8.0]


def _daily_series(n: int, base: date | None = None) -> list[dict]:
    base = base or date(2023, 1, 1)
    return [
        {
            "ds": (base + timedelta(days=i)).isoformat(),
            "y": 50.0 + _WEEKLY_SEASON[i % 7],
        }
        for i in range(n)
    ]


class TestPaydayDriverHttpEndToEnd:
    def test_payday_driver_present_and_deduped_over_http(self, mock_weather_success):
        # 2026-03-01 .. 2026-03-31 (31 days), enough history for "auto" to
        # stay on statsforecast (driver logic is engine-independent).
        history = _daily_series(90, base=date(2025, 12, 1))
        payload = {
            "series_id": "payday-http",
            "frequency": "D",
            "horizon": 10,  # covers 2026-03-01..2026-03-10, includes Mar 1 spillover
            "history": history,
            "use_context": True,
            "engine": "statsforecast",
        }
        data = client.post("/forecast/run", json=payload).json()

        payday_drivers = [d for d in data["drivers"] if d["kind"] == "payday"]
        assert len(payday_drivers) == 1
        assert payday_drivers[0]["label"] == "Fin de mes"

    def test_calendar_only_still_produces_payday_driver(self, mock_weather_failure):
        history = _daily_series(90, base=date(2026, 1, 1))
        payload = {
            "series_id": "payday-calendar-only",
            "frequency": "D",
            "horizon": 10,  # 2026-04-01..2026-04-10, includes Apr 1 spillover of Mar 31
            "history": history,
            "use_context": True,
        }
        data = client.post("/forecast/run", json=payload).json()

        assert data["context_status"] == "calendar_only"
        payday_drivers = [d for d in data["drivers"] if d["kind"] == "payday"]
        assert len(payday_drivers) == 1
        assert payday_drivers[0]["label"] == "Fin de mes"

    def test_default_request_never_emits_a_payday_driver(self):
        # use_context omitted -> drivers must stay [] (backward compatibility,
        # same invariant already covered for other kinds in
        # test_context_contract.py::TestBackwardCompatibility).
        history = _daily_series(60, base=date(2026, 3, 1))
        payload = {
            "series_id": "payday-legacy",
            "frequency": "D",
            "horizon": 10,
            "history": history,
        }
        data = client.post("/forecast/run", json=payload).json()
        assert data["drivers"] == []
