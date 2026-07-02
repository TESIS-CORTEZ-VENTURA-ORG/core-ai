"""Tests for the LightGBM engine (app/forecasting/engines/ml_engine.py).

Any scenario using `use_context=True` either goes through the direct
`service.forecast(..., weather_client=...)` injection point, or (for
HTTP-level tests) the `mock_weather_success`/`mock_weather_failure` fixtures
from `conftest.py` — never the real Open-Meteo network.
"""

from __future__ import annotations

from datetime import date, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from app.forecasting.engines.base import EngineNotAvailableError
from app.forecasting.engines.ml_engine import MLEngine
from app.forecasting.features.calendar import build_date_features
from app.forecasting.features.weather import WeatherClient
from app.forecasting.schemas import ForecastRequest, HistoryPoint
from app.forecasting.service import forecast
from app.main import app

client = TestClient(app)

_WEEKLY_SEASON = [10.0, 12.0, 15.0, 20.0, 25.0, 22.0, 8.0]  # Mon-Sun


def _daily_series(n: int, base: date | None = None) -> list[dict]:
    base = base or date(2023, 1, 1)
    return [
        {
            "ds": (base + timedelta(days=i)).isoformat(),
            "y": 50.0 + _WEEKLY_SEASON[i % 7],
        }
        for i in range(n)
    ]


def _spiked_daily_series(
    start: date, end: date, spike_multiplier: float = 1.8
) -> list[HistoryPoint]:
    """Weekly-seasonal series with a deterministic spike every Jul 28-29
    (Fiestas Patrias), repeated across every year in the range — exactly the
    kind of pattern the ML engine's calendar features should learn and a
    fixed-lag seasonal-naive baseline cannot.
    """
    n = (end - start).days + 1
    series: list[HistoryPoint] = []
    for i in range(n):
        d = start + timedelta(days=i)
        y = 50.0 + _WEEKLY_SEASON[i % 7]
        if d.month == 7 and d.day in (28, 29):
            y *= spike_multiplier
        series.append(HistoryPoint(ds=d, y=round(y, 2)))
    return series


def _payday_spiked_daily_series(
    start: date, end: date, spike_multiplier: float = 1.6
) -> list[HistoryPoint]:
    """Weekly-seasonal series with a deterministic spike on every payday
    window day (quincena/fin-de-mes +-1, see features/calendar.py) — a
    twice-a-month pattern a fixed weekly lag cannot anticipate, but the
    `is_payday_window` calendar feature can.
    """
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    payday_flags = build_date_features(dates)
    series: list[HistoryPoint] = []
    for i, d in enumerate(dates):
        y = 50.0 + _WEEKLY_SEASON[i % 7]
        if payday_flags[d].is_payday_window:
            y *= spike_multiplier
        series.append(HistoryPoint(ds=d, y=round(y, 2)))
    return series


def _mock_weather_client() -> WeatherClient:
    """A WeatherClient wired to a MockTransport returning flat, dry weather —
    used for direct `service.forecast(..., weather_client=...)` calls that
    need `use_context=True` without any real HTTP traffic."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": ["2000-01-01"],
                    "temperature_2m_max": [22.0],
                    "precipitation_sum": [0.0],
                }
            },
        )

    return WeatherClient(transport=httpx.MockTransport(handler))


class TestMinimumHistoryGuard:
    def test_explicit_ml_with_short_history_returns_501(self):
        payload = {
            "series_id": "short",
            "frequency": "D",
            "horizon": 3,
            "history": _daily_series(10),
            "engine": "ml",
        }
        resp = client.post("/forecast/run", json=payload)
        assert resp.status_code == 501
        assert "observations" in resp.json()["detail"].lower()

    def test_engine_raises_engine_not_available_directly(self):
        engine = MLEngine()
        short_history = [
            HistoryPoint(ds=date(2026, 1, i + 1), y=float(i)) for i in range(5)
        ]
        with pytest.raises(EngineNotAvailableError):
            engine.forecast(short_history, "D", 3, None)


class TestMLForecastShape:
    def test_returns_correct_number_of_points(self):
        payload = {
            "series_id": "ml-shape",
            "frequency": "D",
            "horizon": 7,
            "history": _daily_series(60),
            "engine": "ml",
        }
        resp = client.post("/forecast/run", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["engine"] == "ml"
        assert data["model"] == "LightGBM"
        assert len(data["points"]) == 7

    def test_prediction_intervals_ordered(self):
        payload = {
            "series_id": "ml-bands",
            "frequency": "D",
            "horizon": 7,
            "history": _daily_series(60),
            "engine": "ml",
        }
        data = client.post("/forecast/run", json=payload).json()
        for pt in data["points"]:
            assert pt["yhat_lo"] <= pt["yhat"] <= pt["yhat_hi"]


class TestAutoSelectionRespectsUseContext:
    def test_default_auto_request_does_not_pick_ml(self):
        # >= 28 obs (enough for MLEngine's own minimum) but use_context is
        # omitted -> "auto" must keep behaving exactly as before this feature.
        payload = {
            "series_id": "auto-no-context",
            "frequency": "D",
            "horizon": 7,
            "history": _daily_series(60),
        }
        data = client.post("/forecast/run", json=payload).json()
        assert data["engine"] != "ml"

    def test_auto_with_use_context_and_enough_history_picks_ml(
        self, mock_weather_success
    ):
        payload = {
            "series_id": "auto-with-context",
            "frequency": "D",
            "horizon": 7,
            "history": _daily_series(60),
            "use_context": True,
        }
        data = client.post("/forecast/run", json=payload).json()
        assert data["engine"] == "ml"

    def test_auto_with_use_context_but_short_history_falls_back(
        self, mock_weather_success
    ):
        payload = {
            "series_id": "auto-context-short",
            "frequency": "D",
            "horizon": 3,
            "history": _daily_series(10),
            "use_context": True,
        }
        resp = client.post("/forecast/run", json=payload)
        assert resp.status_code == 200
        assert resp.json()["engine"] != "ml"


class TestBacktestBeatsNaiveWithContext:
    """Core thesis claim (ticket §4): on a series with a real holiday-anchored
    spike that a fixed weekly lag cannot see coming, the context-aware ML
    engine should out-perform both the seasonal-naive baseline AND its own
    context-free run on the same holdout.
    """

    def setup_method(self):
        # 3 occurrences of the Jul 28-29 spike (2023, 2024, 2025) in training;
        # the 14-day holdout window (last 14 days ending 2025-08-04) covers
        # the 3rd one, so the model has 2 prior occurrences to learn from.
        history = _spiked_daily_series(date(2023, 1, 1), date(2025, 8, 4))
        self.request = ForecastRequest(
            series_id="spiked-holiday-series",
            frequency="D",
            horizon=14,
            history=history,
            engine="ml",
            use_context=True,
        )
        self.response = forecast(self.request, weather_client=_mock_weather_client())

    def test_context_status_is_full(self):
        assert self.response.context_status == "full"

    def test_backtest_is_present(self):
        assert self.response.backtest is not None

    def test_context_model_beats_seasonal_naive(self):
        bt = self.response.backtest
        assert bt.model_smape < bt.baseline_smape
        assert bt.improvement_pct > 0

    def test_context_model_beats_its_own_context_free_run(self):
        bt = self.response.backtest
        assert bt.model_smape_no_context is not None
        assert bt.model_smape < bt.model_smape_no_context


class TestBacktestBeatsNaiveWithPaydayContext:
    """Same thesis claim as `TestBacktestBeatsNaiveWithContext` but for the
    payday (quincena/fin-de-mes) signal instead of a yearly holiday: a
    twice-a-month spike a fixed weekly lag cannot see coming, which
    `is_payday_window` should let the ML engine anticipate.
    """

    def setup_method(self):
        # ~2.5 years of daily history -> dozens of payday-window occurrences
        # in training; the 14-day holdout (ending 2025-08-04) covers at least
        # one quincena/fin-de-mes window.
        history = _payday_spiked_daily_series(date(2023, 1, 1), date(2025, 8, 4))
        self.request = ForecastRequest(
            series_id="spiked-payday-series",
            frequency="D",
            horizon=14,
            history=history,
            engine="ml",
            use_context=True,
        )
        self.response = forecast(self.request, weather_client=_mock_weather_client())

    def test_context_status_is_full(self):
        assert self.response.context_status == "full"

    def test_backtest_is_present(self):
        assert self.response.backtest is not None

    def test_context_model_beats_seasonal_naive(self):
        bt = self.response.backtest
        assert bt.model_smape < bt.baseline_smape
        assert bt.improvement_pct > 0

    def test_context_model_beats_its_own_context_free_run(self):
        bt = self.response.backtest
        assert bt.model_smape_no_context is not None
        assert bt.model_smape < bt.model_smape_no_context

    def test_payday_driver_present_in_response(self):
        kinds = {d.kind for d in self.response.drivers}
        assert "payday" in kinds
