"""Integration tests for the forecasting API."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily_series(n_points: int, base_date: date | None = None) -> list[dict]:
    """
    Generate a synthetic daily series with weekly seasonality.

    Pattern: base value + day-of-week seasonal component + small noise-free trend.
    """
    if base_date is None:
        base_date = date(2023, 1, 1)

    season = [10.0, 12.0, 15.0, 20.0, 25.0, 22.0, 8.0]  # Mon-Sun
    series = []
    for i in range(n_points):
        d = base_date + timedelta(days=i)
        y = 50.0 + season[i % 7] + i * 0.05  # trend + seasonality
        series.append({"ds": d.isoformat(), "y": round(y, 2)})
    return series


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestForecastHappyPath:
    """120-point daily series, horizon=14, expects full response with backtest."""

    def setup_method(self):
        history = _make_daily_series(120)
        self.payload = {
            "series_id": "test-series-01",
            "frequency": "D",
            "horizon": 14,
            "history": history,
        }
        self.response = client.post("/forecast/run", json=self.payload)

    def test_status_200(self):
        assert self.response.status_code == 200

    def test_correct_number_of_points(self):
        data = self.response.json()
        assert len(data["points"]) == 14

    def test_prediction_intervals_ordered(self):
        data = self.response.json()
        for pt in data["points"]:
            assert pt["yhat_lo"] <= pt["yhat"], (
                f"yhat_lo={pt['yhat_lo']} > yhat={pt['yhat']} on {pt['target_date']}"
            )
            assert pt["yhat"] <= pt["yhat_hi"], (
                f"yhat={pt['yhat']} > yhat_hi={pt['yhat_hi']} on {pt['target_date']}"
            )

    def test_target_dates_strictly_increasing(self):
        data = self.response.json()
        dates = [date.fromisoformat(pt["target_date"]) for pt in data["points"]]
        for i in range(1, len(dates)):
            assert dates[i] > dates[i - 1], (
                f"Dates not strictly increasing: {dates[i - 1]} >= {dates[i]}"
            )

    def test_target_dates_after_last_history_date(self):
        data = self.response.json()
        last_history = date.fromisoformat(self.payload["history"][-1]["ds"])
        first_forecast = date.fromisoformat(data["points"][0]["target_date"])
        assert first_forecast > last_history, (
            f"First forecast date {first_forecast} is not after last history {last_history}"
        )

    def test_backtest_is_present(self):
        data = self.response.json()
        assert data["backtest"] is not None

    def test_backtest_model_smape_is_non_negative_float(self):
        data = self.response.json()
        smape_val = data["backtest"]["model_smape"]
        assert isinstance(smape_val, (int, float))
        assert not math.isnan(smape_val)
        assert smape_val >= 0.0

    def test_series_id_echoed(self):
        data = self.response.json()
        assert data["series_id"] == "test-series-01"

    def test_model_and_baseline_strings_present(self):
        data = self.response.json()
        assert isinstance(data["model"], str) and data["model"]
        assert isinstance(data["baseline"], str) and data["baseline"]


class TestInsufficientDataForBacktest:
    """2-point history with large horizon: backtest must be None, forecast must succeed."""

    def setup_method(self):
        history = _make_daily_series(2)
        self.payload = {
            "series_id": "test-series-short",
            "frequency": "D",
            "horizon": 7,
            "history": history,
        }
        self.response = client.post("/forecast/run", json=self.payload)

    def test_status_200(self):
        assert self.response.status_code == 200

    def test_backtest_is_none(self):
        data = self.response.json()
        assert data["backtest"] is None

    def test_forecast_points_returned(self):
        data = self.response.json()
        assert len(data["points"]) == 7
