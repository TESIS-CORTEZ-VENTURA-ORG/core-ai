"""Contract tests for HU-08-07 (exogenous context): backward compatibility,
`drivers`, `context_status`, and graceful weather degradation.

Weather is always mocked via `mock_weather_success`/`mock_weather_failure`
(conftest.py) — never the real network.
"""

from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

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


class TestBackwardCompatibility:
    """A request without `use_context`/`location` must respond exactly like
    the pre-HU-08-07 API: no drivers, context "off", no network calls."""

    def setup_method(self):
        self.payload = {
            "series_id": "legacy-series",
            "frequency": "D",
            "horizon": 14,
            "history": _daily_series(120),
        }
        self.response = client.post("/forecast/run", json=self.payload)

    def test_status_200(self):
        assert self.response.status_code == 200

    def test_drivers_is_empty_list(self):
        assert self.response.json()["drivers"] == []

    def test_context_status_is_off(self):
        assert self.response.json()["context_status"] == "off"

    def test_backtest_model_smape_no_context_is_none(self):
        data = self.response.json()
        assert data["backtest"] is not None
        assert data["backtest"]["model_smape_no_context"] is None

    def test_pre_existing_fields_unchanged_shape(self):
        data = self.response.json()
        for field in (
            "series_id",
            "engine",
            "model",
            "baseline",
            "frequency",
            "points",
            "backtest",
        ):
            assert field in data


class TestUseContextFullPath:
    def test_status_200_and_context_full(self, mock_weather_success):
        payload = {
            "series_id": "context-full",
            "frequency": "D",
            "horizon": 7,
            "history": _daily_series(60),
            "use_context": True,
        }
        resp = client.post("/forecast/run", json=payload)
        assert resp.status_code == 200
        assert resp.json()["context_status"] == "full"

    def test_custom_location_is_accepted(self, mock_weather_success):
        payload = {
            "series_id": "context-custom-location",
            "frequency": "D",
            "horizon": 7,
            "history": _daily_series(60),
            "use_context": True,
            "location": {"latitude": -8.11, "longitude": -79.03},  # Trujillo
        }
        resp = client.post("/forecast/run", json=payload)
        assert resp.status_code == 200


class TestUseContextDegradation:
    def test_weather_failure_degrades_to_calendar_only_without_failing(
        self, mock_weather_failure
    ):
        payload = {
            "series_id": "context-degraded",
            "frequency": "D",
            "horizon": 7,
            "history": _daily_series(60),
            "use_context": True,
        }
        resp = client.post("/forecast/run", json=payload)
        assert resp.status_code == 200
        assert resp.json()["context_status"] == "calendar_only"

    def test_calendar_only_still_produces_calendar_drivers(self, mock_weather_failure):
        # Cover a horizon guaranteed to include Fiestas Patrias so a driver
        # is emitted even without weather.
        history = _daily_series((date(2026, 7, 20) - date(2023, 1, 1)).days)
        payload = {
            "series_id": "context-degraded-drivers",
            "frequency": "D",
            "horizon": 10,  # 2026-07-21 .. 2026-07-30, includes Jul 28-29
            "history": history,
            "use_context": True,
        }
        data = client.post("/forecast/run", json=payload).json()
        assert data["context_status"] == "calendar_only"
        kinds = {d["kind"] for d in data["drivers"]}
        assert "weather" not in kinds  # weather degraded -> no weather drivers
        assert any(d["label"] == "Fiestas Patrias" for d in data["drivers"])


class TestDriverImpactHonesty:
    def test_impact_pct_present_when_history_has_evidence(self, mock_weather_success):
        # Multi-year history with the SAME spike on Jul 28-29 every year ->
        # the driver for the next Jul 28-29 occurrence should carry impact_pct.
        base = date(2023, 1, 1)
        n_days = (date(2026, 7, 20) - base).days
        series = []
        for i in range(n_days):
            d = base + timedelta(days=i)
            y = 50.0 + _WEEKLY_SEASON[i % 7]
            if d.month == 7 and d.day in (28, 29):
                y *= 1.7
            series.append({"ds": d.isoformat(), "y": round(y, 2)})

        payload = {
            "series_id": "impact-evidence",
            "frequency": "D",
            "horizon": 10,  # covers 2026-07-28/29
            "history": series,
            "use_context": True,
            "engine": "statsforecast",  # engine choice is irrelevant to drivers
        }
        data = client.post("/forecast/run", json=payload).json()
        fiestas = [d for d in data["drivers"] if d["label"] == "Fiestas Patrias"]
        assert fiestas, "expected a Fiestas Patrias driver within the horizon"
        assert any(d["impact_pct"] is not None for d in fiestas)
        # The synthetic spike is a genuine uplift -> impact_pct must be positive.
        assert all(d["impact_pct"] > 0 for d in fiestas if d["impact_pct"] is not None)

    def test_impact_pct_omitted_when_history_has_no_evidence(
        self, mock_weather_success
    ):
        # Short, flat history that never actually saw the event -> the driver
        # should still be narrated (so the UI can say "Fiestas Patrias en X
        # días") but WITHOUT a fabricated impact_pct.
        base = date(2026, 6, 1)
        series = _daily_series(40, base=base)  # ends 2026-07-10, no Jul 28/29 seen
        payload = {
            "series_id": "impact-no-evidence",
            "frequency": "D",
            "horizon": 25,  # 2026-07-11 .. 2026-08-04, covers Jul 28-29
            "history": series,
            "use_context": True,
        }
        data = client.post("/forecast/run", json=payload).json()
        fiestas = [d for d in data["drivers"] if d["label"] == "Fiestas Patrias"]
        assert fiestas, "expected a Fiestas Patrias driver within the horizon"
        assert all(d["impact_pct"] is None for d in fiestas)
