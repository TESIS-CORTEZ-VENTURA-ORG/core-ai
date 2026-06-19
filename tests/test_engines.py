"""Tests for pluggable engine selection.

These prove the strategy seam is wired: the response reports which engine ran,
explicit selection works, the future models (TimesFM / Chronos) are reachable
and fail with a clear 501 until their adapters land, and unknown engines 400.
"""

from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.forecasting.engines import available_engines, list_engine_keys, resolve_engine
from app.main import app

client = TestClient(app)


def _series(n: int) -> list[dict]:
    base = date(2023, 1, 1)
    season = [10.0, 12.0, 15.0, 20.0, 25.0, 22.0, 8.0]
    return [
        {"ds": (base + timedelta(days=i)).isoformat(), "y": 50.0 + season[i % 7]}
        for i in range(n)
    ]


def _payload(engine: str | None = None, n: int = 60, horizon: int = 7) -> dict:
    body: dict = {
        "series_id": "s1",
        "frequency": "D",
        "horizon": horizon,
        "history": _series(n),
    }
    if engine is not None:
        body["engine"] = engine
    return body


class TestEngineSelection:
    def test_registry_lists_all_engines(self):
        keys = list_engine_keys()
        for expected in ("auto", "statsforecast", "seasonalnaive", "timesfm", "chronos"):
            assert expected in keys

    def test_auto_degrades_to_statsforecast_when_heavy_models_absent(self):
        # TimesFM / Chronos are not installed in CI, so auto must fall back.
        resolved = resolve_engine(None, "auto")
        assert resolved.key == "statsforecast"

    def test_default_request_reports_statsforecast_engine(self):
        data = client.post("/forecast/run", json=_payload()).json()
        assert data["engine"] == "statsforecast"

    def test_explicit_statsforecast(self):
        resp = client.post("/forecast/run", json=_payload("statsforecast"))
        assert resp.status_code == 200
        assert resp.json()["engine"] == "statsforecast"

    def test_explicit_seasonalnaive_baseline(self):
        resp = client.post("/forecast/run", json=_payload("seasonalnaive"))
        assert resp.status_code == 200
        body = resp.json()
        assert body["engine"] == "seasonalnaive"
        assert body["model"] == "SeasonalNaive"


class TestFutureEnginesWired:
    """TimesFM and Chronos are reachable but not yet implemented -> 501."""

    def test_timesfm_returns_501_with_hint(self):
        resp = client.post("/forecast/run", json=_payload("timesfm"))
        assert resp.status_code == 501
        assert "timesfm" in resp.json()["detail"].lower()

    def test_chronos_returns_501_with_hint(self):
        resp = client.post("/forecast/run", json=_payload("chronos"))
        assert resp.status_code == 501
        assert "chronos" in resp.json()["detail"].lower()

    def test_future_engines_are_registered(self):
        avail = available_engines()
        assert "timesfm" in avail
        assert "chronos" in avail


class TestUnknownEngine:
    def test_unknown_engine_returns_400(self):
        resp = client.post("/forecast/run", json=_payload("does-not-exist"))
        assert resp.status_code == 400
        assert "unknown engine" in resp.json()["detail"].lower()
