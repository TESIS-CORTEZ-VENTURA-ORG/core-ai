"""Shared pytest fixtures.

Critically, these prevent tests from ever hitting the real Open-Meteo network
for `use_context=True` requests exercised through the HTTP `TestClient` —
where there's no way to inject a `WeatherClient` directly the way
`app.forecasting.service.forecast(..., weather_client=...)` allows for
direct/unit-level calls. Patching `app.forecasting.service.fetch_weather_context`
(the name as imported into the service module) covers every HTTP-level test.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.forecasting.features.weather import WeatherPoint, WeatherUnavailableError


def _fake_weather_success(
    dates: list[date], latitude: float, longitude: float, client: object | None = None
) -> dict[date, WeatherPoint]:
    """Deterministic, dry, 22°C weather for every requested date — no network."""
    return {d: WeatherPoint(ds=d, temp_max_c=22.0, precip_mm=0.0) for d in dates}


def _fake_weather_failure(
    dates: list[date], latitude: float, longitude: float, client: object | None = None
) -> dict[date, WeatherPoint]:
    raise WeatherUnavailableError("simulated Open-Meteo outage (test)")


@pytest.fixture
def mock_weather_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Weather context succeeds with deterministic data — exercises the
    "full" context_status path without touching the network."""
    monkeypatch.setattr(
        "app.forecasting.service.fetch_weather_context", _fake_weather_success
    )


@pytest.fixture
def mock_weather_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Weather context always fails — exercises the "calendar_only"
    degradation path without touching the network."""
    monkeypatch.setattr(
        "app.forecasting.service.fetch_weather_context", _fake_weather_failure
    )
