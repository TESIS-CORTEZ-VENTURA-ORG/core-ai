"""Tests for the Open-Meteo weather client (app/forecasting/features/weather.py).

Never touches the real network — every HTTP call goes through an
`httpx.MockTransport` handler.
"""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from app.forecasting.features.weather import (
    WeatherClient,
    WeatherUnavailableError,
    fetch_weather_context,
)


def _daily_payload(
    dates: list[str], temps: list[float | None], precip: list[float | None]
) -> dict:
    return {
        "latitude": -12.05,
        "longitude": -77.05,
        "daily": {
            "time": dates,
            "temperature_2m_max": temps,
            "precipitation_sum": precip,
        },
    }


class TestFetchHistorical:
    def test_parses_points_from_archive_api(self):
        payload = _daily_payload(
            ["2026-01-01", "2026-01-02"],
            [28.5, 27.1],
            [0.0, 3.2],
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert "archive-api.open-meteo.com" in str(request.url)
            return httpx.Response(200, json=payload)

        client = WeatherClient(transport=httpx.MockTransport(handler))
        points = client.fetch_historical(
            date(2026, 1, 1), date(2026, 1, 2), -12.05, -77.05
        )

        assert len(points) == 2
        assert points[0].ds == date(2026, 1, 1)
        assert points[0].temp_max_c == 28.5
        assert points[1].precip_mm == 3.2

    def test_cache_avoids_duplicate_requests(self):
        payload = _daily_payload(["2026-01-01"], [28.5], [0.0])
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(200, json=payload)

        client = WeatherClient(transport=httpx.MockTransport(handler))
        client.fetch_historical(date(2026, 1, 1), date(2026, 1, 1), -12.05, -77.05)
        client.fetch_historical(date(2026, 1, 1), date(2026, 1, 1), -12.05, -77.05)

        assert call_count["n"] == 1


class TestFetchForecast:
    def test_parses_points_from_forecast_api(self):
        payload = _daily_payload(["2026-07-01", "2026-07-02"], [22.0, 21.5], [1.0, 0.0])

        def handler(request: httpx.Request) -> httpx.Response:
            assert "api.open-meteo.com" in str(request.url)
            assert "archive" not in str(request.url)
            return httpx.Response(200, json=payload)

        client = WeatherClient(transport=httpx.MockTransport(handler))
        points = client.fetch_forecast(2, -12.05, -77.05)

        assert len(points) == 2
        assert points[0].precip_mm == 1.0

    def test_caps_forecast_days_at_16(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["forecast_days"] = request.url.params.get("forecast_days")
            return httpx.Response(200, json=_daily_payload([], [], []))

        client = WeatherClient(transport=httpx.MockTransport(handler))
        client.fetch_forecast(30, -12.05, -77.05)

        assert captured["forecast_days"] == "16"


class TestGracefulDegradation:
    def test_timeout_raises_weather_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("simulated timeout", request=request)

        client = WeatherClient(transport=httpx.MockTransport(handler))
        with pytest.raises(WeatherUnavailableError):
            client.fetch_historical(date(2026, 1, 1), date(2026, 1, 2), -12.05, -77.05)

    def test_http_error_status_raises_weather_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="service unavailable")

        client = WeatherClient(transport=httpx.MockTransport(handler))
        with pytest.raises(WeatherUnavailableError):
            client.fetch_forecast(7, -12.05, -77.05)

    def test_malformed_payload_raises_weather_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": "shape"})

        client = WeatherClient(transport=httpx.MockTransport(handler))
        with pytest.raises(WeatherUnavailableError):
            client.fetch_forecast(7, -12.05, -77.05)


class TestFetchWeatherContext:
    def test_splits_past_and_future_dates_across_endpoints(self):
        urls_hit = []

        def handler(request: httpx.Request) -> httpx.Response:
            urls_hit.append(str(request.url))
            if "archive" in str(request.url):
                return httpx.Response(
                    200, json=_daily_payload(["2020-01-01"], [25.0], [0.0])
                )
            return httpx.Response(
                200, json=_daily_payload(["2099-01-01"], [26.0], [0.0])
            )

        client = WeatherClient(transport=httpx.MockTransport(handler))
        result = fetch_weather_context(
            [date(2020, 1, 1), date(2099, 1, 1)], -12.05, -77.05, client=client
        )

        assert any("archive" in u for u in urls_hit)
        assert any("archive" not in u and "api.open-meteo.com" in u for u in urls_hit)
        assert date(2020, 1, 1) in result
        assert date(2099, 1, 1) in result

    def test_empty_dates_returns_empty_dict_without_any_request(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not call Open-Meteo for an empty date list")

        client = WeatherClient(transport=httpx.MockTransport(handler))
        assert fetch_weather_context([], -12.05, -77.05, client=client) == {}
