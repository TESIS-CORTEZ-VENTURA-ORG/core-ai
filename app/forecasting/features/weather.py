"""Open-Meteo weather client (no API key) with mandatory graceful degradation.

Fetches daily max temperature + precipitation for a restaurant location
(default Lima) to feed the ML engine and the response `drivers`. Weather is
inherently unreliable to fetch (public rate limits, timeouts, outages) — every
caller MUST be prepared for :class:`WeatherUnavailableError` and fall back to
calendar-only context. A forecast request must NEVER fail solely because
Open-Meteo is down; see `context_status="calendar_only"` on `ForecastResponse`.

Two endpoints, matching Open-Meteo's split between historical and forecast
data:
- Archive API (`archive-api.open-meteo.com`) — arbitrary past date ranges,
  used for training-history weather.
- Forecast API (`api.open-meteo.com`) — up to 16 days ahead, used for the
  forecast horizon.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date

import httpx

logger = logging.getLogger(__name__)

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_MAX_FORECAST_DAYS = 16
_DEFAULT_TIMEOUT_SECONDS = 4.0
# 15 min: long enough that a single request's backtest pass + final full-history
# pass (which can overlap in date range) reuse one fetch instead of hammering a
# keyless, rate-limited public API twice for the same data.
_DEFAULT_CACHE_TTL_SECONDS = 900
_DAILY_PARAMS = "temperature_2m_max,precipitation_sum"
_TIMEZONE = "America/Lima"


class WeatherUnavailableError(RuntimeError):
    """Raised when Open-Meteo cannot be reached or returns an unusable response.

    Callers MUST catch this and degrade to calendar-only context — see
    `app/forecasting/service.py`.
    """


@dataclass(frozen=True)
class WeatherPoint:
    ds: date
    temp_max_c: float | None
    precip_mm: float | None


class WeatherClient:
    """Thin Open-Meteo client with a short timeout and an in-memory TTL cache."""

    def __init__(
        self,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        cache_ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self._client = httpx.Client(transport=transport, timeout=timeout_seconds)
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[tuple, tuple[float, list[WeatherPoint]]] = {}

    def close(self) -> None:
        self._client.close()

    def fetch_historical(
        self, start: date, end: date, latitude: float, longitude: float
    ) -> list[WeatherPoint]:
        """Daily weather for a past date range via the archive API (one call)."""
        cache_key = ("archive", latitude, longitude, start, end)
        return self._get_or_fetch(
            cache_key,
            lambda: self._request(
                _ARCHIVE_URL,
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "daily": _DAILY_PARAMS,
                    "timezone": _TIMEZONE,
                },
            ),
        )

    def fetch_forecast(
        self, days_ahead: int, latitude: float, longitude: float
    ) -> list[WeatherPoint]:
        """Daily weather for today..+N days via the forecast API (max 16 days)."""
        forecast_days = max(1, min(days_ahead, _MAX_FORECAST_DAYS))
        cache_key = ("forecast", latitude, longitude, forecast_days, date.today())
        return self._get_or_fetch(
            cache_key,
            lambda: self._request(
                _FORECAST_URL,
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "daily": _DAILY_PARAMS,
                    "timezone": _TIMEZONE,
                    "forecast_days": forecast_days,
                },
            ),
        )

    def _get_or_fetch(self, cache_key: tuple, fetch) -> list[WeatherPoint]:  # noqa: ANN001
        cached = self._cache.get(cache_key)
        if cached is not None:
            expires_at, points = cached
            if time.monotonic() < expires_at:
                return points
        points = fetch()
        self._cache[cache_key] = (time.monotonic() + self._cache_ttl_seconds, points)
        return points

    def _request(self, url: str, params: dict) -> list[WeatherPoint]:
        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            logger.warning("Open-Meteo request to %s timed out: %s", url, exc)
            raise WeatherUnavailableError(
                f"Open-Meteo timeout calling {url}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning("Open-Meteo request to %s failed: %s", url, exc)
            raise WeatherUnavailableError(
                f"Open-Meteo error calling {url}: {exc}"
            ) from exc
        except ValueError as exc:  # invalid JSON body
            logger.warning(
                "Open-Meteo response from %s was not valid JSON: %s", url, exc
            )
            raise WeatherUnavailableError(
                f"Open-Meteo invalid JSON from {url}: {exc}"
            ) from exc

        try:
            daily = payload["daily"]
            dates = [date.fromisoformat(s) for s in daily["time"]]
            temps = daily.get("temperature_2m_max", [None] * len(dates))
            precip = daily.get("precipitation_sum", [None] * len(dates))
        except (KeyError, TypeError) as exc:
            logger.warning(
                "Open-Meteo response from %s had an unexpected shape: %s", url, exc
            )
            raise WeatherUnavailableError(
                f"Open-Meteo malformed payload from {url}: {exc}"
            ) from exc

        return [
            WeatherPoint(ds=d, temp_max_c=t, precip_mm=p)
            for d, t, p in zip(dates, temps, precip)
        ]


_default_client: WeatherClient | None = None


def get_default_client() -> WeatherClient:
    """Process-wide singleton so the TTL cache is actually shared across requests."""
    global _default_client
    if _default_client is None:
        _default_client = WeatherClient()
    return _default_client


def fetch_weather_context(
    dates: list[date],
    latitude: float,
    longitude: float,
    client: WeatherClient | None = None,
) -> dict[date, WeatherPoint]:
    """Fetch weather for *dates*, routed to the archive or forecast API.

    Raises `WeatherUnavailableError` if either leg fails — the caller (the
    forecasting service) is responsible for catching it and degrading to
    calendar-only context so a flaky weather provider never breaks a forecast.
    """
    if not dates:
        return {}

    client = client or get_default_client()
    today = date.today()
    past_dates = sorted(d for d in dates if d < today)
    future_dates = sorted(d for d in dates if d >= today)

    points: dict[date, WeatherPoint] = {}
    if past_dates:
        historical = client.fetch_historical(
            past_dates[0], past_dates[-1], latitude, longitude
        )
        points.update({p.ds: p for p in historical})
    if future_dates:
        days_ahead = (future_dates[-1] - today).days + 1
        forecast = client.fetch_forecast(days_ahead, latitude, longitude)
        points.update({p.ds: p for p in forecast})
    return points
