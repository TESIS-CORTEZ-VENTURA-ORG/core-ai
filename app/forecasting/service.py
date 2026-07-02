"""Forecasting orchestration layer — sits between the router and the engine."""

from __future__ import annotations

import logging

from fastapi import HTTPException

from app.config import get_settings
from app.forecasting.engine import baseline_name, date_range, run_seasonal_naive
from app.forecasting.engines import (
    EngineNotAvailableError,
    ForecastEngine,
    resolve_engine,
)
from app.forecasting.engines.ml_engine import MLEngine
from app.forecasting.engines.registry import UnknownEngineError
from app.forecasting.features.calendar import build_date_features
from app.forecasting.features.context import ForecastContext
from app.forecasting.features.drivers import build_drivers
from app.forecasting.features.weather import (
    WeatherClient,
    WeatherUnavailableError,
    fetch_weather_context,
)
from app.forecasting.schemas import (
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    BacktestMetrics,
    Driver,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HistoryPoint,
)
from app.metrics import smape

logger = logging.getLogger(__name__)


def forecast(
    request: ForecastRequest,
    *,
    weather_client: WeatherClient | None = None,
) -> ForecastResponse:
    """
    Orchestrate a full forecast run.

    Steps
    -----
    1. Validate that horizon <= configured max.
    2. Resolve the engine (auto-selection is history/context-aware — see
       `resolve_engine`).
    3. When `use_context=True` (HU-08-07): build calendar features (always
       available) and attempt a weather fetch (best-effort, degrades to
       calendar-only on any failure — a forecast must never fail solely
       because Open-Meteo is down).
    4. Optionally run a holdout backtest when data is sufficient, including a
       context-free re-run for the "ml" engine so the response can show the
       univariate-vs-exogenous comparison.
    5. Run the primary forecast on the full history.
    6. Return a ForecastResponse with optional backtest metrics and drivers.

    `weather_client` is exposed only for test injection (a fresh, mockable
    `WeatherClient`); production callers (the router) never pass it, so the
    process-wide cached singleton is used.
    """
    settings = get_settings()

    if request.horizon > settings.forecast_max_horizon:
        raise HTTPException(
            status_code=400,
            detail=(
                f"horizon={request.horizon} exceeds the maximum allowed "
                f"value of {settings.forecast_max_horizon}."
            ),
        )

    history = sorted(request.history, key=lambda p: p.ds)

    try:
        engine = resolve_engine(
            request.engine,
            settings.forecast_engine,
            history_length=len(history),
            frequency=request.frequency,
            season_length=request.season_length,
            use_context=request.use_context,
        )
    except UnknownEngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # -----------------------------------------------------------------------
    # Exogenous context (HU-08-07) — opt-in via use_context, additive only.
    # -----------------------------------------------------------------------
    context: ForecastContext | None = None
    context_status: str = "off"
    future_dates = date_range(history[-1].ds, request.horizon, request.frequency)

    if request.use_context:
        latitude = request.location.latitude if request.location else DEFAULT_LATITUDE
        longitude = (
            request.location.longitude if request.location else DEFAULT_LONGITUDE
        )
        all_dates = [p.ds for p in history] + future_dates
        date_features = build_date_features(all_dates)

        try:
            weather_by_date = fetch_weather_context(
                all_dates, latitude, longitude, client=weather_client
            )
            context_status = "full"
        except WeatherUnavailableError as exc:
            logger.warning(
                "Weather context unavailable for series '%s'; degrading to "
                "calendar-only context: %s",
                request.series_id,
                exc,
            )
            weather_by_date = {}
            context_status = "calendar_only"

        context = ForecastContext(
            date_features=date_features,
            weather_by_date=weather_by_date,
            context_status=context_status,  # type: ignore[arg-type]
        )

    # -----------------------------------------------------------------------
    # Backtest (holdout) — only when len(history) >= 2 * horizon
    # -----------------------------------------------------------------------
    backtest: BacktestMetrics | None = None
    holdout_size = request.horizon

    if len(history) >= 2 * holdout_size:
        train = history[:-holdout_size]
        test = history[-holdout_size:]

        model_preds = _run_engine(
            engine,
            train,
            request.frequency,
            holdout_size,
            request.season_length,
            context,
        )
        baseline_preds = run_seasonal_naive(
            train,
            request.frequency,
            holdout_size,
            request.season_length,
        )

        y_true = [p.y for p in test]
        model_yhats = [p.yhat for p in model_preds]
        baseline_yhats = [p.yhat for p in baseline_preds]

        model_err = smape(y_true, model_yhats)
        baseline_err = smape(y_true, baseline_yhats)

        if baseline_err > 0:
            improvement = (baseline_err - model_err) / baseline_err * 100.0
        else:
            improvement = 0.0

        # Academic comparison (ticket §4): when context is active and the
        # resolved engine actually consumes it, re-run the SAME holdout
        # without context so the response shows "univariate vs +exogenous".
        model_smape_no_context: float | None = None
        if context is not None and engine.key == MLEngine.key:
            no_context_preds = _run_engine(
                engine,
                train,
                request.frequency,
                holdout_size,
                request.season_length,
                None,
            )
            model_smape_no_context = round(
                smape(y_true, [p.yhat for p in no_context_preds]), 4
            )

        backtest = BacktestMetrics(
            holdout_size=holdout_size,
            model_smape=round(model_err, 4),
            baseline_smape=round(baseline_err, 4),
            improvement_pct=round(improvement, 4),
            model_smape_no_context=model_smape_no_context,
        )

    # -----------------------------------------------------------------------
    # Full forecast on complete history
    # -----------------------------------------------------------------------
    points = _run_engine(
        engine,
        history,
        request.frequency,
        request.horizon,
        request.season_length,
        context,
    )

    drivers: list[Driver] = (
        build_drivers(history, points, context) if context is not None else []
    )

    return ForecastResponse(
        series_id=request.series_id,
        engine=engine.key,
        model=engine.model_name(),
        baseline=baseline_name(),
        frequency=request.frequency,
        points=points,
        backtest=backtest,
        drivers=drivers,
        context_status=context_status,  # type: ignore[arg-type]
    )


def _run_engine(
    engine: ForecastEngine,
    history: list[HistoryPoint],
    frequency: str,
    horizon: int,
    season_length: int | None,
    context: ForecastContext | None,
) -> list[ForecastPoint]:
    """Run an engine, mapping a missing/unimplemented model to HTTP 501."""
    try:
        return engine.forecast(history, frequency, horizon, season_length, context)
    except (EngineNotAvailableError, NotImplementedError) as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
