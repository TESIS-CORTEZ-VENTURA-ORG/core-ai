"""Forecasting orchestration layer — sits between the router and the engine."""

from __future__ import annotations

from fastapi import HTTPException

from app.config import get_settings
from app.forecasting.engine import baseline_name, run_seasonal_naive
from app.forecasting.engines import (
    EngineNotAvailableError,
    ForecastEngine,
    resolve_engine,
)
from app.forecasting.engines.registry import UnknownEngineError
from app.forecasting.schemas import (
    BacktestMetrics,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HistoryPoint,
)
from app.metrics import smape


def forecast(request: ForecastRequest) -> ForecastResponse:
    """
    Orchestrate a full forecast run.

    Steps
    -----
    1. Validate that horizon <= configured max.
    2. Optionally run a holdout backtest when data is sufficient.
    3. Run the primary forecast on the full history.
    4. Return a ForecastResponse with optional backtest metrics.
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

    try:
        engine = resolve_engine(request.engine, settings.forecast_engine)
    except UnknownEngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    history = sorted(request.history, key=lambda p: p.ds)

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

        backtest = BacktestMetrics(
            holdout_size=holdout_size,
            model_smape=round(model_err, 4),
            baseline_smape=round(baseline_err, 4),
            improvement_pct=round(improvement, 4),
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
    )

    return ForecastResponse(
        series_id=request.series_id,
        engine=engine.key,
        model=engine.model_name(),
        baseline=baseline_name(),
        frequency=request.frequency,
        points=points,
        backtest=backtest,
    )


def _run_engine(
    engine: ForecastEngine,
    history: list[HistoryPoint],
    frequency: str,
    horizon: int,
    season_length: int | None,
) -> list[ForecastPoint]:
    """Run an engine, mapping a missing/unimplemented model to HTTP 501."""
    try:
        return engine.forecast(history, frequency, horizon, season_length)
    except (EngineNotAvailableError, NotImplementedError) as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
