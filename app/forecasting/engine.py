"""
Forecasting engine — baseline implementation.

This module exposes a single public function:

    run_forecast(history, frequency, horizon, season_length) -> list[ForecastPoint]

The current engine uses statsforecast (AutoETS + SeasonalNaive) when the library
is available, falling back to a pure numpy/pandas implementation when it is not
(e.g. build environments where Numba / statsforecast cannot be installed).

Chronos-2 will replace the primary model behind this same signature in a future
increment (E08-C2 or later).  The interface contract must not change.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from app.forecasting.schemas import ForecastPoint, HistoryPoint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SEASON_LENGTH: dict[str, int] = {"D": 7, "W": 52}
_MODEL_NAME_STATSFORECAST = "AutoETS"
_MODEL_NAME_NUMPY = "SeasonalNaive-numpy"
_BASELINE_NAME = "SeasonalNaive"
_LEVEL = 80  # prediction interval level (P10/P90)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def model_name() -> str:
    """Return the name of the primary model when it is unknown for a request."""
    return _MODEL_NAME_STATSFORECAST if _STATSFORECAST_AVAILABLE else _MODEL_NAME_NUMPY


def baseline_name() -> str:
    """Return the name of the baseline model."""
    return _BASELINE_NAME


def resolve_model_name(
    history: list[HistoryPoint],
    frequency: str,
    season_length: int | None,
) -> str:
    """Return the model that will ACTUALLY run for *this* series.

    The statsforecast engine downgrades AutoETS -> SeasonalNaive when the series
    is too short to estimate a seasonal model (len < 2 * season_length). This
    mirrors that decision so the response reports the real model used instead of
    a static label (otherwise a short series would be reported as "AutoETS" while
    SeasonalNaive actually ran).
    """
    if not _STATSFORECAST_AVAILABLE:
        return _MODEL_NAME_NUMPY
    sl = season_length if season_length is not None else _DEFAULT_SEASON_LENGTH[frequency]
    if len(history) >= 2 * sl:
        return _MODEL_NAME_STATSFORECAST  # AutoETS
    return _BASELINE_NAME  # SeasonalNaive used as primary fallback on short series


# ---------------------------------------------------------------------------
# statsforecast availability check
# ---------------------------------------------------------------------------

_STATSFORECAST_AVAILABLE = False

try:
    from statsforecast import StatsForecast  # noqa: F401
    from statsforecast.models import AutoETS, SeasonalNaive  # noqa: F401

    _STATSFORECAST_AVAILABLE = True
    logger.info("statsforecast is available — using AutoETS as primary model.")
except Exception as exc:  # pragma: no cover
    logger.warning(
        "statsforecast is NOT available (%s). Falling back to numpy/pandas engine.",
        exc,
    )

# ---------------------------------------------------------------------------
# statsforecast engine
# ---------------------------------------------------------------------------


def _date_range(start: date, steps: int, frequency: str) -> list[date]:
    """Generate a list of future dates starting the day/week after *start*."""
    delta = timedelta(days=1) if frequency == "D" else timedelta(weeks=1)
    return [start + delta * (i + 1) for i in range(steps)]


if _STATSFORECAST_AVAILABLE:
    import pandas as pd
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, SeasonalNaive

    def _build_sf_df(history: list[HistoryPoint]) -> "pd.DataFrame":
        return pd.DataFrame(
            {
                "unique_id": ["series"] * len(history),
                "ds": pd.to_datetime([p.ds for p in history]),
                "y": [p.y for p in history],
            }
        )

    def _run_statsforecast(
        history: list[HistoryPoint],
        frequency: str,
        horizon: int,
        season_length: int,
        models: list,
    ) -> "pd.DataFrame":
        df = _build_sf_df(history)
        sf_freq = "D" if frequency == "D" else "W"
        sf = StatsForecast(models=models, freq=sf_freq, n_jobs=1)
        fc = sf.forecast(df=df, h=horizon, level=[_LEVEL])
        return fc

    def _statsforecast_forecast(
        history: list[HistoryPoint],
        frequency: str,
        horizon: int,
        season_length: int,
    ) -> list[ForecastPoint]:
        min_season_obs = 2 * season_length

        if len(history) >= min_season_obs:
            primary_model = AutoETS(season_length=season_length)
            used_model_col = "AutoETS"
        else:
            logger.warning(
                "Series too short for AutoETS (len=%d, need %d). "
                "Using SeasonalNaive as primary.",
                len(history),
                min_season_obs,
            )
            primary_model = SeasonalNaive(season_length=season_length)
            used_model_col = "SeasonalNaive"

        fc = _run_statsforecast(
            history, frequency, horizon, season_length, [primary_model]
        )

        future_dates = _date_range(history[-1].ds, horizon, frequency)

        lo_col = f"{used_model_col}-lo-{_LEVEL}"
        hi_col = f"{used_model_col}-hi-{_LEVEL}"
        yhat_col = used_model_col

        # Use .reset_index() + column-label access to avoid itertuples() mangling
        # hyphens in column names (e.g. "AutoETS-lo-80" -> invalid attribute).
        fc_reset = fc.reset_index()
        yhat_vals = fc_reset[yhat_col].tolist()
        lo_vals = fc_reset[lo_col].tolist()
        hi_vals = fc_reset[hi_col].tolist()

        points: list[ForecastPoint] = []
        for i in range(horizon):
            points.append(
                ForecastPoint(
                    target_date=future_dates[i],
                    yhat=float(yhat_vals[i]),
                    yhat_lo=float(lo_vals[i]),
                    yhat_hi=float(hi_vals[i]),
                )
            )

        return points

# ---------------------------------------------------------------------------
# numpy/pandas fallback engine
# ---------------------------------------------------------------------------

import math  # noqa: E402 — always available

import numpy as np  # noqa: E402 — always available
import pandas as pd  # noqa: E402 — always available (pandas bundled with statsforecast anyway)


def _seasonal_naive_predict(values: list[float], season_length: int, horizon: int) -> list[float]:
    """Repeat the last full season's values to fill *horizon* steps."""
    n = len(values)
    preds: list[float] = []
    for h in range(1, horizon + 1):
        idx = n - season_length + ((h - 1) % season_length)
        preds.append(values[idx])
    return preds


def _empirical_bands(
    residuals: list[float], preds: list[float], alpha: float = 0.10
) -> tuple[list[float], list[float]]:
    """
    Build prediction bands from in-sample residuals using empirical quantiles.

    alpha=0.10 → 80% interval (P10/P90).
    """
    arr = np.array(residuals)
    q_lo = float(np.quantile(arr, alpha))
    q_hi = float(np.quantile(arr, 1.0 - alpha))
    lo = [p + q_lo for p in preds]
    hi = [p + q_hi for p in preds]
    return lo, hi


def _numpy_forecast(
    history: list[HistoryPoint],
    frequency: str,
    horizon: int,
    season_length: int,
) -> list[ForecastPoint]:
    """Pure numpy/pandas SeasonalNaive with empirical prediction bands."""
    values = [p.y for p in history]

    # In-sample residuals: one-step-ahead seasonal naive on training set
    in_sample_preds: list[float] = []
    for i in range(season_length, len(values)):
        in_sample_preds.append(values[i - season_length])
    residuals = [
        values[season_length + i] - in_sample_preds[i]
        for i in range(len(in_sample_preds))
    ]
    if not residuals:
        residuals = [0.0]

    preds = _seasonal_naive_predict(values, season_length, horizon)
    lo, hi = _empirical_bands(residuals, preds)

    future_dates = _date_range(history[-1].ds, horizon, frequency)

    return [
        ForecastPoint(
            target_date=future_dates[i],
            yhat=preds[i],
            yhat_lo=lo[i],
            yhat_hi=hi[i],
        )
        for i in range(horizon)
    ]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def run_forecast(
    history: list[HistoryPoint],
    frequency: str,
    horizon: int,
    season_length: int | None,
) -> list[ForecastPoint]:
    """
    Generate a demand forecast.

    Parameters
    ----------
    history:
        Ordered list of (date, value) observations, oldest first.
    frequency:
        "D" for daily, "W" for weekly.
    horizon:
        Number of future periods to forecast.
    season_length:
        Override the default seasonal period.  When None the default is
        7 for "D" and 52 for "W".

    Returns
    -------
    list[ForecastPoint]
        Exactly *horizon* forecast points with yhat, yhat_lo, yhat_hi.

    Notes
    -----
    This function is the integration seam for future model upgrades.
    Chronos-2 (or any other model) will replace the internals here while
    keeping the signature and return type unchanged so callers are unaffected.
    """
    sl = season_length if season_length is not None else _DEFAULT_SEASON_LENGTH[frequency]

    if _STATSFORECAST_AVAILABLE:
        return _statsforecast_forecast(history, frequency, horizon, sl)
    else:
        return _numpy_forecast(history, frequency, horizon, sl)


def run_seasonal_naive(
    history: list[HistoryPoint],
    frequency: str,
    horizon: int,
    season_length: int | None,
) -> list[ForecastPoint]:
    """
    Run SeasonalNaive baseline explicitly.

    Always uses the numpy implementation regardless of statsforecast availability
    so that the baseline is consistent in backtest comparisons.
    """
    sl = season_length if season_length is not None else _DEFAULT_SEASON_LENGTH[frequency]
    return _numpy_forecast(history, frequency, horizon, sl)
