"""LightGBM demand-forecasting engine with optional exogenous context.

Design decision — plain LightGBM over `mlforecast` (documented per the ticket,
both are Nixtla-adjacent so either was acceptable):

- `mlforecast`'s exogenous-feature API (future exogenous frames) assumes ALL
  future covariate values are known and reliable. Our calendar covariates are
  fully deterministic, but the weather covariate is best-effort (Open-Meteo,
  degrades gracefully to "no weather" mid-request) — mixing a reliable and an
  unreliable exogenous source is simpler to reason about, and to unit test,
  with a small hand-rolled pandas feature matrix than by fighting
  `mlforecast`'s "future frame" contract for a partially-available input.
- `mlforecast`'s main selling point is fast batched multi-series backtesting.
  This service always trains exactly one series per request (and retrains
  from scratch every time — see below), so that advantage doesn't apply here.
- A hand-rolled matrix keeps every feature (lags, calendar, weather) visible
  in one place, which matters for a thesis that has to explain *why* a number
  moved, not just produce it.

Retraining: this engine is retrained on the full submitted `history` on every
single call (explicit product requirement — the model must "relearn" from
whatever history the caller sends, not from a previously fitted artifact).
There is intentionally no persisted model/state between requests.

Forecasting strategy: recursive one-step-ahead. Each future step is predicted
from lag/rolling features built off the true history plus previously
predicted future values, then the prediction is appended to the working
series before building the next step's features — the standard recursive
multi-horizon strategy for tree-based models.

Prediction intervals: NOT quantile regression (which would need 2-3x the
LightGBM fits per request, on top of the mandatory full retrain — too
expensive to pay on every request). Instead, the same technique the numpy
fallback baseline already uses (`app/forecasting/engine.py::_empirical_bands`):
empirical quantiles of in-sample one-step-ahead residuals, added to each
point prediction. Keeps intervals honest without tripling training cost.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import ClassVar

import numpy as np
import pandas as pd

from app.forecasting.engine import date_range
from app.forecasting.engines.base import EngineNotAvailableError, ForecastEngine
from app.forecasting.features.context import ForecastContext
from app.forecasting.features.weather import WeatherPoint
from app.forecasting.schemas import ForecastPoint, HistoryPoint

logger = logging.getLogger(__name__)

_DEFAULT_SEASON_LENGTH: dict[str, int] = {"D": 7, "W": 52}

# An engine needs enough *usable* training rows (after the lag/rolling warmup
# window is dropped) for LightGBM to learn anything beyond noise. Two seasons
# is not enough once ~1 season is consumed as warmup (lag_{season_length}
# leaves the first `season_length` rows as NaN) — four seasons leaves at
# least three full seasons of clean training data, which is the minimum we
# consider "reasonable" for a several-feature boosted-tree model.
_MIN_SEASONS_FOR_ML = 4

# Sentinel used for `days_to_next_event` when no event falls within the
# calendar module's lookahead window — keeps the feature numeric (LightGBM
# splits on it like any other continuous feature) without conflating "no
# event nearby" with "event is today" (which would be 0).
_NO_EVENT_SENTINEL_DAYS = 999

# Matches the 80% (P10/P90) interval convention used by the other engines.
_RESIDUAL_ALPHA = 0.10

_LGBM_PARAMS: dict[str, object] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "num_leaves": 15,
    # Default min_child_samples (20) is too high for the small per-series
    # datasets this service trains on (tens to low hundreds of rows) and
    # would silently degenerate to a near-constant model.
    "min_child_samples": 5,
    "random_state": 42,
    "verbosity": -1,  # LightGBM logs to stdout by default; route errors via the app logger instead.
}

_INSTALL_HINT = (
    "ML engine selected but the 'lightgbm' package is not importable. On "
    "Debian-based images this also requires the 'libgomp1' system package "
    "(LightGBM's OpenMP runtime dependency — see Dockerfile). Install both "
    "or use engine='statsforecast'."
)


def _is_lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except Exception:
        return False
    return True


class MLEngine(ForecastEngine):
    """LightGBM engine with lag/rolling/calendar/weather features, retrained per request."""

    key: ClassVar[str] = "ml"
    auto_selectable: ClassVar[bool] = True

    @classmethod
    def is_available(cls) -> bool:
        return _is_lightgbm_available()

    @classmethod
    def auto_selectable_for(
        cls,
        history_length: int,
        frequency: str,
        season_length: int | None,
        use_context: bool = False,
    ) -> bool:
        if not use_context:
            # Preserve "auto"'s pre-existing default (statsforecast) when the
            # caller hasn't opted into exogenous context. ML's edge comes
            # from calendar/weather features; without them there's no reason
            # to prefer it over the proven baseline, and — critically — a
            # default request (no new fields set) must keep responding
            # exactly as it did before this feature shipped.
            return False
        sl = season_length or _DEFAULT_SEASON_LENGTH.get(frequency, 7)
        return history_length >= _MIN_SEASONS_FOR_ML * sl

    def model_name(self) -> str:
        return "LightGBM"

    def forecast(
        self,
        history: list[HistoryPoint],
        frequency: str,
        horizon: int,
        season_length: int | None,
        context: ForecastContext | None = None,
    ) -> list[ForecastPoint]:
        if not self.is_available():
            raise EngineNotAvailableError(_INSTALL_HINT)

        sl = season_length or _DEFAULT_SEASON_LENGTH.get(frequency, 7)
        min_required = _MIN_SEASONS_FOR_ML * sl
        if len(history) < min_required:
            raise EngineNotAvailableError(
                f"ML engine needs at least {min_required} observations "
                f"({_MIN_SEASONS_FOR_ML} seasons of length {sl}) for frequency "
                f"'{frequency}'; got {len(history)}. Submit more history or use "
                f"engine='statsforecast'."
            )

        from lightgbm import (
            LGBMRegressor,
        )  # local import: keeps is_available() truthful
        # even when the dependency is present but misconfigured (e.g. missing libgomp1),
        # since importing at module load time would break the whole app, not just this engine.

        sorted_history = sorted(history, key=lambda p: p.ds)
        future_dates = date_range(sorted_history[-1].ds, horizon, frequency)

        df = _build_training_frame(sorted_history, sl, context)
        feature_cols = [c for c in df.columns if c not in ("ds", "y")]
        categorical_cols = [c for c in ("day_of_week", "month") if c in feature_cols]

        train_df = df.dropna(subset=["lag_1", f"lag_{sl}"]).reset_index(drop=True)
        if train_df.empty:
            raise EngineNotAvailableError(
                "ML engine has no usable training rows after building lag features; "
                "history is too short or irregular for the requested season_length."
            )

        model = LGBMRegressor(**_LGBM_PARAMS)
        model.fit(
            train_df[feature_cols], train_df["y"], categorical_feature=categorical_cols
        )

        in_sample_pred = model.predict(train_df[feature_cols])
        residuals = (train_df["y"].to_numpy() - in_sample_pred).tolist()
        lo_offset, hi_offset = _residual_offsets(residuals)

        temp_fallback = 0.0
        if "temp_max_c" in df.columns:
            temp_mean = df["temp_max_c"].mean()
            temp_fallback = float(temp_mean) if not pd.isna(temp_mean) else 0.0

        values = [p.y for p in sorted_history]
        dates = [p.ds for p in sorted_history]
        points: list[ForecastPoint] = []
        for target_date in future_dates:
            row = _build_prediction_row(target_date, values, sl, context, temp_fallback)
            x = pd.DataFrame([row], columns=feature_cols)
            yhat = float(model.predict(x)[0])
            points.append(
                ForecastPoint(
                    target_date=target_date,
                    yhat=yhat,
                    yhat_lo=yhat + lo_offset,
                    yhat_hi=yhat + hi_offset,
                )
            )
            values.append(yhat)
            dates.append(target_date)

        return points


def _residual_offsets(residuals: list[float]) -> tuple[float, float]:
    """Empirical P10/P90 offsets from in-sample one-step-ahead residuals."""
    if not residuals:
        return 0.0, 0.0
    arr = np.array(residuals)
    lo = float(np.quantile(arr, _RESIDUAL_ALPHA))
    hi = float(np.quantile(arr, 1.0 - _RESIDUAL_ALPHA))
    return lo, hi


def _weather_field(
    weather_by_date: dict[date, WeatherPoint], d: date, field: str
) -> float | None:
    point = weather_by_date.get(d)
    return getattr(point, field) if point is not None else None


def _build_training_frame(
    history: list[HistoryPoint],
    season_length: int,
    context: ForecastContext | None,
) -> pd.DataFrame:
    """Build the (ds, y, features...) training frame for the full history.

    Column set mirrors `_build_prediction_row` exactly (same `context`
    instance threaded through both), so the training frame's columns and the
    single-row prediction frame always line up.
    """
    dates = [p.ds for p in history]
    values = [p.y for p in history]
    df = pd.DataFrame({"ds": dates, "y": values})
    df["lag_1"] = df["y"].shift(1)
    df[f"lag_{season_length}"] = df["y"].shift(season_length)
    df[f"rolling_mean_{season_length}"] = (
        df["y"].shift(1).rolling(season_length, min_periods=1).mean()
    )
    df["day_of_week"] = df["ds"].apply(lambda d: d.weekday())
    df["month"] = df["ds"].apply(lambda d: d.month)

    if context is not None:
        df["is_holiday"] = df["ds"].map(
            lambda d: (
                int(context.date_features[d].is_holiday)
                if d in context.date_features
                else 0
            )
        )
        df["is_weekend"] = df["ds"].map(
            lambda d: (
                int(context.date_features[d].is_weekend)
                if d in context.date_features
                else int(d.weekday() >= 5)
            )
        )
        df["days_to_next_event"] = df["ds"].map(
            lambda d: (
                context.date_features[d].days_to_next_event
                if d in context.date_features
                and context.date_features[d].days_to_next_event is not None
                else _NO_EVENT_SENTINEL_DAYS
            )
        )
        # Payday signal (quincena/fin-de-mes +-1 day, see features/calendar.py)
        # — plain 0/1 flag, same treatment as is_holiday/is_weekend above.
        df["is_payday_window"] = df["ds"].map(
            lambda d: (
                int(context.date_features[d].is_payday_window)
                if d in context.date_features
                else 0
            )
        )
        if context.weather_by_date:
            # `pd.to_numeric` forces a proper float64 column (NaN, not the
            # `None` produced by `.map`), so `.fillna()` below doesn't hit
            # pandas' "downcasting object dtype" deprecation path.
            df["temp_max_c"] = pd.to_numeric(
                df["ds"].map(
                    lambda d: _weather_field(context.weather_by_date, d, "temp_max_c")
                ),
                errors="coerce",
            )
            df["precip_mm"] = pd.to_numeric(
                df["ds"].map(
                    lambda d: _weather_field(context.weather_by_date, d, "precip_mm")
                ),
                errors="coerce",
            )
            temp_mean = df["temp_max_c"].mean()
            df["temp_max_c"] = df["temp_max_c"].fillna(
                temp_mean if not pd.isna(temp_mean) else 0.0
            )
            # No precipitation reading -> assume dry (documented, honest fallback,
            # not an invented value: 0mm is the neutral/no-signal case).
            df["precip_mm"] = df["precip_mm"].fillna(0.0)

    return df


def _build_prediction_row(
    target_date: date,
    values: list[float],
    season_length: int,
    context: ForecastContext | None,
    temp_fallback: float,
) -> dict[str, object]:
    """Build a single feature row for *target_date* from the running series.

    `values` is the true history plus any already-predicted future steps
    (recursive strategy) — lag/rolling features read off its tail exactly
    like `_build_training_frame` reads off `df["y"].shift(...)`.
    """
    row: dict[str, object] = {
        "lag_1": values[-1],
        f"lag_{season_length}": values[-season_length]
        if len(values) >= season_length
        else values[0],
        f"rolling_mean_{season_length}": float(np.mean(values[-season_length:]))
        if len(values) >= season_length
        else float(np.mean(values)),
        "day_of_week": target_date.weekday(),
        "month": target_date.month,
    }

    if context is not None:
        feat = context.date_features.get(target_date)
        row["is_holiday"] = int(feat.is_holiday) if feat is not None else 0
        row["is_weekend"] = (
            int(feat.is_weekend)
            if feat is not None
            else int(target_date.weekday() >= 5)
        )
        row["days_to_next_event"] = (
            feat.days_to_next_event
            if feat is not None and feat.days_to_next_event is not None
            else _NO_EVENT_SENTINEL_DAYS
        )
        row["is_payday_window"] = int(feat.is_payday_window) if feat is not None else 0
        if context.weather_by_date:
            temp = _weather_field(context.weather_by_date, target_date, "temp_max_c")
            precip = _weather_field(context.weather_by_date, target_date, "precip_mm")
            row["temp_max_c"] = temp if temp is not None else temp_fallback
            row["precip_mm"] = precip if precip is not None else 0.0

    return row
