# core-ai — GastronomIA Demand Forecasting Microservice

Stateless FastAPI service that receives a historical time series in the POST body
and returns demand forecasts with prediction intervals.

## Stack

- **Python 3.12**
- **FastAPI** + **Uvicorn**
- **statsforecast** (AutoETS + SeasonalNaive) — current baseline engine
- **Pydantic v2** for validation
- **uv** for dependency management

> **Note — current engine**: The primary forecasting model is `AutoETS` from
> `statsforecast`.  This is the baseline increment.  A stronger model (TimesFM 2.5
> or Chronos-2) plugs in behind the same engine contract in a future epic
> increment without changing any callers — see *Pluggable engines* below.

---

## Pluggable engines

The forecasting model is a **swappable strategy**. Every engine implements the
same `ForecastEngine` contract (`app/forecasting/engines/base.py`), so the model
can change without touching request/response shapes, the backtest, or callers.

| Engine key      | Model              | Status                                  |
|-----------------|--------------------|-----------------------------------------|
| `statsforecast` | AutoETS (+ numpy fallback) | ✅ implemented (default baseline) |
| `seasonalnaive` | SeasonalNaive      | ✅ implemented (thesis baseline)        |
| `timesfm`       | TimesFM 2.5        | 🔌 wired — adapter pending (E08)        |
| `chronos`       | Chronos-2          | 🔌 wired — adapter pending              |
| `auto`          | best available     | ✅ default — degrades to `statsforecast` |

Select per request with the optional `engine` field, or set the service default
via `CORE_AI_FORECAST_ENGINE`. Requesting a wired-but-unimplemented engine returns
**501** with an install/implementation hint; an unknown key returns **400**.

Adding the real model is two steps and changes nothing else:

1. Install the model dependency (e.g. `uv add timesfm` / `uv add chronos-forecasting`).
2. Fill in `forecast()` in `engines/timesfm_engine.py` (or `chronos_engine.py`),
   mapping the model output to `ForecastPoint(target_date, yhat=q50, yhat_lo=q10,
   yhat_hi=q90)`. Then add the key to `_AUTO_ORDER` / flip `auto_selectable`.

---

## Setup and run

```bash
uv sync --extra dev
uv run uvicorn app.main:app --reload --port 8000
```

---

## Endpoints

### `GET /health`

Liveness probe.

```json
{ "status": "ok" }
```

---

### `POST /forecast/run`

Run a demand forecast.

#### Request body

```json
{
  "series_id": "restaurant-01-covers",
  "frequency": "D",
  "horizon": 14,
  "history": [
    { "ds": "2024-01-01", "y": 120.0 },
    { "ds": "2024-01-02", "y": 135.0 }
  ],
  "season_length": null,
  "engine": "auto"
}
```

| Field           | Type             | Description                                            |
|-----------------|------------------|--------------------------------------------------------|
| `series_id`     | string           | Identifier echoed back in the response                 |
| `frequency`     | "D" or "W"       | Daily or weekly series                                 |
| `horizon`       | int > 0          | Number of future periods to forecast (max 365)         |
| `history`       | HistoryPoint[]   | {"ds": "YYYY-MM-DD", "y": float}, minimum 2 points    |
| `season_length` | int or null      | Override seasonal period (default: 7 for D, 52 for W) |
| `engine`        | string or null   | Engine key (see *Pluggable engines*); null -> default  |

#### Response body

```json
{
  "series_id": "restaurant-01-covers",
  "engine": "statsforecast",
  "model": "AutoETS",
  "baseline": "SeasonalNaive",
  "frequency": "D",
  "points": [
    {
      "target_date": "2024-05-01",
      "yhat": 142.3,
      "yhat_lo": 128.1,
      "yhat_hi": 156.5
    }
  ],
  "backtest": {
    "holdout_size": 14,
    "model_smape": 8.42,
    "baseline_smape": 11.73,
    "improvement_pct": 28.22
  }
}
```

`backtest` is `null` when `len(history) < 2 * horizon`.

---

## Running tests

```bash
uv run pytest -q
```

---

## Environment variables

All variables use the `CORE_AI_` prefix.

| Variable                       | Default | Description                        |
|--------------------------------|---------|------------------------------------|
| `CORE_AI_APP_NAME`             | core-ai | Service name shown in OpenAPI docs |
| `CORE_AI_DEFAULT_LEVELS`       | [80]    | Prediction interval levels         |
| `CORE_AI_FORECAST_MAX_HORIZON` | 365     | Maximum allowed horizon            |
| `CORE_AI_FORECAST_ENGINE`      | auto    | Default engine when none requested |
