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
> `statsforecast`.  This is the baseline increment.  Chronos-2 will replace the
> model internals behind the same `engine.run_forecast` signature in a future
> epic increment without changing any callers.

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
  "season_length": null
}
```

| Field           | Type             | Description                                            |
|-----------------|------------------|--------------------------------------------------------|
| `series_id`     | string           | Identifier echoed back in the response                 |
| `frequency`     | "D" or "W"       | Daily or weekly series                                 |
| `horizon`       | int > 0          | Number of future periods to forecast (max 365)         |
| `history`       | HistoryPoint[]   | {"ds": "YYYY-MM-DD", "y": float}, minimum 2 points    |
| `season_length` | int or null      | Override seasonal period (default: 7 for D, 52 for W) |

#### Response body

```json
{
  "series_id": "restaurant-01-covers",
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
