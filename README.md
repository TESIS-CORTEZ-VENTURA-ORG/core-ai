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
> `statsforecast`. This is the baseline increment. A stronger model (TimesFM 2.5
> or Chronos-2) plugs in behind the same engine contract in a future epic
> increment without changing any callers — see _Pluggable engines_ below.

---

## Pluggable engines

The forecasting model is a **swappable strategy**. Every engine implements the
same `ForecastEngine` contract (`app/forecasting/engines/base.py`), so the model
can change without touching request/response shapes, the backtest, or callers.

| Engine key      | Model                                  | Status                                   |
| --------------- | -------------------------------------- | ---------------------------------------- |
| `statsforecast` | AutoETS (+ numpy fallback)             | ✅ implemented (default baseline)        |
| `seasonalnaive` | SeasonalNaive                          | ✅ implemented (thesis baseline)         |
| `ml`            | LightGBM (+ calendar/weather features) | ✅ implemented (HU-08-07)                |
| `timesfm`       | TimesFM 2.5                            | 🔌 wired — adapter pending (E08)         |
| `chronos`       | Chronos-2                              | 🔌 wired — adapter pending               |
| `auto`          | best available                         | ✅ default — degrades to `statsforecast` |

Select per request with the optional `engine` field, or set the service default
via `CORE_AI_FORECAST_ENGINE`. Requesting a wired-but-unimplemented engine returns
**501** with an install/implementation hint; an unknown key returns **400**.

`auto` only prefers `ml` when the request opts into exogenous context
(`use_context: true`, see below) **and** has enough history (>= 4 seasons —
28 daily / 208 weekly observations). Without `use_context`, `auto` behaves
exactly as it did before HU-08-07 (degrades to `statsforecast`).

Adding the real model is two steps and changes nothing else:

1. Install the model dependency (e.g. `uv add timesfm` / `uv add chronos-forecasting`).
2. Fill in `forecast()` in `engines/timesfm_engine.py` (or `chronos_engine.py`),
   mapping the model output to `ForecastPoint(target_date, yhat=q50, yhat_lo=q10,
yhat_hi=q90)`. Then add the key to `_AUTO_ORDER` / flip `auto_selectable`.

---

## Exogenous context — Peruvian calendar + weather (HU-08-07)

Opt-in via `use_context: true` on `POST /forecast/run`. **Additive and fully
backward-compatible**: omitting `use_context`/`location` reproduces the exact
response this API returned before this increment (`drivers: []`,
`context_status: "off"`, zero network calls).

**What it adds:**

- **Calendar** (`app/forecasting/features/calendar.py`, always available, no
  network): official Peru holidays (`holidays.PE`) + a curated gastronomic
  calendar (Día del Ceviche 28-jun, Día del Pisco Sour 1st Sat of Feb, San
  Valentín 14-feb, Día de la Madre 2nd Sun of May, Día del Padre 3rd Sun of
  Jun, Fiestas Patrias 28/29-jul, Halloween/Día de la Canción Criolla 31-oct,
  Nochebuena/Navidad, Nochevieja/Año Nuevo).
- **Weather** (`app/forecasting/features/weather.py`, Open-Meteo, no API key):
  daily max temperature + precipitation, historical (archive API) for
  training and forecast (up to 16 days) for the horizon. Short timeout (4s),
  in-memory TTL cache, **mandatory graceful degradation** — if Open-Meteo is
  unreachable the forecast still succeeds with calendar-only context
  (`context_status: "calendar_only"`), never a 5xx.
- **`ml` engine** (LightGBM, retrained on every request from the full
  submitted `history` — see `ml_engine.py` docstring for the mlforecast-vs-
  plain-LightGBM decision): lag/rolling features + day-of-week/month +
  calendar + weather, recursive multi-horizon forecasting, empirical-residual
  prediction bands.
- **`drivers`**: the context factors that fall inside the forecast horizon,
  narratable by the UI (`"Fiestas Patrias en 12 días: +35% demanda
proyectada"`). `impact_pct` is the average historical uplift of that event
  vs. equivalent non-event days **computed only when the submitted history
  actually contains a prior occurrence of it** — omitted (`null`), never
  guessed, otherwise.
- **`backtest.model_smape_no_context`**: when `use_context=true` and the
  resolved engine is `ml`, the same holdout re-run WITHOUT context features —
  the univariate-vs-exogenous comparison.

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

| Field           | Type                            | Description                                                                          |
| --------------- | ------------------------------- | ------------------------------------------------------------------------------------ |
| `series_id`     | string                          | Identifier echoed back in the response                                               |
| `frequency`     | "D" or "W"                      | Daily or weekly series                                                               |
| `horizon`       | int > 0                         | Number of future periods to forecast (max 365)                                       |
| `history`       | HistoryPoint[]                  | {"ds": "YYYY-MM-DD", "y": float}, minimum 2 points                                   |
| `season_length` | int or null                     | Override seasonal period (default: 7 for D, 52 for W)                                |
| `engine`        | string or null                  | Engine key (see _Pluggable engines_); null -> default                                |
| `use_context`   | bool (default `false`)          | HU-08-07: opt into calendar + weather context and the `ml` engine's auto-eligibility |
| `location`      | `{latitude, longitude}` or null | Weather coordinates when `use_context=true`; defaults to Lima (-12.046, -77.043)     |

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
    "improvement_pct": 28.22,
    "model_smape_no_context": null
  },
  "drivers": [],
  "context_status": "off"
}
```

`backtest` is `null` when `len(history) < 2 * horizon`. `drivers`/
`context_status` are always present (`[]`/`"off"` by default) so the shape
never changes based on `use_context` — only the _values_ do.

#### Request/response with `use_context: true` (HU-08-07)

```jsonc
// POST /forecast/run
{
  "series_id": "motif-restobar-covers",
  "frequency": "D",
  "horizon": 14,
  "history": [
    /* ...daily covers, 2+ years recommended for the "ml" engine... */
  ],
  "engine": "ml",
  "use_context": true,
  "location": { "latitude": -12.046, "longitude": -77.043 },
}
```

```jsonc
// 200 OK
{
  "series_id": "motif-restobar-covers",
  "engine": "ml",
  "model": "LightGBM",
  "baseline": "SeasonalNaive",
  "frequency": "D",
  "points": [
    {
      "target_date": "2025-08-05",
      "yhat": 65.4,
      "yhat_lo": 65.26,
      "yhat_hi": 65.53,
    },
    // ...14 points
  ],
  "backtest": {
    "holdout_size": 14,
    "model_smape": 6.8487,
    "baseline_smape": 8.1633,
    "improvement_pct": 16.104,
    "model_smape_no_context": 11.0394, // same holdout, engine="ml" WITHOUT context -> proves context helps
  },
  "drivers": [
    {
      "date": "2025-08-06",
      "kind": "holiday",
      "label": "Batalla de Junín",
      "impact_pct": -5.85,
    },
    {
      "date": "2025-08-09",
      "kind": "weekend",
      "label": "Fin de semana",
      "impact_pct": -13.7,
    },
    {
      "date": "2025-08-09",
      "kind": "weather",
      "label": "Lluvia esperada (8.5 mm)",
      "impact_pct": null,
    },
    // Driver.kind: "holiday" | "gastro_event" | "weather" | "weekend"
    // impact_pct is null when `history` has no prior occurrence of that event (never guessed).
  ],
  "context_status": "full", // "full" | "calendar_only" (Open-Meteo degraded) | "off" (use_context=false)
}
```

Captured live against a running `gastronomia-core-ai` container (Docker,
`POST http://localhost:8000/forecast/run`) with a 3-year synthetic daily
series carrying a real Jul 28-29 (Fiestas Patrias) spike — see
`tests/test_ml_engine.py::TestBacktestBeatsNaiveWithContext` for the
pytest-level version of the same scenario.

---

## Running tests

```bash
uv run pytest -q
```

---

## Environment variables

All variables use the `CORE_AI_` prefix.

| Variable                       | Default | Description                        |
| ------------------------------ | ------- | ---------------------------------- |
| `CORE_AI_APP_NAME`             | core-ai | Service name shown in OpenAPI docs |
| `CORE_AI_DEFAULT_LEVELS`       | [80]    | Prediction interval levels         |
| `CORE_AI_FORECAST_MAX_HORIZON` | 365     | Maximum allowed horizon            |
| `CORE_AI_FORECAST_ENGINE`      | auto    | Default engine when none requested |
