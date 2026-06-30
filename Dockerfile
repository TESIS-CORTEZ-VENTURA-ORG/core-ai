# core-ai — microservicio de forecasting (FastAPI). Imagen basada en uv (Python 3.12,
# fijado por statsforecast/numba). Build en dos capas: dependencias (cacheable) y código.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Capa de dependencias (cacheable): se resuelve antes de copiar el código fuente.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# Código de la app + sync final (instala el propio proyecto).
COPY . .
RUN uv sync --frozen --no-dev

# Correr como usuario no-root (A3).
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Liveness: el orquestador espera a que /health responda 200 antes de enrutar.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:${PORT:-8000}/health').getcode()==200 else 1)"

CMD uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
