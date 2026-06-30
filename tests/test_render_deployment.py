"""Structural deployment checks for Render-safe core-ai artifacts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_artifact(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_env_example_documents_core_ai_settings_with_safe_defaults() -> None:
    env_example = read_artifact(".env.example")

    expected_lines = {
        "CORE_AI_APP_NAME=core-ai",
        "CORE_AI_DEFAULT_LEVELS=80",
        "CORE_AI_FORECAST_MAX_HORIZON=365",
        "CORE_AI_FORECAST_ENGINE=auto",
    }

    for line in expected_lines:
        assert line in env_example


def test_env_example_contains_only_core_ai_names_and_no_secret_placeholders() -> None:
    env_example = read_artifact(".env.example")
    assignment_names = [
        line.split("=", 1)[0]
        for line in env_example.splitlines()
        if line and not line.startswith("#") and "=" in line
    ]

    assert assignment_names == [
        "CORE_AI_APP_NAME",
        "CORE_AI_DEFAULT_LEVELS",
        "CORE_AI_FORECAST_MAX_HORIZON",
        "CORE_AI_FORECAST_ENGINE",
        "PORT",
    ]
    assert "SECRET" not in env_example.upper()
    assert "TOKEN" not in env_example.upper()
    assert "PASSWORD" not in env_example.upper()


def test_dockerfile_uses_render_port_for_healthcheck_and_uvicorn() -> None:
    dockerfile = read_artifact("Dockerfile")

    assert "${PORT:-8000}" in dockerfile
    assert "localhost:${PORT:-8000}/health" in dockerfile
    assert '--port ${PORT:-8000}' in dockerfile
    assert '"--port", "8000"' not in dockerfile
    assert "localhost:8000/health" not in dockerfile
