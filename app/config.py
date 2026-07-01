"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CORE_AI_", case_sensitive=False)

    app_name: str = "core-ai"
    default_levels: list[int] = [80]
    forecast_max_horizon: int = 365
    # Default forecasting engine when the request does not specify one.
    # "auto" picks the best available model, degrading to the baseline.
    forecast_engine: str = "auto"

    # E09 — Chat / Text-to-SQL configuration.
    # Provider: "openai" | "anthropic" | "xai" | "mock".
    # Empty string triggers auto-select (first key present → mock).
    chat_provider: str = ""
    # Model override for the active chat provider (e.g. "gpt-4o", "claude-sonnet-4-6").
    # Empty string means each adapter uses its own default.
    chat_model: str = ""


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
