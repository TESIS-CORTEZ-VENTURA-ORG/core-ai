"""Pluggable LLM adapters for the Text-to-SQL feature.

Each adapter implements the same :class:`LLMAdapter` contract so the active
model can be swapped (openai, anthropic, xai, mock) via env vars without
changing the orchestration or validation logic. This mirrors the
ForecastEngine pattern used in app/forecasting/engines/.
"""

from app.chat.adapters.base import AdapterNotAvailableError, LLMAdapter

__all__ = ["AdapterNotAvailableError", "LLMAdapter"]
