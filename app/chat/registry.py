"""LLM adapter registry and auto-selection.

Mirrors the ForecastEngine registry pattern. Resolution order:
1. *requested* key if provided explicitly (e.g. from a request field).
2. CORE_AI_CHAT_PROVIDER environment variable.
3. Auto: first adapter in _AUTO_ORDER whose API key is present.
4. Final fallback: mock (always available, no key needed).

The mock fallback guarantees the service NEVER fails for lack of a provider;
it just degrades gracefully to deterministic canned queries.
"""

from __future__ import annotations

import os

from app.chat.adapters.anthropic_adapter import AnthropicAdapter
from app.chat.adapters.base import LLMAdapter
from app.chat.adapters.mock_adapter import MockAdapter
from app.chat.adapters.openai_adapter import OpenAIAdapter
from app.chat.adapters.xai_adapter import XAIAdapter

#: All registered adapters, keyed by their stable identifier.
_ADAPTERS: dict[str, type[LLMAdapter]] = {
    OpenAIAdapter.key: OpenAIAdapter,
    AnthropicAdapter.key: AnthropicAdapter,
    XAIAdapter.key: XAIAdapter,
    MockAdapter.key: MockAdapter,
}

#: Auto-selection preference order (best/most capable first).
_AUTO_ORDER: list[str] = [
    AnthropicAdapter.key,  # primary per ADR-003 (Claude Sonnet 4.6 / Haiku 4.5)
    OpenAIAdapter.key,
    XAIAdapter.key,
    MockAdapter.key,  # always last — the guaranteed fallback
]


class UnknownAdapterError(ValueError):
    """Raised when the requested adapter key is not registered."""


def resolve_adapter(requested: str | None = None) -> LLMAdapter:
    """Resolve a concrete adapter instance.

    Args:
        requested: Explicit adapter key (e.g. from a request field).
                   If None, falls back to env var, then auto-select.
    """
    key = (requested or os.environ.get("CORE_AI_CHAT_PROVIDER", "")).strip().lower()

    if not key:
        # Auto-select: first available adapter in preference order.
        for candidate in _AUTO_ORDER:
            cls = _ADAPTERS[candidate]
            if cls.is_available():
                return cls()
        return MockAdapter()  # unreachable in practice (mock is always available)

    cls = _ADAPTERS.get(key)
    if cls is None:
        valid = ", ".join(_ADAPTERS.keys())
        raise UnknownAdapterError(
            f"Unknown chat provider '{key}'. Valid options: {valid}."
        )
    return cls()


def list_adapter_keys() -> list[str]:
    """Return all registered adapter keys."""
    return list(_ADAPTERS.keys())


def available_adapters() -> dict[str, bool]:
    """Map each adapter key to whether its API key / SDK is present."""
    return {key: cls.is_available() for key, cls in _ADAPTERS.items()}
