"""Abstract base contract for LLM adapters used in the Text-to-SQL feature.

An LLM adapter is a swappable strategy that translates a natural-language
question + curated schema context into a single read-only PostgreSQL SELECT.
The seam is identical in spirit to ForecastEngine: the active model can
change without touching orchestration, validation, or response shaping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from app.chat.schemas import Nl2SqlRequest, Nl2SqlResponse


class AdapterNotAvailableError(RuntimeError):
    """Raised when an adapter's API key or SDK dependency is missing.

    The service maps this to HTTP 503 so the caller gets a clear, actionable
    error instead of an opaque import traceback or auth failure.
    """


class LLMAdapter(ABC):
    """Strategy contract every LLM adapter must satisfy."""

    #: Stable identifier used to select the adapter (matches CORE_AI_CHAT_PROVIDER).
    key: ClassVar[str]

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True when the adapter's API key / SDK is present."""

    @abstractmethod
    def model_name(self) -> str:
        """Human-facing model identifier surfaced in the response."""

    @abstractmethod
    def nl2sql(self, request: Nl2SqlRequest) -> Nl2SqlResponse:
        """Translate *request.question* into a single read-only SELECT.

        The service validates the output before returning it. The NestJS
        backend applies a stricter 9-rule gate before executing anything.
        """

    @abstractmethod
    def answer(self, question: str, columns: list[str], rows: list[list]) -> str:
        """Return a short Spanish summary of the query result.

        This is the second LLM call — optional, non-fatal. If the adapter
        cannot produce an answer the service catches the error and returns
        a generic fallback to the NestJS backend.
        """
