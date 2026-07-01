"""Abstract base contract for LLM adapters used in Text-to-SQL and document extraction.

An LLM adapter is a swappable strategy that:
  - nl2sql:  translates a natural-language question into a read-only SELECT (E09).
  - answer:  humanises a query result in Spanish (E09, optional).
  - extract: extracts structured menu/ingredient data from document text (E11 Smart
             Onboarding). Reuses the same adapter registry — no new provider system.

The seam is identical in spirit to ForecastEngine: the active model can change
without touching orchestration, validation, or response shaping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, TypedDict

from app.chat.schemas import Nl2SqlRequest, Nl2SqlResponse


class AdapterNotAvailableError(RuntimeError):
    """Raised when an adapter's API key or SDK dependency is missing.

    The service maps this to HTTP 503 so the caller gets a clear, actionable
    error instead of an opaque import traceback or auth failure.
    """


# ---------------------------------------------------------------------------
# Shared types for document extraction (E11 Smart Onboarding)
# ---------------------------------------------------------------------------


class ExtractedMenuItem(TypedDict, total=False):
    """A single menu item extracted from a restaurant document."""

    name: str  # required
    price: float  # required — always positive PEN amount
    category: str  # optional
    description: str  # optional


class ExtractedIngredient(TypedDict, total=False):
    """A single ingredient extracted from a restaurant document."""

    name: str  # required
    unit: str  # required (kg, g, litro, unidad, …)
    estimatedCost: float  # optional — PEN per unit


class ExtractResult(TypedDict):
    """Structured data extracted from a restaurant document.

    Both lists may be empty — the adapter is expected to be conservative:
    only clearly-present items are returned; nothing is invented.
    """

    menuItems: list[ExtractedMenuItem]
    ingredients: list[ExtractedIngredient]


# ---------------------------------------------------------------------------
# Adapter contract
# ---------------------------------------------------------------------------


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

    @abstractmethod
    def extract(self, text: str, target: str, currency: str = "PEN") -> ExtractResult:
        """Extract menu items and/or ingredients from restaurant document text.

        Args:
            text:     Raw text content of the document (PDF, xlsx, csv converted
                      to plaintext by the NestJS backend).
            target:   'menu' → only menuItems; 'ingredients' → only ingredients;
                      'auto' → both.
            currency: ISO currency code; always 'PEN' for the Peruvian market.

        Returns:
            ExtractResult with menuItems and ingredients lists. Adapters MUST
            be conservative: only clearly-present items are returned; prices
            are never invented; lists may be empty on uncertain input.

        Raises:
            AdapterNotAvailableError: if the adapter's SDK or key is missing.
        """
