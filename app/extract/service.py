"""Document extraction service — orchestrates between the router and LLM adapters.

Responsibilities:
1. Resolve the active LLM adapter via the shared chat registry (reuse, no new system).
2. Call adapter.extract() to get raw ExtractResult from the document text.
3. Validate and coerce the result into the typed ExtractResponse.
4. Any invalid/missing items are silently dropped (conservative: malformed input
   yields empty arrays, never a crash).

The chat registry auto-selects the best available provider (Anthropic > OpenAI >
xAI > mock). The mock adapter always succeeds with canned data — CI never fails.
"""

from __future__ import annotations

import json
import logging

from fastapi import HTTPException

from app.chat.adapters.base import AdapterNotAvailableError
from app.chat.registry import UnknownAdapterError, resolve_adapter
from app.extract.schemas import (
    ExtractedIngredient,
    ExtractedMenuItem,
    ExtractRequest,
    ExtractResponse,
)

logger = logging.getLogger(__name__)


def extract_document(request: ExtractRequest) -> ExtractResponse:
    """Extract structured restaurant data from plain-text document content.

    Uses the shared LLM adapter registry (same adapters as E09 chat).
    The adapter's extract() method calls the LLM with a strict JSON-only prompt.
    The raw LLM output is validated item-by-item; invalid items are dropped
    instead of raising, making the endpoint resilient to partial LLM failures.
    """
    try:
        adapter = resolve_adapter()
    except UnknownAdapterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        raw = adapter.extract(request.text, request.target, request.currency)
    except AdapterNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # Catch unexpected LLM errors (network, timeout, JSON parse in adapter).
        # Log for observability; return empty arrays instead of 500 so the backend
        # can display a "no items extracted" preview rather than crashing onboarding.
        logger.warning("extract() adapter error: %s", exc)
        return ExtractResponse(
            menuItems=[],
            ingredients=[],
            provider=adapter.key,
            model=adapter.model_name(),
        )

    menu_items = _coerce_menu_items(raw.get("menuItems", []))
    ingredients = _coerce_ingredients(raw.get("ingredients", []))

    return ExtractResponse(
        menuItems=menu_items,
        ingredients=ingredients,
        provider=adapter.key,
        model=adapter.model_name(),
    )


def _coerce_menu_items(raw: object) -> list[ExtractedMenuItem]:
    """Validate and filter raw menu item data from the adapter.

    Each item is validated independently; invalid items are dropped with a
    warning instead of failing the entire request (conservative approach).
    """
    if not isinstance(raw, list):
        return []
    result: list[ExtractedMenuItem] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        price_raw = item.get("price")
        if not isinstance(name, str) or not name.strip():
            logger.debug("extract: dropped menuItem[%d] — missing name", i)
            continue
        try:
            price = float(price_raw) if price_raw is not None else 0.0
        except (TypeError, ValueError):
            logger.debug(
                "extract: dropped menuItem[%d] — invalid price: %r", i, price_raw
            )
            continue
        if price < 0:
            logger.debug(
                "extract: dropped menuItem[%d] — negative price %.2f", i, price
            )
            continue
        category = item.get("category")
        description = item.get("description")
        result.append(
            ExtractedMenuItem(
                name=name.strip(),
                price=price,
                category=str(category).strip() if category else None,
                description=str(description).strip() if description else None,
            )
        )
    return result


def _coerce_ingredients(raw: object) -> list[ExtractedIngredient]:
    """Validate and filter raw ingredient data from the adapter."""
    if not isinstance(raw, list):
        return []
    result: list[ExtractedIngredient] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        unit = item.get("unit", "")
        if not isinstance(name, str) or not name.strip():
            logger.debug("extract: dropped ingredient[%d] — missing name", i)
            continue
        if not isinstance(unit, str) or not unit.strip():
            logger.debug("extract: dropped ingredient[%d] — missing unit", i)
            continue
        cost_raw = item.get("estimatedCost")
        estimated_cost: float | None = None
        if cost_raw is not None:
            try:
                estimated_cost = float(cost_raw)
                if estimated_cost < 0:
                    estimated_cost = None  # drop invalid cost but keep the ingredient
            except (TypeError, ValueError):
                estimated_cost = None
        result.append(
            ExtractedIngredient(
                name=name.strip(),
                unit=unit.strip(),
                estimatedCost=estimated_cost,
            )
        )
    return result


def _safe_parse_json(text: str) -> object:
    """Try to parse JSON; return None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
