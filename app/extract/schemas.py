"""Pydantic v2 schemas for POST /extract/document.

These are the Pydantic mirror of the Zod schemas in:
  team-backend/src/shared/ingestion/import-document.ts

Both sides must stay in sync (same field names, same constraints).
All monetary values are PEN (Peruvian sol).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    """Body for POST /extract/document.

    The NestJS backend converts the uploaded file to plaintext and sends it
    here. core-ai NEVER touches files or the business database — it only
    runs LLM inference and returns structured data.
    """

    text: str = Field(
        min_length=1,
        max_length=200_000,
        description="Raw text extracted from the uploaded document.",
    )
    target: Literal["menu", "ingredients", "auto"] = Field(
        default="auto",
        description=(
            "'menu' → extract menu items only; "
            "'ingredients' → extract ingredients only; "
            "'auto' → extract both."
        ),
    )
    currency: str = Field(
        default="PEN",
        description="ISO currency code for prices (always PEN for the Peruvian market).",
    )


class ExtractedMenuItem(BaseModel):
    """A single menu item extracted from the document."""

    name: str = Field(min_length=1, description="Dish name as it appears in the menu.")
    price: float = Field(ge=0, description="Sell price in PEN.")
    category: str | None = Field(
        default=None, description="Menu category (e.g. 'Platos de fondo')."
    )
    description: str | None = Field(
        default=None, description="Short dish description if present."
    )


class ExtractedIngredient(BaseModel):
    """A single ingredient extracted from the document."""

    name: str = Field(min_length=1, description="Ingredient name.")
    unit: str = Field(
        min_length=1, description="Unit of measure (kg, g, litro, unidad…)."
    )
    estimatedCost: float | None = Field(  # noqa: N815 — camelCase mirrors the TS contract
        default=None,
        ge=0,
        description="Estimated cost per unit in PEN, if present in the document.",
    )


class ExtractResponse(BaseModel):
    """core-ai response for POST /extract/document.

    The NestJS backend presents this to the user as the preview payload.
    Nothing is written to the database until the user confirms via
    POST /import/document/commit.
    """

    menuItems: list[ExtractedMenuItem] = Field(  # noqa: N815
        default_factory=list,
        description="Extracted menu items. Empty if none found or target='ingredients'.",
    )
    ingredients: list[ExtractedIngredient] = Field(
        default_factory=list,
        description="Extracted ingredients. Empty if none found or target='menu'.",
    )
    provider: str = Field(
        description="LLM provider key (openai | anthropic | xai | mock)."
    )
    model: str = Field(description="Model identifier reported by the provider.")
