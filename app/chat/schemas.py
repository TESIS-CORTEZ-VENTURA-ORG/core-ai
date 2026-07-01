"""Pydantic v2 schemas for the chat / Text-to-SQL API.

These are the Pydantic mirror of the Zod schemas in:
  team-backend/src/shared/chat/chat.ts

Both sides must stay in sync (same field names, same constraints).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Nl2SqlRequest(BaseModel):
    """Body for POST /chat/nl2sql — translate a question into a SELECT."""

    question: str = Field(min_length=1, max_length=2000)
    # Curated table/column descriptions passed by the NestJS backend.
    # Never raw DB metadata — the backend controls what the LLM can "see".
    schema_context: str = Field(min_length=1, max_length=50_000)
    dialect: Literal["postgresql"] = "postgresql"
    max_rows: int = Field(default=200, ge=1, le=500)


class Nl2SqlResponse(BaseModel):
    """core-ai response for POST /chat/nl2sql."""

    sql: str
    provider: str
    model: str
    notes: str = ""


class AnswerRequest(BaseModel):
    """Body for POST /chat/answer — humanize a query result."""

    question: str = Field(min_length=1, max_length=2000)
    columns: list[str]
    # rows is a 2D list of primitive-serializable values fetched by the backend.
    rows: list[list]
    # Pass the same provider used for nl2sql so the same adapter answers.
    provider: str | None = None


class AnswerResponse(BaseModel):
    """core-ai response for POST /chat/answer."""

    answer: str
    provider: str
