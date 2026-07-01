"""Chat orchestration service — sits between the router and the LLM adapters.

Responsibilities:
1. Resolve the active LLM adapter via the registry.
2. Call adapter.nl2sql() to get a SQL string from the question.
3. Apply a first-pass validation (SELECT/WITH check, no blocked keywords).
4. Return the Nl2SqlResponse to the router (NestJS applies the full 9-rule gate).

The service-level check is a quick sanity filter. The NestJS backend's
sql-validator.util.ts is the authoritative hard gate before DB execution.
"""

from __future__ import annotations

import re

from fastapi import HTTPException

from app.chat.adapters.base import AdapterNotAvailableError
from app.chat.registry import UnknownAdapterError, resolve_adapter
from app.chat.schemas import (
    AnswerRequest,
    AnswerResponse,
    Nl2SqlRequest,
    Nl2SqlResponse,
)

# Must start with SELECT or WITH (optional leading whitespace).
_SELECT_RE = re.compile(r"^\s*(WITH|SELECT)\b", re.IGNORECASE)

# Blocked DDL/DML keywords as whole words. Applied after comment stripping.
_BLOCKED_KEYWORDS_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|COPY"
    r"|CALL|DO|VACUUM|EXECUTE|PREPARE|DEALLOCATE|FETCH|MOVE|DECLARE"
    r"|DISCARD|RESET|UNLISTEN|LISTEN|NOTIFY|LOAD|MERGE)\b",
    re.IGNORECASE,
)


def nl2sql(request: Nl2SqlRequest) -> Nl2SqlResponse:
    """Generate a read-only SQL query from a natural-language question.

    The returned SQL is always a single SELECT / WITH statement. A service-
    level sanity check prevents obviously malicious responses from reaching the
    NestJS backend. The backend applies the complete 9-rule validation gate
    before DB execution.
    """
    try:
        adapter = resolve_adapter()
    except UnknownAdapterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = adapter.nl2sql(request)
    except AdapterNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Strip trailing semicolons (the NestJS gate will also do this, but be
    # defensive here too so we don't forward garbage to the backend).
    sql = result.sql.strip().rstrip(";").strip()

    if not _SELECT_RE.match(sql):
        raise HTTPException(
            status_code=422,
            detail=(
                f"The LLM returned a non-SELECT statement. "
                f"Refusing to forward it. Provider: {result.provider}."
            ),
        )

    if _BLOCKED_KEYWORDS_RE.search(sql):
        raise HTTPException(
            status_code=422,
            detail=(
                "The LLM output contains blocked DDL/DML keywords. "
                f"Provider: {result.provider}."
            ),
        )

    result.sql = sql
    return result


def answer(request: AnswerRequest) -> AnswerResponse:
    """Generate a Spanish natural-language answer from a query result.

    The NestJS backend fetches the rows, then calls this endpoint to produce
    a human-friendly sentence. Non-fatal: if this endpoint returns an error
    the backend falls back to a generic 'N registros encontrados' message.
    """
    try:
        adapter = resolve_adapter(request.provider)
    except UnknownAdapterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        text = adapter.answer(request.question, request.columns, request.rows)
    except AdapterNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AnswerResponse(answer=text, provider=adapter.key)
