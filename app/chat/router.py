"""Chat / Text-to-SQL API router.

Two endpoints:
  POST /chat/nl2sql  — natural language → read-only SQL (primary call)
  POST /chat/answer  — query result rows → human-friendly Spanish sentence (optional)

The NestJS backend is the ONLY caller. Endpoint access is not guarded here
because core-ai runs in an internal network — the NestJS service is the
perimeter. The full security gate (9-rule SQL validation + RLS FORCE
execution) lives in the NestJS ChatService.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.chat import service
from app.chat.schemas import (
    AnswerRequest,
    AnswerResponse,
    Nl2SqlRequest,
    Nl2SqlResponse,
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/nl2sql", response_model=Nl2SqlResponse)
def nl2sql_endpoint(request: Nl2SqlRequest) -> Nl2SqlResponse:
    """Translate a natural-language question into a single read-only SELECT.

    The active LLM adapter is resolved at request time from the environment
    (CORE_AI_CHAT_PROVIDER → first key present → mock). The response SQL
    has already passed a first-pass sanity check; the NestJS backend applies
    the authoritative 9-rule validation gate before executing anything.
    """
    return service.nl2sql(request)


@router.post("/answer", response_model=AnswerResponse)
def answer_endpoint(request: AnswerRequest) -> AnswerResponse:
    """Generate a short Spanish sentence summarising a query result.

    Called by the NestJS backend after fetching the rows. This is optional:
    the backend handles a 5xx gracefully and returns a generic message.
    """
    return service.answer(request)
