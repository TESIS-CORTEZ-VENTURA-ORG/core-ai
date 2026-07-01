"""Anthropic (Claude) LLM adapter for Text-to-SQL.

Requires ANTHROPIC_API_KEY in the environment. Uses the anthropic Python SDK.
The system prompt is sent as the Anthropic `system` parameter (not as a
message), which is the recommended approach for instruction-following tasks.

Model is configurable via CORE_AI_CHAT_MODEL; defaults to claude-haiku-4-5
(fast, inexpensive, strong instruction-following for structured output like SQL).
"""

from __future__ import annotations

import os

from app.chat.adapters.base import AdapterNotAvailableError, LLMAdapter
from app.chat.prompt import build_answer_prompt, build_nl2sql_prompt
from app.chat.schemas import Nl2SqlRequest, Nl2SqlResponse

DEFAULT_MODEL = "claude-haiku-4-5"


class AnthropicAdapter(LLMAdapter):
    """Claude adapter — requires ANTHROPIC_API_KEY."""

    key = "anthropic"

    @classmethod
    def is_available(cls) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def model_name(self) -> str:
        return os.environ.get("CORE_AI_CHAT_MODEL") or DEFAULT_MODEL

    def nl2sql(self, request: Nl2SqlRequest) -> Nl2SqlResponse:
        client = self._client()
        system_prompt, user_prompt = build_nl2sql_prompt(request)
        message = client.messages.create(
            model=self.model_name(),
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        sql = message.content[0].text.strip() if message.content else ""
        return Nl2SqlResponse(sql=sql, provider=self.key, model=self.model_name())

    def answer(self, question: str, columns: list[str], rows: list[list]) -> str:
        client = self._client()
        prompt = build_answer_prompt(question, columns, rows)
        message = client.messages.create(
            model=self.model_name(),
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip() if message.content else ""

    def _client(self):  # type: ignore[no-untyped-def]
        try:
            import anthropic
        except ImportError as exc:
            raise AdapterNotAvailableError(
                "anthropic SDK is not installed. Run: uv add anthropic"
            ) from exc
        return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
