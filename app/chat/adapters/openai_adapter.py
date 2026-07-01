"""OpenAI (GPT) LLM adapter for Text-to-SQL.

Requires OPENAI_API_KEY in the environment. Uses the openai Python SDK
(version >= 1.0) with the standard chat completions API.

Model is configurable via CORE_AI_CHAT_MODEL; defaults to gpt-4o-mini
(cost-efficient, adequate reasoning for SQL generation on constrained schemas).
"""

from __future__ import annotations

import os

from app.chat.adapters.base import AdapterNotAvailableError, LLMAdapter
from app.chat.prompt import build_answer_prompt, build_nl2sql_prompt
from app.chat.schemas import Nl2SqlRequest, Nl2SqlResponse

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIAdapter(LLMAdapter):
    """GPT adapter — requires OPENAI_API_KEY."""

    key = "openai"

    @classmethod
    def is_available(cls) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def model_name(self) -> str:
        return os.environ.get("CORE_AI_CHAT_MODEL", DEFAULT_MODEL)

    def nl2sql(self, request: Nl2SqlRequest) -> Nl2SqlResponse:
        client = self._client()
        system_prompt, user_prompt = build_nl2sql_prompt(request)
        completion = client.chat.completions.create(
            model=self.model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0,  # deterministic — SQL generation must be precise
        )
        sql = (completion.choices[0].message.content or "").strip()
        return Nl2SqlResponse(sql=sql, provider=self.key, model=self.model_name())

    def answer(self, question: str, columns: list[str], rows: list[list]) -> str:
        client = self._client()
        prompt = build_answer_prompt(question, columns, rows)
        completion = client.chat.completions.create(
            model=self.model_name(),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.3,
        )
        return (completion.choices[0].message.content or "").strip()

    def _client(self):  # type: ignore[no-untyped-def]
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AdapterNotAvailableError(
                "openai SDK is not installed. Run: uv add openai"
            ) from exc
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
