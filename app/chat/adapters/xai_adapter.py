"""xAI (Grok) LLM adapter for Text-to-SQL and document extraction.

Requires XAI_API_KEY in the environment. xAI's API is OpenAI-compatible,
so we reuse the openai Python SDK with a custom base_url. No separate SDK
is needed.

Model is configurable via CORE_AI_CHAT_MODEL; defaults to grok-3-mini.
"""

from __future__ import annotations

import json
import logging
import os

from app.chat.adapters.base import AdapterNotAvailableError, ExtractResult, LLMAdapter
from app.chat.prompt import build_answer_prompt, build_nl2sql_prompt
from app.chat.schemas import Nl2SqlRequest, Nl2SqlResponse
from app.extract.prompt import build_extract_prompt

XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-3-mini"
logger = logging.getLogger(__name__)


class XAIAdapter(LLMAdapter):
    """Grok adapter — requires XAI_API_KEY, uses OpenAI SDK with xAI base_url."""

    key = "xai"

    @classmethod
    def is_available(cls) -> bool:
        return bool(os.environ.get("XAI_API_KEY"))

    def model_name(self) -> str:
        return os.environ.get("CORE_AI_CHAT_MODEL") or DEFAULT_MODEL

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
            temperature=0,
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

    def extract(self, text: str, target: str, currency: str = "PEN") -> ExtractResult:
        """Extract menu items and/or ingredients from restaurant document text.

        Uses response_format=json_object (supported by xAI's OpenAI-compatible API)
        to guarantee the model outputs valid JSON.
        """
        client = self._client()
        system_prompt, user_prompt = build_extract_prompt(text, target, currency)
        try:
            completion = client.chat.completions.create(
                model=self.model_name(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2048,
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw_text = (completion.choices[0].message.content or "").strip()
            parsed = json.loads(raw_text)
            return {
                "menuItems": parsed.get("menuItems", []),
                "ingredients": parsed.get("ingredients", []),
            }
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("xAI extract: failed to parse response — %s", exc)
            return {"menuItems": [], "ingredients": []}

    def _client(self):  # type: ignore[no-untyped-def]
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AdapterNotAvailableError(
                "openai SDK is not installed. Run: uv add openai"
            ) from exc
        return OpenAI(
            api_key=os.environ.get("XAI_API_KEY"),
            base_url=XAI_BASE_URL,
        )
