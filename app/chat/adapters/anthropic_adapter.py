"""Anthropic (Claude) LLM adapter for Text-to-SQL and document extraction.

Requires ANTHROPIC_API_KEY in the environment. Uses the anthropic Python SDK.
The system prompt is sent as the Anthropic `system` parameter (not as a
message), which is the recommended approach for instruction-following tasks.

Model is configurable via CORE_AI_CHAT_MODEL; defaults to claude-haiku-4-5
(fast, inexpensive, strong instruction-following for structured output like SQL
and JSON extraction).
"""

from __future__ import annotations

import json
import logging
import os

from app.chat.adapters.base import AdapterNotAvailableError, ExtractResult, LLMAdapter
from app.chat.prompt import build_answer_prompt, build_nl2sql_prompt
from app.chat.schemas import Nl2SqlRequest, Nl2SqlResponse
from app.extract.prompt import build_extract_prompt

DEFAULT_MODEL = "claude-haiku-4-5"
logger = logging.getLogger(__name__)


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

    def extract(self, text: str, target: str, currency: str = "PEN") -> ExtractResult:
        """Extract menu items and/or ingredients from restaurant document text.

        Claude's strong instruction-following makes it reliable for strict JSON
        output without needing a forced json_object mode. The caller validates
        each item individually so partial responses degrade gracefully.
        """
        client = self._client()
        system_prompt, user_prompt = build_extract_prompt(text, target, currency)
        try:
            message = client.messages.create(
                model=self.model_name(),
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = message.content[0].text.strip() if message.content else ""
            # Strip markdown fences if Claude adds them despite instructions
            raw_text = _strip_fences(raw_text)
            parsed = json.loads(raw_text)
            return {
                "menuItems": parsed.get("menuItems", []),
                "ingredients": parsed.get("ingredients", []),
            }
        except (json.JSONDecodeError, KeyError, TypeError, IndexError) as exc:
            logger.warning("Anthropic extract: failed to parse response — %s", exc)
            return {"menuItems": [], "ingredients": []}

    def _client(self):  # type: ignore[no-untyped-def]
        try:
            import anthropic
        except ImportError as exc:
            raise AdapterNotAvailableError(
                "anthropic SDK is not installed. Run: uv add anthropic"
            ) from exc
        return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _strip_fences(text: str) -> str:
    """Remove optional markdown code fences that LLMs sometimes prepend.

    Claude respects the 'no code fences' instruction most of the time, but
    defensively stripping ensures we always get parseable JSON.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (```json or ```) and last line (```)
        inner = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    return text
