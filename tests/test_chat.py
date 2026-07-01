"""Tests for the E09 chat / Text-to-SQL feature.

Coverage:
- Adapter registry: auto-selection (no key → mock), explicit key, unknown key.
- MockAdapter: nl2sql returns a valid SELECT for known patterns + generic fallback.
- MockAdapter: answer returns a non-empty string.
- Service-level gate: non-SELECT output is rejected with HTTP 422.
- POST /chat/nl2sql: happy path with mock adapter.
- POST /chat/answer: happy path with mock adapter.
- POST /chat/nl2sql: unknown provider → 400.

All tests run with no external API keys — the mock adapter is always used.
"""

from __future__ import annotations


import pytest
from fastapi.testclient import TestClient

from app.chat.adapters.mock_adapter import MockAdapter
from app.chat.registry import available_adapters, list_adapter_keys, resolve_adapter
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestAdapterRegistry:
    def test_all_expected_keys_registered(self) -> None:
        keys = list_adapter_keys()
        for expected in ("openai", "anthropic", "xai", "mock"):
            assert expected in keys, f"Expected adapter '{expected}' to be registered"

    def test_available_adapters_includes_mock(self) -> None:
        avail = available_adapters()
        assert avail["mock"] is True, "mock adapter must always be available"

    def test_auto_select_without_any_key_returns_mock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Remove all API keys to force auto-select to reach the mock fallback.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.delenv("CORE_AI_CHAT_PROVIDER", raising=False)
        adapter = resolve_adapter()
        assert adapter.key == "mock"

    def test_explicit_mock_key_returns_mock(self) -> None:
        adapter = resolve_adapter("mock")
        assert isinstance(adapter, MockAdapter)

    def test_unknown_key_raises(self) -> None:
        from app.chat.registry import UnknownAdapterError

        with pytest.raises(UnknownAdapterError, match="Unknown chat provider"):
            resolve_adapter("does-not-exist")

    def test_env_var_selects_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORE_AI_CHAT_PROVIDER", "mock")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        adapter = resolve_adapter()
        assert adapter.key == "mock"


# ---------------------------------------------------------------------------
# MockAdapter unit tests
# ---------------------------------------------------------------------------


class TestMockAdapterNl2Sql:
    def setup_method(self) -> None:
        self.adapter = MockAdapter()
        self.schema_ctx = "Table: sales_history (dish_name, qty, total)"

    def _request(self, question: str):  # type: ignore[no-untyped-def]
        from app.chat.schemas import Nl2SqlRequest

        return Nl2SqlRequest(
            question=question,
            schema_context=self.schema_ctx,
            dialect="postgresql",
            max_rows=200,
        )

    def test_profitability_pattern_returns_select(self) -> None:
        result = self.adapter.nl2sql(self._request("¿cuál es mi plato más rentable?"))
        assert result.sql.upper().startswith("SELECT")
        assert result.provider == "mock"

    def test_sales_pattern_returns_select(self) -> None:
        result = self.adapter.nl2sql(self._request("muéstrame las ventas del mes"))
        assert result.sql.upper().startswith("SELECT")

    def test_stock_pattern_returns_select(self) -> None:
        result = self.adapter.nl2sql(self._request("¿qué insumos tienen bajo stock?"))
        assert result.sql.upper().startswith("SELECT")

    def test_dish_pattern_returns_select(self) -> None:
        result = self.adapter.nl2sql(self._request("platos más vendidos"))
        assert result.sql.upper().startswith("SELECT")

    def test_generic_fallback_returns_select(self) -> None:
        result = self.adapter.nl2sql(self._request("xyz completamente desconocido"))
        assert result.sql.upper().startswith("SELECT")

    def test_result_has_limit(self) -> None:
        result = self.adapter.nl2sql(self._request("ventas"))
        assert "LIMIT" in result.sql.upper()

    def test_result_never_contains_dml(self) -> None:
        for question in [
            "ventas",
            "rentabilidad",
            "stock",
            "platos",
            "pedidos",
            "xyz desconocido",
        ]:
            result = self.adapter.nl2sql(self._request(question))
            upper = result.sql.upper()
            for blocked in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE"):
                assert blocked not in upper, (
                    f"Mock returned '{blocked}' for question '{question}'"
                )

    def test_result_never_references_blocked_tables(self) -> None:
        blocked_tables = {"users", "refresh_tokens", "audit_logs", "tenants"}
        for question in ["ventas", "rentabilidad", "stock", "platos", "xyz"]:
            result = self.adapter.nl2sql(self._request(question))
            sql_lower = result.sql.lower()
            for tbl in blocked_tables:
                assert tbl not in sql_lower, (
                    f"Mock referenced blocked table '{tbl}' for question '{question}'"
                )

    def test_result_never_references_salary(self) -> None:
        for question in ["empleados", "staff", "personal", "sueldos"]:
            result = self.adapter.nl2sql(self._request(question))
            assert "salary" not in result.sql.lower()


class TestMockAdapterAnswer:
    def setup_method(self) -> None:
        self.adapter = MockAdapter()

    def test_empty_rows_returns_no_records_message(self) -> None:
        result = self.adapter.answer("¿ventas?", [], [])
        assert "No se encontraron" in result

    def test_single_row_returns_result_summary(self) -> None:
        result = self.adapter.answer("pregunta", ["name", "qty"], [["Lomo Saltado", 5]])
        assert len(result) > 0

    def test_multiple_rows_returns_count(self) -> None:
        rows = [["Dish A", 10], ["Dish B", 8], ["Dish C", 3]]
        result = self.adapter.answer("¿platos?", ["name", "qty"], rows)
        assert "3" in result or "registro" in result.lower()


# ---------------------------------------------------------------------------
# Service-level gate (non-SELECT rejection)
# ---------------------------------------------------------------------------


class TestServiceGate:
    """The service must reject non-SELECT output BEFORE it leaves core-ai."""

    def test_nl2sql_rejects_delete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If an adapter returned DELETE (shouldn't happen, but gate must catch it)."""
        from app.chat import service
        from app.chat.schemas import Nl2SqlRequest
        from unittest.mock import MagicMock

        bad_resp = MagicMock()
        bad_resp.sql = "DELETE FROM sales_history"
        bad_resp.provider = "mock"
        bad_resp.model = "mock-v1"
        bad_resp.notes = ""

        monkeypatch.setattr(
            "app.chat.service.resolve_adapter",
            lambda *_: MagicMock(nl2sql=lambda _r: bad_resp),
        )
        from fastapi import HTTPException

        req = Nl2SqlRequest(
            question="delete all",
            schema_context="table: sales_history",
            dialect="postgresql",
            max_rows=200,
        )
        with pytest.raises(HTTPException) as exc_info:
            service.nl2sql(req)
        assert exc_info.value.status_code == 422

    def test_nl2sql_rejects_drop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.chat import service
        from app.chat.schemas import Nl2SqlRequest
        from unittest.mock import MagicMock

        bad_resp = MagicMock()
        bad_resp.sql = "SELECT 1 DROP TABLE users"
        bad_resp.provider = "mock"
        bad_resp.model = "mock-v1"
        bad_resp.notes = ""

        monkeypatch.setattr(
            "app.chat.service.resolve_adapter",
            lambda *_: MagicMock(nl2sql=lambda _r: bad_resp),
        )
        from fastapi import HTTPException

        req = Nl2SqlRequest(
            question="drop users",
            schema_context="table: users",
            dialect="postgresql",
            max_rows=200,
        )
        with pytest.raises(HTTPException) as exc_info:
            service.nl2sql(req)
        assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# HTTP endpoint tests (FastAPI TestClient, mock adapter via env)
# ---------------------------------------------------------------------------


class TestNl2SqlEndpoint:
    SCHEMA_CTX = "Table sales_history: dish_name (text), qty (int), total (numeric), sold_on (date)"

    def _payload(self, question: str = "¿cuáles son mis ventas?") -> dict:
        return {
            "question": question,
            "schema_context": self.SCHEMA_CTX,
            "dialect": "postgresql",
            "max_rows": 50,
        }

    def test_happy_path_returns_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CORE_AI_CHAT_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        resp = client.post("/chat/nl2sql", json=self._payload())
        assert resp.status_code == 200

    def test_response_contains_required_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CORE_AI_CHAT_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        resp = client.post("/chat/nl2sql", json=self._payload())
        body = resp.json()
        assert "sql" in body
        assert "provider" in body
        assert "model" in body

    def test_provider_is_mock_without_api_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CORE_AI_CHAT_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        resp = client.post("/chat/nl2sql", json=self._payload())
        assert resp.json()["provider"] == "mock"

    def test_sql_starts_with_select(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CORE_AI_CHAT_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        resp = client.post("/chat/nl2sql", json=self._payload("ventas del mes"))
        assert resp.json()["sql"].upper().startswith("SELECT")

    def test_explicit_mock_provider_via_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORE_AI_CHAT_PROVIDER", "mock")
        resp = client.post("/chat/nl2sql", json=self._payload())
        assert resp.status_code == 200
        assert resp.json()["provider"] == "mock"

    def test_unknown_provider_env_returns_400(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORE_AI_CHAT_PROVIDER", "nonexistent-llm")
        resp = client.post("/chat/nl2sql", json=self._payload())
        assert resp.status_code == 400

    def test_empty_question_returns_422(self) -> None:
        resp = client.post(
            "/chat/nl2sql",
            json={"question": "", "schema_context": "x", "dialect": "postgresql"},
        )
        assert resp.status_code == 422


class TestAnswerEndpoint:
    def test_happy_path_returns_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORE_AI_CHAT_PROVIDER", "mock")
        resp = client.post(
            "/chat/answer",
            json={
                "question": "¿ventas?",
                "columns": ["dish_name", "total"],
                "rows": [["Lomo Saltado", "50.00"], ["Ceviche", "30.00"]],
                "provider": "mock",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "provider" in body
        assert len(body["answer"]) > 0

    def test_empty_rows_returns_no_data_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORE_AI_CHAT_PROVIDER", "mock")
        resp = client.post(
            "/chat/answer",
            json={
                "question": "¿ventas?",
                "columns": [],
                "rows": [],
                "provider": "mock",
            },
        )
        assert resp.status_code == 200
        assert "No se encontraron" in resp.json()["answer"]
