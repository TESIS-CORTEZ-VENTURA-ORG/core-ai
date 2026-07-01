"""Mock LLM adapter — no API key required.

Returns deterministic, valid responses for a small set of question patterns
recognised by regex (nl2sql/answer) and fixed canned data for document
extraction (extract). This enables:
  - CI tests with no external service and no cost.
  - A fully functional $0 demo of the end-to-end flow.
  - Regression tests that fix exact output.

Every canned SQL query uses only tables in the NestJS analytics allowlist and
includes a LIMIT clause. None of them reference blocked tables (users,
refresh_tokens, tenants, audit_logs) or the salary column.
"""

from __future__ import annotations

import re

from app.chat.adapters.base import ExtractResult, LLMAdapter
from app.chat.schemas import Nl2SqlRequest, Nl2SqlResponse

# (compiled_regex, canned_sql) pairs — first match wins.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Profitability / margin queries
    (
        re.compile(r"rentab|profit|margin|ganan", re.IGNORECASE),
        (
            "SELECT mi.name, "
            "SUM(oi.unit_price * oi.qty) AS revenue, "
            "COUNT(oi.id) AS orders "
            "FROM order_items oi "
            "JOIN menu_items mi ON oi.menu_item_id = mi.id "
            "GROUP BY mi.name "
            "ORDER BY revenue DESC NULLS LAST "
            "LIMIT 10"
        ),
    ),
    # Sales / revenue queries
    (
        re.compile(r"venta|sales|ingres|revenue|factura", re.IGNORECASE),
        (
            "SELECT dish_name, "
            "SUM(qty) AS total_qty, "
            "SUM(total) AS total_revenue "
            "FROM sales_history "
            "GROUP BY dish_name "
            "ORDER BY total_revenue DESC "
            "LIMIT 20"
        ),
    ),
    # Stock / inventory queries
    (
        re.compile(r"stock|inventar|insumo|ingredient|ingredient", re.IGNORECASE),
        (
            "SELECT i.name AS ingredient, "
            "COALESCE(SUM("
            "CASE WHEN im.type = 'in' THEN im.qty ELSE -im.qty END"
            "), 0) AS current_stock "
            "FROM ingredients i "
            "LEFT JOIN inventory_movements im ON im.ingredient_id = i.id "
            "GROUP BY i.id, i.name "
            "ORDER BY current_stock ASC "
            "LIMIT 20"
        ),
    ),
    # Popular dishes / best-sellers
    (
        re.compile(r"plato|dish|menu|item|popular|vendido|sold", re.IGNORECASE),
        (
            "SELECT dish_name, "
            "SUM(qty) AS total_sold, "
            "SUM(total) AS total_revenue "
            "FROM sales_history "
            "GROUP BY dish_name "
            "ORDER BY total_sold DESC "
            "LIMIT 10"
        ),
    ),
    # Orders / purchases
    (
        re.compile(r"pedido|order|compra|purchase", re.IGNORECASE),
        (
            "SELECT DATE(created_at) AS day, "
            "COUNT(*) AS total_orders, "
            "SUM(total) AS revenue "
            "FROM orders "
            "GROUP BY day "
            "ORDER BY day DESC "
            "LIMIT 30"
        ),
    ),
]

# Used when no pattern matches — safe, always valid.
_FALLBACK_SQL = (
    "SELECT dish_name, "
    "SUM(qty) AS total_sold, "
    "SUM(total) AS total_revenue "
    "FROM sales_history "
    "GROUP BY dish_name "
    "ORDER BY total_sold DESC "
    "LIMIT 10"
)

# Deterministic canned extraction result for CI / no-key environments.
# Mirrors a realistic Peruvian restaurant menu to make tests meaningful.
_MOCK_EXTRACT_RESULT: ExtractResult = {
    "menuItems": [
        {
            "name": "Lomo Saltado",
            "price": 32.50,
            "category": "Platos de fondo",
            "description": "Carne de res salteada con verduras y papas fritas",
        },
        {
            "name": "Ceviche Mixto",
            "price": 28.00,
            "category": "Entradas",
            "description": None,
        },
        {
            "name": "Inca Kola 500ml",
            "price": 5.00,
            "category": "Bebidas",
            "description": None,
        },
    ],
    "ingredients": [
        {"name": "Carne de res", "unit": "kg", "estimatedCost": 32.00},
        {"name": "Limón", "unit": "kg", "estimatedCost": 4.50},
    ],
}


class MockAdapter(LLMAdapter):
    """Deterministic mock adapter — always available, no API key needed."""

    key = "mock"

    @classmethod
    def is_available(cls) -> bool:
        # The mock adapter is always available — it is the final fallback.
        return True

    def model_name(self) -> str:
        return "mock-v1"

    def nl2sql(self, request: Nl2SqlRequest) -> Nl2SqlResponse:
        for pattern, sql in _PATTERNS:
            if pattern.search(request.question):
                return Nl2SqlResponse(
                    sql=sql,
                    provider=self.key,
                    model=self.model_name(),
                    notes="Mock adapter — deterministic canned response.",
                )
        return Nl2SqlResponse(
            sql=_FALLBACK_SQL,
            provider=self.key,
            model=self.model_name(),
            notes="Mock adapter — generic fallback.",
        )

    def answer(self, question: str, columns: list[str], rows: list[list]) -> str:
        n = len(rows)
        if n == 0:
            return "No se encontraron registros para esa consulta."
        if n == 1 and columns and rows[0]:
            pairs = ", ".join(
                f"{col}: {val}" for col, val in zip(columns, rows[0], strict=False)
            )
            return f"Resultado: {pairs}."
        top = rows[0][0] if rows[0] else "—"
        return f"Se encontraron {n} registro(s). Primer resultado destacado: {top}."

    def extract(self, text: str, target: str, currency: str = "PEN") -> ExtractResult:
        """Return deterministic canned extraction data — no LLM call, no API key needed.

        The canned result is intentionally realistic (a Peruvian restaurant menu)
        so that e2e tests against the mock can assert meaningful field values.
        """
        result: ExtractResult = {"menuItems": [], "ingredients": []}
        if target in ("menu", "auto"):
            result["menuItems"] = list(_MOCK_EXTRACT_RESULT["menuItems"])
        if target in ("ingredients", "auto"):
            result["ingredients"] = list(_MOCK_EXTRACT_RESULT["ingredients"])
        return result
