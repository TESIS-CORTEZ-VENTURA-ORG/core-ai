"""LLM prompt builders for the Text-to-SQL feature.

Centralised here so every adapter sends the same system instructions.
The system prompt is intentionally strict: the LLM is told exactly what
it MAY and MUST NOT do. This is the first line of defence; the NestJS
backend's SQL validator is the second, stricter gate.
"""

from __future__ import annotations

from app.chat.schemas import Nl2SqlRequest

# System prompt enforced for every provider. Rules are numbered so logs and
# errors can reference them precisely. Critically: we tell the LLM NOT to add
# a tenant_id filter — RLS FORCE handles tenant isolation, and an LLM-added
# filter could be wrong or missing.
NL2SQL_SYSTEM = """\
You are a PostgreSQL analytics expert for a restaurant management system.

STRICT OUTPUT RULES — output exactly one raw SQL statement, nothing else:
1. Output ONLY a single valid PostgreSQL SELECT (or WITH ... SELECT) statement.
2. No semicolons anywhere in the output.
3. No multiple statements separated by semicolons.
4. No DDL: no CREATE, DROP, ALTER, TRUNCATE.
5. No DML: no INSERT, UPDATE, DELETE, MERGE.
6. Only reference tables and columns that appear in the Schema Context below.
7. Never use pg_ system tables, information_schema, or pg_catalog.
8. Never reference the column "salary".
9. Always include a LIMIT clause with at most 200 rows.
10. Do NOT add a tenant_id filter — Row Level Security handles tenant isolation automatically.
11. No markdown, no code fences, no explanations — output the raw SQL only.
"""


def build_nl2sql_prompt(request: Nl2SqlRequest) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the nl2sql LLM call."""
    user_prompt = (
        f"Schema Context:\n{request.schema_context}\n\n"
        f"User question: {request.question}\n\n"
        f"Return a single read-only PostgreSQL SELECT query. "
        f"LIMIT must be at most {request.max_rows} rows. "
        f"Output the SQL only — no markdown, no explanation."
    )
    return NL2SQL_SYSTEM, user_prompt


def build_answer_prompt(question: str, columns: list[str], rows: list[list]) -> str:
    """Build a prompt for the natural-language answer summarization call.

    We limit the row preview to 5 rows to keep token cost manageable. The
    LLM gets enough context to produce a useful sentence without being sent
    thousands of rows.
    """
    header = ", ".join(columns) if columns else "(no columns)"
    preview = rows[:5]
    rows_text = "\n".join(str(r) for r in preview)
    extra = f"\n... y {len(rows) - 5} filas más." if len(rows) > 5 else ""
    total = len(rows)
    return (
        f"El usuario preguntó: {question!r}\n\n"
        f"Resultado de la consulta ({total} fila(s)):\n"
        f"Columnas: {header}\n"
        f"{rows_text}{extra}\n\n"
        "Responde en español con una sola oración concisa que resuma el hallazgo "
        "principal. No incluyas SQL ni detalles técnicos."
    )
