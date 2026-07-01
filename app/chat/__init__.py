"""Text-to-SQL chat feature for GastronomIA core-ai.

Mirrors the forecasting module structure: router + schemas + service +
adapters (pluggable LLM strategy) + registry (auto-selection).

core-ai NEVER connects to the business database. It only runs LLM inference
and returns a single read-only SELECT query. The NestJS backend owns DB
execution, SQL validation, RLS enforcement, and the API response envelope.
"""
