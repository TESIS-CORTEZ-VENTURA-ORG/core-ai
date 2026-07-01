"""Document extraction API router — E11 Smart Onboarding.

POST /extract/document — extract structured menu/ingredient data from plain text.

The NestJS backend is the ONLY caller. It converts the uploaded file (PDF/xlsx/csv)
to plain text and sends it here. core-ai runs LLM inference and returns structured
JSON; the backend presents a preview to the user and, upon confirmation, commits
the data to the tenant's catalog (runInTenant, RLS FORCE).

Access is not guarded here because core-ai runs in an internal network — the NestJS
service is the perimeter (CASL manage Catalog: owner/manager only).
"""

from __future__ import annotations

from fastapi import APIRouter

from app.extract import service
from app.extract.schemas import ExtractRequest, ExtractResponse

router = APIRouter(prefix="/extract", tags=["extract"])


@router.post("/document", response_model=ExtractResponse)
def extract_document_endpoint(request: ExtractRequest) -> ExtractResponse:
    """Extract structured menu items and/or ingredients from restaurant document text.

    The active LLM adapter is auto-resolved from the environment
    (CORE_AI_CHAT_PROVIDER → first key present → mock). The mock adapter is
    always available for CI and no-key environments, returning deterministic
    canned data.
    """
    return service.extract_document(request)
