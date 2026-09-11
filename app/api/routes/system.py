"""System-level endpoints that do not expose application secrets."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(app_name=settings.app_name, environment=settings.environment)
