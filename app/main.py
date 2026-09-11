"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.config import Settings, get_settings
from app.db import Database


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an application with injectable settings for tests."""
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_settings.ensure_local_directories()
        database = Database(active_settings.database_path)
        await database.initialize()
        app.state.settings = active_settings
        app.state.database = database
        yield
        await database.dispose()

    app = FastAPI(title=active_settings.app_name, version="0.1.0", lifespan=lifespan)
    app.include_router(api_router, prefix=active_settings.api_prefix)
    return app


app = create_app()
