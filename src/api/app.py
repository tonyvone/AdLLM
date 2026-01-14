"""
FastAPI Application for AdTech LLM.

Provides REST API endpoints for all AdTech LLM functionality
including ad generation, targeting, fraud detection, and bidding.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import (
    ads_router,
    audiences_router,
    bids_router,
    contextual_router,
    fraud_router,
    health_router,
    training_router,
    workflows_router,
)
from src.config.settings import settings
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    # Startup
    setup_logging()
    logger.info(
        "Starting AdTech LLM API",
        version=settings.app_version,
        environment=settings.environment,
    )

    # Initialize components
    app.state.initialized = True

    yield

    # Shutdown
    logger.info("Shutting down AdTech LLM API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description="Domain-Specific AI Model for Advertising Technology",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health_router, tags=["Health"])
    app.include_router(ads_router, prefix=settings.api_prefix, tags=["Ad Creatives"])
    app.include_router(audiences_router, prefix=settings.api_prefix, tags=["Audiences"])
    app.include_router(bids_router, prefix=settings.api_prefix, tags=["Bidding"])
    app.include_router(fraud_router, prefix=settings.api_prefix, tags=["Fraud Detection"])
    app.include_router(contextual_router, prefix=settings.api_prefix, tags=["Contextual"])
    app.include_router(training_router, prefix=settings.api_prefix, tags=["Training"])
    app.include_router(workflows_router, prefix=settings.api_prefix, tags=["Workflows"])

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    return app


# Application instance
app = create_app()
