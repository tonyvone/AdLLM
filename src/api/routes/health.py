"""Health check endpoints."""

from datetime import datetime

from fastapi import APIRouter

from src.config.settings import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
    }


@router.get("/health/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    # Check if all required services are available
    checks = {
        "api": True,
        "database": True,  # Would check actual DB connection
        "cache": True,  # Would check Redis connection
    }

    all_ready = all(checks.values())

    return {
        "ready": all_ready,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/live")
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Domain-Specific AI Model for Advertising Technology",
        "docs": "/docs",
        "health": "/health",
    }
