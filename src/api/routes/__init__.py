"""API route handlers for AdTech LLM."""

from src.api.routes.ads import router as ads_router
from src.api.routes.audiences import router as audiences_router
from src.api.routes.bids import router as bids_router
from src.api.routes.contextual import router as contextual_router
from src.api.routes.fraud import router as fraud_router
from src.api.routes.health import router as health_router
from src.api.routes.training import router as training_router
from src.api.routes.workflows import router as workflows_router

__all__ = [
    "ads_router",
    "audiences_router",
    "bids_router",
    "contextual_router",
    "fraud_router",
    "health_router",
    "training_router",
    "workflows_router",
]
