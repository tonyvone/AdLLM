"""Audience targeting endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from src.data.schemas import (
    AudienceTargetingRequest,
    AudienceTargetingResponse,
    CTRPredictionRequest,
    CTRPredictionResponse,
)
from src.modules.audience_targeting import AudienceTargetingModule
from src.modules.ctr_prediction import CTRPredictionModule
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Module instances
_targeting_module: AudienceTargetingModule | None = None
_ctr_module: CTRPredictionModule | None = None


def get_targeting_module() -> AudienceTargetingModule:
    """Get or create the audience targeting module."""
    global _targeting_module
    if _targeting_module is None:
        _targeting_module = AudienceTargetingModule()
    return _targeting_module


def get_ctr_module() -> CTRPredictionModule:
    """Get or create the CTR prediction module."""
    global _ctr_module
    if _ctr_module is None:
        _ctr_module = CTRPredictionModule()
    return _ctr_module


@router.post("/audiences/recommendations", response_model=AudienceTargetingResponse)
async def get_audience_recommendations(
    request: AudienceTargetingRequest,
    module: Annotated[AudienceTargetingModule, Depends(get_targeting_module)],
):
    """
    Get audience targeting recommendations.

    Analyzes campaign objectives and returns optimal audience segments
    with predicted performance metrics.
    """
    logger.info(
        "Getting audience recommendations",
        product_category=request.product_category,
        objective=request.campaign_objective,
    )

    try:
        response = await module.get_targeting_recommendations(request)
        return response
    except Exception as e:
        logger.error(f"Targeting recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audiences/predict-ctr", response_model=CTRPredictionResponse)
async def predict_ctr(
    request: CTRPredictionRequest,
    module: Annotated[CTRPredictionModule, Depends(get_ctr_module)],
):
    """
    Predict click-through rate for ad placement.

    Returns predicted CTR with confidence interval and feature importance.
    """
    logger.info("Predicting CTR")

    try:
        response = await module.predict(request)
        return response
    except Exception as e:
        logger.error(f"CTR prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audiences/segments")
async def list_segments():
    """
    List available audience segments.

    Returns predefined and custom audience segments.
    """
    return {
        "segments": [
            {
                "id": "tech_enthusiasts",
                "name": "Tech Enthusiasts",
                "description": "Users interested in technology and gadgets",
                "estimated_reach": 50000000,
                "type": "behavioral",
            },
            {
                "id": "young_professionals",
                "name": "Young Professionals",
                "description": "Career-focused individuals aged 25-35",
                "estimated_reach": 80000000,
                "type": "demographic",
            },
            {
                "id": "budget_shoppers",
                "name": "Budget-Conscious Shoppers",
                "description": "Price-sensitive deal seekers",
                "estimated_reach": 100000000,
                "type": "behavioral",
            },
            {
                "id": "luxury_consumers",
                "name": "Luxury Consumers",
                "description": "High-income premium buyers",
                "estimated_reach": 20000000,
                "type": "demographic",
            },
            {
                "id": "health_fitness",
                "name": "Health & Fitness",
                "description": "Wellness and fitness enthusiasts",
                "estimated_reach": 60000000,
                "type": "behavioral",
            },
        ],
        "total": 5,
    }


@router.get("/audiences/segments/{segment_id}")
async def get_segment(segment_id: str):
    """
    Get details of a specific audience segment.

    Returns segment criteria, reach estimates, and performance data.
    """
    # Placeholder response
    return {
        "segment_id": segment_id,
        "name": segment_id.replace("_", " ").title(),
        "criteria": {
            "interests": ["example"],
            "behaviors": ["example"],
        },
        "metrics": {
            "estimated_reach": 50000000,
            "avg_ctr": 0.025,
            "avg_cvr": 0.03,
        },
    }


@router.post("/audiences/lookalike")
async def create_lookalike_audience(
    seed_audience_id: str,
    expansion_ratio: float = 1.0,
):
    """
    Create a lookalike audience based on a seed audience.

    Expands targeting to similar users based on behavioral patterns.
    """
    return {
        "status": "created",
        "seed_audience_id": seed_audience_id,
        "lookalike_id": f"lookalike_{seed_audience_id}",
        "expansion_ratio": expansion_ratio,
        "estimated_reach": int(50000000 * expansion_ratio),
    }
