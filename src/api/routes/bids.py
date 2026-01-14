"""Bid optimization endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.data.schemas import BidOptimizationRequest, BidOptimizationResponse
from src.modules.bid_optimization import BidOptimizationModule
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Module instance
_bid_module: BidOptimizationModule | None = None


def get_bid_module() -> BidOptimizationModule:
    """Get or create the bid optimization module."""
    global _bid_module
    if _bid_module is None:
        _bid_module = BidOptimizationModule()
    return _bid_module


@router.post("/bids/optimize", response_model=BidOptimizationResponse)
async def optimize_bid(
    request: BidOptimizationRequest,
    module: Annotated[BidOptimizationModule, Depends(get_bid_module)],
):
    """
    Optimize bid for an RTB auction.

    Analyzes auction opportunity and returns optimal bid recommendation
    based on campaign strategy and predicted value.
    """
    logger.info(
        "Optimizing bid",
        campaign_id=str(request.campaign_id),
        strategy=request.strategy.value,
    )

    try:
        response = await module.optimize(request)
        return response
    except Exception as e:
        logger.error(f"Bid optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bids/optimize/batch")
async def optimize_batch(
    requests: list[BidOptimizationRequest],
    module: Annotated[BidOptimizationModule, Depends(get_bid_module)],
):
    """
    Optimize bids for multiple auctions in batch.

    Processes multiple bid requests in parallel for high-throughput scenarios.
    """
    if len(requests) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 requests per batch",
        )

    try:
        responses = await module.batch_optimize(requests)
        return {
            "results": responses,
            "total": len(responses),
            "participated": sum(1 for r in responses if r.should_bid),
        }
    except Exception as e:
        logger.error(f"Batch optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bids/strategies")
async def list_strategies():
    """
    List available bidding strategies.

    Returns supported strategies with descriptions and use cases.
    """
    return {
        "strategies": [
            {
                "id": "manual",
                "name": "Manual Bidding",
                "description": "Fixed bid amounts set manually",
                "use_case": "Full control over bid prices",
            },
            {
                "id": "auto_cpc",
                "name": "Automatic CPC",
                "description": "Automatically optimize for clicks",
                "use_case": "Maximize click volume within budget",
            },
            {
                "id": "target_cpa",
                "name": "Target CPA",
                "description": "Optimize to achieve target cost per acquisition",
                "use_case": "Focus on conversions at specific cost",
            },
            {
                "id": "target_roas",
                "name": "Target ROAS",
                "description": "Optimize for return on ad spend",
                "use_case": "Maximize revenue relative to spend",
            },
            {
                "id": "maximize_conversions",
                "name": "Maximize Conversions",
                "description": "Get most conversions within budget",
                "use_case": "Conversion-focused campaigns",
            },
            {
                "id": "maximize_clicks",
                "name": "Maximize Clicks",
                "description": "Get most clicks within budget",
                "use_case": "Awareness and traffic campaigns",
            },
        ],
    }


@router.get("/bids/market-insights")
async def get_market_insights():
    """
    Get current market insights for bidding.

    Returns average CPMs, competition levels, and trends.
    """
    return {
        "average_cpm_by_format": {
            "display": 2.50,
            "video": 8.00,
            "native": 4.00,
            "audio": 3.50,
        },
        "competition_levels": {
            "finance": "high",
            "technology": "high",
            "entertainment": "medium",
            "news": "medium",
            "gaming": "low",
        },
        "time_multipliers": {
            "prime_time": 1.3,
            "business_hours": 1.1,
            "late_night": 0.6,
        },
        "trends": {
            "mobile_growth": "+15% YoY",
            "video_demand": "+25% YoY",
            "ctv_emerging": "+50% YoY",
        },
    }
