"""Ad creative generation endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from src.data.schemas import AdCreativeRequest, AdCreativeResponse
from src.modules.creative_generation import AdCreativeGenerator
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Module instance (would be injected via dependency injection in production)
_generator: AdCreativeGenerator | None = None


def get_generator() -> AdCreativeGenerator:
    """Get or create the ad creative generator."""
    global _generator
    if _generator is None:
        _generator = AdCreativeGenerator()
    return _generator


@router.post("/ads/generate", response_model=AdCreativeResponse)
async def generate_ad_creative(
    request: AdCreativeRequest,
    generator: Annotated[AdCreativeGenerator, Depends(get_generator)],
):
    """
    Generate ad creative variants.

    Creates multiple ad variants including headlines, descriptions,
    and CTAs based on product information and targeting parameters.
    """
    logger.info(
        "Generating ad creatives",
        product=request.product_name,
        num_variants=request.num_variants,
    )

    try:
        response = await generator.generate(request)
        return response
    except Exception as e:
        logger.error(f"Ad generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ads/generate/batch")
async def generate_batch(
    requests: list[AdCreativeRequest],
    generator: Annotated[AdCreativeGenerator, Depends(get_generator)],
):
    """
    Generate ad creatives for multiple products in batch.

    Processes multiple generation requests in parallel for efficiency.
    """
    if len(requests) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 requests per batch",
        )

    try:
        responses = await generator.generate_batch(requests)
        return {"results": responses, "total": len(responses)}
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ads/templates")
async def get_templates():
    """
    Get available ad templates and formats.

    Returns supported ad formats, tone options, and template guidelines.
    """
    return {
        "formats": [
            {"id": "display", "name": "Display Banner", "sizes": ["300x250", "728x90", "160x600"]},
            {"id": "native", "name": "Native Ad", "sizes": ["flexible"]},
            {"id": "video", "name": "Video Ad", "sizes": ["16:9", "1:1", "9:16"]},
            {"id": "rich_media", "name": "Rich Media", "sizes": ["expandable"]},
        ],
        "tones": [
            {"id": "professional", "description": "Business-focused, formal"},
            {"id": "casual", "description": "Friendly, conversational"},
            {"id": "urgent", "description": "Time-sensitive, action-oriented"},
            {"id": "luxurious", "description": "Premium, exclusive feel"},
            {"id": "playful", "description": "Fun, energetic tone"},
        ],
        "best_practices": {
            "headline_length": "60-90 characters",
            "description_length": "120-180 characters",
            "cta_words": ["Shop Now", "Learn More", "Get Started", "Try Free"],
        },
    }


@router.get("/ads/{creative_id}")
async def get_creative(creative_id: UUID):
    """
    Get details of a specific ad creative.

    Returns creative content, performance metrics, and variants.
    """
    # In production, would fetch from database
    return {
        "creative_id": str(creative_id),
        "status": "active",
        "message": "Creative retrieval not yet implemented - would fetch from database",
    }
