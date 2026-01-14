"""Contextual analysis endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.data.schemas import ContextualAnalysisRequest, ContextualAnalysisResponse
from src.modules.contextual_analysis import ContextualAnalysisModule
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Module instance
_contextual_module: ContextualAnalysisModule | None = None


def get_contextual_module() -> ContextualAnalysisModule:
    """Get or create the contextual analysis module."""
    global _contextual_module
    if _contextual_module is None:
        _contextual_module = ContextualAnalysisModule()
    return _contextual_module


@router.post("/contextual/analyze", response_model=ContextualAnalysisResponse)
async def analyze_context(
    request: ContextualAnalysisRequest,
    module: Annotated[ContextualAnalysisModule, Depends(get_contextual_module)],
):
    """
    Analyze content for ad placement suitability.

    Extracts topics, sentiment, and brand safety information from
    page content or URL to determine appropriate ad categories.
    """
    logger.info(
        "Analyzing context",
        url=request.url,
        has_content=bool(request.content),
    )

    try:
        response = await module.analyze(request)
        return response
    except Exception as e:
        logger.error(f"Contextual analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/contextual/analyze/batch")
async def analyze_batch(
    requests: list[ContextualAnalysisRequest],
    module: Annotated[ContextualAnalysisModule, Depends(get_contextual_module)],
):
    """
    Analyze multiple pages/content in batch.

    Processes multiple contextual analyses in parallel.
    """
    if len(requests) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 requests per batch",
        )

    try:
        responses = await module.batch_analyze(requests)
        return {
            "results": responses,
            "total": len(responses),
            "safe_count": sum(1 for r in responses if r.brand_safety_score >= 0.7),
        }
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/contextual/categories")
async def list_ad_categories():
    """
    List available ad categories for targeting.

    Returns categories that can be matched to content topics.
    """
    return {
        "categories": [
            {"id": "tech", "name": "Technology", "parent": None},
            {"id": "electronics", "name": "Electronics", "parent": "tech"},
            {"id": "software", "name": "Software", "parent": "tech"},
            {"id": "finance", "name": "Financial Services", "parent": None},
            {"id": "banking", "name": "Banking", "parent": "finance"},
            {"id": "insurance", "name": "Insurance", "parent": "finance"},
            {"id": "health", "name": "Healthcare", "parent": None},
            {"id": "pharma", "name": "Pharmaceuticals", "parent": "health"},
            {"id": "wellness", "name": "Wellness", "parent": "health"},
            {"id": "automotive", "name": "Automotive", "parent": None},
            {"id": "travel", "name": "Travel", "parent": None},
            {"id": "retail", "name": "Retail", "parent": None},
            {"id": "fashion", "name": "Fashion", "parent": "retail"},
            {"id": "food", "name": "Food & Beverage", "parent": "retail"},
            {"id": "entertainment", "name": "Entertainment", "parent": None},
            {"id": "gaming", "name": "Gaming", "parent": "entertainment"},
            {"id": "education", "name": "Education", "parent": None},
            {"id": "b2b", "name": "B2B Services", "parent": None},
        ],
    }


@router.get("/contextual/brand-safety-levels")
async def list_safety_levels():
    """
    List brand safety levels and their criteria.

    Returns available safety configurations.
    """
    return {
        "levels": [
            {
                "id": "permissive",
                "name": "Permissive",
                "description": "Minimal restrictions, allows most content",
                "blocked_categories": ["adult_content", "illegal_activity"],
            },
            {
                "id": "standard",
                "name": "Standard",
                "description": "Balanced approach for most advertisers",
                "blocked_categories": [
                    "adult_content",
                    "violence",
                    "hate_speech",
                    "illegal_activity",
                ],
            },
            {
                "id": "strict",
                "name": "Strict",
                "description": "Conservative approach for sensitive brands",
                "blocked_categories": [
                    "adult_content",
                    "violence",
                    "hate_speech",
                    "illegal_activity",
                    "gambling",
                    "weapons",
                    "misinformation",
                ],
            },
        ],
    }


@router.post("/contextual/keyword-extract")
async def extract_keywords(
    content: str,
    max_keywords: int = 20,
):
    """
    Extract keywords from content.

    Returns important keywords for contextual targeting.
    """
    import re

    # Simple keyword extraction
    words = re.findall(r"\b[a-zA-Z]{4,}\b", content.lower())
    stop_words = {
        "this", "that", "with", "from", "have", "been", "were", "will",
        "would", "could", "should", "their", "there", "about", "which",
    }
    filtered = [w for w in words if w not in stop_words]

    # Count frequencies
    freq = {}
    for word in filtered:
        freq[word] = freq.get(word, 0) + 1

    # Get top keywords
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [{"keyword": word, "frequency": count} for word, count in sorted_words[:max_keywords]]

    return {"keywords": keywords, "total_words": len(words)}
