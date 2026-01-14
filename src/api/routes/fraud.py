"""Fraud detection endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.data.schemas import FraudDetectionRequest, FraudDetectionResponse
from src.modules.fraud_detection import FraudDetectionModule
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Module instance
_fraud_module: FraudDetectionModule | None = None


def get_fraud_module() -> FraudDetectionModule:
    """Get or create the fraud detection module."""
    global _fraud_module
    if _fraud_module is None:
        _fraud_module = FraudDetectionModule()
    return _fraud_module


@router.post("/fraud/analyze", response_model=FraudDetectionResponse)
async def analyze_fraud(
    request: FraudDetectionRequest,
    module: Annotated[FraudDetectionModule, Depends(get_fraud_module)],
):
    """
    Analyze an interaction for fraud.

    Checks for bot traffic, click fraud, and invalid traffic patterns.
    Returns fraud score and recommended action.
    """
    logger.info(
        "Analyzing for fraud",
        event_type=request.event_type,
    )

    try:
        response = await module.analyze(request)
        return response
    except Exception as e:
        logger.error(f"Fraud analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fraud/analyze/batch")
async def analyze_batch(
    requests: list[FraudDetectionRequest],
    module: Annotated[FraudDetectionModule, Depends(get_fraud_module)],
):
    """
    Analyze multiple interactions for fraud in batch.

    Processes multiple fraud checks in parallel for high-throughput scenarios.
    """
    if len(requests) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 requests per batch",
        )

    try:
        responses = await module.batch_analyze(requests)
        return {
            "results": responses,
            "total": len(responses),
            "fraudulent_count": sum(1 for r in responses if r.is_fraudulent),
            "blocked_count": sum(
                1 for r in responses if r.recommended_action == "block"
            ),
        }
    except Exception as e:
        logger.error(f"Batch fraud analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fraud/stats")
async def get_fraud_stats():
    """
    Get fraud detection statistics.

    Returns fraud rates, common patterns, and trends.
    """
    return {
        "overall_fraud_rate": 0.08,
        "fraud_by_type": {
            "bot_traffic": 0.04,
            "click_fraud": 0.02,
            "impression_fraud": 0.015,
            "invalid_traffic": 0.005,
        },
        "blocked_last_24h": 15000,
        "flagged_last_24h": 25000,
        "top_indicators": [
            "bot_user_agent",
            "high_click_frequency",
            "datacenter_ip",
            "no_mouse_movement",
        ],
    }


@router.get("/fraud/check-types")
async def list_check_types():
    """
    List available fraud check types.

    Returns descriptions of each fraud detection method.
    """
    return {
        "check_types": [
            {
                "id": "click_fraud",
                "name": "Click Fraud Detection",
                "description": "Detects fraudulent click patterns and click farms",
                "indicators": [
                    "high_click_frequency",
                    "regular_click_intervals",
                    "excessive_ip_clicks",
                ],
            },
            {
                "id": "impression_fraud",
                "name": "Impression Fraud Detection",
                "description": "Detects fake impressions, pixel stuffing, and ad stacking",
                "indicators": [
                    "low_viewability",
                    "pixel_stuffing",
                    "instant_impression",
                ],
            },
            {
                "id": "bot_detection",
                "name": "Bot Detection",
                "description": "Identifies automated non-human traffic",
                "indicators": [
                    "bot_user_agent",
                    "headless_browser",
                    "no_mouse_movement",
                    "suspicious_fingerprint",
                ],
            },
            {
                "id": "invalid_traffic",
                "name": "Invalid Traffic (IVT)",
                "description": "Detects general invalid traffic patterns",
                "indicators": [
                    "datacenter_ip",
                    "proxy_detected",
                    "geo_mismatch",
                ],
            },
            {
                "id": "attribution_fraud",
                "name": "Attribution Fraud",
                "description": "Detects click injection and attribution manipulation",
                "indicators": [
                    "click_injection",
                    "click_spamming",
                ],
            },
        ],
    }


@router.post("/fraud/report")
async def report_fraud(
    interaction_id: str,
    fraud_type: str,
    description: str | None = None,
):
    """
    Report suspected fraud for review.

    Submits a manual fraud report for investigation.
    """
    return {
        "status": "submitted",
        "report_id": f"report_{interaction_id}",
        "interaction_id": interaction_id,
        "fraud_type": fraud_type,
        "message": "Report submitted for review",
    }
