"""Data models and schemas for AdTech LLM."""

from src.data.models import (
    AdCreative,
    AudienceSegment,
    BidRequest,
    BidResponse,
    Campaign,
    FraudSignal,
    UserInteraction,
)
from src.data.schemas import (
    AdCreativeRequest,
    AdCreativeResponse,
    AudienceTargetingRequest,
    AudienceTargetingResponse,
    BidOptimizationRequest,
    BidOptimizationResponse,
    FraudDetectionRequest,
    FraudDetectionResponse,
)

__all__ = [
    # Models
    "AdCreative",
    "AudienceSegment",
    "BidRequest",
    "BidResponse",
    "Campaign",
    "FraudSignal",
    "UserInteraction",
    # Schemas
    "AdCreativeRequest",
    "AdCreativeResponse",
    "AudienceTargetingRequest",
    "AudienceTargetingResponse",
    "BidOptimizationRequest",
    "BidOptimizationResponse",
    "FraudDetectionRequest",
    "FraudDetectionResponse",
]
