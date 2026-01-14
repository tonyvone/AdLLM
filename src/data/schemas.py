"""
Pydantic schemas for API request/response validation.

These schemas define the contract between the API and clients,
ensuring type safety and validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Base Schemas
# ============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration."""

    class Config:
        from_attributes = True
        populate_by_name = True


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""

    created_at: datetime | None = None
    updated_at: datetime | None = None


# ============================================================================
# Ad Creative Schemas
# ============================================================================

class AdFormatEnum(str, Enum):
    """Supported ad formats."""

    DISPLAY = "display"
    VIDEO = "video"
    NATIVE = "native"
    AUDIO = "audio"
    RICH_MEDIA = "rich_media"


class AdToneEnum(str, Enum):
    """Ad copy tone options."""

    PROFESSIONAL = "professional"
    CASUAL = "casual"
    URGENT = "urgent"
    FRIENDLY = "friendly"
    LUXURIOUS = "luxurious"
    PLAYFUL = "playful"


class AdCreativeRequest(BaseSchema):
    """Request schema for ad creative generation."""

    product_name: str = Field(..., min_length=1, max_length=255, description="Product/service name")
    product_description: str = Field(..., min_length=10, description="Product description")
    target_audience: dict[str, Any] = Field(
        default_factory=dict,
        description="Target audience demographics and interests",
    )
    brand_guidelines: dict[str, Any] | None = Field(
        default=None,
        description="Brand voice and style guidelines",
    )
    ad_format: AdFormatEnum = Field(default=AdFormatEnum.DISPLAY, description="Ad format type")
    tone: AdToneEnum = Field(default=AdToneEnum.PROFESSIONAL, description="Desired tone")
    keywords: list[str] = Field(default_factory=list, description="Target keywords")
    num_variants: int = Field(default=5, ge=1, le=100, description="Number of variants to generate")
    max_headline_length: int = Field(default=90, ge=10, le=150)
    max_description_length: int = Field(default=300, ge=50, le=500)
    include_cta: bool = Field(default=True, description="Include call-to-action")
    cta_options: list[str] | None = Field(default=None, description="Preferred CTA options")
    language: str = Field(default="en", description="Target language code")
    context: dict[str, Any] | None = Field(default=None, description="Additional context")


class AdVariant(BaseSchema):
    """Single ad creative variant."""

    variant_id: str
    headline: str
    description: str
    body: str | None = None
    cta_text: str | None = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    predicted_ctr: float | None = Field(default=None, ge=0.0, le=1.0)
    sentiment_score: float | None = None
    relevance_score: float | None = None


class AdCreativeResponse(BaseSchema):
    """Response schema for ad creative generation."""

    request_id: str
    product_name: str
    variants: list[AdVariant]
    total_variants: int
    generation_time_ms: int
    model_version: str
    warnings: list[str] = Field(default_factory=list)


# ============================================================================
# Audience Targeting Schemas
# ============================================================================

class DemographicCriteria(BaseSchema):
    """Demographic targeting criteria."""

    age_min: int | None = Field(default=None, ge=13, le=100)
    age_max: int | None = Field(default=None, ge=13, le=100)
    genders: list[str] | None = None
    income_levels: list[str] | None = None
    education_levels: list[str] | None = None
    marital_status: list[str] | None = None
    parental_status: list[str] | None = None


class GeographicCriteria(BaseSchema):
    """Geographic targeting criteria."""

    countries: list[str] | None = None
    regions: list[str] | None = None
    cities: list[str] | None = None
    postal_codes: list[str] | None = None
    radius_km: float | None = Field(default=None, ge=0)
    exclude_locations: list[str] | None = None


class BehavioralCriteria(BaseSchema):
    """Behavioral targeting criteria."""

    interests: list[str] | None = None
    purchase_intent: list[str] | None = None
    browsing_history_categories: list[str] | None = None
    device_types: list[str] | None = None
    time_of_day: list[str] | None = None
    days_of_week: list[str] | None = None


class AudienceTargetingRequest(BaseSchema):
    """Request schema for audience targeting recommendations."""

    campaign_id: UUID | None = None
    product_category: str = Field(..., description="Product category")
    product_description: str = Field(..., description="Product description")
    campaign_objective: str = Field(
        default="conversions",
        description="Campaign objective: awareness, consideration, conversions",
    )
    budget: float = Field(default=1000.0, ge=0)
    demographics: DemographicCriteria | None = None
    geography: GeographicCriteria | None = None
    behavior: BehavioralCriteria | None = None
    existing_audiences: list[dict[str, Any]] | None = None
    exclude_audiences: list[str] | None = None


class AudienceSegmentResult(BaseSchema):
    """Single audience segment recommendation."""

    segment_id: str
    name: str
    description: str
    segment_type: str
    criteria: dict[str, Any]
    estimated_reach: int
    estimated_cpm: float
    predicted_ctr: float
    predicted_conversion_rate: float
    match_score: float = Field(ge=0.0, le=1.0)


class AudienceTargetingResponse(BaseSchema):
    """Response schema for audience targeting."""

    request_id: str
    segments: list[AudienceSegmentResult]
    total_estimated_reach: int
    recommended_budget_allocation: dict[str, float]
    insights: list[str]
    generation_time_ms: int


# ============================================================================
# Bid Optimization Schemas
# ============================================================================

class BidStrategyEnum(str, Enum):
    """Bidding strategy options."""

    MANUAL = "manual"
    AUTO_CPC = "auto_cpc"
    TARGET_CPA = "target_cpa"
    TARGET_ROAS = "target_roas"
    MAXIMIZE_CONVERSIONS = "maximize_conversions"
    MAXIMIZE_CLICKS = "maximize_clicks"


class BidOptimizationRequest(BaseSchema):
    """Request schema for bid optimization."""

    campaign_id: UUID
    ad_slot: dict[str, Any] = Field(..., description="Ad slot information")
    user_signals: dict[str, Any] = Field(default_factory=dict, description="User signals")
    context_signals: dict[str, Any] = Field(default_factory=dict, description="Context signals")
    floor_price: float = Field(default=0.0, ge=0)
    max_bid: float = Field(default=10.0, ge=0)
    strategy: BidStrategyEnum = Field(default=BidStrategyEnum.AUTO_CPC)
    target_cpa: float | None = Field(default=None, ge=0)
    target_roas: float | None = Field(default=None, ge=0)
    historical_performance: dict[str, Any] | None = None
    competitor_signals: dict[str, Any] | None = None


class BidRecommendation(BaseSchema):
    """Single bid recommendation."""

    creative_id: UUID
    recommended_bid: float
    bid_confidence: float = Field(ge=0.0, le=1.0)
    predicted_win_rate: float = Field(ge=0.0, le=1.0)
    predicted_ctr: float = Field(ge=0.0, le=1.0)
    predicted_cvr: float = Field(ge=0.0, le=1.0)
    expected_value: float
    reasoning: list[str]


class BidOptimizationResponse(BaseSchema):
    """Response schema for bid optimization."""

    request_id: str
    should_bid: bool
    recommendations: list[BidRecommendation]
    auction_insights: dict[str, Any]
    latency_ms: int
    model_version: str


# ============================================================================
# Fraud Detection Schemas
# ============================================================================

class FraudCheckType(str, Enum):
    """Types of fraud checks."""

    CLICK_FRAUD = "click_fraud"
    IMPRESSION_FRAUD = "impression_fraud"
    BOT_DETECTION = "bot_detection"
    INVALID_TRAFFIC = "invalid_traffic"
    ATTRIBUTION_FRAUD = "attribution_fraud"


class FraudDetectionRequest(BaseSchema):
    """Request schema for fraud detection."""

    interaction_id: UUID | None = None
    event_type: str = Field(..., description="Event type: impression, click, conversion")
    ip_address: str | None = Field(default=None, description="IP address (will be hashed)")
    user_agent: str | None = None
    device_fingerprint: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_data: dict[str, Any] | None = None
    behavioral_signals: dict[str, Any] | None = None
    check_types: list[FraudCheckType] = Field(
        default_factory=lambda: list(FraudCheckType),
        description="Types of fraud checks to perform",
    )


class FraudIndicator(BaseSchema):
    """Individual fraud indicator."""

    indicator_type: str
    severity: str  # low, medium, high, critical
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    evidence: dict[str, Any] | None = None


class FraudDetectionResponse(BaseSchema):
    """Response schema for fraud detection."""

    request_id: str
    is_fraudulent: bool
    fraud_score: float = Field(ge=0.0, le=1.0, description="Overall fraud probability")
    risk_level: str  # safe, low, medium, high, critical
    indicators: list[FraudIndicator]
    recommended_action: str  # allow, flag, block
    analysis_time_ms: int
    model_version: str


# ============================================================================
# Contextual Analysis Schemas
# ============================================================================

class ContextualAnalysisRequest(BaseSchema):
    """Request schema for contextual analysis."""

    url: str | None = Field(default=None, description="Page URL to analyze")
    content: str | None = Field(default=None, description="Direct content to analyze")
    ad_categories: list[str] | None = Field(
        default=None,
        description="Ad categories to check suitability for",
    )
    brand_safety_level: str = Field(
        default="standard",
        description="Brand safety level: permissive, standard, strict",
    )


class ContextualAnalysisResponse(BaseSchema):
    """Response schema for contextual analysis."""

    request_id: str
    url: str | None
    topics: list[dict[str, float]]  # Topic and confidence
    sentiment: dict[str, float]  # positive, negative, neutral scores
    brand_safety_score: float = Field(ge=0.0, le=1.0)
    unsafe_categories: list[str]
    suitable_ad_categories: list[str]
    keywords_extracted: list[str]
    content_quality_score: float = Field(ge=0.0, le=1.0)
    analysis_time_ms: int


# ============================================================================
# CTR Prediction Schemas
# ============================================================================

class CTRPredictionRequest(BaseSchema):
    """Request schema for CTR prediction."""

    creative_id: UUID | None = None
    creative_features: dict[str, Any] = Field(..., description="Creative features")
    audience_features: dict[str, Any] = Field(default_factory=dict)
    context_features: dict[str, Any] = Field(default_factory=dict)
    historical_features: dict[str, Any] | None = None


class CTRPredictionResponse(BaseSchema):
    """Response schema for CTR prediction."""

    request_id: str
    predicted_ctr: float = Field(ge=0.0, le=1.0)
    confidence_interval: tuple[float, float]
    feature_importance: dict[str, float]
    model_version: str
    prediction_time_ms: int


# ============================================================================
# Training and Fine-tuning Schemas
# ============================================================================

class TrainingJobRequest(BaseSchema):
    """Request schema for starting a training job."""

    job_name: str = Field(..., min_length=1, max_length=255)
    base_model: str = Field(default="meta-llama/Llama-3.1-8B-Instruct")
    dataset_ids: list[str] = Field(..., min_length=1)
    training_type: str = Field(default="sft", description="Training type: sft, rlhf, dpo")
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    validation_split: float = Field(default=0.1, ge=0.0, le=0.5)
    max_epochs: int = Field(default=3, ge=1, le=100)
    early_stopping: bool = Field(default=True)


class TrainingJobStatus(BaseSchema):
    """Training job status response."""

    job_id: str
    job_name: str
    status: str  # pending, running, completed, failed
    progress: float = Field(ge=0.0, le=1.0)
    current_epoch: int
    current_step: int
    total_steps: int
    metrics: dict[str, float]
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None = None


# ============================================================================
# Dataset Schemas
# ============================================================================

class DatasetUploadRequest(BaseSchema):
    """Request schema for dataset upload."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    source: str = Field(default="upload", description="Source: upload, kaggle, huggingface, url")
    source_identifier: str | None = Field(
        default=None,
        description="Kaggle dataset ID, HF dataset name, or URL",
    )
    format: str = Field(default="csv", description="Data format: csv, json, parquet")
    schema_mapping: dict[str, str] | None = Field(
        default=None,
        description="Column name mapping to standard schema",
    )


class DatasetInfo(BaseSchema):
    """Dataset information response."""

    dataset_id: str
    name: str
    source: str
    status: str
    row_count: int
    size_bytes: int
    columns: list[dict[str, str]]
    quality_score: float | None
    created_at: datetime
    last_updated: datetime | None
