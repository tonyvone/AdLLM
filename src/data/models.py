"""
Core data models for AdTech LLM.

Defines SQLAlchemy ORM models for persistent storage of campaigns,
creatives, audiences, bids, and fraud signals.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class CampaignStatus(str, Enum):
    """Campaign status enum."""

    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class AdFormat(str, Enum):
    """Ad format types."""

    DISPLAY = "display"
    VIDEO = "video"
    NATIVE = "native"
    AUDIO = "audio"
    RICH_MEDIA = "rich_media"


class Campaign(Base):
    """Campaign model representing an advertising campaign."""

    __tablename__ = "campaigns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    advertiser_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    status = Column(SQLEnum(CampaignStatus), default=CampaignStatus.DRAFT)
    budget_total = Column(Float, default=0.0)
    budget_daily = Column(Float, default=0.0)
    budget_spent = Column(Float, default=0.0)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    targeting_config = Column(JSON, default=dict)
    optimization_goal = Column(String(50))  # ctr, conversions, impressions
    bid_strategy = Column(String(50))  # manual, auto, target_cpa
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    creatives = relationship("AdCreative", back_populates="campaign", lazy="dynamic")
    audience_segments = relationship("AudienceSegment", back_populates="campaign", lazy="dynamic")

    __table_args__ = (
        Index("idx_campaign_advertiser_status", "advertiser_id", "status"),
    )


class AdCreative(Base):
    """Ad creative model for storing generated ad content."""

    __tablename__ = "ad_creatives"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False)
    name = Column(String(255), nullable=False)
    format = Column(SQLEnum(AdFormat), default=AdFormat.DISPLAY)
    headline = Column(String(255))
    description = Column(Text)
    body = Column(Text)
    cta_text = Column(String(100))  # Call to action
    image_url = Column(String(512))
    video_url = Column(String(512))
    landing_url = Column(String(512))
    dimensions = Column(JSON)  # {"width": 300, "height": 250}
    variants = Column(JSON, default=list)  # A/B test variants
    performance_score = Column(Float, default=0.0)
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    campaign = relationship("Campaign", back_populates="creatives")

    __table_args__ = (
        Index("idx_creative_campaign_active", "campaign_id", "is_active"),
    )

    @property
    def ctr(self) -> float:
        """Calculate click-through rate."""
        if self.impressions == 0:
            return 0.0
        return self.clicks / self.impressions

    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        if self.clicks == 0:
            return 0.0
        return self.conversions / self.clicks


class AudienceSegment(Base):
    """Audience segment for targeting."""

    __tablename__ = "audience_segments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    segment_type = Column(String(50))  # demographic, behavioral, contextual, lookalike
    criteria = Column(JSON, default=dict)  # Targeting criteria
    estimated_reach = Column(Integer, default=0)
    match_rate = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    campaign = relationship("Campaign", back_populates="audience_segments")


class BidRequest(Base):
    """Real-time bidding request model."""

    __tablename__ = "bid_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    request_id = Column(String(100), unique=True, index=True)
    exchange = Column(String(50))  # Google, OpenX, etc.
    publisher_id = Column(String(100))
    site_domain = Column(String(255))
    page_url = Column(String(512))
    ad_slot = Column(JSON)  # Slot dimensions and position
    user_data = Column(JSON)  # Anonymized user signals
    device_info = Column(JSON)  # Device type, OS, etc.
    geo_info = Column(JSON)  # Country, region, city
    floor_price = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    response = relationship("BidResponse", back_populates="request", uselist=False)

    __table_args__ = (
        Index("idx_bid_request_timestamp", "timestamp"),
    )


class BidResponse(Base):
    """Real-time bidding response model."""

    __tablename__ = "bid_responses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    request_id = Column(UUID(as_uuid=True), ForeignKey("bid_requests.id"), unique=True)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"))
    creative_id = Column(UUID(as_uuid=True), ForeignKey("ad_creatives.id"))
    bid_price = Column(Float, nullable=False)
    win_price = Column(Float)  # Actual price if won
    bid_status = Column(String(20))  # submitted, won, lost, timeout
    latency_ms = Column(Integer)  # Response latency
    model_confidence = Column(Float)  # AI model confidence score
    features_used = Column(JSON)  # Features that influenced bid
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    request = relationship("BidRequest", back_populates="response")


class UserInteraction(Base):
    """User interaction events (impressions, clicks, conversions)."""

    __tablename__ = "user_interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    creative_id = Column(UUID(as_uuid=True), ForeignKey("ad_creatives.id"), index=True)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), index=True)
    event_type = Column(String(20), index=True)  # impression, click, conversion
    user_id_hash = Column(String(64))  # Anonymized user ID
    session_id = Column(String(64))
    device_type = Column(String(20))
    browser = Column(String(50))
    os = Column(String(50))
    country = Column(String(3))
    region = Column(String(100))
    page_url = Column(String(512))
    referrer_url = Column(String(512))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metadata = Column(JSON)

    __table_args__ = (
        Index("idx_interaction_campaign_event", "campaign_id", "event_type", "timestamp"),
    )


class FraudSignal(Base):
    """Fraud detection signals and analysis results."""

    __tablename__ = "fraud_signals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    interaction_id = Column(UUID(as_uuid=True), ForeignKey("user_interactions.id"))
    signal_type = Column(String(50), index=True)  # bot, click_farm, invalid_traffic
    severity = Column(String(20))  # low, medium, high, critical
    confidence_score = Column(Float, default=0.0)
    indicators = Column(JSON)  # Specific fraud indicators found
    ip_address_hash = Column(String(64))
    is_blocked = Column(Boolean, default=False)
    reviewed_by = Column(String(100))  # Human reviewer if any
    review_notes = Column(Text)
    detected_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_fraud_type_severity", "signal_type", "severity"),
    )


class DatasetMetadata(Base):
    """Metadata for ingested datasets."""

    __tablename__ = "dataset_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, unique=True)
    source = Column(String(50))  # kaggle, huggingface, partner, scraped
    source_url = Column(String(512))
    description = Column(Text)
    schema_info = Column(JSON)  # Column names, types
    row_count = Column(Integer, default=0)
    size_bytes = Column(Integer, default=0)
    file_path = Column(String(512))
    status = Column(String(20))  # pending, processing, ready, failed
    quality_score = Column(Float)
    last_updated = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelCheckpoint(Base):
    """Model training checkpoints and versioning."""

    __tablename__ = "model_checkpoints"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    base_model = Column(String(255))
    checkpoint_path = Column(String(512))
    training_config = Column(JSON)
    metrics = Column(JSON)  # Evaluation metrics
    dataset_ids = Column(JSON)  # Datasets used for training
    is_production = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_model_name_version", "model_name", "version"),
    )
