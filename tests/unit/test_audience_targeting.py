"""Unit tests for Audience Targeting module."""

import pytest

from src.modules.audience_targeting import AudienceTargetingModule
from src.data.schemas import (
    AudienceTargetingRequest,
    DemographicCriteria,
    GeographicCriteria,
    BehavioralCriteria,
)


@pytest.fixture
def targeting_module():
    """Create an audience targeting module instance."""
    return AudienceTargetingModule()


@pytest.fixture
def sample_request():
    """Create a sample targeting request."""
    return AudienceTargetingRequest(
        product_category="technology",
        product_description="A smart home device for modern living",
        campaign_objective="conversions",
        budget=5000.0,
        demographics=DemographicCriteria(
            age_min=25,
            age_max=45,
        ),
        geography=GeographicCriteria(
            countries=["US", "UK"],
        ),
        behavior=BehavioralCriteria(
            interests=["technology", "smart_home", "gadgets"],
        ),
    )


class TestAudienceTargetingModule:
    """Tests for AudienceTargetingModule class."""

    @pytest.mark.asyncio
    async def test_get_recommendations_returns_response(
        self, targeting_module, sample_request
    ):
        """Test that recommendations are returned."""
        response = await targeting_module.get_targeting_recommendations(sample_request)

        assert response.request_id is not None
        assert len(response.segments) > 0
        assert response.total_estimated_reach > 0
        assert response.generation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_segments_have_required_fields(
        self, targeting_module, sample_request
    ):
        """Test that segments have all required fields."""
        response = await targeting_module.get_targeting_recommendations(sample_request)

        for segment in response.segments:
            assert segment.segment_id is not None
            assert segment.name is not None
            assert segment.segment_type is not None
            assert segment.estimated_reach > 0
            assert segment.estimated_cpm > 0
            assert 0 <= segment.match_score <= 1

    @pytest.mark.asyncio
    async def test_budget_allocation_sums_to_budget(
        self, targeting_module, sample_request
    ):
        """Test that budget allocation sums to total budget."""
        response = await targeting_module.get_targeting_recommendations(sample_request)

        total_allocated = sum(response.recommended_budget_allocation.values())

        # Allow small floating point tolerance
        assert abs(total_allocated - sample_request.budget) < 1.0

    @pytest.mark.asyncio
    async def test_insights_are_generated(self, targeting_module, sample_request):
        """Test that insights are generated."""
        response = await targeting_module.get_targeting_recommendations(sample_request)

        assert len(response.insights) > 0

    @pytest.mark.asyncio
    async def test_tech_product_matches_tech_segments(
        self, targeting_module, sample_request
    ):
        """Test that tech products match tech-related segments."""
        response = await targeting_module.get_targeting_recommendations(sample_request)

        segment_names = [s.name.lower() for s in response.segments]
        assert any("tech" in name for name in segment_names)

    @pytest.mark.asyncio
    async def test_conversion_objective_prioritizes_cvr(
        self, targeting_module, sample_request
    ):
        """Test that conversion objective prioritizes conversion rate."""
        sample_request.campaign_objective = "conversions"
        response = await targeting_module.get_targeting_recommendations(sample_request)

        # Check that insights mention conversions
        insights_text = " ".join(response.insights).lower()
        assert "conversion" in insights_text or "converting" in insights_text


class TestCTRPrediction:
    """Tests for CTR prediction functionality."""

    @pytest.mark.asyncio
    async def test_predict_ctr_returns_valid_range(self, targeting_module):
        """Test that CTR prediction returns value in valid range."""
        features = {
            "device_type": "mobile",
            "time_of_day": 20,
            "is_weekend": True,
        }

        ctr, (low, high) = await targeting_module.predict_ctr(features)

        assert 0 <= ctr <= 1
        assert low <= ctr <= high

    @pytest.mark.asyncio
    async def test_mobile_has_higher_ctr(self, targeting_module):
        """Test that mobile devices tend to have higher CTR."""
        mobile_features = {"device_type": "mobile"}
        desktop_features = {"device_type": "desktop"}

        mobile_ctr, _ = await targeting_module.predict_ctr(mobile_features)
        desktop_ctr, _ = await targeting_module.predict_ctr(desktop_features)

        # Mobile typically has higher CTR
        assert mobile_ctr >= desktop_ctr * 0.9  # Allow some variance

    @pytest.mark.asyncio
    async def test_prime_time_boost(self, targeting_module):
        """Test that prime time hours boost CTR."""
        prime_features = {"time_of_day": 20}  # 8 PM
        off_peak_features = {"time_of_day": 4}  # 4 AM

        prime_ctr, _ = await targeting_module.predict_ctr(prime_features)
        off_peak_ctr, _ = await targeting_module.predict_ctr(off_peak_features)

        assert prime_ctr > off_peak_ctr


class TestSegmentMatching:
    """Tests for segment matching logic."""

    @pytest.mark.asyncio
    async def test_fitness_product_matches_health_segment(self, targeting_module):
        """Test that fitness products match health segments."""
        request = AudienceTargetingRequest(
            product_category="fitness",
            product_description="Smart fitness tracker with heart rate monitoring",
            campaign_objective="conversions",
            budget=1000.0,
        )

        response = await targeting_module.get_targeting_recommendations(request)

        segment_names = [s.name.lower() for s in response.segments]
        assert any("health" in name or "fitness" in name for name in segment_names)

    @pytest.mark.asyncio
    async def test_empty_request_returns_segments(self, targeting_module):
        """Test that minimal request still returns segments."""
        request = AudienceTargetingRequest(
            product_category="general",
            product_description="A product",
            campaign_objective="awareness",
            budget=100.0,
        )

        response = await targeting_module.get_targeting_recommendations(request)

        assert len(response.segments) > 0
