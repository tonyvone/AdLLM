"""Unit tests for Ad Creative Generation module."""

import pytest
from unittest.mock import AsyncMock, patch

from src.modules.creative_generation import AdCreativeGenerator
from src.data.schemas import (
    AdCreativeRequest,
    AdCreativeResponse,
    AdFormatEnum,
    AdToneEnum,
)


@pytest.fixture
def generator():
    """Create a generator instance for testing."""
    return AdCreativeGenerator()


@pytest.fixture
def sample_request():
    """Create a sample ad creative request."""
    return AdCreativeRequest(
        product_name="Test Product",
        product_description="A great test product for testing purposes.",
        target_audience={"age_range": "25-45", "interests": ["technology"]},
        ad_format=AdFormatEnum.DISPLAY,
        tone=AdToneEnum.PROFESSIONAL,
        keywords=["test", "product", "quality"],
        num_variants=3,
        max_headline_length=90,
        max_description_length=300,
        include_cta=True,
    )


class TestAdCreativeGenerator:
    """Tests for AdCreativeGenerator class."""

    @pytest.mark.asyncio
    async def test_generate_returns_response(self, generator, sample_request):
        """Test that generate returns a valid response."""
        response = await generator.generate(sample_request)

        assert isinstance(response, AdCreativeResponse)
        assert response.product_name == sample_request.product_name
        assert len(response.variants) > 0
        assert response.generation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_respects_num_variants(self, generator, sample_request):
        """Test that generate returns the requested number of variants."""
        sample_request.num_variants = 5
        response = await generator.generate(sample_request)

        assert len(response.variants) <= sample_request.num_variants

    @pytest.mark.asyncio
    async def test_variants_have_required_fields(self, generator, sample_request):
        """Test that each variant has required fields."""
        response = await generator.generate(sample_request)

        for variant in response.variants:
            assert variant.variant_id is not None
            assert variant.headline is not None
            assert variant.description is not None
            assert 0 <= variant.confidence_score <= 1

    @pytest.mark.asyncio
    async def test_headline_respects_max_length(self, generator, sample_request):
        """Test that headlines respect maximum length."""
        sample_request.max_headline_length = 50
        response = await generator.generate(sample_request)

        for variant in response.variants:
            assert len(variant.headline) <= sample_request.max_headline_length

    @pytest.mark.asyncio
    async def test_description_respects_max_length(self, generator, sample_request):
        """Test that descriptions respect maximum length."""
        sample_request.max_description_length = 150
        response = await generator.generate(sample_request)

        for variant in response.variants:
            assert len(variant.description) <= sample_request.max_description_length

    @pytest.mark.asyncio
    async def test_cta_included_when_requested(self, generator, sample_request):
        """Test that CTA is included when requested."""
        sample_request.include_cta = True
        response = await generator.generate(sample_request)

        # At least one variant should have CTA
        has_cta = any(v.cta_text is not None for v in response.variants)
        assert has_cta

    @pytest.mark.asyncio
    async def test_batch_generation(self, generator):
        """Test batch generation of multiple requests."""
        requests = [
            AdCreativeRequest(
                product_name=f"Product {i}",
                product_description=f"Description for product {i}",
                num_variants=2,
            )
            for i in range(3)
        ]

        responses = await generator.generate_batch(requests)

        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert response.product_name == f"Product {i}"


class TestPromptBuilding:
    """Tests for prompt building functionality."""

    def test_format_audience_with_data(self, generator):
        """Test audience formatting with data."""
        audience = {
            "age_range": "25-35",
            "interests": ["tech", "gaming"],
            "location": "USA",
        }
        formatted = generator._format_audience(audience)

        assert "25-35" in formatted
        assert "tech" in formatted
        assert "USA" in formatted

    def test_format_audience_empty(self, generator):
        """Test audience formatting with empty data."""
        formatted = generator._format_audience({})
        assert formatted == "General audience"


class TestVariantScoring:
    """Tests for variant scoring functionality."""

    @pytest.mark.asyncio
    async def test_variants_are_scored(self, generator, sample_request):
        """Test that variants receive confidence scores."""
        response = await generator.generate(sample_request)

        for variant in response.variants:
            assert variant.confidence_score is not None
            assert 0 <= variant.confidence_score <= 1

    @pytest.mark.asyncio
    async def test_variants_sorted_by_score(self, generator, sample_request):
        """Test that variants are sorted by confidence score."""
        response = await generator.generate(sample_request)

        scores = [v.confidence_score for v in response.variants]
        assert scores == sorted(scores, reverse=True)
