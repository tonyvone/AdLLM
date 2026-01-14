"""Integration tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_readiness_check(self, client):
        """Test readiness check."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "checks" in data

    def test_liveness_check(self, client):
        """Test liveness check."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestAdCreativeEndpoints:
    """Tests for ad creative endpoints."""

    def test_generate_ad_creative(self, client):
        """Test ad creative generation."""
        request_data = {
            "product_name": "Test Product",
            "product_description": "A great product for testing",
            "num_variants": 3,
            "tone": "professional",
        }

        response = client.post("/api/v1/ads/generate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "variants" in data
        assert len(data["variants"]) > 0

    def test_get_templates(self, client):
        """Test getting ad templates."""
        response = client.get("/api/v1/ads/templates")
        assert response.status_code == 200
        data = response.json()
        assert "formats" in data
        assert "tones" in data


class TestAudienceEndpoints:
    """Tests for audience targeting endpoints."""

    def test_get_audience_recommendations(self, client):
        """Test audience recommendation."""
        request_data = {
            "product_category": "technology",
            "product_description": "Smart home device",
            "campaign_objective": "conversions",
            "budget": 1000.0,
        }

        response = client.post("/api/v1/audiences/recommendations", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "segments" in data
        assert "total_estimated_reach" in data

    def test_list_segments(self, client):
        """Test listing audience segments."""
        response = client.get("/api/v1/audiences/segments")
        assert response.status_code == 200
        data = response.json()
        assert "segments" in data

    def test_predict_ctr(self, client):
        """Test CTR prediction."""
        request_data = {
            "creative_features": {"format": "display"},
            "audience_features": {"device_type": "mobile"},
            "context_features": {"hour_of_day": 20},
        }

        response = client.post("/api/v1/audiences/predict-ctr", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_ctr" in data
        assert 0 <= data["predicted_ctr"] <= 1


class TestFraudEndpoints:
    """Tests for fraud detection endpoints."""

    def test_analyze_fraud(self, client):
        """Test fraud analysis."""
        request_data = {
            "event_type": "click",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "check_types": ["bot_detection", "click_fraud"],
        }

        response = client.post("/api/v1/fraud/analyze", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "is_fraudulent" in data
        assert "fraud_score" in data
        assert "risk_level" in data

    def test_get_fraud_stats(self, client):
        """Test getting fraud statistics."""
        response = client.get("/api/v1/fraud/stats")
        assert response.status_code == 200
        data = response.json()
        assert "overall_fraud_rate" in data

    def test_list_check_types(self, client):
        """Test listing fraud check types."""
        response = client.get("/api/v1/fraud/check-types")
        assert response.status_code == 200
        data = response.json()
        assert "check_types" in data


class TestBidEndpoints:
    """Tests for bid optimization endpoints."""

    def test_optimize_bid(self, client):
        """Test bid optimization."""
        request_data = {
            "campaign_id": "00000000-0000-0000-0000-000000000001",
            "ad_slot": {"size": {"width": 300, "height": 250}},
            "floor_price": 0.5,
            "max_bid": 5.0,
            "strategy": "auto_cpc",
        }

        response = client.post("/api/v1/bids/optimize", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "should_bid" in data
        assert "recommendations" in data

    def test_list_strategies(self, client):
        """Test listing bid strategies."""
        response = client.get("/api/v1/bids/strategies")
        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data


class TestContextualEndpoints:
    """Tests for contextual analysis endpoints."""

    def test_analyze_context(self, client):
        """Test contextual analysis."""
        request_data = {
            "content": "This is a technology news article about smartphones and gadgets.",
            "brand_safety_level": "standard",
        }

        response = client.post("/api/v1/contextual/analyze", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "topics" in data
        assert "sentiment" in data
        assert "brand_safety_score" in data

    def test_list_categories(self, client):
        """Test listing ad categories."""
        response = client.get("/api/v1/contextual/categories")
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
