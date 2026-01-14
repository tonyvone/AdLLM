"""Unit tests for Fraud Detection module."""

import pytest
from datetime import datetime

from src.modules.fraud_detection import FraudDetectionModule
from src.data.schemas import FraudCheckType, FraudDetectionRequest


@pytest.fixture
def fraud_module():
    """Create a fraud detection module instance."""
    return FraudDetectionModule()


@pytest.fixture
def sample_request():
    """Create a sample fraud detection request."""
    return FraudDetectionRequest(
        event_type="click",
        ip_address="192.168.1.1",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        timestamp=datetime.utcnow(),
        check_types=[FraudCheckType.BOT_DETECTION, FraudCheckType.CLICK_FRAUD],
    )


class TestFraudDetectionModule:
    """Tests for FraudDetectionModule class."""

    @pytest.mark.asyncio
    async def test_analyze_returns_response(self, fraud_module, sample_request):
        """Test that analyze returns a valid response."""
        response = await fraud_module.analyze(sample_request)

        assert response.request_id is not None
        assert isinstance(response.is_fraudulent, bool)
        assert 0 <= response.fraud_score <= 1
        assert response.risk_level in ["safe", "low", "medium", "high", "critical"]
        assert response.recommended_action in ["allow", "flag", "block"]

    @pytest.mark.asyncio
    async def test_bot_detection_catches_bots(self, fraud_module):
        """Test that bot detection catches known bot patterns."""
        request = FraudDetectionRequest(
            event_type="impression",
            user_agent="Mozilla/5.0 (compatible; Googlebot/2.1)",
            check_types=[FraudCheckType.BOT_DETECTION],
        )

        response = await fraud_module.analyze(request)

        assert response.fraud_score > 0.5
        assert any(i.indicator_type == "bot_user_agent" for i in response.indicators)

    @pytest.mark.asyncio
    async def test_headless_browser_detection(self, fraud_module):
        """Test detection of headless browsers."""
        request = FraudDetectionRequest(
            event_type="click",
            user_agent="Mozilla/5.0 HeadlessChrome/91.0.4472.124",
            check_types=[FraudCheckType.BOT_DETECTION],
        )

        response = await fraud_module.analyze(request)

        assert response.fraud_score > 0.5
        assert any(i.indicator_type == "headless_browser" for i in response.indicators)

    @pytest.mark.asyncio
    async def test_legitimate_traffic_passes(self, fraud_module, sample_request):
        """Test that legitimate traffic is not flagged."""
        # Use realistic user agent and behavioral signals
        sample_request.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        sample_request.behavioral_signals = {
            "mouse_movements": 50,
            "time_on_page": 30,
        }

        response = await fraud_module.analyze(sample_request)

        assert response.recommended_action != "block"

    @pytest.mark.asyncio
    async def test_click_fraud_detection(self, fraud_module):
        """Test click fraud detection with suspicious patterns."""
        request = FraudDetectionRequest(
            event_type="click",
            session_data={
                "click_count": 100,
                "duration_seconds": 60,
                "click_timestamps": [1.0, 2.0, 3.0, 4.0, 5.0],  # Regular intervals
            },
            check_types=[FraudCheckType.CLICK_FRAUD],
        )

        response = await fraud_module.analyze(request)

        assert response.fraud_score > 0.5
        has_click_indicator = any(
            "click" in i.indicator_type.lower() for i in response.indicators
        )
        assert has_click_indicator

    @pytest.mark.asyncio
    async def test_instant_interaction_detection(self, fraud_module):
        """Test detection of impossibly fast interactions."""
        request = FraudDetectionRequest(
            event_type="click",
            behavioral_signals={"time_on_page": 0.1},  # Too fast
            check_types=[FraudCheckType.BOT_DETECTION],
        )

        response = await fraud_module.analyze(request)

        assert any(i.indicator_type == "instant_interaction" for i in response.indicators)

    @pytest.mark.asyncio
    async def test_batch_analysis(self, fraud_module):
        """Test batch fraud analysis."""
        requests = [
            FraudDetectionRequest(
                event_type="click",
                user_agent=f"User-Agent-{i}",
                check_types=[FraudCheckType.BOT_DETECTION],
            )
            for i in range(5)
        ]

        responses = await fraud_module.batch_analyze(requests)

        assert len(responses) == 5
        for response in responses:
            assert response.request_id is not None


class TestRiskLevelDetermination:
    """Tests for risk level determination."""

    @pytest.mark.asyncio
    async def test_high_score_critical_level(self, fraud_module):
        """Test that very high scores result in critical risk level."""
        request = FraudDetectionRequest(
            event_type="click",
            user_agent="curl/7.68.0",  # Clear bot
            check_types=[FraudCheckType.BOT_DETECTION],
        )

        response = await fraud_module.analyze(request)

        assert response.risk_level in ["high", "critical"]
        assert response.recommended_action == "block"

    @pytest.mark.asyncio
    async def test_low_score_safe_level(self, fraud_module):
        """Test that low scores result in safe risk level."""
        request = FraudDetectionRequest(
            event_type="impression",
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            behavioral_signals={
                "mouse_movements": 100,
                "time_on_page": 45,
            },
            check_types=[FraudCheckType.BOT_DETECTION],
        )

        response = await fraud_module.analyze(request)

        assert response.risk_level in ["safe", "low"]
