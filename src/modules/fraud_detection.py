"""
Fraud Detection Module for AdTech LLM.

Analyzes ad interactions to detect fraudulent activity including
bot traffic, click farms, and invalid traffic patterns.
"""

import hashlib
import re
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import numpy as np

from src.config.settings import settings
from src.data.schemas import (
    FraudCheckType,
    FraudDetectionRequest,
    FraudDetectionResponse,
    FraudIndicator,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FraudDetectionModule:
    """
    Module for detecting ad fraud and invalid traffic.

    Features:
    - Bot detection
    - Click fraud analysis
    - Invalid traffic identification
    - Behavioral anomaly detection
    - IP reputation checking
    """

    def __init__(self):
        """Initialize the fraud detection module."""
        self.logger = get_logger("FraudDetectionModule")
        self._known_bot_patterns = self._load_bot_patterns()
        self._suspicious_ip_ranges = self._load_suspicious_ips()
        self._session_cache: dict[str, list[dict]] = {}

    def _load_bot_patterns(self) -> list[re.Pattern]:
        """Load known bot user agent patterns."""
        patterns = [
            r"bot|crawler|spider|scraper",
            r"headless|phantom|selenium|puppeteer",
            r"curl|wget|python-requests|httpie",
            r"facebookexternalhit|twitterbot|linkedinbot",
            r"googlebot|bingbot|yandexbot|baiduspider",
            r"ahrefsbot|semrushbot|mj12bot",
        ]
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def _load_suspicious_ips(self) -> set[str]:
        """Load known suspicious IP ranges (hashed for privacy)."""
        # In production, this would load from a database or service
        return set()

    async def analyze(
        self,
        request: FraudDetectionRequest,
    ) -> FraudDetectionResponse:
        """
        Analyze an interaction for fraud.

        Args:
            request: Fraud detection request

        Returns:
            Fraud analysis response
        """
        start_time = datetime.utcnow()
        request_id = str(uuid4())

        self.logger.info(
            "Analyzing for fraud",
            event_type=request.event_type,
            checks=request.check_types,
        )

        indicators = []
        fraud_scores = []

        # Run requested checks
        for check_type in request.check_types:
            check_result = await self._run_check(check_type, request)
            if check_result:
                indicators.extend(check_result["indicators"])
                fraud_scores.append(check_result["score"])

        # Calculate overall fraud score
        if fraud_scores:
            overall_score = np.mean(fraud_scores) * 0.7 + max(fraud_scores) * 0.3
        else:
            overall_score = 0.0

        # Determine risk level
        risk_level = self._determine_risk_level(overall_score, indicators)

        # Determine recommended action
        action = self._determine_action(risk_level, overall_score)

        analysis_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return FraudDetectionResponse(
            request_id=request_id,
            is_fraudulent=overall_score > 0.7,
            fraud_score=round(overall_score, 4),
            risk_level=risk_level,
            indicators=indicators,
            recommended_action=action,
            analysis_time_ms=analysis_time,
            model_version="fraud_detector_v1.0",
        )

    async def _run_check(
        self,
        check_type: FraudCheckType,
        request: FraudDetectionRequest,
    ) -> dict[str, Any] | None:
        """Run a specific fraud check."""
        if check_type == FraudCheckType.BOT_DETECTION:
            return await self._check_bot(request)
        elif check_type == FraudCheckType.CLICK_FRAUD:
            return await self._check_click_fraud(request)
        elif check_type == FraudCheckType.IMPRESSION_FRAUD:
            return await self._check_impression_fraud(request)
        elif check_type == FraudCheckType.INVALID_TRAFFIC:
            return await self._check_invalid_traffic(request)
        elif check_type == FraudCheckType.ATTRIBUTION_FRAUD:
            return await self._check_attribution_fraud(request)
        return None

    async def _check_bot(self, request: FraudDetectionRequest) -> dict[str, Any]:
        """Check for bot traffic."""
        indicators = []
        score = 0.0

        # Check user agent
        if request.user_agent:
            ua_lower = request.user_agent.lower()

            # Check against known bot patterns
            for pattern in self._known_bot_patterns:
                if pattern.search(ua_lower):
                    indicators.append(
                        FraudIndicator(
                            indicator_type="bot_user_agent",
                            severity="high",
                            confidence=0.9,
                            description=f"User agent matches known bot pattern",
                            evidence={"pattern_matched": pattern.pattern},
                        )
                    )
                    score = max(score, 0.9)
                    break

            # Check for missing/suspicious UA characteristics
            if len(request.user_agent) < 20:
                indicators.append(
                    FraudIndicator(
                        indicator_type="short_user_agent",
                        severity="medium",
                        confidence=0.6,
                        description="Unusually short user agent string",
                    )
                )
                score = max(score, 0.5)

            # Check for headless browser indicators
            headless_indicators = ["headless", "phantom", "selenium"]
            if any(h in ua_lower for h in headless_indicators):
                indicators.append(
                    FraudIndicator(
                        indicator_type="headless_browser",
                        severity="high",
                        confidence=0.85,
                        description="Headless browser detected",
                    )
                )
                score = max(score, 0.85)

        # Check device fingerprint
        if request.device_fingerprint:
            # Check for known fake fingerprints
            if self._is_suspicious_fingerprint(request.device_fingerprint):
                indicators.append(
                    FraudIndicator(
                        indicator_type="suspicious_fingerprint",
                        severity="medium",
                        confidence=0.7,
                        description="Device fingerprint shows suspicious characteristics",
                    )
                )
                score = max(score, 0.6)

        # Check behavioral signals
        if request.behavioral_signals:
            signals = request.behavioral_signals

            # Check for non-human behavior
            if signals.get("mouse_movements", 0) == 0:
                indicators.append(
                    FraudIndicator(
                        indicator_type="no_mouse_movement",
                        severity="medium",
                        confidence=0.5,
                        description="No mouse movement detected",
                    )
                )
                score = max(score, 0.4)

            if signals.get("time_on_page", 0) < 1:
                indicators.append(
                    FraudIndicator(
                        indicator_type="instant_interaction",
                        severity="high",
                        confidence=0.8,
                        description="Interaction occurred too quickly to be human",
                    )
                )
                score = max(score, 0.75)

        return {"indicators": indicators, "score": score}

    async def _check_click_fraud(self, request: FraudDetectionRequest) -> dict[str, Any]:
        """Check for click fraud patterns."""
        indicators = []
        score = 0.0

        # Analyze session data for click patterns
        if request.session_data:
            session = request.session_data

            # Check click frequency
            clicks_in_session = session.get("click_count", 0)
            session_duration = session.get("duration_seconds", 1)

            if session_duration > 0:
                clicks_per_minute = (clicks_in_session / session_duration) * 60
                if clicks_per_minute > 10:
                    indicators.append(
                        FraudIndicator(
                            indicator_type="high_click_frequency",
                            severity="high",
                            confidence=0.85,
                            description=f"Abnormally high click rate: {clicks_per_minute:.1f}/min",
                            evidence={"clicks_per_minute": clicks_per_minute},
                        )
                    )
                    score = max(score, 0.8)

            # Check for click patterns
            click_timestamps = session.get("click_timestamps", [])
            if len(click_timestamps) >= 3:
                # Check for regular intervals (bot behavior)
                if self._has_regular_intervals(click_timestamps):
                    indicators.append(
                        FraudIndicator(
                            indicator_type="regular_click_intervals",
                            severity="high",
                            confidence=0.9,
                            description="Clicks occur at suspiciously regular intervals",
                        )
                    )
                    score = max(score, 0.85)

        # Check IP for click farm patterns
        if request.ip_address:
            ip_hash = hashlib.sha256(request.ip_address.encode()).hexdigest()[:16]

            # Check for multiple clicks from same IP
            if self._check_ip_click_history(ip_hash):
                indicators.append(
                    FraudIndicator(
                        indicator_type="excessive_ip_clicks",
                        severity="high",
                        confidence=0.8,
                        description="Excessive clicks from this IP address",
                    )
                )
                score = max(score, 0.75)

        return {"indicators": indicators, "score": score}

    async def _check_impression_fraud(self, request: FraudDetectionRequest) -> dict[str, Any]:
        """Check for impression fraud."""
        indicators = []
        score = 0.0

        if request.session_data:
            session = request.session_data

            # Check for ad stacking (multiple ads in same position)
            if session.get("visible_area_percentage", 100) < 50:
                indicators.append(
                    FraudIndicator(
                        indicator_type="low_viewability",
                        severity="medium",
                        confidence=0.7,
                        description="Ad not sufficiently visible to user",
                        evidence={"visible_percentage": session.get("visible_area_percentage")},
                    )
                )
                score = max(score, 0.5)

            # Check for pixel stuffing
            if session.get("ad_size", {}).get("width", 300) < 10:
                indicators.append(
                    FraudIndicator(
                        indicator_type="pixel_stuffing",
                        severity="critical",
                        confidence=0.95,
                        description="Ad rendered at extremely small size",
                    )
                )
                score = max(score, 0.95)

            # Check impression timing
            if session.get("time_to_impression_ms", 1000) < 50:
                indicators.append(
                    FraudIndicator(
                        indicator_type="instant_impression",
                        severity="medium",
                        confidence=0.6,
                        description="Impression registered too quickly",
                    )
                )
                score = max(score, 0.5)

        return {"indicators": indicators, "score": score}

    async def _check_invalid_traffic(self, request: FraudDetectionRequest) -> dict[str, Any]:
        """Check for general invalid traffic (IVT)."""
        indicators = []
        score = 0.0

        # Check for datacenter IPs
        if request.ip_address:
            if self._is_datacenter_ip(request.ip_address):
                indicators.append(
                    FraudIndicator(
                        indicator_type="datacenter_ip",
                        severity="high",
                        confidence=0.8,
                        description="Traffic from datacenter IP (likely non-human)",
                    )
                )
                score = max(score, 0.7)

        # Check for proxy/VPN
        if request.behavioral_signals:
            if request.behavioral_signals.get("is_proxy"):
                indicators.append(
                    FraudIndicator(
                        indicator_type="proxy_detected",
                        severity="medium",
                        confidence=0.7,
                        description="Traffic routed through proxy or VPN",
                    )
                )
                score = max(score, 0.5)

        # Check geographic anomalies
        if request.session_data:
            claimed_country = request.session_data.get("claimed_country")
            ip_country = request.session_data.get("ip_country")
            if claimed_country and ip_country and claimed_country != ip_country:
                indicators.append(
                    FraudIndicator(
                        indicator_type="geo_mismatch",
                        severity="medium",
                        confidence=0.6,
                        description="Geographic location mismatch",
                        evidence={
                            "claimed": claimed_country,
                            "detected": ip_country,
                        },
                    )
                )
                score = max(score, 0.4)

        return {"indicators": indicators, "score": score}

    async def _check_attribution_fraud(self, request: FraudDetectionRequest) -> dict[str, Any]:
        """Check for attribution fraud (click injection, etc.)."""
        indicators = []
        score = 0.0

        if request.session_data:
            session = request.session_data

            # Check for click injection
            if session.get("click_to_install_time_ms", 10000) < 1000:
                indicators.append(
                    FraudIndicator(
                        indicator_type="click_injection",
                        severity="critical",
                        confidence=0.9,
                        description="Suspiciously short click-to-install time",
                        evidence={
                            "click_to_install_ms": session.get("click_to_install_time_ms")
                        },
                    )
                )
                score = max(score, 0.9)

            # Check for click spamming
            clicks_before_install = session.get("clicks_before_install", 0)
            if clicks_before_install > 10:
                indicators.append(
                    FraudIndicator(
                        indicator_type="click_spamming",
                        severity="high",
                        confidence=0.85,
                        description="Excessive clicks before conversion",
                        evidence={"click_count": clicks_before_install},
                    )
                )
                score = max(score, 0.8)

        return {"indicators": indicators, "score": score}

    def _is_suspicious_fingerprint(self, fingerprint: str) -> bool:
        """Check if device fingerprint is suspicious."""
        # Check for known emulator fingerprints
        suspicious_patterns = ["generic", "emulator", "sdk_gphone", "vbox"]
        return any(p in fingerprint.lower() for p in suspicious_patterns)

    def _has_regular_intervals(self, timestamps: list[float]) -> bool:
        """Check if timestamps have suspiciously regular intervals."""
        if len(timestamps) < 3:
            return False

        intervals = [
            timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
        ]

        # Check coefficient of variation
        if np.std(intervals) == 0:
            return True

        cv = np.std(intervals) / np.mean(intervals)
        return cv < 0.1  # Very low variation suggests automated clicks

    def _check_ip_click_history(self, ip_hash: str) -> bool:
        """Check IP click history for suspicious patterns."""
        # In production, this would query a database
        # For now, return False (no suspicious history)
        return False

    def _is_datacenter_ip(self, ip_address: str) -> bool:
        """Check if IP belongs to a known datacenter."""
        # Simplified check - in production would use IP intelligence service
        datacenter_patterns = ["10.", "172.16.", "192.168."]
        return any(ip_address.startswith(p) for p in datacenter_patterns)

    def _determine_risk_level(
        self,
        score: float,
        indicators: list[FraudIndicator],
    ) -> str:
        """Determine overall risk level."""
        if score >= 0.9:
            return "critical"
        elif score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "low"
        else:
            return "safe"

    def _determine_action(self, risk_level: str, score: float) -> str:
        """Determine recommended action based on risk."""
        if risk_level == "critical":
            return "block"
        elif risk_level == "high":
            return "block"
        elif risk_level == "medium":
            return "flag"
        elif risk_level == "low":
            return "flag"
        else:
            return "allow"

    async def batch_analyze(
        self,
        requests: list[FraudDetectionRequest],
    ) -> list[FraudDetectionResponse]:
        """Analyze multiple interactions in batch."""
        import asyncio
        tasks = [self.analyze(req) for req in requests]
        return await asyncio.gather(*tasks)
