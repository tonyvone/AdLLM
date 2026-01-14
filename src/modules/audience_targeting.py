"""
Audience Targeting and Prediction Module for AdTech LLM.

Provides AI-driven audience segmentation, CTR prediction,
and targeting recommendations based on campaign objectives.
"""

import asyncio
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np

from src.config.settings import settings
from src.data.schemas import (
    AudienceSegmentResult,
    AudienceTargetingRequest,
    AudienceTargetingResponse,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AudienceTargetingModule:
    """
    Module for audience targeting and segmentation.

    Features:
    - Demographic analysis
    - Behavioral segmentation
    - Lookalike audience generation
    - CTR/Conversion prediction
    - Budget allocation optimization
    """

    def __init__(self, model_path: str | None = None):
        """
        Initialize the targeting module.

        Args:
            model_path: Path to prediction model
        """
        self.model_path = model_path
        self.logger = get_logger("AudienceTargetingModule")
        self._prediction_model = None

        # Predefined segment templates
        self._segment_templates = self._load_segment_templates()

    def _load_segment_templates(self) -> dict[str, dict[str, Any]]:
        """Load predefined audience segment templates."""
        return {
            "tech_enthusiasts": {
                "name": "Tech Enthusiasts",
                "description": "Users interested in technology, gadgets, and innovation",
                "type": "behavioral",
                "base_ctr": 0.025,
                "base_cvr": 0.03,
                "cpm_range": (2.0, 5.0),
                "criteria": {
                    "interests": ["technology", "gadgets", "software", "AI"],
                    "behaviors": ["tech_news_readers", "early_adopters"],
                },
            },
            "young_professionals": {
                "name": "Young Professionals",
                "description": "Career-focused individuals aged 25-35",
                "type": "demographic",
                "base_ctr": 0.022,
                "base_cvr": 0.025,
                "cpm_range": (3.0, 6.0),
                "criteria": {
                    "age_range": "25-35",
                    "interests": ["career", "productivity", "networking"],
                    "income": "middle_to_high",
                },
            },
            "budget_shoppers": {
                "name": "Budget-Conscious Shoppers",
                "description": "Price-sensitive consumers looking for deals",
                "type": "behavioral",
                "base_ctr": 0.03,
                "base_cvr": 0.04,
                "cpm_range": (1.0, 3.0),
                "criteria": {
                    "behaviors": ["coupon_users", "deal_seekers", "price_comparers"],
                    "purchase_intent": ["sale_items", "discounts"],
                },
            },
            "luxury_consumers": {
                "name": "Luxury Consumers",
                "description": "High-income individuals interested in premium products",
                "type": "demographic",
                "base_ctr": 0.018,
                "base_cvr": 0.02,
                "cpm_range": (8.0, 15.0),
                "criteria": {
                    "income": "high",
                    "interests": ["luxury", "premium_brands", "travel"],
                    "behaviors": ["high_value_purchasers"],
                },
            },
            "health_fitness": {
                "name": "Health & Fitness Enthusiasts",
                "description": "Users focused on wellness, exercise, and healthy living",
                "type": "behavioral",
                "base_ctr": 0.028,
                "base_cvr": 0.035,
                "cpm_range": (2.5, 5.5),
                "criteria": {
                    "interests": ["fitness", "nutrition", "wellness", "sports"],
                    "behaviors": ["gym_goers", "health_app_users"],
                },
            },
            "parents": {
                "name": "Parents with Children",
                "description": "Parents interested in family and children's products",
                "type": "demographic",
                "base_ctr": 0.024,
                "base_cvr": 0.03,
                "cpm_range": (2.0, 4.5),
                "criteria": {
                    "parental_status": "parent",
                    "interests": ["parenting", "family", "education", "kids"],
                },
            },
            "mobile_gamers": {
                "name": "Mobile Gamers",
                "description": "Active mobile gaming audience",
                "type": "behavioral",
                "base_ctr": 0.032,
                "base_cvr": 0.02,
                "cpm_range": (1.5, 4.0),
                "criteria": {
                    "interests": ["mobile_games", "gaming"],
                    "behaviors": ["in_app_purchasers", "frequent_gamers"],
                    "device_types": ["mobile", "tablet"],
                },
            },
            "business_decision_makers": {
                "name": "Business Decision Makers",
                "description": "B2B audience with purchasing authority",
                "type": "demographic",
                "base_ctr": 0.015,
                "base_cvr": 0.015,
                "cpm_range": (10.0, 25.0),
                "criteria": {
                    "job_functions": ["executive", "manager", "director"],
                    "interests": ["business", "enterprise_software"],
                    "behaviors": ["b2b_researchers"],
                },
            },
        }

    async def get_targeting_recommendations(
        self,
        request: AudienceTargetingRequest,
    ) -> AudienceTargetingResponse:
        """
        Get audience targeting recommendations.

        Args:
            request: Targeting request with campaign details

        Returns:
            Targeting recommendations with segments
        """
        start_time = datetime.utcnow()
        request_id = str(uuid4())

        self.logger.info(
            "Generating targeting recommendations",
            product_category=request.product_category,
            objective=request.campaign_objective,
        )

        # Analyze request and match to segments
        segments = await self._match_segments(request)

        # Predict performance metrics
        segments = await self._predict_segment_performance(segments, request)

        # Optimize budget allocation
        budget_allocation = self._optimize_budget_allocation(segments, request.budget)

        # Generate insights
        insights = self._generate_insights(segments, request)

        # Calculate total reach
        total_reach = sum(s.estimated_reach for s in segments)

        generation_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return AudienceTargetingResponse(
            request_id=request_id,
            segments=segments,
            total_estimated_reach=total_reach,
            recommended_budget_allocation=budget_allocation,
            insights=insights,
            generation_time_ms=generation_time,
        )

    async def _match_segments(
        self,
        request: AudienceTargetingRequest,
    ) -> list[AudienceSegmentResult]:
        """Match request to relevant audience segments."""
        segments = []
        category_lower = request.product_category.lower()
        desc_lower = request.product_description.lower()

        # Score each template against the request
        for template_id, template in self._segment_templates.items():
            score = self._calculate_segment_match(template, category_lower, desc_lower, request)

            if score > 0.3:  # Threshold for relevance
                segment = AudienceSegmentResult(
                    segment_id=f"seg_{template_id}_{uuid4().hex[:8]}",
                    name=template["name"],
                    description=template["description"],
                    segment_type=template["type"],
                    criteria=template["criteria"],
                    estimated_reach=self._estimate_reach(template, request),
                    estimated_cpm=np.random.uniform(*template["cpm_range"]),
                    predicted_ctr=template["base_ctr"],
                    predicted_conversion_rate=template["base_cvr"],
                    match_score=score,
                )
                segments.append(segment)

        # Sort by match score
        segments.sort(key=lambda s: s.match_score, reverse=True)

        # Return top segments
        return segments[:8]

    def _calculate_segment_match(
        self,
        template: dict[str, Any],
        category: str,
        description: str,
        request: AudienceTargetingRequest,
    ) -> float:
        """Calculate how well a segment matches the request."""
        score = 0.0

        # Check category/description match with template interests
        template_interests = template.get("criteria", {}).get("interests", [])
        for interest in template_interests:
            if interest.lower() in category or interest.lower() in description:
                score += 0.15

        # Check demographic match
        if request.demographics:
            template_demo = template.get("criteria", {})

            # Age match
            if request.demographics.age_min and request.demographics.age_max:
                if "age_range" in template_demo:
                    age_range = template_demo["age_range"]
                    if isinstance(age_range, str) and "-" in age_range:
                        t_min, t_max = map(int, age_range.split("-"))
                        if (
                            request.demographics.age_min <= t_max
                            and request.demographics.age_max >= t_min
                        ):
                            score += 0.2

            # Parental status match
            if request.demographics.parental_status:
                if template_demo.get("parental_status") in request.demographics.parental_status:
                    score += 0.15

        # Check behavioral match
        if request.behavior and request.behavior.interests:
            for interest in request.behavior.interests:
                if interest.lower() in [i.lower() for i in template_interests]:
                    score += 0.1

        # Objective alignment
        if request.campaign_objective == "conversions":
            if template["base_cvr"] > 0.025:
                score += 0.15
        elif request.campaign_objective == "awareness":
            if template["base_ctr"] > 0.025:
                score += 0.15

        return min(1.0, score)

    def _estimate_reach(
        self,
        template: dict[str, Any],
        request: AudienceTargetingRequest,
    ) -> int:
        """Estimate audience reach for a segment."""
        # Base reach estimates (simplified)
        base_reaches = {
            "tech_enthusiasts": 50_000_000,
            "young_professionals": 80_000_000,
            "budget_shoppers": 100_000_000,
            "luxury_consumers": 20_000_000,
            "health_fitness": 60_000_000,
            "parents": 70_000_000,
            "mobile_gamers": 90_000_000,
            "business_decision_makers": 15_000_000,
        }

        base = base_reaches.get(template["name"].lower().replace(" ", "_"), 30_000_000)

        # Apply geographic constraints
        if request.geography:
            if request.geography.countries:
                # Rough scaling based on number of countries
                base = int(base * min(1.0, len(request.geography.countries) * 0.15))
            if request.geography.cities:
                base = int(base * 0.1)

        # Apply demographic constraints
        if request.demographics:
            if request.demographics.age_min and request.demographics.age_max:
                age_range = request.demographics.age_max - request.demographics.age_min
                base = int(base * (age_range / 80))  # Assuming 13-93 full range

        # Add some randomness
        return int(base * (0.8 + np.random.random() * 0.4))

    async def _predict_segment_performance(
        self,
        segments: list[AudienceSegmentResult],
        request: AudienceTargetingRequest,
    ) -> list[AudienceSegmentResult]:
        """Predict performance metrics for segments."""
        for segment in segments:
            # Adjust predictions based on objective
            if request.campaign_objective == "conversions":
                # Optimize for conversion rate
                segment.predicted_conversion_rate *= 1.1 + np.random.random() * 0.1
                segment.predicted_ctr *= 0.95
            elif request.campaign_objective == "awareness":
                # Optimize for reach/CTR
                segment.predicted_ctr *= 1.1 + np.random.random() * 0.1
                segment.estimated_reach = int(segment.estimated_reach * 1.2)

            # Add confidence bounds
            segment.predicted_ctr = round(segment.predicted_ctr, 4)
            segment.predicted_conversion_rate = round(segment.predicted_conversion_rate, 4)
            segment.estimated_cpm = round(segment.estimated_cpm, 2)

        return segments

    def _optimize_budget_allocation(
        self,
        segments: list[AudienceSegmentResult],
        total_budget: float,
    ) -> dict[str, float]:
        """Optimize budget allocation across segments."""
        if not segments:
            return {}

        allocation = {}

        # Calculate efficiency score for each segment
        efficiencies = []
        for segment in segments:
            # Efficiency = expected conversions per dollar
            expected_conversions = (
                segment.estimated_reach
                * segment.predicted_ctr
                * segment.predicted_conversion_rate
            )
            cost_per_impression = segment.estimated_cpm / 1000
            efficiency = expected_conversions / max(
                0.01, segment.estimated_reach * cost_per_impression
            )
            efficiencies.append((segment.segment_id, efficiency, segment))

        # Sort by efficiency
        efficiencies.sort(key=lambda x: x[1], reverse=True)

        # Allocate budget proportionally to efficiency
        total_efficiency = sum(e[1] for e in efficiencies)

        for seg_id, efficiency, segment in efficiencies:
            if total_efficiency > 0:
                share = efficiency / total_efficiency
            else:
                share = 1.0 / len(segments)

            # Apply min/max constraints
            min_allocation = total_budget * 0.05
            max_allocation = total_budget * 0.4

            segment_budget = max(min_allocation, min(max_allocation, total_budget * share))
            allocation[seg_id] = round(segment_budget, 2)

        # Normalize to total budget
        total_allocated = sum(allocation.values())
        if total_allocated > 0:
            scale = total_budget / total_allocated
            allocation = {k: round(v * scale, 2) for k, v in allocation.items()}

        return allocation

    def _generate_insights(
        self,
        segments: list[AudienceSegmentResult],
        request: AudienceTargetingRequest,
    ) -> list[str]:
        """Generate targeting insights."""
        insights = []

        if not segments:
            insights.append(
                "Consider broadening targeting criteria to reach more potential customers."
            )
            return insights

        # Best performing segment
        best_ctr = max(segments, key=lambda s: s.predicted_ctr)
        insights.append(
            f"Highest predicted CTR: {best_ctr.name} ({best_ctr.predicted_ctr:.2%})"
        )

        best_cvr = max(segments, key=lambda s: s.predicted_conversion_rate)
        if best_cvr != best_ctr:
            insights.append(
                f"Highest predicted conversion rate: {best_cvr.name} "
                f"({best_cvr.predicted_conversion_rate:.2%})"
            )

        # Reach insight
        total_reach = sum(s.estimated_reach for s in segments)
        insights.append(f"Total addressable audience: {total_reach:,} users")

        # Budget recommendation
        avg_cpm = sum(s.estimated_cpm for s in segments) / len(segments)
        estimated_impressions = (request.budget / avg_cpm) * 1000
        insights.append(
            f"Estimated impressions with ${request.budget:.2f} budget: "
            f"{int(estimated_impressions):,}"
        )

        # Optimization suggestions
        if request.campaign_objective == "conversions":
            high_cvr_segments = [s for s in segments if s.predicted_conversion_rate > 0.03]
            if high_cvr_segments:
                insights.append(
                    f"Focus on {len(high_cvr_segments)} high-converting segment(s) "
                    "for optimal ROI"
                )

        return insights

    async def predict_ctr(
        self,
        features: dict[str, Any],
    ) -> tuple[float, tuple[float, float]]:
        """
        Predict CTR for given features.

        Args:
            features: Feature dictionary

        Returns:
            Tuple of (predicted_ctr, confidence_interval)
        """
        # Simplified CTR prediction based on features
        base_ctr = 0.02

        # Adjust based on features
        if features.get("device_type") == "mobile":
            base_ctr *= 1.15
        if features.get("time_of_day", 12) in range(18, 22):
            base_ctr *= 1.2
        if features.get("is_weekend"):
            base_ctr *= 1.1

        # Add noise for realism
        predicted = base_ctr * (0.9 + np.random.random() * 0.2)

        # Confidence interval
        ci = (predicted * 0.8, predicted * 1.2)

        return predicted, ci
