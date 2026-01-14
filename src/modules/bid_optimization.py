"""
Bid Optimization Module for AdTech LLM.

Provides real-time bid optimization for RTB (Real-Time Bidding)
auctions using AI-driven decision making.
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from src.config.settings import settings
from src.data.schemas import (
    BidOptimizationRequest,
    BidOptimizationResponse,
    BidRecommendation,
    BidStrategyEnum,
)
from src.modules.ctr_prediction import CTRPredictionModule
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BidOptimizationModule:
    """
    Module for real-time bid optimization.

    Features:
    - Win rate prediction
    - Value-based bidding
    - Budget pacing
    - Multi-objective optimization
    - Auction dynamics modeling
    """

    def __init__(self, ctr_module: CTRPredictionModule | None = None):
        """
        Initialize the bid optimization module.

        Args:
            ctr_module: CTR prediction module for value estimation
        """
        self.logger = get_logger("BidOptimizationModule")
        self.ctr_module = ctr_module or CTRPredictionModule()

        # Auction parameters
        self._win_rate_model = None
        self._market_prices: dict[str, list[float]] = {}
        self._campaign_budgets: dict[str, dict[str, float]] = {}

    async def optimize(
        self,
        request: BidOptimizationRequest,
    ) -> BidOptimizationResponse:
        """
        Optimize bid for an auction opportunity.

        Args:
            request: Bid optimization request

        Returns:
            Bid recommendation response
        """
        start_time = datetime.utcnow()
        request_id = str(uuid4())

        self.logger.info(
            "Optimizing bid",
            campaign_id=str(request.campaign_id),
            strategy=request.strategy.value,
        )

        # Analyze auction opportunity
        auction_analysis = await self._analyze_auction(request)

        # Get value estimate
        value_estimate = await self._estimate_value(request)

        # Calculate optimal bid based on strategy
        recommendations = await self._calculate_bids(
            request,
            auction_analysis,
            value_estimate,
        )

        # Determine if we should bid
        should_bid = self._should_bid(recommendations, request, auction_analysis)

        latency = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return BidOptimizationResponse(
            request_id=request_id,
            should_bid=should_bid,
            recommendations=recommendations,
            auction_insights=auction_analysis,
            latency_ms=latency,
            model_version="bid_optimizer_v1.0",
        )

    async def _analyze_auction(
        self,
        request: BidOptimizationRequest,
    ) -> dict[str, Any]:
        """Analyze auction characteristics."""
        slot = request.ad_slot
        context = request.context_signals

        # Estimate market clearing price
        market_estimate = self._estimate_market_price(slot, context)

        # Calculate competition level
        competition = self._estimate_competition(slot, context)

        # Win probability function
        win_probs = self._calculate_win_probabilities(
            request.floor_price,
            request.max_bid,
            market_estimate,
        )

        return {
            "estimated_market_price": market_estimate,
            "competition_level": competition,
            "floor_price": request.floor_price,
            "win_probability_at_floor": win_probs["at_floor"],
            "win_probability_at_market": win_probs["at_market"],
            "slot_quality": self._assess_slot_quality(slot),
            "recommended_price_range": {
                "min": max(request.floor_price, market_estimate * 0.7),
                "max": min(request.max_bid, market_estimate * 1.5),
            },
        }

    def _estimate_market_price(
        self,
        slot: dict[str, Any],
        context: dict[str, Any],
    ) -> float:
        """Estimate market clearing price for this slot."""
        # Base price by slot characteristics
        base_price = 0.5  # $0.50 CPM base

        # Adjust by slot size
        slot_size = slot.get("size", {})
        width = slot_size.get("width", 300)
        height = slot_size.get("height", 250)

        size_premium = 1.0
        if width >= 728 or height >= 600:
            size_premium = 1.5
        elif width >= 300 and height >= 250:
            size_premium = 1.2

        # Adjust by position
        position = slot.get("position", "middle")
        position_premiums = {
            "above_fold": 1.8,
            "top": 1.5,
            "middle": 1.0,
            "bottom": 0.7,
            "below_fold": 0.5,
        }
        position_premium = position_premiums.get(position, 1.0)

        # Adjust by context
        page_category = context.get("page_category", "general")
        category_premiums = {
            "finance": 2.0,
            "technology": 1.5,
            "news": 1.3,
            "entertainment": 1.0,
            "gaming": 0.9,
        }
        category_premium = category_premiums.get(page_category, 1.0)

        # Time-based adjustment
        hour = context.get("hour_of_day", 12)
        time_premium = 1.0
        if 18 <= hour <= 22:  # Prime time
            time_premium = 1.3
        elif 2 <= hour <= 6:  # Late night
            time_premium = 0.6

        estimated_price = (
            base_price *
            size_premium *
            position_premium *
            category_premium *
            time_premium
        )

        # Add market noise
        noise = np.random.normal(0, estimated_price * 0.15)

        return max(0.1, round(estimated_price + noise, 4))

    def _estimate_competition(
        self,
        slot: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        """Estimate competition level for auction."""
        # High-value inventory has more competition
        position = slot.get("position", "middle")
        page_category = context.get("page_category", "general")

        score = 0.5

        if position == "above_fold":
            score += 0.2
        if page_category in ["finance", "technology"]:
            score += 0.15

        hour = context.get("hour_of_day", 12)
        if 9 <= hour <= 21:
            score += 0.1

        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"

    def _calculate_win_probabilities(
        self,
        floor_price: float,
        max_bid: float,
        market_price: float,
    ) -> dict[str, float]:
        """Calculate win probabilities at different price points."""
        # Simple logistic win probability model
        def win_prob(bid: float) -> float:
            if bid < floor_price:
                return 0.0
            ratio = bid / market_price
            prob = 1 / (1 + np.exp(-3 * (ratio - 1)))
            return round(min(0.95, prob), 3)

        return {
            "at_floor": win_prob(floor_price),
            "at_market": win_prob(market_price),
            "at_max": win_prob(max_bid),
        }

    def _assess_slot_quality(self, slot: dict[str, Any]) -> float:
        """Assess the quality of an ad slot."""
        score = 0.5

        # Position quality
        position = slot.get("position", "middle")
        position_scores = {
            "above_fold": 1.0,
            "top": 0.9,
            "middle": 0.7,
            "bottom": 0.5,
            "below_fold": 0.3,
        }
        score = position_scores.get(position, 0.5)

        # Size quality
        size = slot.get("size", {})
        width = size.get("width", 300)
        height = size.get("height", 250)

        if width >= 300 and height >= 250:
            score *= 1.1
        if width >= 728:
            score *= 1.2

        # Viewability
        viewability = slot.get("viewability_rate", 0.7)
        score *= (0.5 + viewability * 0.5)

        return round(min(1.0, score), 2)

    async def _estimate_value(
        self,
        request: BidOptimizationRequest,
    ) -> dict[str, float]:
        """Estimate value of winning this auction."""
        from src.data.schemas import CTRPredictionRequest

        # Get CTR prediction
        ctr_request = CTRPredictionRequest(
            creative_id=None,
            creative_features={},
            audience_features=request.user_signals,
            context_features=request.context_signals,
            historical_features=request.historical_performance,
        )

        ctr_response = await self.ctr_module.predict(ctr_request)
        predicted_ctr = ctr_response.predicted_ctr

        # Estimate conversion rate
        if request.historical_performance:
            cvr = request.historical_performance.get("avg_cvr", 0.02)
        else:
            cvr = 0.02  # Default 2% CVR

        # Calculate expected value
        if request.strategy == BidStrategyEnum.TARGET_CPA:
            target_cpa = request.target_cpa or 10.0
            expected_value = predicted_ctr * cvr * target_cpa
        elif request.strategy == BidStrategyEnum.TARGET_ROAS:
            target_roas = request.target_roas or 4.0
            avg_order_value = request.historical_performance.get("avg_order_value", 50.0) if request.historical_performance else 50.0
            expected_value = predicted_ctr * cvr * avg_order_value / target_roas
        else:
            # CPC-based value
            expected_value = predicted_ctr * 0.5  # $0.50 per click baseline

        return {
            "predicted_ctr": predicted_ctr,
            "predicted_cvr": cvr,
            "expected_value": round(expected_value, 4),
            "value_per_click": round(expected_value / max(0.001, predicted_ctr), 4),
        }

    async def _calculate_bids(
        self,
        request: BidOptimizationRequest,
        auction_analysis: dict[str, Any],
        value_estimate: dict[str, float],
    ) -> list[BidRecommendation]:
        """Calculate bid recommendations."""
        recommendations = []

        expected_value = value_estimate["expected_value"]
        market_price = auction_analysis["estimated_market_price"]
        floor_price = request.floor_price

        # Generate bid recommendation for each available creative
        # For simplicity, we'll create one recommendation
        creative_id = UUID("00000000-0000-0000-0000-000000000001")

        # Calculate optimal bid based on strategy
        if request.strategy == BidStrategyEnum.MANUAL:
            optimal_bid = market_price
        elif request.strategy == BidStrategyEnum.AUTO_CPC:
            optimal_bid = min(expected_value * 0.8, request.max_bid)
        elif request.strategy == BidStrategyEnum.TARGET_CPA:
            target_cpa = request.target_cpa or 10.0
            optimal_bid = expected_value * 0.7
        elif request.strategy == BidStrategyEnum.TARGET_ROAS:
            optimal_bid = expected_value * 0.6
        elif request.strategy == BidStrategyEnum.MAXIMIZE_CONVERSIONS:
            optimal_bid = min(expected_value * 0.9, request.max_bid)
        elif request.strategy == BidStrategyEnum.MAXIMIZE_CLICKS:
            optimal_bid = market_price * 1.1
        else:
            optimal_bid = market_price

        # Ensure bid is within bounds
        optimal_bid = max(floor_price, min(request.max_bid, optimal_bid))

        # Calculate confidence
        bid_confidence = self._calculate_bid_confidence(
            optimal_bid,
            market_price,
            expected_value,
            auction_analysis,
        )

        # Win rate at optimal bid
        win_rate = self._estimate_win_rate(optimal_bid, market_price)

        # Reasoning
        reasoning = self._generate_bid_reasoning(
            request.strategy,
            optimal_bid,
            market_price,
            expected_value,
            auction_analysis,
        )

        recommendations.append(
            BidRecommendation(
                creative_id=creative_id,
                recommended_bid=round(optimal_bid, 4),
                bid_confidence=round(bid_confidence, 3),
                predicted_win_rate=round(win_rate, 3),
                predicted_ctr=value_estimate["predicted_ctr"],
                predicted_cvr=value_estimate["predicted_cvr"],
                expected_value=round(expected_value, 4),
                reasoning=reasoning,
            )
        )

        return recommendations

    def _calculate_bid_confidence(
        self,
        bid: float,
        market_price: float,
        expected_value: float,
        auction_analysis: dict[str, Any],
    ) -> float:
        """Calculate confidence in bid recommendation."""
        confidence = 0.7

        # Adjust based on value margin
        value_margin = (expected_value - bid) / max(0.01, bid)
        if value_margin > 0.3:
            confidence += 0.15
        elif value_margin < 0:
            confidence -= 0.2

        # Adjust based on market alignment
        price_ratio = bid / max(0.01, market_price)
        if 0.8 <= price_ratio <= 1.3:
            confidence += 0.1

        # Adjust based on slot quality
        slot_quality = auction_analysis.get("slot_quality", 0.5)
        confidence *= (0.8 + slot_quality * 0.4)

        return max(0.1, min(0.95, confidence))

    def _estimate_win_rate(self, bid: float, market_price: float) -> float:
        """Estimate win rate at given bid."""
        ratio = bid / max(0.01, market_price)
        win_rate = 1 / (1 + np.exp(-3 * (ratio - 1)))
        return max(0.05, min(0.95, win_rate))

    def _generate_bid_reasoning(
        self,
        strategy: BidStrategyEnum,
        bid: float,
        market_price: float,
        expected_value: float,
        auction_analysis: dict[str, Any],
    ) -> list[str]:
        """Generate human-readable bid reasoning."""
        reasoning = []

        # Strategy explanation
        strategy_reasons = {
            BidStrategyEnum.AUTO_CPC: "Optimizing for clicks while maintaining profitability",
            BidStrategyEnum.TARGET_CPA: "Bid adjusted to achieve target CPA",
            BidStrategyEnum.TARGET_ROAS: "Bid set to maximize return on ad spend",
            BidStrategyEnum.MAXIMIZE_CONVERSIONS: "Aggressive bidding to maximize conversions",
            BidStrategyEnum.MAXIMIZE_CLICKS: "Bidding to maximize click volume",
            BidStrategyEnum.MANUAL: "Following manual bid settings",
        }
        reasoning.append(strategy_reasons.get(strategy, "Standard optimization"))

        # Market position
        if bid > market_price * 1.2:
            reasoning.append("Bidding above market to increase win rate")
        elif bid < market_price * 0.8:
            reasoning.append("Conservative bid due to low expected value")
        else:
            reasoning.append("Bid aligned with market expectations")

        # Value assessment
        if expected_value > bid * 1.5:
            reasoning.append("High-value opportunity with strong margin")
        elif expected_value < bid:
            reasoning.append("Marginal value - consider skipping")

        # Competition
        competition = auction_analysis.get("competition_level", "medium")
        if competition == "high":
            reasoning.append("High competition in this auction")

        return reasoning

    def _should_bid(
        self,
        recommendations: list[BidRecommendation],
        request: BidOptimizationRequest,
        auction_analysis: dict[str, Any],
    ) -> bool:
        """Determine if we should participate in this auction."""
        if not recommendations:
            return False

        best_rec = recommendations[0]

        # Check profitability
        if best_rec.expected_value < best_rec.recommended_bid * 0.5:
            return False

        # Check confidence
        if best_rec.bid_confidence < 0.3:
            return False

        # Check win probability
        if best_rec.predicted_win_rate < 0.1:
            return False

        # Check slot quality
        if auction_analysis.get("slot_quality", 0.5) < 0.2:
            return False

        return True

    async def batch_optimize(
        self,
        requests: list[BidOptimizationRequest],
    ) -> list[BidOptimizationResponse]:
        """Optimize bids for multiple auctions."""
        import asyncio
        tasks = [self.optimize(req) for req in requests]
        return await asyncio.gather(*tasks)
