"""
Core modules for AdTech LLM.

Each module provides specialized functionality for key adtech tasks:
- Ad Creative Generation
- Audience Targeting and Prediction
- Fraud Detection
- Contextual Analysis
- CTR Prediction
- Real-Time Bidding Optimization
"""

from src.modules.creative_generation import AdCreativeGenerator
from src.modules.audience_targeting import AudienceTargetingModule
from src.modules.fraud_detection import FraudDetectionModule
from src.modules.contextual_analysis import ContextualAnalysisModule
from src.modules.ctr_prediction import CTRPredictionModule
from src.modules.bid_optimization import BidOptimizationModule

__all__ = [
    "AdCreativeGenerator",
    "AudienceTargetingModule",
    "FraudDetectionModule",
    "ContextualAnalysisModule",
    "CTRPredictionModule",
    "BidOptimizationModule",
]
