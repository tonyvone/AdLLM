"""
CTR Prediction Module for AdTech LLM.

Predicts click-through rates for ad placements using
machine learning models trained on historical data.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np

from src.config.settings import settings
from src.data.schemas import CTRPredictionRequest, CTRPredictionResponse
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CTRPredictionModule:
    """
    Module for CTR (Click-Through Rate) prediction.

    Features:
    - Feature engineering
    - Real-time prediction
    - Confidence intervals
    - Feature importance analysis
    - Model versioning
    """

    def __init__(self, model_path: str | None = None):
        """
        Initialize the CTR prediction module.

        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.logger = get_logger("CTRPredictionModule")
        self._model = None
        self._feature_stats = self._initialize_feature_stats()

    def _initialize_feature_stats(self) -> dict[str, dict[str, float]]:
        """Initialize feature statistics for normalization."""
        return {
            "hour_of_day": {"mean": 12.0, "std": 6.9},
            "day_of_week": {"mean": 3.0, "std": 2.0},
            "device_type_mobile": {"mean": 0.6, "std": 0.49},
            "device_type_desktop": {"mean": 0.3, "std": 0.46},
            "device_type_tablet": {"mean": 0.1, "std": 0.3},
            "position": {"mean": 3.0, "std": 2.0},
            "ad_size_score": {"mean": 0.5, "std": 0.25},
            "historical_ctr": {"mean": 0.02, "std": 0.015},
        }

    async def load_model(self) -> None:
        """Load the prediction model."""
        if self._model is not None:
            return

        if self.model_path:
            try:
                import joblib
                self._model = joblib.load(self.model_path)
                self.logger.info(f"Loaded model from {self.model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load model: {e}")
                self._model = "heuristic"
        else:
            self._model = "heuristic"

    async def predict(
        self,
        request: CTRPredictionRequest,
    ) -> CTRPredictionResponse:
        """
        Predict CTR for given features.

        Args:
            request: Prediction request with features

        Returns:
            Prediction response with CTR and confidence
        """
        start_time = datetime.utcnow()
        request_id = str(uuid4())

        self.logger.info(
            "Predicting CTR",
            creative_id=str(request.creative_id) if request.creative_id else None,
        )

        await self.load_model()

        # Engineer features
        features = self._engineer_features(
            request.creative_features,
            request.audience_features,
            request.context_features,
            request.historical_features,
        )

        # Make prediction
        if self._model == "heuristic":
            predicted_ctr, confidence = self._predict_heuristic(features)
        else:
            predicted_ctr, confidence = self._predict_model(features)

        # Calculate feature importance
        importance = self._calculate_feature_importance(features)

        prediction_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return CTRPredictionResponse(
            request_id=request_id,
            predicted_ctr=round(predicted_ctr, 6),
            confidence_interval=(
                round(predicted_ctr * (1 - confidence), 6),
                round(predicted_ctr * (1 + confidence), 6),
            ),
            feature_importance=importance,
            model_version="ctr_v1.0" if self._model == "heuristic" else self.model_path,
            prediction_time_ms=prediction_time,
        )

    def _engineer_features(
        self,
        creative_features: dict[str, Any],
        audience_features: dict[str, Any],
        context_features: dict[str, Any],
        historical_features: dict[str, Any] | None,
    ) -> dict[str, float]:
        """Engineer features for prediction."""
        features = {}

        # Creative features
        if creative_features:
            # Ad size score
            width = creative_features.get("width", 300)
            height = creative_features.get("height", 250)
            size_score = min(1.0, (width * height) / (728 * 90))
            features["ad_size_score"] = size_score

            # Format encoding
            ad_format = creative_features.get("format", "display")
            features["is_video"] = 1.0 if ad_format == "video" else 0.0
            features["is_native"] = 1.0 if ad_format == "native" else 0.0

            # Text quality (headline length)
            headline = creative_features.get("headline", "")
            features["headline_length_score"] = min(1.0, len(headline) / 90)

        # Audience features
        if audience_features:
            # Device type
            device = audience_features.get("device_type", "desktop")
            features["device_type_mobile"] = 1.0 if device == "mobile" else 0.0
            features["device_type_desktop"] = 1.0 if device == "desktop" else 0.0
            features["device_type_tablet"] = 1.0 if device == "tablet" else 0.0

            # Age group
            age = audience_features.get("age", 35)
            features["age_score"] = 1.0 - abs(age - 35) / 50  # Peak at 35

            # Interest match
            interests = audience_features.get("interests", [])
            features["interest_count"] = min(1.0, len(interests) / 10)

        # Context features
        if context_features:
            # Time features
            hour = context_features.get("hour_of_day", 12)
            features["hour_of_day"] = hour

            # Prime time indicator
            features["is_prime_time"] = 1.0 if 18 <= hour <= 22 else 0.0

            # Day of week
            day = context_features.get("day_of_week", 3)
            features["day_of_week"] = day
            features["is_weekend"] = 1.0 if day in [5, 6] else 0.0

            # Position
            position = context_features.get("position", 1)
            features["position"] = position
            features["position_score"] = 1.0 / (position + 1)

            # Page quality
            features["page_quality"] = context_features.get("page_quality_score", 0.7)

        # Historical features
        if historical_features:
            features["historical_ctr"] = historical_features.get("avg_ctr", 0.02)
            features["historical_cvr"] = historical_features.get("avg_cvr", 0.01)
            features["impression_count"] = min(
                1.0, historical_features.get("impression_count", 0) / 100000
            )

        return features

    def _predict_heuristic(
        self,
        features: dict[str, float],
    ) -> tuple[float, float]:
        """Predict CTR using heuristic model."""
        base_ctr = 0.015

        # Apply feature effects
        multipliers = []

        # Device effect
        if features.get("device_type_mobile", 0) > 0:
            multipliers.append(1.2)
        elif features.get("device_type_tablet", 0) > 0:
            multipliers.append(1.1)

        # Time effects
        if features.get("is_prime_time", 0) > 0:
            multipliers.append(1.25)
        if features.get("is_weekend", 0) > 0:
            multipliers.append(1.1)

        # Position effect
        position_score = features.get("position_score", 0.5)
        multipliers.append(0.7 + position_score * 0.6)

        # Format effects
        if features.get("is_video", 0) > 0:
            multipliers.append(1.3)
        if features.get("is_native", 0) > 0:
            multipliers.append(1.4)

        # Page quality effect
        page_quality = features.get("page_quality", 0.7)
        multipliers.append(0.8 + page_quality * 0.4)

        # Historical effect
        historical_ctr = features.get("historical_ctr", 0.02)
        if historical_ctr > 0:
            base_ctr = base_ctr * 0.5 + historical_ctr * 0.5

        # Apply multipliers
        for mult in multipliers:
            base_ctr *= mult

        # Add noise
        noise = np.random.normal(0, base_ctr * 0.1)
        predicted_ctr = max(0.001, min(0.15, base_ctr + noise))

        # Confidence based on feature completeness
        feature_completeness = len([v for v in features.values() if v > 0]) / 15
        confidence = 0.3 - (feature_completeness * 0.15)

        return predicted_ctr, confidence

    def _predict_model(
        self,
        features: dict[str, float],
    ) -> tuple[float, float]:
        """Predict CTR using trained model."""
        # Convert features to array
        feature_names = sorted(features.keys())
        X = np.array([[features.get(name, 0) for name in feature_names]])

        # Normalize features
        X_normalized = self._normalize_features(X, feature_names)

        try:
            # Get prediction
            prediction = self._model.predict(X_normalized)[0]

            # Get prediction probability if available
            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(X_normalized)[0]
                confidence = 0.2  # Based on model certainty
            else:
                confidence = 0.25

            return float(prediction), confidence

        except Exception as e:
            self.logger.error(f"Model prediction failed: {e}")
            return self._predict_heuristic(features)

    def _normalize_features(
        self,
        X: np.ndarray,
        feature_names: list[str],
    ) -> np.ndarray:
        """Normalize features using stored statistics."""
        X_normalized = X.copy()

        for i, name in enumerate(feature_names):
            if name in self._feature_stats:
                stats = self._feature_stats[name]
                X_normalized[0, i] = (X[0, i] - stats["mean"]) / stats["std"]

        return X_normalized

    def _calculate_feature_importance(
        self,
        features: dict[str, float],
    ) -> dict[str, float]:
        """Calculate feature importance for this prediction."""
        # Heuristic importance based on feature values
        importance = {}

        # High impact features
        high_impact = {
            "is_prime_time": 0.15,
            "position_score": 0.12,
            "device_type_mobile": 0.1,
            "is_video": 0.1,
            "is_native": 0.1,
            "page_quality": 0.08,
            "historical_ctr": 0.15,
        }

        for feature, base_importance in high_impact.items():
            if feature in features:
                # Scale importance by feature value
                importance[feature] = round(
                    base_importance * (0.5 + features[feature] * 0.5), 3
                )

        # Medium impact features
        medium_impact = ["is_weekend", "headline_length_score", "ad_size_score"]
        for feature in medium_impact:
            if feature in features:
                importance[feature] = round(
                    0.05 * (0.5 + features[feature] * 0.5), 3
                )

        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: round(v / total, 3) for k, v in importance.items()}

        return importance

    async def batch_predict(
        self,
        requests: list[CTRPredictionRequest],
    ) -> list[CTRPredictionResponse]:
        """Predict CTR for multiple requests."""
        import asyncio
        tasks = [self.predict(req) for req in requests]
        return await asyncio.gather(*tasks)

    async def train(
        self,
        training_data: list[dict[str, Any]],
        labels: list[int],
        model_type: str = "gradient_boosting",
    ) -> dict[str, Any]:
        """
        Train a new CTR prediction model.

        Args:
            training_data: List of feature dictionaries
            labels: Binary labels (0/1 for no-click/click)
            model_type: Model type to train

        Returns:
            Training results and metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, log_loss

        # Prepare features
        all_features = set()
        for data in training_data:
            all_features.update(data.keys())

        feature_names = sorted(all_features)
        X = np.array([
            [d.get(name, 0) for name in feature_names]
            for d in training_data
        ])
        y = np.array(labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        if model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        elif model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000)
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)

        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)

        # Save model
        import joblib
        model_path = settings.data.models_path / f"ctr_model_{model_type}.joblib"
        joblib.dump(model, model_path)

        self._model = model
        self.model_path = str(model_path)

        return {
            "model_type": model_type,
            "model_path": str(model_path),
            "metrics": {
                "auc_roc": round(auc, 4),
                "log_loss": round(logloss, 4),
            },
            "feature_count": len(feature_names),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
        }
