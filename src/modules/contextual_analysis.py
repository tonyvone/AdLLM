"""
Contextual Analysis Module for AdTech LLM.

Analyzes web content for ad placement suitability, brand safety,
topic classification, and sentiment analysis.
"""

import asyncio
import re
from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import aiohttp

from src.config.settings import settings
from src.data.schemas import ContextualAnalysisRequest, ContextualAnalysisResponse
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ContextualAnalysisModule:
    """
    Module for contextual content analysis.

    Features:
    - Topic extraction
    - Sentiment analysis
    - Brand safety scoring
    - Keyword extraction
    - Content quality assessment
    """

    def __init__(self, model_path: str | None = None):
        """
        Initialize the contextual analysis module.

        Args:
            model_path: Path to analysis model
        """
        self.model_path = model_path
        self.logger = get_logger("ContextualAnalysisModule")
        self._nlp = None

        # Brand safety categories
        self._unsafe_categories = self._load_unsafe_categories()
        self._topic_keywords = self._load_topic_keywords()

    def _load_unsafe_categories(self) -> dict[str, list[str]]:
        """Load brand safety categories and keywords."""
        return {
            "adult_content": [
                "porn", "xxx", "adult content", "explicit", "nsfw",
                "erotic", "nude", "sex"
            ],
            "violence": [
                "murder", "killing", "massacre", "terrorist", "bombing",
                "shooting", "violence", "gore", "death"
            ],
            "hate_speech": [
                "racist", "hatred", "discrimination", "supremacist",
                "bigot", "extremist"
            ],
            "illegal_activity": [
                "drugs", "cocaine", "heroin", "illegal", "piracy",
                "counterfeit", "fraud"
            ],
            "misinformation": [
                "fake news", "hoax", "conspiracy", "debunked"
            ],
            "gambling": [
                "casino", "betting", "gambling", "poker", "slots"
            ],
            "weapons": [
                "guns for sale", "buy weapons", "ammunition", "explosives"
            ],
        }

    def _load_topic_keywords(self) -> dict[str, list[str]]:
        """Load topic classification keywords."""
        return {
            "technology": [
                "software", "app", "tech", "computer", "smartphone", "AI",
                "machine learning", "digital", "internet", "cloud"
            ],
            "finance": [
                "investment", "stock", "banking", "money", "credit",
                "loan", "insurance", "financial", "trading"
            ],
            "health": [
                "health", "medical", "doctor", "hospital", "wellness",
                "fitness", "nutrition", "medicine", "treatment"
            ],
            "entertainment": [
                "movie", "music", "game", "celebrity", "tv show",
                "streaming", "entertainment", "concert"
            ],
            "sports": [
                "football", "basketball", "soccer", "tennis", "sports",
                "athlete", "championship", "tournament", "league"
            ],
            "travel": [
                "travel", "vacation", "hotel", "flight", "destination",
                "tourism", "trip", "resort", "booking"
            ],
            "food": [
                "recipe", "restaurant", "cooking", "food", "dining",
                "cuisine", "chef", "meal"
            ],
            "fashion": [
                "fashion", "clothing", "style", "designer", "trend",
                "outfit", "wear", "brand"
            ],
            "automotive": [
                "car", "vehicle", "automotive", "driving", "auto",
                "motorcycle", "truck", "electric vehicle"
            ],
            "real_estate": [
                "home", "property", "real estate", "housing", "apartment",
                "mortgage", "rent", "buy house"
            ],
            "education": [
                "education", "school", "university", "learning", "course",
                "student", "teacher", "degree"
            ],
            "business": [
                "business", "company", "startup", "entrepreneur",
                "corporate", "management", "CEO"
            ],
        }

    async def analyze(
        self,
        request: ContextualAnalysisRequest,
    ) -> ContextualAnalysisResponse:
        """
        Analyze content for ad placement.

        Args:
            request: Analysis request with URL or content

        Returns:
            Analysis response with topics, sentiment, safety
        """
        start_time = datetime.utcnow()
        request_id = str(uuid4())

        self.logger.info(
            "Analyzing content",
            url=request.url,
            has_content=bool(request.content),
        )

        # Get content to analyze
        content = request.content
        if request.url and not content:
            content = await self._fetch_content(request.url)

        if not content:
            return ContextualAnalysisResponse(
                request_id=request_id,
                url=request.url,
                topics=[],
                sentiment={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                brand_safety_score=0.5,
                unsafe_categories=[],
                suitable_ad_categories=[],
                keywords_extracted=[],
                content_quality_score=0.0,
                analysis_time_ms=0,
            )

        # Run analysis
        topics = await self._extract_topics(content)
        sentiment = await self._analyze_sentiment(content)
        safety_result = await self._check_brand_safety(content, request.brand_safety_level)
        keywords = await self._extract_keywords(content)
        quality_score = await self._assess_quality(content)

        # Determine suitable ad categories
        suitable_categories = self._determine_suitable_categories(
            topics,
            safety_result["score"],
            request.ad_categories,
        )

        analysis_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return ContextualAnalysisResponse(
            request_id=request_id,
            url=request.url,
            topics=topics,
            sentiment=sentiment,
            brand_safety_score=safety_result["score"],
            unsafe_categories=safety_result["categories"],
            suitable_ad_categories=suitable_categories,
            keywords_extracted=keywords,
            content_quality_score=quality_score,
            analysis_time_ms=analysis_time,
        )

    async def _fetch_content(self, url: str) -> str | None:
        """Fetch content from URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={"User-Agent": "AdTechLLM/1.0 ContextAnalyzer"},
                ) as response:
                    if response.status != 200:
                        return None

                    html = await response.text()

                    # Extract text from HTML
                    return self._extract_text_from_html(html)

        except Exception as e:
            self.logger.warning(f"Failed to fetch URL: {e}")
            return None

    def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Get text
            text = soup.get_text(separator=" ", strip=True)

            # Clean up whitespace
            text = re.sub(r"\s+", " ", text)

            return text[:10000]  # Limit for analysis

        except ImportError:
            # Fallback: basic regex extraction
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text[:10000]

    async def _extract_topics(self, content: str) -> list[dict[str, float]]:
        """Extract topics from content."""
        content_lower = content.lower()
        topics = []

        for topic, keywords in self._topic_keywords.items():
            matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            if matches > 0:
                confidence = min(1.0, matches * 0.15)
                topics.append({"topic": topic, "confidence": round(confidence, 2)})

        # Sort by confidence
        topics.sort(key=lambda x: x["confidence"], reverse=True)

        return topics[:5]  # Return top 5 topics

    async def _analyze_sentiment(self, content: str) -> dict[str, float]:
        """Analyze content sentiment."""
        # Simple keyword-based sentiment analysis
        positive_words = [
            "great", "excellent", "amazing", "wonderful", "best", "love",
            "fantastic", "awesome", "good", "happy", "success", "beautiful"
        ]
        negative_words = [
            "bad", "terrible", "awful", "worst", "hate", "poor", "horrible",
            "disappointing", "fail", "sad", "angry", "wrong"
        ]

        content_lower = content.lower()
        words = content_lower.split()
        total_words = len(words)

        if total_words == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

        positive_count = sum(1 for w in words if any(pw in w for pw in positive_words))
        negative_count = sum(1 for w in words if any(nw in w for nw in negative_words))

        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words

        # Normalize
        total = positive_ratio + negative_ratio
        if total > 0:
            positive_score = (positive_ratio / total) * 0.8
            negative_score = (negative_ratio / total) * 0.8
            neutral_score = 0.2
        else:
            positive_score = 0.2
            negative_score = 0.2
            neutral_score = 0.6

        return {
            "positive": round(positive_score, 2),
            "negative": round(negative_score, 2),
            "neutral": round(neutral_score, 2),
        }

    async def _check_brand_safety(
        self,
        content: str,
        safety_level: str,
    ) -> dict[str, Any]:
        """Check content for brand safety issues."""
        content_lower = content.lower()
        found_categories = []
        severity_scores = []

        # Define thresholds based on safety level
        thresholds = {
            "permissive": 3,
            "standard": 2,
            "strict": 1,
        }
        threshold = thresholds.get(safety_level, 2)

        for category, keywords in self._unsafe_categories.items():
            matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            if matches >= threshold:
                found_categories.append(category)
                severity_scores.append(min(1.0, matches * 0.2))

        # Calculate safety score (1.0 = safe, 0.0 = unsafe)
        if severity_scores:
            safety_score = max(0.0, 1.0 - max(severity_scores))
        else:
            safety_score = 1.0

        # Apply safety level adjustment
        if safety_level == "strict" and safety_score > 0.9:
            # More conservative for strict mode
            safety_score = min(safety_score, 0.95)

        return {
            "score": round(safety_score, 2),
            "categories": found_categories,
        }

    async def _extract_keywords(self, content: str) -> list[str]:
        """Extract important keywords from content."""
        # Simple TF-based keyword extraction
        words = re.findall(r"\b[a-zA-Z]{4,}\b", content.lower())

        # Remove common stop words
        stop_words = {
            "this", "that", "with", "from", "have", "been", "were", "will",
            "would", "could", "should", "their", "there", "about", "which",
            "when", "what", "where", "into", "more", "some", "than", "also",
            "just", "only", "other", "such", "like", "your", "these", "them",
        }

        filtered = [w for w in words if w not in stop_words]

        # Count frequencies
        freq = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1

        # Get top keywords
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        return [word for word, count in sorted_words[:20] if count > 1]

    async def _assess_quality(self, content: str) -> float:
        """Assess content quality."""
        # Quality indicators
        word_count = len(content.split())

        # Length score
        if word_count < 100:
            length_score = 0.3
        elif word_count < 500:
            length_score = 0.6
        elif word_count < 2000:
            length_score = 0.9
        else:
            length_score = 1.0

        # Readability (simplified)
        sentences = len(re.findall(r"[.!?]+", content))
        if sentences > 0:
            avg_sentence_length = word_count / sentences
            if 15 <= avg_sentence_length <= 25:
                readability_score = 1.0
            elif 10 <= avg_sentence_length <= 35:
                readability_score = 0.7
            else:
                readability_score = 0.4
        else:
            readability_score = 0.3

        # Vocabulary diversity
        words = content.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            diversity_score = min(1.0, unique_ratio * 2)
        else:
            diversity_score = 0.0

        # Combined score
        quality_score = (
            length_score * 0.3 +
            readability_score * 0.4 +
            diversity_score * 0.3
        )

        return round(quality_score, 2)

    def _determine_suitable_categories(
        self,
        topics: list[dict[str, float]],
        safety_score: float,
        requested_categories: list[str] | None,
    ) -> list[str]:
        """Determine suitable ad categories for this content."""
        suitable = []

        # Topic-to-category mapping
        topic_categories = {
            "technology": ["tech", "electronics", "software", "gadgets"],
            "finance": ["financial_services", "banking", "insurance", "investment"],
            "health": ["healthcare", "pharma", "wellness", "fitness"],
            "entertainment": ["media", "streaming", "gaming", "movies"],
            "sports": ["sports_equipment", "athletic_wear", "fitness"],
            "travel": ["travel", "hospitality", "airlines", "tourism"],
            "food": ["food_beverage", "restaurants", "grocery"],
            "fashion": ["apparel", "beauty", "luxury", "accessories"],
            "automotive": ["automotive", "car_accessories", "ev"],
            "real_estate": ["real_estate", "home_improvement", "mortgage"],
            "education": ["education", "edtech", "online_learning"],
            "business": ["b2b", "enterprise", "professional_services"],
        }

        # Add categories based on topics
        for topic_info in topics:
            topic = topic_info.get("topic", "")
            confidence = topic_info.get("confidence", 0)

            if confidence >= 0.3 and topic in topic_categories:
                suitable.extend(topic_categories[topic])

        # Filter by safety
        if safety_score < 0.5:
            suitable = []  # Too unsafe for any ads
        elif safety_score < 0.7:
            # Limit to less sensitive categories
            sensitive = ["pharma", "financial_services", "insurance"]
            suitable = [c for c in suitable if c not in sensitive]

        # Filter by requested categories if provided
        if requested_categories:
            suitable = [c for c in suitable if c in requested_categories]

        return list(set(suitable))[:10]

    async def batch_analyze(
        self,
        requests: list[ContextualAnalysisRequest],
    ) -> list[ContextualAnalysisResponse]:
        """Analyze multiple pieces of content in batch."""
        tasks = [self.analyze(req) for req in requests]
        return await asyncio.gather(*tasks)
