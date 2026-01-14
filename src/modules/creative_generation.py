"""
Ad Creative Generation Module for AdTech LLM.

Generates personalized ad creatives including headlines, descriptions,
body copy, and CTAs using LLM-based generation with adtech optimization.
"""

import asyncio
import hashlib
import re
from datetime import datetime
from typing import Any
from uuid import uuid4

from src.config.settings import settings
from src.data.schemas import (
    AdCreativeRequest,
    AdCreativeResponse,
    AdFormatEnum,
    AdToneEnum,
    AdVariant,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AdCreativeGenerator:
    """
    Module for generating ad creatives using LLM.

    Features:
    - Multi-variant generation
    - Tone and style customization
    - Brand guideline compliance
    - A/B test variant creation
    - Quality scoring
    """

    def __init__(self, model_path: str | None = None):
        """
        Initialize the creative generator.

        Args:
            model_path: Path to fine-tuned model. Uses default if None.
        """
        self.model_path = model_path or settings.llm.model_name
        self.logger = get_logger("AdCreativeGenerator")
        self._model = None
        self._tokenizer = None

    async def load_model(self) -> None:
        """Load the generation model."""
        if self._model is not None:
            return

        try:
            # Try to load local fine-tuned model first
            await self._load_local_model()
        except Exception as e:
            self.logger.warning(f"Failed to load local model: {e}")
            # Fall back to API-based generation
            self._model = "api"

    async def _load_local_model(self) -> None:
        """Load local HuggingFace model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        if device == "cpu":
            self._model = self._model.to(device)

        self.logger.info(f"Loaded model on {device}")

    async def generate(self, request: AdCreativeRequest) -> AdCreativeResponse:
        """
        Generate ad creative variants.

        Args:
            request: Creative generation request

        Returns:
            Response with generated variants
        """
        start_time = datetime.utcnow()
        request_id = str(uuid4())

        self.logger.info(
            "Generating ad creatives",
            product=request.product_name,
            num_variants=request.num_variants,
        )

        await self.load_model()

        # Generate variants
        variants = await self._generate_variants(request)

        # Score and rank variants
        scored_variants = await self._score_variants(variants, request)

        # Sort by score
        scored_variants.sort(key=lambda v: v.confidence_score, reverse=True)

        generation_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return AdCreativeResponse(
            request_id=request_id,
            product_name=request.product_name,
            variants=scored_variants[: request.num_variants],
            total_variants=len(scored_variants),
            generation_time_ms=generation_time,
            model_version=self.model_path,
            warnings=self._check_brand_compliance(scored_variants, request),
        )

    async def _generate_variants(
        self,
        request: AdCreativeRequest,
    ) -> list[AdVariant]:
        """Generate raw creative variants."""
        variants = []

        # Build prompts for different styles
        prompts = self._build_prompts(request)

        for i, prompt in enumerate(prompts):
            try:
                if self._model == "api":
                    generated = await self._generate_via_api(prompt, request)
                else:
                    generated = await self._generate_via_local(prompt, request)

                # Parse generated text into structured variant
                variant = self._parse_generated_output(generated, i, request)
                if variant:
                    variants.append(variant)

            except Exception as e:
                self.logger.error(f"Generation failed for prompt {i}: {e}")

        # If no variants generated, create template-based fallbacks
        if not variants:
            variants = self._generate_template_variants(request)

        return variants

    def _build_prompts(self, request: AdCreativeRequest) -> list[str]:
        """Build generation prompts for the LLM."""
        base_context = f"""Product: {request.product_name}
Description: {request.product_description}
Target Audience: {self._format_audience(request.target_audience)}
Tone: {request.tone.value}
Format: {request.ad_format.value}
Keywords: {', '.join(request.keywords) if request.keywords else 'None specified'}
Language: {request.language}"""

        prompts = []

        # Generate multiple prompts for variety
        prompt_templates = [
            f"""Generate a compelling {request.ad_format.value} ad creative.

{base_context}

Requirements:
- Headline: Maximum {request.max_headline_length} characters
- Description: Maximum {request.max_description_length} characters
- Include a clear call-to-action
- Match the {request.tone.value} tone

Generate the ad in this format:
HEADLINE: [headline text]
DESCRIPTION: [description text]
CTA: [call-to-action text]""",

            f"""You are an expert ad copywriter. Create a high-converting ad for:

{base_context}

Focus on:
- Emotional appeal
- Clear value proposition
- Urgency when appropriate

Format:
HEADLINE: [compelling headline]
DESCRIPTION: [persuasive description]
CTA: [action-oriented CTA]""",

            f"""Create an ad that will maximize click-through rate.

{base_context}

Optimize for:
- Attention-grabbing headline
- Benefit-focused description
- Strong call-to-action

Output:
HEADLINE: [headline]
DESCRIPTION: [description]
CTA: [cta]""",
        ]

        # Add variations based on requested number
        num_needed = request.num_variants + 5  # Generate extra for selection
        for i in range(num_needed):
            template_idx = i % len(prompt_templates)
            prompts.append(prompt_templates[template_idx])

        return prompts

    def _format_audience(self, audience: dict[str, Any]) -> str:
        """Format audience dictionary to string."""
        if not audience:
            return "General audience"

        parts = []
        if "age_range" in audience:
            parts.append(f"Age: {audience['age_range']}")
        if "interests" in audience:
            parts.append(f"Interests: {', '.join(audience['interests'][:5])}")
        if "location" in audience:
            parts.append(f"Location: {audience['location']}")

        return "; ".join(parts) if parts else "General audience"

    async def _generate_via_api(
        self,
        prompt: str,
        request: AdCreativeRequest,
    ) -> str:
        """Generate using OpenAI API fallback."""
        if not settings.llm.openai_api_key:
            # Use template generation as fallback
            return self._generate_template_response(request)

        try:
            import openai

            client = openai.AsyncOpenAI(api_key=settings.llm.openai_api_key)
            response = await client.chat.completions.create(
                model=settings.llm.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert advertising copywriter specializing in digital ads.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.8,
            )
            return response.choices[0].message.content

        except Exception as e:
            self.logger.warning(f"API generation failed: {e}")
            return self._generate_template_response(request)

    async def _generate_via_local(
        self,
        prompt: str,
        request: AdCreativeRequest,
    ) -> str:
        """Generate using local model."""
        inputs = self._tokenizer(prompt, return_tensors="pt")

        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=settings.llm.temperature,
            top_p=settings.llm.top_p,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part
        return generated[len(prompt) :]

    def _generate_template_response(self, request: AdCreativeRequest) -> str:
        """Generate template-based response as fallback."""
        templates = {
            AdToneEnum.PROFESSIONAL: {
                "headline": f"Discover {request.product_name} - Professional Solutions",
                "description": f"{request.product_description[:200]}. Trusted by industry leaders.",
                "cta": "Learn More",
            },
            AdToneEnum.CASUAL: {
                "headline": f"Check out {request.product_name}!",
                "description": f"Looking for something great? {request.product_description[:150]}",
                "cta": "See What's New",
            },
            AdToneEnum.URGENT: {
                "headline": f"Don't Miss Out on {request.product_name}!",
                "description": f"Limited time offer! {request.product_description[:150]}",
                "cta": "Act Now",
            },
            AdToneEnum.FRIENDLY: {
                "headline": f"Meet {request.product_name} - Your New Favorite",
                "description": f"We think you'll love this. {request.product_description[:150]}",
                "cta": "Get Started",
            },
            AdToneEnum.LUXURIOUS: {
                "headline": f"Experience {request.product_name} - Exclusive Excellence",
                "description": f"Elevate your expectations. {request.product_description[:150]}",
                "cta": "Discover More",
            },
            AdToneEnum.PLAYFUL: {
                "headline": f"{request.product_name} - Fun Starts Here!",
                "description": f"Ready for something awesome? {request.product_description[:150]}",
                "cta": "Let's Go!",
            },
        }

        template = templates.get(request.tone, templates[AdToneEnum.PROFESSIONAL])

        return f"""HEADLINE: {template['headline']}
DESCRIPTION: {template['description']}
CTA: {template['cta']}"""

    def _parse_generated_output(
        self,
        output: str,
        variant_idx: int,
        request: AdCreativeRequest,
    ) -> AdVariant | None:
        """Parse LLM output into structured variant."""
        try:
            # Extract components using regex
            headline_match = re.search(r"HEADLINE:\s*(.+?)(?:\n|$)", output, re.IGNORECASE)
            desc_match = re.search(
                r"DESCRIPTION:\s*(.+?)(?:CTA:|$)", output, re.IGNORECASE | re.DOTALL
            )
            cta_match = re.search(r"CTA:\s*(.+?)(?:\n|$)", output, re.IGNORECASE)

            headline = headline_match.group(1).strip() if headline_match else ""
            description = desc_match.group(1).strip() if desc_match else ""
            cta = cta_match.group(1).strip() if cta_match else "Learn More"

            # Truncate if needed
            headline = headline[: request.max_headline_length]
            description = description[: request.max_description_length]

            if not headline or not description:
                return None

            return AdVariant(
                variant_id=f"var_{variant_idx}_{hashlib.md5(headline.encode()).hexdigest()[:8]}",
                headline=headline,
                description=description,
                body=None,
                cta_text=cta if request.include_cta else None,
                confidence_score=0.0,  # Will be scored later
                predicted_ctr=None,
                sentiment_score=None,
                relevance_score=None,
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse output: {e}")
            return None

    def _generate_template_variants(
        self,
        request: AdCreativeRequest,
    ) -> list[AdVariant]:
        """Generate template-based variants as fallback."""
        templates = [
            {
                "headline": f"Discover {request.product_name} Today",
                "description": f"{request.product_description[:200]}",
                "cta": "Shop Now",
            },
            {
                "headline": f"{request.product_name} - Transform Your Experience",
                "description": f"Experience the difference. {request.product_description[:150]}",
                "cta": "Get Started",
            },
            {
                "headline": f"Why Choose {request.product_name}?",
                "description": f"The smart choice for you. {request.product_description[:150]}",
                "cta": "Learn More",
            },
            {
                "headline": f"Introducing {request.product_name}",
                "description": f"Something new awaits. {request.product_description[:150]}",
                "cta": "Explore Now",
            },
            {
                "headline": f"{request.product_name} - Made for You",
                "description": f"Perfectly designed for your needs. {request.product_description[:150]}",
                "cta": "See Details",
            },
        ]

        variants = []
        for i, template in enumerate(templates):
            variants.append(
                AdVariant(
                    variant_id=f"template_{i}",
                    headline=template["headline"][: request.max_headline_length],
                    description=template["description"][: request.max_description_length],
                    body=None,
                    cta_text=template["cta"] if request.include_cta else None,
                    confidence_score=0.7 - (i * 0.05),  # Decreasing confidence
                    predicted_ctr=0.02 + (0.005 * (4 - i)),
                )
            )

        return variants

    async def _score_variants(
        self,
        variants: list[AdVariant],
        request: AdCreativeRequest,
    ) -> list[AdVariant]:
        """Score and rank generated variants."""
        for variant in variants:
            scores = []

            # Length appropriateness
            headline_score = 1.0 - abs(len(variant.headline) - 60) / 100
            desc_score = 1.0 - abs(len(variant.description) - 150) / 200
            scores.append(headline_score * 0.3 + desc_score * 0.3)

            # Keyword presence
            keyword_score = 0.0
            if request.keywords:
                text = f"{variant.headline} {variant.description}".lower()
                matches = sum(1 for kw in request.keywords if kw.lower() in text)
                keyword_score = min(1.0, matches / max(1, len(request.keywords)))
            scores.append(keyword_score * 0.2)

            # CTA presence and quality
            cta_score = 0.8 if variant.cta_text else 0.4
            if variant.cta_text and any(
                word in variant.cta_text.lower()
                for word in ["now", "today", "free", "start", "get"]
            ):
                cta_score = 1.0
            scores.append(cta_score * 0.2)

            # Calculate final confidence
            variant.confidence_score = sum(scores)
            variant.relevance_score = keyword_score

            # Estimate CTR based on heuristics
            variant.predicted_ctr = 0.015 + (variant.confidence_score * 0.02)

        return variants

    def _check_brand_compliance(
        self,
        variants: list[AdVariant],
        request: AdCreativeRequest,
    ) -> list[str]:
        """Check variants against brand guidelines."""
        warnings = []

        if not request.brand_guidelines:
            return warnings

        forbidden_words = request.brand_guidelines.get("forbidden_words", [])
        required_elements = request.brand_guidelines.get("required_elements", [])

        for variant in variants:
            text = f"{variant.headline} {variant.description}".lower()

            # Check forbidden words
            for word in forbidden_words:
                if word.lower() in text:
                    warnings.append(
                        f"Variant {variant.variant_id} contains forbidden word: {word}"
                    )

            # Check required elements
            for element in required_elements:
                if element.lower() not in text:
                    warnings.append(
                        f"Variant {variant.variant_id} missing required element: {element}"
                    )

        return warnings

    async def generate_batch(
        self,
        requests: list[AdCreativeRequest],
    ) -> list[AdCreativeResponse]:
        """Generate creatives for multiple requests in parallel."""
        tasks = [self.generate(req) for req in requests]
        return await asyncio.gather(*tasks)
