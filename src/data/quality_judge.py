"""Deterministic rubric quality judge performing rejection sampling across multiple quality dimensions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.api.schemas import CurationReport, SyntheticSample


class QualityJudge:
    """Multi-dimension rejection sampling quality judge for synthetic instruction datasets."""

    def __init__(
        self,
        quality_threshold: float = 0.70,
        min_response_chars: int = 20,
        max_response_chars: int = 15000,
        schema_weight: float = 0.35,
        completeness_weight: float = 0.30,
        token_density_weight: float = 0.25,
        instruction_following_weight: float = 0.10,
    ) -> None:
        self.quality_threshold = quality_threshold
        self.min_response_chars = min_response_chars
        self.max_response_chars = max_response_chars
        self.schema_weight = schema_weight
        self.completeness_weight = completeness_weight
        self.token_density_weight = token_density_weight
        self.instruction_following_weight = instruction_following_weight

    def score_schema_adherence(self, sample: SyntheticSample) -> float:
        """Evaluate ChatML schema validity, alternation, and non-emptiness."""
        if not sample.messages or len(sample.messages) < 2:
            return 0.0

        score = 1.0
        roles = [m.role for m in sample.messages]

        # Valid roles
        for r in roles:
            if r not in ("system", "user", "assistant"):
                score -= 0.4

        # Non-empty content
        for m in sample.messages:
            if not m.content or not m.content.strip():
                score -= 0.5

        # Check that user prompt precedes assistant response
        user_indices = [i for i, r in enumerate(roles) if r == "user"]
        assistant_indices = [i for i, r in enumerate(roles) if r == "assistant"]

        if not user_indices or not assistant_indices:
            return 0.0

        if user_indices[-1] > assistant_indices[-1]:
            # Prompt appeared after response
            score -= 0.4

        return max(0.0, min(1.0, score))

    def score_completeness(self, sample: SyntheticSample) -> float:
        """Evaluate response completeness, structure, and absence of truncation."""
        response = sample.get_response().strip()
        prompt = sample.get_prompt().strip()

        if len(response) < self.min_response_chars or len(response) > self.max_response_chars:
            return 0.1

        score = 1.0

        # Penalize truncation artifacts (e.g. trailing "...")
        if response.endswith("...") or response.endswith("…"):
            score = min(score, 0.2)

        # Markdown code fence matching: unclosed code fence indicates severe truncation / malformed generation
        code_fence_count = response.count("```")
        if code_fence_count % 2 != 0:
            score = min(score, 0.1)

        # If user asked for JSON, check if JSON block or valid JSON exists
        if "json" in prompt.lower():
            json_matches = re.findall(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            found_valid_json = False
            for match in json_matches:
                try:
                    json.loads(match)
                    found_valid_json = True
                    break
                except Exception:
                    pass
            if not found_valid_json:
                # Try parsing raw response
                try:
                    json.loads(response)
                    found_valid_json = True
                except Exception:
                    pass
            if not found_valid_json:
                score -= 0.25

        # Minimum length scaling according to complexity level
        expected_min_words = sample.complexity_level * 10
        actual_words = len(response.split())
        if actual_words < expected_min_words:
            deficit_penalty = min(0.3, (expected_min_words - actual_words) * 0.02)
            score -= deficit_penalty

        return max(0.0, min(1.0, score))

    def score_token_density(self, sample: SyntheticSample) -> float:
        """Evaluate information entropy, repetition penalty, and token density."""
        response = sample.get_response().strip()
        words = re.findall(r"\w+", response.lower())

        if not words:
            return 0.0

        # Type-token ratio (unique tokens / total tokens)
        unique_words = set(words)
        ttr = len(unique_words) / len(words)

        score = min(1.0, ttr * 1.5)

        # Detect repetitive n-gram loops (e.g. identical 4-grams repeated >= 3 times)
        if len(words) >= 8:
            four_grams: Dict[str, int] = {}
            for i in range(len(words) - 3):
                fg = " ".join(words[i : i + 4])
                four_grams[fg] = four_grams.get(fg, 0) + 1

            max_repetition = max(four_grams.values()) if four_grams else 1
            if max_repetition >= 4:
                score -= 0.45
            elif max_repetition >= 3:
                score -= 0.25

        # Penalize excessive generic filler
        fillers = ["um", "uh", "like", "basically"]
        filler_count = sum(words.count(f) for f in fillers)
        if len(words) > 0 and (filler_count / len(words)) > 0.08:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def score_instruction_following(self, sample: SyntheticSample) -> float:
        """Evaluate lexical alignment and constraint relevance."""
        prompt = sample.get_prompt()
        response = sample.get_response()

        prompt_keywords = set(re.findall(r"\b[a-zA-Z]{4,}\b", prompt.lower()))
        response_keywords = set(re.findall(r"\b[a-zA-Z]{4,}\b", response.lower()))

        if not prompt_keywords:
            return 1.0

        overlap = prompt_keywords.intersection(response_keywords)
        overlap_ratio = len(overlap) / len(prompt_keywords)

        return min(1.0, overlap_ratio * 1.2)

    def evaluate_sample(self, sample: SyntheticSample) -> Tuple[float, Dict[str, float]]:
        """Compute composite score and dimension breakdown for a single sample."""
        s_schema = self.score_schema_adherence(sample)
        s_comp = self.score_completeness(sample)
        s_dens = self.score_token_density(sample)
        s_inst = self.score_instruction_following(sample)

        composite = (
            self.schema_weight * s_schema
            + self.completeness_weight * s_comp
            + self.token_density_weight * s_dens
            + self.instruction_following_weight * s_inst
        )

        scores = {
            "schema_adherence": round(s_schema, 4),
            "completeness": round(s_comp, 4),
            "token_density": round(s_dens, 4),
            "instruction_following": round(s_inst, 4),
            "composite": round(composite, 4),
        }
        return composite, scores

    def filter(
        self,
        samples: List[SyntheticSample],
    ) -> Tuple[List[SyntheticSample], Dict[str, Any]]:
        """Perform rejection sampling on candidate samples based on rubric scores.

        Args:
            samples: Samples to filter.

        Returns:
            Tuple of (accepted_samples, summary_stats)
        """
        accepted: List[SyntheticSample] = []
        total_input = len(samples)
        rejected_count = 0

        for sample in samples:
            composite, scores = self.evaluate_sample(sample)
            sample.quality_scores = scores

            if composite >= self.quality_threshold:
                sample.passed_filters = True
                accepted.append(sample)
            else:
                sample.passed_filters = False
                rejected_count += 1

        stats = {
            "total_input": total_input,
            "quality_rejected": rejected_count,
            "accepted_count": len(accepted),
            "pass_rate": len(accepted) / total_input if total_input > 0 else 0.0,
            "quality_threshold": self.quality_threshold,
        }

        return accepted, stats

    def curate_pipeline(
        self,
        samples: List[SyntheticSample],
        deduplicator: Any,
        duration_seconds: float = 0.0,
    ) -> Tuple[List[SyntheticSample], CurationReport]:
        """Execute end-to-end deduplication, contamination gating, and quality rejection sampling."""
        total_input = len(samples)

        # 1. Deduplication & Contamination Gate
        deduped_samples, dedup_stats = deduplicator.filter(samples)

        # 2. Quality Rubric Judge Rejection Sampling
        curated_samples, quality_stats = self.filter(deduped_samples)

        yield_count = len(curated_samples)
        yield_rate = yield_count / total_input if total_input > 0 else 0.0

        report = CurationReport(
            total_input=total_input,
            duplicates_removed=dedup_stats["duplicates_removed"],
            contaminated_removed=dedup_stats["contaminated_removed"],
            quality_rejected=quality_stats["quality_rejected"],
            final_yield_count=yield_count,
            yield_rate=round(yield_rate, 4),
            duration_seconds=round(duration_seconds, 4),
            filtering_breakdown={
                "raw_generated": total_input,
                "duplicates_removed": dedup_stats["duplicates_removed"],
                "contaminated_removed": dedup_stats["contaminated_removed"],
                "quality_rejected": quality_stats["quality_rejected"],
                "final_clean_curated": yield_count,
            },
        )

        return curated_samples, report
