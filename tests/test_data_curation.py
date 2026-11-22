"""Unit tests for Evol-Instruct synthetic data generation, deduplication, and quality judge."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.api.schemas import ChatMessage, SyntheticSample
from src.data.deduplicator import Deduplicator, BENCHMARK_EVAL_SET, jaccard_similarity, tokenize_ngrams
from src.data.quality_judge import QualityJudge
from src.data.synthetic_generator import DOMAIN_SEEDS, SyntheticDataGenerator


class TestSyntheticDataGenerator:
    """Verifies Evol-Instruct generation, mutations, and serialization."""

    def test_generator_initialization(self):
        generator = SyntheticDataGenerator(seed=42)
        assert generator.simulation_mode is True
        assert generator.rng is not None

    def test_generate_samples_structure(self):
        generator = SyntheticDataGenerator(seed=123)
        samples = generator.generate(num_samples=10, noise_rate=0.0)

        assert len(samples) == 10
        for sample in samples:
            assert isinstance(sample, SyntheticSample)
            assert len(sample.messages) >= 2
            roles = [m.role for m in sample.messages]
            assert "user" in roles
            assert "assistant" in roles
            assert sample.domain in DOMAIN_SEEDS.keys()
            assert 1 <= sample.complexity_level <= 5
            assert sample.get_prompt() != ""
            assert sample.get_response() != ""

    def test_evol_instruct_complexity_levels(self):
        generator = SyntheticDataGenerator(seed=42)
        base_item = DOMAIN_SEEDS["reasoning"][0]

        for level in range(1, 6):
            prompt, response, mutation_type = generator.evolve_instruction(base_item, "reasoning", level)
            assert len(prompt) > 0
            assert len(response) > 0
            if level == 1:
                assert mutation_type == "seed"
            elif level == 2:
                assert mutation_type == "in_depth_constraint"
                assert "Complexity Analysis" in response
            elif level == 3:
                assert mutation_type == "in_depth_deepening"
                assert "Chain of Thought" in response
            elif level == 4:
                assert mutation_type == "in_breadth_domain_transfer"
                assert "Distributed" in response
            elif level == 5:
                assert mutation_type == "concretizing"
                assert "Failure Mode" in response

    def test_noise_injection(self):
        generator = SyntheticDataGenerator(seed=777)
        text = "The quick brown fox jumps over the lazy dog repeatedly"
        noisy_typo = generator.inject_noise(text, "typo")
        assert isinstance(noisy_typo, str)

        noisy_filler = generator.inject_noise(text, "filler")
        assert any(f in noisy_filler for f in ["Um", "Basically", "So"])

        noisy_loop = generator.inject_noise(text, "repetition")
        assert len(noisy_loop) > len(text)

        truncated = generator.inject_noise(text, "truncation")
        assert truncated.endswith("...")

    def test_jsonl_save_and_load(self):
        generator = SyntheticDataGenerator(seed=42)
        samples = generator.generate(num_samples=5, noise_rate=0.0)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            SyntheticDataGenerator.save_to_jsonl(samples, tmp_path)
            loaded_samples = SyntheticDataGenerator.load_from_jsonl(tmp_path)

            assert len(loaded_samples) == 5
            for orig, loaded in zip(samples, loaded_samples):
                assert orig.id == loaded.id
                assert orig.domain == loaded.domain
                assert orig.get_prompt() == loaded.get_prompt()
                assert orig.get_response() == loaded.get_response()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestDeduplicator:
    """Verifies near-duplicate detection and benchmark contamination filtering."""

    def test_jaccard_similarity(self):
        set_a = {"alpha", "beta", "gamma"}
        set_b = {"alpha", "beta", "gamma"}
        assert jaccard_similarity(set_a, set_b) == 1.0

        set_c = {"delta", "epsilon"}
        assert jaccard_similarity(set_a, set_c) == 0.0

        set_d = {"alpha", "beta", "delta"}
        # Intersection: {alpha, beta} (2), Union: {alpha, beta, gamma, delta} (4) -> 0.5
        assert jaccard_similarity(set_a, set_d) == 0.5

    def test_exact_and_near_duplicate_removal(self):
        dedup = Deduplicator(dedup_threshold=0.75, contamination_threshold=0.70)
        messages_orig = [
            ChatMessage(role="user", content="Explain how quicksort partitions elements in an array."),
            ChatMessage(role="assistant", content="Quicksort selects a pivot and partitions elements into two sub-arrays."),
        ]
        messages_dup = [
            ChatMessage(role="user", content="Explain how quicksort partitions elements in an array."),
            ChatMessage(role="assistant", content="Quicksort selects a pivot and partitions elements into two sub-arrays."),
        ]
        messages_unique = [
            ChatMessage(role="user", content="Discuss photosynthesis and chlorophyll in chloroplasts."),
            ChatMessage(role="assistant", content="Photosynthesis converts solar photon flux into chemical ATP energy."),
        ]

        sample1 = SyntheticSample(id="s1", messages=messages_orig, domain="code")
        sample2 = SyntheticSample(id="s2", messages=messages_dup, domain="code")
        sample3 = SyntheticSample(id="s3", messages=messages_unique, domain="biology")

        accepted, stats = dedup.filter([sample1, sample2, sample3])

        assert len(accepted) == 2
        assert stats["duplicates_removed"] == 1
        assert stats["contaminated_removed"] == 0
        assert accepted[0].id == "s1"
        assert accepted[1].id == "s3"

    def test_benchmark_contamination_gating(self):
        dedup = Deduplicator(contamination_threshold=0.70)

        # Contaminated prompt verbatim from BENCHMARK_EVAL_SET
        bench_prompt = BENCHMARK_EVAL_SET[0]["prompt"]
        contaminated_sample = SyntheticSample(
            id="contam_01",
            messages=[
                ChatMessage(role="user", content=bench_prompt),
                ChatMessage(role="assistant", content="Safety always takes precedence."),
            ],
            domain="safety",
        )

        clean_sample = SyntheticSample(
            id="clean_01",
            messages=[
                ChatMessage(role="user", content="How do airplanes achieve aerodynamic lift during take off?"),
                ChatMessage(role="assistant", content="Airplanes generate lift through Bernoulli pressure differential."),
            ],
            domain="physics",
        )

        accepted, stats = dedup.filter([contaminated_sample, clean_sample])

        assert len(accepted) == 1
        assert stats["contaminated_removed"] == 1
        assert accepted[0].id == "clean_01"
        assert contaminated_sample.is_contaminated is True


class TestQualityJudge:
    """Verifies rubric scoring and rejection sampling."""

    def test_schema_adherence_scoring(self):
        judge = QualityJudge()

        # Valid sample
        valid = SyntheticSample(
            id="v1",
            messages=[
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="What is 2+2?"),
                ChatMessage(role="assistant", content="2+2 is 4."),
            ],
            domain="math",
        )
        assert judge.score_schema_adherence(valid) == 1.0

    def test_rejection_of_unclosed_code_blocks(self):
        judge = QualityJudge(quality_threshold=0.70)

        bad_code_sample = SyntheticSample(
            id="bad_code",
            messages=[
                ChatMessage(role="user", content="Write a python function to add two numbers."),
                ChatMessage(role="assistant", content="Here is code: ```python\ndef add(a, b): return a + b"),
            ],
            domain="code",
        )

        composite, scores = judge.evaluate_sample(bad_code_sample)
        assert scores["completeness"] < 0.70
        accepted, stats = judge.filter([bad_code_sample])
        assert len(accepted) == 0
        assert stats["quality_rejected"] == 1

    def test_rejection_of_repetitive_loop_tokens(self):
        judge = QualityJudge(quality_threshold=0.70)

        loop_sample = SyntheticSample(
            id="loop_01",
            messages=[
                ChatMessage(role="user", content="Explain quantum entanglement."),
                ChatMessage(
                    role="assistant",
                    content="entanglement is cool entanglement is cool entanglement is cool entanglement is cool entanglement is cool",
                ),
            ],
            domain="physics",
        )

        composite, scores = judge.evaluate_sample(loop_sample)
        assert scores["token_density"] < 0.60
        accepted, stats = judge.filter([loop_sample])
        assert len(accepted) == 0
        assert stats["quality_rejected"] == 1

    def test_end_to_end_curate_pipeline(self):
        generator = SyntheticDataGenerator(seed=42)
        raw_samples = generator.generate(num_samples=20, noise_rate=0.2)

        dedup = Deduplicator()
        judge = QualityJudge(quality_threshold=0.70)

        curated, report = judge.curate_pipeline(raw_samples, dedup, duration_seconds=0.15)

        assert report.total_input == 20
        assert report.final_yield_count == len(curated)
        assert 0.0 <= report.yield_rate <= 1.0
        assert report.final_yield_count > 0
        for c in curated:
            assert c.passed_filters is True
            assert c.quality_scores["composite"] >= 0.70
