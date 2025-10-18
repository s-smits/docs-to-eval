"""
High-level workflow tests that exercise the core building blocks without relying on
network calls or heavyweight manual scripts.
"""

import random

import pytest

from docs_to_eval.core.classification import EvaluationTypeClassifier
from docs_to_eval.core.agentic_question_generator import AgenticQuestionGenerator
from docs_to_eval.core.verification import VerificationOrchestrator


def test_agentic_generation_and_verification_flow(sample_corpus):
    """End-to-end smoke test from classification to verification."""
    random.seed(42)

    classifier = EvaluationTypeClassifier()
    classification = classifier.classify_corpus(sample_corpus)

    assert classification.primary_type is not None
    assert 0.0 <= classification.confidence <= 1.0

    generator = AgenticQuestionGenerator()
    benchmark = generator.generate_comprehensive_benchmark(
        corpus_text=sample_corpus,
        num_questions=3,
        eval_type=classification.primary_type,
    )

    assert benchmark["total_generated"] == 3
    assert benchmark["questions"]

    first_item = benchmark["questions"][0]
    assert first_item["question"]
    assert first_item["answer"]

    orchestrator = VerificationOrchestrator()
    verification = orchestrator.verify(
        prediction=first_item["answer"],
        ground_truth=first_item["answer"],
        eval_type=classification.primary_type.value,
        question=first_item["question"],
    )

    assert verification.score >= 0.9
    assert "overall_score" in verification.metrics or verification.metrics


def test_batch_verification_and_metrics():
    """The orchestrator should return stable scores and aggregate metrics."""
    orchestrator = VerificationOrchestrator()

    predictions = ["Answer one", "Answer two"]
    truths = ["Answer one", "Different answer"]

    results = orchestrator.verify_batch(
        predictions=predictions,
        ground_truths=truths,
        eval_type="domain_knowledge",
    )

    assert len(results) == 2
    assert results[0].score >= 0.7
    assert 0.0 <= results[1].score <= 1.0

    aggregates = orchestrator.compute_aggregate_metrics(results)
    assert aggregates["num_samples"] == 2
    assert "mean_score" in aggregates
