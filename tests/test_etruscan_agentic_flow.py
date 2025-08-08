"""
Integration test: Etruscan corpus → file retrieval → smart chunking (chonkie) → agentic generation (10 questions)
"""

import os
from pathlib import Path
import pytest

from docs_to_eval.utils.text_processing import create_smart_chunks_from_files
from docs_to_eval.utils.config import ChunkingConfig
from docs_to_eval.core.agentic.orchestrator import AgenticBenchmarkOrchestrator
from docs_to_eval.core.agentic.models import PipelineConfig, DifficultyLevel
from docs_to_eval.core.evaluation import EvaluationType
from docs_to_eval.llm.mock_interface import MockLLMInterface


@pytest.mark.asyncio
@pytest.mark.integration
def test_smoke_marker_only():
    # This keeps the module collected for markers; actual async test below
    assert True


@pytest.mark.asyncio
@pytest.mark.integration
async def test_etruscan_full_flow_chunk_then_agentic():
    # Location of domain corpus texts
    base_dir = Path("/Users/air/Developer/docs-to-eval/domain_spcfc_general_corpus/etruscan_texts")
    assert base_dir.exists(), f"Corpus directory not found: {base_dir}"

    # Collect .txt files
    file_contents = []
    for p in sorted(base_dir.glob("**/*.txt")):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                file_contents.append({"filename": p.name, "content": text})
        except Exception:
            continue

    # Ensure we have content
    assert len(file_contents) > 0, "No .txt files with content found in corpus directory"

    # Smart chunking with chonkie enabled; allow token-aware chunking
    os.environ["DOCS_TO_EVAL_TESTING_MODE"] = "true"  # force fresh chunking, helpful logs
    chunk_cfg = ChunkingConfig(
        enable_chonkie=True,
        chunking_strategy="semantic",
        use_token_chunking=True,
        target_token_size=1500,  # Slightly smaller for test speed
        min_token_size=800,
        max_token_size=2500,
        overlap_tokens=150,
    )

    chunks = create_smart_chunks_from_files(file_contents, chunk_cfg)
    assert isinstance(chunks, list) and len(chunks) > 0, "Chunking should produce at least one chunk"

    # Build corpus_text by concatenating top chunks by semantic score (cap to avoid huge strings)
    chunks_sorted = sorted(chunks, key=lambda c: c.get("semantic_score", 0), reverse=True)
    selected = chunks_sorted[:5] if len(chunks_sorted) > 5 else chunks_sorted
    corpus_text = "\n\n".join(c.get("text", "") for c in selected)
    assert len(corpus_text.strip()) > 0, "Concatenated corpus_text should not be empty"

    # Agentic generation with Mock LLMs
    mock_llm = MockLLMInterface(model_name="test-mock", temperature=0.5)
    llm_pool = {
        "retriever": mock_llm,
        "creator": mock_llm,
        "adversary": mock_llm,
        "refiner": mock_llm,
    }

    pipe_cfg = PipelineConfig(
        difficulty=DifficultyLevel.INTERMEDIATE,
        num_questions=10,
        parallel_batch_size=2,
        max_retry_cycles=1,
    )

    orchestrator = AgenticBenchmarkOrchestrator(llm_pool, pipe_cfg)

    items = await orchestrator.generate(
        corpus_text=corpus_text,
        eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
        num_questions=10,
        difficulty=DifficultyLevel.INTERMEDIATE,
    )

    # Basic validations
    assert len(items) > 0, "Should generate at least one item"
    # Print a compact preview for debugging (not required assertions)
    for i, item in enumerate(items[:3]):
        print(f"Q{i+1}: {item.question[:100]} | A: {item.answer[:80]}")


