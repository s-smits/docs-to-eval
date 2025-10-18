"""
Agentic Pipeline Walkthrough

Runs the agentic benchmarking pipeline end-to-end with mock LLM components and the FastAPI interface.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from docs_to_eval.core.agentic.orchestrator import AgenticBenchmarkOrchestrator
from docs_to_eval.core.agentic.models import DifficultyLevel, PipelineConfig
from docs_to_eval.core.evaluation import EvaluationType
from docs_to_eval.core.verification import VerificationOrchestrator
from docs_to_eval.llm.mock_interface import MockLLMInterface
from docs_to_eval.ui_api.main import app
from docs_to_eval.utils.text_processing import ChunkingConfig, create_smart_chunks_from_files


FAKE_CORPUS = (
    "The Helios Resonator is a high-efficiency solar energy converter developed in 2024. "
    "It integrates perovskite tandem cells with a micro-lens array to increase photon capture in low-light conditions. "
    "The system includes a thermal buffer to stabilize output and a smart MPPT controller tuned for partial shading. "
    "A standard Helios Resonator module operates at 23-26% conversion efficiency under diffuse daylight and can reach 31% under direct sunlight. "
    "Compared to legacy mono-junction silicon panels, the Helios design reduces temperature-induced efficiency losses by approximately 40%. "
)


def run_agentic_generation() -> None:
    print("\n=== Agentic Orchestrator Generation (Mock LLM) ===")
    mock_llm = MockLLMInterface(model_name="mock-agent", temperature=0.3)
    llm_pool = {
        "retriever": mock_llm,
        "creator": mock_llm,
        "adversary": mock_llm,
        "refiner": mock_llm,
    }
    config = PipelineConfig(
        difficulty=DifficultyLevel.INTERMEDIATE,
        num_questions=3,
        min_validation_score=0.6,
        parallel_batch_size=2,
        max_retry_cycles=1,
    )

    orchestrator = AgenticBenchmarkOrchestrator(llm_pool, config)

    async def _run():
        items = await orchestrator.generate(
            corpus_text=FAKE_CORPUS,
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
            num_questions=3,
            difficulty=DifficultyLevel.INTERMEDIATE,
        )
        print(f"Generated {len(items)} items")
        verifier = VerificationOrchestrator()

        for idx, item in enumerate(items, 1):
            # LLM answers only with the question (no access to gold answer)
            pred = (await mock_llm.generate_response(item.question)).text
            # Verify
            vr = verifier.verify(
                prediction=pred,
                ground_truth=item.answer,
                eval_type=EvaluationType.FACTUAL_QA.value,
                options=item.options,
                question=item.question,
            )
            print(f"\nItem {idx}:")
            print(f"Question: {item.question}")
            print(f"Context: {item.context}")
            print(f"Expected: {item.answer}")
            print(f"LLM Answer: {pred}")
            print(f"Score: {vr.score:.3f} | Method: {vr.method}")

    asyncio.run(_run())


def run_api_route_flow() -> None:
    print("\n=== FastAPI Route Evaluation (Mock) ===")
    client = TestClient(app)

    payload = {
        "corpus_text": FAKE_CORPUS,
        "eval_type": "domain_knowledge",
        "num_questions": 2,
        "use_agentic": False,
        "run_name": "Helios Resonator E2E",
    }

    start = client.post("/api/v1/evaluation/start", json=payload)
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    print(f"Started run: {run_id}")

    # Poll for completion
    for _ in range(200):
        status = client.get(f"/api/v1/evaluation/{run_id}/status")
        assert status.status_code == 200, status.text
        data = status.json()
        if data.get("status") == "completed":
            results = data.get("results") or {}
            # Print a compact summary
            print("Status: completed")
            print("Aggregate metrics:")
            agg = (results.get("aggregate_metrics") or {})
            print(json.dumps(agg, indent=2))

            # Show first couple of individual results if present
            ind: List[dict] = (results.get("individual_results") or [])
            for i, r in enumerate(ind[:2], 1):
                print(f"\nRoute Result {i}:")
                # Keys differ depending on route path; print robustly
                print(json.dumps(r, indent=2, default=str))
            break
        elif data.get("status") == "error":
            print("Status: error")
            print(json.dumps(data, indent=2))
            break


if __name__ == "__main__":
    run_agentic_generation()
    run_api_route_flow()
    
    # Chunk directory and run agentic on a selected chunk
    def run_chunking_and_agentic_on_real_files() -> None:
        print("\n=== Chunking Directory then Agentic on One Chunk ===")
        corpus_dir = Path("/Users/air/Developer/docs-to-eval/data")
        files = []
        for p in sorted(corpus_dir.glob("**/*")):
            if p.is_file() and p.suffix.lower() in {".txt", ".md", ".json"}:
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    if text.strip():
                        files.append({"filename": str(p.name), "content": text})
                except Exception:
                    continue
            if len(files) >= 5:
                break
        if not files:
            print("No files found to chunk.")
            return

        chunk_cfg = ChunkingConfig(
            enable_chonkie=False,
            use_token_chunking=False,
            target_chunk_size=4000,
            min_chunk_size=1200,
            max_chunk_size=8000,
            overlap_size=200,
        )
        chunks = create_smart_chunks_from_files(files, chunk_cfg)
        print(f"Created {len(chunks)} chunks from {len(files)} files")
        if not chunks:
            return

        # Pick the most semantically coherent chunk
        best = max(chunks, key=lambda c: c.get("semantic_score", 0))
        context = best["text"][:500]

        mock_llm = MockLLMInterface(model_name="mock-agent", temperature=0.3)
        llm_pool = {"retriever": mock_llm, "creator": mock_llm, "adversary": mock_llm, "refiner": mock_llm}
        config = PipelineConfig(
            difficulty=DifficultyLevel.INTERMEDIATE,
            num_questions=1,
            min_validation_score=0.6,
            parallel_batch_size=1,
            max_retry_cycles=0,
        )
        orchestrator = AgenticBenchmarkOrchestrator(llm_pool, config)

        async def _run_one():
            concept = "core_topic"
            draft = await orchestrator.question_writer.produce(
                concept, context, EvaluationType.DOMAIN_KNOWLEDGE, context
            )
            candidate = await orchestrator.adversary.produce(draft, DifficultyLevel.INTERMEDIATE)
            refined = await orchestrator.refiner.produce(candidate)

            pred = (await mock_llm.generate_response(refined.question)).text
            verifier = VerificationOrchestrator()
            vr = verifier.verify(
                prediction=pred,
                ground_truth=refined.answer,
                eval_type=EvaluationType.FACTUAL_QA.value,
                options=refined.options,
                question=refined.question,
            )

            print("\n[Selected Chunk Preview]\n" + context[:400] + ("..." if len(context) > 400 else ""))
            print("\nGenerated Q&A from selected chunk:")
            print(f"Question: {refined.question}")
            print(f"Context (trimmed): {context[:200]}...")
            print(f"Expected: {refined.answer}")
            print(f"LLM Answer: {pred}")
            print(f"Score: {vr.score:.3f} | Method: {vr.method}")

        asyncio.run(_run_one())

    run_chunking_and_agentic_on_real_files()
