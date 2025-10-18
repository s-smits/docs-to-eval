<img width="911" height="911" alt="image" src="https://github.com/user-attachments/assets/d79500c7-55ff-4c5e-8ba4-049ed3617127" />

# docs-to-eval

docs-to-eval builds tailor-made evaluation sets from raw documentation and scores model outputs with the right verification strategy. It combines lightweight FastAPI services, reusable Python components, and optional agentic workflows.

## Why docs-to-eval
- Classifies corpora into deterministic vs. generative evaluation modes and picks the right metrics automatically.
- Generates benchmarks with agentic strategies, quality scoring, and domain-aware difficulty balancing.
- Verifies answers with exact matching, execution sandboxes, similarity scoring, or LLM judging depending on the task.
- Ships a React front-end and REST API so you can run evaluations from a browser or another system.

## Quick start
Prerequisites: Python 3.11+, [uv](https://github.com/astral-sh/uv), Node.js 20+ (for the frontend).

```bash
# Set up Python environment
uv venv .venv
uv sync

# Start the API + UI
uv run python run_server.py

# Or invoke the CLI
uv run python -m docs_to_eval.cli.main --help
```

The API is available at `http://localhost:8080`; the React UI is served from the same process during development.

## Examples
- `examples/agentic_pipeline_walkthrough.py` – end-to-end agentic benchmark generation with the FastAPI facade.
- `examples/etruscan_corpus_agentic_demo.py` – semantic chunking plus concept mining on the Etruscan sample corpus.
- `examples/mixed_verification_showcase.py` – illustrates numeric, factual, and mixed verification improvements.
- `examples/local_qwen_pipeline_demo.py` – runs the evaluation pipeline against a local or mock Qwen model.
- `examples/backend_agent_loop_demo.py` – drives the complete backend agent loop and writes reports to `tests/manual/results/`.

## Manual diagnostics
- `tests/manual/llm_provider_diagnostics.py` – interactive Groq/Gemini connectivity and pipeline sanity checks.
- `tests/manual/critical_fixes_walkthrough.py` – validates question generation, verification, and context alignment fixes.
- `tests/manual/lm_eval_harness_adapter.py` – exports dynamic corpora into lm-eval-harness format for leaderboard runs.

All manual scripts persist their JSON outputs in `tests/manual/results/` (ignored by git).

## Programmatic example
```python
from pathlib import Path

from docs_to_eval.core.classification import classify_evaluation_type
from docs_to_eval.core.agentic_question_generator import AgenticQuestionGenerator
from docs_to_eval.core.verification import VerificationOrchestrator

corpus = Path("domain_docs/intro.txt").read_text()

classification = classify_evaluation_type(corpus)
eval_type = classification["primary_type"]

generator = AgenticQuestionGenerator()
benchmark = generator.generate_comprehensive_benchmark(
    corpus_text=corpus,
    eval_type=eval_type,
    num_questions=25,
)

verifier = VerificationOrchestrator()
result = verifier.verify("predicted answer", benchmark[0]["answer"], eval_type)
print(result.score, result.metadata)
```

## Architecture at a glance
```
docs_to_eval/
├── core/               # classification, agentic generation, verification
├── llm/                # OpenRouter & mock interfaces
├── ui_api/             # FastAPI routes + websocket notifications
├── utils/              # config, text processing, caching helpers
└── cli/                # Typer-based command-line tools
frontend/               # React UI for browsing corpora and evaluation runs
```

## Testing

The test suite is now split between fast automated checks and heavier manual flows:

```bash
# Run the automated fast suite (used in CI)
uv run pytest tests/automated

# Run the full automated suite, mirroring CI
uv run pytest
```

Manual walk-through scripts live in `tests/manual/`; they are intentionally excluded from
the default pytest discovery because they hit external services or require API keys. Run
those directly via `uv run python tests/manual/<script>.py` when you need the longer
scenarios.

Use `DOCS_TO_EVAL_TESTING_MODE=true` to force deterministic chunking and disable caches
when running tests locally.

## Contributing
- Keep dependencies in sync with `uv lock` and `uv sync`.
- Aim for Pydantic v2-style validators (`field_validator`, `model_validator`) when adding models.
- Open an issue if you need support for additional evaluation strategies or new verification plugins.
