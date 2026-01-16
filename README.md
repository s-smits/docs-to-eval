# docs-to-eval

docs-to-eval builds tailor-made evaluation sets from raw documentation and scores model outputs with the right verification strategy. It combines lightweight FastAPI services, reusable Python components, and optional agentic workflows.

<img width="1428" height="744" alt="image" src="https://github.com/user-attachments/assets/fba0bcc2-d6da-48ea-9c7c-03f61451e4c1" />

## Why docs-to-eval
- Classifies corpora into deterministic vs. generative evaluation modes and picks the right metrics automatically.
- Builds benchmarks with agentic strategies, nuanced quality scoring, and domain-aware difficulty tuning.
- Verifies answers with exact matching, execution sandboxes, similarity scoring, or LLM judging depending on the task.
- Ships a React front-end and REST API so you can run evaluations from a browser or another system.

## Quick start
Prerequisites: Python 3.11+, [uv](https://github.com/astral-sh/uv), Node.js 20+ (for the frontend).

```bash
# Set up Python environment
uv venv .venv
uv sync

# Start the FastAPI backend
uv run uvicorn docs_to_eval.app:app --host 0.0.0.0 --port 8080 --reload

# In a second terminal, start the React frontend
cd frontend
npm install
npm run dev

# CLI entry point (optional)
uv run docs-to-eval evaluate --help
```

The API is available at `http://localhost:8080`. Point the Next.js app at it by creating
`frontend/.env.local`:

```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8080
NEXT_PUBLIC_API_WS_BASE=ws://localhost:8080
```

## Groq and Gemini in docs-to-eval
- Install SDKs alongside the core project: `uv add groq google-generativeai`.
- Provide credentials before running local scripts or the FastAPI app:
  - `export GROQ_API_KEY=<your key>`
  - `export GEMINI_API_KEY=<your key>` (or `GOOGLE_API_KEY`)
- Create an interface on demand with `create_llm_interface('groq', model='compound')` or `create_llm_interface('gemini', model='gemini-2.5-flash')` and pass it into pipelines, CLI runs, or custom scripts.
  - **For Groq, you can choose**: `groq/compound`
  - **For Gemini, you can choose**: `gemini-2.5-flash` (fast, cost-effective), `gemini-2.5-pro` (advanced reasoning, 1M token context), `gemini-2.0-flash` (multimodal streaming)
- Batch helpers (`GroqBatchInterface`, `GeminiBatchInterface`) support configurable concurrency; inspect `get_batch_stats()` for throughput data.
- Use `tests/manual/llm_provider_diagnostics.py` to verify credentials, latency, and error handling before wiring providers into production flows.

## Examples
- `examples/agentic_pipeline_walkthrough.py` – end-to-end agentic benchmark generation with the FastAPI facade.
- `examples/etruscan_corpus_agentic_demo.py` – semantic chunking plus concept mining on the Etruscan sample corpus.
- `examples/mixed_verification_showcase.py` – illustrates numeric, factual, and mixed verification improvements.
- `examples/mixed_verification_showcase.py` – illustrates numeric, factual, and mixed verification improvements.
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
├── llm/                # MLX, OpenRouter, Qwen & mock interfaces
├── ui_api/             # Modular routers (corpus, config, evaluation, status)
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

# Run frontend unit tests (Vitest + Testing Library)
cd frontend
npm test -- --run
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
