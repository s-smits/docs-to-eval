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

```bash
uv run pytest
```

Use `DOCS_TO_EVAL_TESTING_MODE=true` to force deterministic chunking and disable caches when running tests locally.

## Contributing
- Keep dependencies in sync with `uv lock` and `uv sync`.
- Aim for Pydantic v2-style validators (`field_validator`, `model_validator`) when adding models.
- Open an issue if you need support for additional evaluation strategies or new verification plugins.
