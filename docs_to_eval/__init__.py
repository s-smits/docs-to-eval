"""
docs-to-eval: Automated LLM Evaluation System

A comprehensive framework for generating domain-specific benchmarks from text corpora
and evaluating LLMs using appropriate metrics and verification methods.
"""

# Ensure environment variables from .env are available as early as possible
try:
    from pathlib import Path
    from dotenv import load_dotenv
    # Load from current working directory and repository root
    load_dotenv(override=False)
    repo_root_env = Path(__file__).resolve().parents[1] / ".env"
    if repo_root_env.exists():
        load_dotenv(dotenv_path=repo_root_env, override=False)
except Exception:
    # Do not fail import if dotenv is unavailable
    pass

__version__ = "1.0.0"
__author__ = "docs-to-eval Team"

from .core.evaluation import EvaluationFramework
from .core.classification import EvaluationTypeClassifier
from .core.benchmarks import BenchmarkGeneratorFactory
from .llm.mock_interface import MockLLMInterface

# Optional FastAPI components
try:
    from .ui_api.main import create_app
    from .app import app  # FastAPI ASGI app
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    create_app = None
    app = None

__all__ = [
    "EvaluationFramework",
    "EvaluationTypeClassifier", 
    "BenchmarkGeneratorFactory",
    "MockLLMInterface"
]

if _HAS_FASTAPI:
    __all__.extend(["create_app", "app"])
