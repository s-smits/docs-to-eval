"""
Utilities and manual integration workflows that are useful during
development but are not part of the automated pytest suite.

Each module in this package is intentionally interactive or network
dependent. Run them manually when you need deep diagnostics or to
capture rich validation artifacts.
"""

from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


__all__ = ["RESULTS_DIR"]
