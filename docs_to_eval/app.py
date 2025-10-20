"""ASGI entrypoint so `uvicorn docs_to_eval.app:app` works out of the box."""

from .ui_api.main import create_app

# Expose the FastAPI instance for ASGI servers
app = create_app()
