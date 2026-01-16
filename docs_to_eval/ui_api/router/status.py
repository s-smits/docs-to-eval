
from fastapi import APIRouter, WebSocket
import os
from datetime import datetime
from ..websockets import handle_websocket_connection
from ...utils.logging import get_logger

router = APIRouter()
logger = get_logger("status_routes")

@router.get("/")
async def api_root():
    """API root endpoint"""
    return {
        "message": "docs-to-eval API v1",
        "endpoints": {
            "corpus": "/corpus",
            "evaluation": "/evaluation",
            "results": "/results",
            "websocket": "/ws/{run_id}"
        }
    }

@router.get("/health")
async def health_check():
    """API health check"""
    from .evaluation import evaluation_runs
    from ..websockets import websocket_manager
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_runs": len(await evaluation_runs.list_runs()),
        "websocket_connections": len(websocket_manager.active_connections)
    }

@router.get("/debug/env")
async def debug_env():
    """Debug endpoint to check environment variables"""
    return {
        "openrouter_key_set": bool(os.getenv("OPENROUTER_API_KEY")),
        "openrouter_key_length": len(os.getenv("OPENROUTER_API_KEY", "")),
        "pytorch_mps_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK"),
        "docs_to_eval_provider": os.getenv("DOCS_TO_EVAL_PROVIDER"),
        "docs_to_eval_model_name": os.getenv("DOCS_TO_EVAL_MODEL_NAME"),
        "env_vars_count": len([k for k in os.environ.keys() if k.startswith(("OPENROUTER", "PYTORCH", "DOCS_TO_EVAL"))])
    }

@router.get("/debug/env_check") # Note: was env-check in routes.py, let's keep it hyphenated if needed
@router.get("/debug/env-check")
async def debug_env_check():
    """Debug endpoint to check environment variable configuration status"""
    api_key_or = os.getenv("OPENROUTER_API_KEY")
    api_key_gte = os.getenv("DOCS_TO_EVAL_API_KEY")

    provider = os.getenv("DOCS_TO_EVAL_PROVIDER", "NOT SET")
    model_name = os.getenv("DOCS_TO_EVAL_MODEL_NAME", "NOT SET")

    configured = False
    message = "LLM API key needs to be set."

    if api_key_gte and api_key_gte != "your_api_key_here" and len(api_key_gte) > 10:
        configured = True
        message = f"LLM configured with DOCS_TO_EVAL_API_KEY for provider: {provider}, model: {model_name}"
    elif api_key_or and api_key_or != "your_api_key_here" and len(api_key_or) > 10:
        configured = True
        message = f"LLM configured with OPENROUTER_API_KEY for provider: {provider}, model: {model_name}"

    return {
        "status": "configured" if configured else "not_configured",
        "docs_to_eval_api_key_set": bool(api_key_gte),
        "openrouter_api_key_set": bool(api_key_or),
        "docs_to_eval_provider": provider,
        "docs_to_eval_model_name": model_name,
        "message": message
    }

@router.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await handle_websocket_connection(websocket, run_id)
