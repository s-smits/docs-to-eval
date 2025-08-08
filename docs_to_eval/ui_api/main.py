"""
FastAPI main application for docs-to-eval system
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path

from .routes import router
from ..utils.logging import setup_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    import os
    
    # Startup
    setup_logging("INFO", "logs")
    logger = get_logger("app")
    logger.info("Starting docs-to-eval API server")
    
    # Verify environment variables
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        logger.info(f"OPENROUTER_API_KEY configured (length: {len(api_key)})")
    else:
        logger.warning("OPENROUTER_API_KEY not configured - agentic evaluation will not work")
    
    # Create necessary directories
    directories = ["output", "logs", "cache", "uploads"]
    for directory in directories:
        path = Path(directory)
        path.mkdir(exist_ok=True)
        logger.info(f"Ensured directory exists: {path.absolute()}")
    
    # Verify critical components
    # Lazy import in request handlers; avoid unused import warnings here
    try:
        __import__("docs_to_eval.core.pipeline")
        __import__("docs_to_eval.llm.openrouter_interface")
        logger.info("✅ Core components available")
    except ImportError as e:
        logger.error(f"❌ Core components may be unavailable: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down docs-to-eval API server")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="docs-to-eval API",
        description="Automated LLM Evaluation System API",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(router, prefix="/api/v1")
    
    # Mount static files (for serving the frontend)
    static_path = Path(__file__).parent.parent.parent / "frontend" / "dist"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    
    # Root endpoint - serve the frontend
    @app.get("/")
    async def root():
        static_path = Path(__file__).parent.parent.parent / "frontend" / "dist" / "index.html"
        if static_path.exists():
            from fastapi.responses import FileResponse
            return FileResponse(static_path)
        else:
            return {
                "message": "docs-to-eval API",
                "version": "1.0.0",
                "docs_url": "/docs",
                "websocket_url": "/api/v1/ws/{run_id}",
                "frontend": "Frontend files not found"
            }
    
    # Health check
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "docs-to-eval-api"}
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "docs_to_eval.ui_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )