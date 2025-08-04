"""
FastAPI main application for docs-to-eval system
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
from pathlib import Path
from typing import Dict, Any

from .routes import router
from .websockets import websocket_manager
from ..utils.logging import setup_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    setup_logging("INFO", "logs")
    logger = get_logger("app")
    logger.info("Starting docs-to-eval API server")
    
    # Create necessary directories
    Path("output").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("cache").mkdir(exist_ok=True)
    Path("uploads").mkdir(exist_ok=True)
    
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