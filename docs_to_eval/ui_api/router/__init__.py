
from fastapi import APIRouter
from .corpus import router as corpus_router
from .evaluation import router as evaluation_router
from .config import router as config_router
from .status import router as status_router

# Create main API router
router = APIRouter()

# Include sub-routers with logical prefixes
router.include_router(status_router, tags=["status"])
router.include_router(corpus_router, prefix="/corpus", tags=["corpus"])
router.include_router(evaluation_router, prefix="/evaluation", tags=["evaluation"])
router.include_router(evaluation_router, tags=["runs"]) # For /runs and /runs/{run_id}
router.include_router(config_router, prefix="/config", tags=["config"])
router.include_router(config_router, tags=["llm", "types"]) # For /llm and /types
