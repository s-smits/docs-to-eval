"""
FastAPI routes for docs-to-eval system
"""

import asyncio
import uuid
import json
import httpx
import re
from pathlib import Path
import os
from collections import OrderedDict
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, Form, status
from fastapi.responses import FileResponse, JSONResponse # Import JSONResponse
from pydantic import BaseModel, Field, ValidationError, validator, root_validator

# Local imports
from .websockets import websocket_manager, handle_websocket_connection, get_progress_tracker
from ..core.classification import EvaluationTypeClassifier, ClassificationResult
from ..utils.config import EvaluationType
from ..core.agentic.agents import Validator # Keep Validator for potential use in evaluation flow
from ..core.agentic.orchestrator import AgenticBenchmarkOrchestrator # No longer directly used, but kept for context if needed
from ..core.agentic.models import PipelineConfig, DifficultyLevel
from ..llm.mock_interface import MockLLMInterface # Keep mock for fallback
from ..llm.openrouter_interface import OpenRouterInterface, OpenRouterConfig # Keep OpenRouterConfig for OpenRouter specific handling in factory
from ..utils.config import EvaluationConfig, create_default_config, ConfigManager
from ..utils.logging import get_logger
from ..utils.text_processing import create_smart_chunks_from_files
from ..llm.llm_factory import get_llm_interface, list_llm_models # Import list_llm_models
from ..llm.base import BaseLLMInterface, LLMResponse # Import BaseLLMInterface for type hinting
from dotenv import load_dotenv, set_key


# Create router
router = APIRouter()
logger = get_logger("api_routes")


# Pydantic models for API
class CorpusUploadRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10*1024*1024, description="Corpus text content")
    name: Optional[str] = Field("corpus", max_length=100, description="Corpus name")
    description: Optional[str] = Field("", max_length=500, description="Corpus description")

    @validator('text')
    def validate_text_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Text content cannot be empty")

        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:.*base64',
            r'\\x[0-9a-fA-F]{2}',
            r'eval\s*\(',
            r'exec\s*\('
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE | re.DOTALL):
                raise ValueError("Text contains potentially malicious content")

        return v.strip()

    @validator('name')
    def validate_name(cls, v):
        if v:
            # Sanitize name by removing special characters and keeping only safe ones
            # Remove all special characters except spaces, hyphens, underscores, and parentheses
            sanitized = re.sub(r'[^\w\s\-_\(\)]', '', v)
            sanitized = re.sub(r'\s+', ' ', sanitized).strip()  # Normalize spaces

            if len(sanitized) < 3:
                # Use a default name if sanitization results in too short name
                sanitized = "Uploaded Corpus"

            return sanitized
        return v.strip() if v else v


class EvaluationRequest(BaseModel):
    corpus_text: str = Field(..., min_length=1, max_length=10*1024*1024, description="Corpus text content")
    eval_type: Optional[str] = Field(None, max_length=50, description="Evaluation type")
    num_questions: int = Field(default=20, ge=1, le=200, description="Number of questions to generate")
    use_agentic: bool = Field(default=True, description="Use agentic generation")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Temperature for generation")
    token_threshold: int = Field(default=2000, ge=500, le=4000, description="Token threshold for chunk concatenation")
    run_name: Optional[str] = Field(None, max_length=100, description="Name for this evaluation run")

    # Finetune test set configuration
    finetune_test_set_enabled: bool = Field(default=True, description="Enable creation of finetune test set")
    finetune_test_set_percentage: float = Field(default=0.2, ge=0.1, le=0.5, description="Percentage of questions for test set (e.g., 0.2 = 20%)")
    finetune_random_seed: int = Field(default=42, ge=0, le=999999, description="Random seed for reproducible splits")

    # LLM Provider and Model Selection
    provider: str = Field(default="openrouter", description="Selected LLM Provider")
    modelName: str = Field(default="anthropic/claude-sonnet-4", description="Selected LLM Model Name")


    @root_validator(pre=True)
    def validate_corpus_text(cls, values):
        corpus_text = values.get("corpus_text")
        if corpus_text is not None:
            if not isinstance(corpus_text, str):
                raise ValueError("corpus_text must be a string")
            if not corpus_text.strip():
                raise ValueError("corpus_text cannot be empty")
        return values

    @root_validator(pre=True)
    def validate_eval_type(cls, values):
        eval_type = values.get("eval_type")
        if eval_type is not None:
            if not isinstance(eval_type, str):
                raise ValueError("eval_type must be a string")
            if eval_type.lower() not in [e.value for e in EvaluationType] + ["auto-detect", "auto"]:
                raise ValueError(f"Invalid eval_type: {eval_type}")
        return values

    @root_validator(pre=True)
    def validate_run_name(cls, values):
        run_name = values.get("run_name")
        if run_name is not None:
            if not isinstance(run_name, str):
                raise ValueError("run_name must be a string")
            if not run_name.strip():
                raise ValueError("run_name cannot be empty if provided")
        return values


class EvaluationStatus(BaseModel):
    run_id: str
    status: str
    phase: Optional[str] = None
    progress_percent: float = 0
    message: str = "Queued"
    error: Optional[str] = None
    start_time: datetime
    estimated_completion: Optional[datetime] = None


class QwenEvaluationRequest(BaseModel):
    corpus_text: str = Field(..., min_length=1, max_length=5*1024*1024, description="Corpus text content")
    num_questions: int = Field(default=5, ge=1, le=20, description="Number of questions to generate")
    use_fictional: bool = Field(default=True, description="Use fictional content enhancement")
    token_threshold: int = Field(default=2000, ge=500, le=4000, description="Token threshold for chunk concatenation")
    run_name: Optional[str] = Field("Qwen Local Test", max_length=100, description="Name for this evaluation run")

    @root_validator(pre=True)
    def validate_corpus_text(cls, values):
        corpus_text = values.get("corpus_text")
        if corpus_text is not None:
            if not isinstance(corpus_text, str):
                raise ValueError("corpus_text must be a string")
            if not corpus_text.strip():
                raise ValueError("corpus_text cannot be empty")
        return values

    @root_validator(pre=True)
    def validate_run_name(cls, values):
        run_name = values.get("run_name")
        if run_name is not None:
            if not isinstance(run_name, str):
                raise ValueError("run_name must be a string")
            if not run_name.strip():
                raise ValueError("run_name cannot be empty if provided")
        return values


class EvaluationResult(BaseModel):
    run_id: str
    status: str
    message: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    # Top-level aggregate metrics for compatibility with tests expecting it at root
    aggregate_metrics: Optional[Dict[str, Any]] = None


# In-memory storage for evaluation runs with size limits

class EvaluationRunManager:
    def __init__(self, max_runs: int = 100, max_age_hours: int = 24):
        self._runs: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_runs = max_runs
        self.max_age_seconds = max_age_hours * 3600
        self.lock = asyncio.Lock() # Initialize the lock here
        # Start cleanup task
        BackgroundTasks().add_task(self._cleanup_old_runs)

    async def add_run(self, run_id: str, run_info: Dict[str, Any]):
        """Add run with automatic cleanup"""
        # Clean old runs first
        self._cleanup_old_runs()

        # Add new run
        async with self.lock:
            self._runs[run_id] = run_info

        # Enforce size limit
        while len(self._runs) > self.max_runs:
            async with self.lock:
                self._runs.popitem(last=False)  # Remove oldest

    async def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run info"""
        async with self.lock:
            return self._runs.get(run_id)

    async def update_run(self, run_id: str, updates: Dict[str, Any]):
        """Update run info"""
        async with self.lock:
            if run_id in self._runs:
                self._runs[run_id].update(updates)

    async def delete_run(self, run_id: str):
        """Delete run"""
        async with self.lock:
            self._runs.pop(run_id, None)

    async def list_runs(self) -> Dict[str, Dict[str, Any]]:
        """List all runs"""
        async with self.lock:
            self._cleanup_old_runs()
            return dict(self._runs)

    def _cleanup_old_runs(self):
        """Remove old completed runs"""
        current_time = time.time()
        to_remove = []

        for run_id, run_info in self._runs.items():
            start_time = run_info.get('start_time')
            if isinstance(start_time, datetime):
                start_time = start_time.timestamp()
            elif not isinstance(start_time, (int, float)):
                start_time = current_time

            # Remove old completed/error runs
            if (run_info.get('status') in ['completed', 'error'] and
                current_time - start_time > self.max_age_seconds):
                to_remove.append(run_id)

        for run_id in to_remove:
            self._runs.pop(run_id, None)

# Replace global dict with managed storage
evaluation_runs = EvaluationRunManager()


@router.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await handle_websocket_connection(websocket, run_id)


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


# NEW ENDPOINT ADDED HER
@router.get("/llm/models", response_model=List[str])
async def get_llm_models(provider: str):
    """Get a list of available LLM models for a given provider."""
    try:
        model_data = list_llm_models(provider)
        # Extract just the model names from the dict format
        if model_data and isinstance(model_data[0], dict):
            models = [item["value"] for item in model_data]
        else:
            models = model_data
        return models
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting LLM models for provider {provider}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/corpus/upload")
async def upload_corpus_text(request: CorpusUploadRequest):
    """Upload corpus text for analysis"""
    try:
        # Analyze corpus
        classifier = EvaluationTypeClassifier()
        classification = classifier.classify_corpus(request.text)

        # Generate corpus ID
        corpus_id = str(uuid.uuid4())

        # Store corpus information
        corpus_info = {
            "id": corpus_id,
            "name": request.name,
            "description": request.description,
            "text": request.text,
            "classification": classification.to_dict(),
            "stats": {
                "characters": len(request.text),
                "words": len(request.text.split()),
                "lines": len(request.text.splitlines())
            },
            "created_at": datetime.now().isoformat()
        }

        logger.info("Corpus uploaded", corpus_id=corpus_id,
                   chars=len(request.text),
                   primary_type=classification.primary_type)

        return {
            "corpus_id": corpus_id,
            "classification": classification.to_dict(),
            "stats": corpus_info["stats"],
            "corpus_text": request.text  # Include the corpus text
        }

    except Exception as e:
        logger.error(f"Error uploading corpus: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/corpus/upload-file")
async def upload_corpus_file(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    """Upload corpus from file with validation"""
    try:
        # Validate file size (10MB limit)
        MAX_FILE_SIZE = 10 * 1024 * 1024
        content = await file.read()

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {len(content)} bytes exceeds {MAX_FILE_SIZE} bytes limit"
            )

        # Validate file extension
        if not file.filename:
            raise HTTPException(status_code=400, detail="File must have a name")

        allowed_extensions = {'.txt', '.md', '.py', '.js', '.json', '.csv', '.html', '.xml', '.yml', '.yaml', '.cfg', '.ini', '.log'}
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not allowed. Supported: {', '.join(allowed_extensions)}"
            )

        # Safely decode content
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text = content.decode('latin-1')
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Cannot decode file as text")

        # Validate text length
        MAX_TEXT_LENGTH = 5 * 1024 * 1024  # 5MB text limit
        if len(text) > MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=413,
                detail=f"Text content too long: {len(text)} characters exceeds {MAX_TEXT_LENGTH} limit"
            )

        # Basic content validation - check for suspicious patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:.*base64',
            r'\\x[0-9a-fA-F]{2}'
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise HTTPException(status_code=400, detail="File contains suspicious content")

        # Create request
        request = CorpusUploadRequest(
            text=text,
            name=name or file.filename or "uploaded_corpus",
            description=f"Uploaded from file: {file.filename}"
        )

        return await upload_corpus_text(request)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/corpus/upload-multiple")
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    name: Optional[str] = Form(None),
    # Optional client-provided chunk thresholds (tokens)
    min_tokens: Optional[int] = Form(None),
    target_tokens: Optional[int] = Form(None),
    max_tokens: Optional[int] = Form(None),
    overlap_tokens: Optional[int] = Form(None),
):
    """Upload multiple files as a single corpus with smart chunking"""
    try:
        file_contents = []
        file_names = []
        total_size = 0

        # Supported text file extensions
        supported_extensions = {'.txt', '.md', '.py', '.js', '.json', '.csv', '.html', '.xml', '.yml', '.yaml', '.cfg', '.ini', '.log'}

        # First pass: collect all files
        for file in files:
            if not file.filename:
                continue

            file_extension = Path(file.filename).suffix.lower()

            # Skip unsupported files
            if file_extension not in supported_extensions:
                logger.warning(f"Skipping unsupported file type: {file.filename}")
                continue

            try:
                # Read file content
                content = await file.read()
                total_size += len(content)

                # Check total size limit (10MB)
                if total_size > 10 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Total file size exceeds 10MB limit")

                # Detect encoding and decode
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        text = content.decode('latin-1')
                    except UnicodeDecodeError:
                        logger.warning(f"Could not decode file: {file.filename}")
                        continue

                # Store file content for smart chunking
                file_contents.append({
                    'filename': file.filename,
                    'content': text.strip()
                })
                file_names.append(file.filename)

            except Exception as e:
                logger.warning(f"Error processing file {file.filename}: {e}")
                continue

        if not file_contents:
            raise HTTPException(status_code=400, detail="No valid text files found")

        # Use smart chunking to concatenate files to optimal chunk sizes
        from ..utils.config import ChunkingConfig

        # disable_chonkie = os.getenv("DISABLE_CHONKIE", "false").lower() in ["true", "1", "yes"]

        # Honor env only: DISABLE_CHONKIE true disables chonkie; otherwise enabled
        disable_chonkie = os.getenv("DISABLE_CHONKIE", "false").lower() in ["true", "1", "yes", "on"]

        # New config: chonkie default enabled unless disabled via env; allow client overrides
        chunking_config = ChunkingConfig(
            use_token_chunking=True,
            target_token_size=3000,
            max_token_size=4000,
            enable_chonkie=not disable_chonkie,
        )

        # Apply client-provided thresholds if present (validated & clamped)
        def clamp(v: int, lo: int, hi: int) -> int:
            return max(lo, min(hi, v))

        # Defaults / bounds
        MIN_ALLOWED = 500
        MAX_ALLOWED = 8192
        DEFAULT_MIN = 2000
        DEFAULT_TARGET = 3000
        DEFAULT_MAX = 4000
        DEFAULT_OVERLAP = 300

        # Start from defaults in config
        current_min = DEFAULT_MIN
        current_target = DEFAULT_TARGET
        current_max = DEFAULT_MAX
        current_overlap = DEFAULT_OVERLAP

        if isinstance(min_tokens, int):
            current_min = clamp(min_tokens, MIN_ALLOWED, MAX_ALLOWED)
        if isinstance(target_tokens, int):
            current_target = clamp(target_tokens, MIN_ALLOWED, MAX_ALLOWED)
        if isinstance(max_tokens, int):
            current_max = clamp(max_tokens, MIN_ALLOWED, MAX_ALLOWED)
        if isinstance(overlap_tokens, int):
            current_overlap = clamp(overlap_tokens, 0, 1024)

        # Ensure ordering: min <= target <= max
        if current_min > current_max:
            current_min, current_max = current_max, current_min
        current_target = clamp(current_target, current_min, current_max)

        # Push into config
        chunking_config.min_token_size = current_min
        chunking_config.target_token_size = current_target
        chunking_config.max_token_size = current_max
        chunking_config.overlap_tokens = current_overlap

        if disable_chonkie:
            logger.info("Chonkie disabled via DISABLE_CHONKIE environment variable")
        else:
            logger.info("Chonkie enabled (default)")

        logger.info(
            "Creating smart chunks from %d files (min=%d, target=%d, max=%d, overlap=%d, chonkie=%s)",
            len(file_contents),
            chunking_config.min_token_size,
            chunking_config.target_token_size,
            chunking_config.max_token_size,
            chunking_config.overlap_tokens,
            str(chunking_config.enable_chonkie),
        )
        chunks = create_smart_chunks_from_files(file_contents, chunking_config)

        # Combine all chunks into final text (for backward compatibility)
        combined_text = ""
        chunk_info = []

        for i, chunk in enumerate(chunks):
            if combined_text:
                combined_text += f"\n\n{'='*80}\nCHUNK {i+1} (from {len(chunk['file_sources'])} files: {', '.join(chunk['file_sources'][:3])}{'...' if len(chunk['file_sources']) > 3 else ''})\n{'='*80}\n\n"
            combined_text += chunk['text']

            chunk_info.append({
                'chunk_index': i,
                'size_chars': chunk['size'],
                'token_count': chunk.get('token_count', 'estimated'),
                'file_count': len(chunk['file_sources']),
                'files': chunk['file_sources'][:5],  # Limit to first 5 for API response
                'method': chunk['method']
            })

        # Create request
        corpus_name = name or f"Smart-chunked corpus ({len(file_names)} files → {len(chunks)} chunks)"
        request = CorpusUploadRequest(
            text=combined_text,
            name=corpus_name,
            description=f"Smart-chunked corpus from {len(file_names)} files into {len(chunks)} optimal chunks (~3k tokens each): {', '.join(file_names[:5])}" +
                       (f" and {len(file_names)-5} more" if len(file_names) > 5 else "")
        )

        result = await upload_corpus_text(request)
        result["files_processed"] = len(file_names)
        result["file_names"] = file_names
        result["chunks_created"] = len(chunks)
        result["chunking_info"] = chunk_info
        result["smart_chunking_enabled"] = True
        result["chunking_thresholds"] = {
            "min_token_size": chunking_config.min_token_size,
            "target_token_size": chunking_config.target_token_size,
            "max_token_size": chunking_config.max_token_size,
            "overlap_tokens": chunking_config.overlap_tokens,
            "chonkie_enabled": chunking_config.enable_chonkie,
        }

        logger.info(f"Smart chunking complete: {len(file_names)} files → {len(chunks)} chunks")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading multiple files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluation/start")
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Start a new evaluation run"""
    try:
        # Debug logging
        logger.info("Evaluation request received",
                   corpus_text_length=len(request.corpus_text) if request.corpus_text else 0,
                   eval_type=request.eval_type,
                   num_questions=request.num_questions,
                   use_agentic=request.use_agentic,
                   temperature=request.temperature,
                   run_name=request.run_name,
                   provider=request.provider, # Log provider
                   modelName=request.modelName) # Log model name

        # Enhanced validation
        if not request.corpus_text or not request.corpus_text.strip():
            raise HTTPException(status_code=422, detail="corpus_text is required and cannot be empty")

        # Validate corpus text length
        MAX_CORPUS_LENGTH = 10 * 1024 * 1024  # 10MB
        if len(request.corpus_text) > MAX_CORPUS_LENGTH:
            raise HTTPException(
                status_code=413,
                detail=f"Corpus text too long: {len(request.corpus_text)} characters exceeds {MAX_CORPUS_LENGTH} limit"
            )

        # Validate number of questions
        if not 1 <= request.num_questions <= 200:
            raise HTTPException(status_code=422, detail="num_questions must be between 1 and 200")

        # Validate temperature
        if not 0 <= request.temperature <= 2:
            raise HTTPException(status_code=422, detail="temperature must be between 0 and 2")

        # Generate run ID
        run_id = str(uuid.uuid4())

        # Create evaluation configuration from env, then override with request values
        config_manager = ConfigManager()
        config_manager.update_from_env()
        config = config_manager.get_config()

        # Override LLM settings with values from the request for this specific run
        config.llm.provider = request.provider
        config.llm.model_name = request.modelName
        config.llm.temperature = request.temperature
        # No request.max_tokens on EvaluationRequest, so use config's default or env
        # config.llm.api_key is handled below from environment

        # Convert string eval_type to enum
        if request.eval_type:
            try:
                # Try to find matching enum value
                eval_type = None
                for et in EvaluationType:
                    if et.value == request.eval_type:
                        eval_type = et
                        break
                config.eval_type = eval_type or EvaluationType.DOMAIN_KNOWLEDGE
            except (AttributeError, ValueError):
                # Failed to parse eval_type, use default
                config.eval_type = EvaluationType.DOMAIN_KNOWLEDGE
        else:
            config.eval_type = EvaluationType.DOMAIN_KNOWLEDGE

        config.generation.num_questions = request.num_questions
        config.generation.use_agentic = request.use_agentic
        # config.llm.temperature = request.temperature # Already set above, but ensuring consistency

        # --- FIX STARTS HERE: Correct API key loading based on requested provider ---
        api_key = None
        if request.provider == "openrouter":
            api_key = os.environ.get('OPENROUTER_API_KEY')
        elif request.provider == "groq":
            api_key = os.environ.get('GROQ_API_KEY')
        elif request.provider == "gemini_sdk": # Assuming a Gemini SDK provider exists and uses GEMINI_API_KEY
            api_key = os.environ.get('GEMINI_API_KEY')
        # Fallback to a generic API key if no provider-specific key is found
        if not api_key:
            api_key = os.environ.get('DOCS_TO_EVAL_API_KEY')

        config.llm.api_key = api_key # Set the api_key in the config object for this run
        # --- FIX ENDS HERE ---

        # Check if API key is available for agentic evaluation
        # Allow agentic evaluation with mock LLMs when no API key is provided
        if request.use_agentic and not config.llm.api_key:
            logger.warning("No API key configured for agentic evaluation - will use mock LLMs for demonstration purposes")
            # Set a flag to indicate mock mode
            config.llm.mock_mode = True
        else:
            config.llm.mock_mode = False # Ensure mock_mode is false if api key is available

        # Store run information
        run_info = {
            "run_id": run_id,
            "status": "queued",
            "phase": None,
            "request": request.dict(),
            "config": config.dict(), # Store the final config for this run
            "start_time": datetime.now(),
            "progress_percent": 0,
            "message": "Evaluation queued",
            "results": None,
            "error": None
        }

        await evaluation_runs.add_run(run_id, run_info)

        # Start evaluation in background
        background_tasks.add_task(run_evaluation, run_id, request, config)

        logger.info("Evaluation started", run_id=run_id, eval_type=config.eval_type, provider=config.llm.provider, model_name=config.llm.model_name)

        return {
            "run_id": run_id,
            "status": "queued",
            "message": "Evaluation started",
            "websocket_url": f"/api/v1/ws/{run_id}"
        }

    except ValidationError as e:
        logger.error(f"Validation error starting evaluation: {e}")
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Error starting evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluation/qwen-local")
async def start_qwen_local_evaluation(request: QwenEvaluationRequest, background_tasks: BackgroundTasks):
    """🤖 Start local Qwen evaluation with fictional content testing"""
    try:
        logger.info("Qwen local evaluation request received",
                   corpus_text_length=len(request.corpus_text) if request.corpus_text else 0,
                   num_questions=request.num_questions,
                   use_fictional=request.use_fictional,
                   run_name=request.run_name)

        # Validate required fields
        if not request.corpus_text or not request.corpus_text.strip():
            raise HTTPException(status_code=422, detail="corpus_text is required and cannot be empty")

        # Generate run ID
        run_id = str(uuid.uuid4())

        # Store run information
        run_info = {
            "run_id": run_id,
            "status": "queued",
            "phase": "qwen_local_testing",
            "request": request.dict(),
            "start_time": datetime.now(),
            "progress_percent": 0,
            "message": "Qwen local evaluation queued",
            "results": None,
            "error": None,
            "evaluation_type": "qwen_local"
        }

        await evaluation_runs.add_run(run_id, run_info)

        # Start Qwen evaluation in background
        background_tasks.add_task(run_qwen_local_evaluation, run_id, request)

        logger.info("Qwen local evaluation started", run_id=run_id)

        return {
            "run_id": run_id,
            "status": "queued",
            "message": "Qwen local evaluation started",
            "websocket_url": f"/api/v1/ws/{run_id}",
            "evaluation_type": "qwen_local"
        }

    except ValidationError as e:
        logger.error(f"Validation error starting Qwen evaluation: {e}")
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Error starting Qwen evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def generate_evaluation_questions(
    corpus_text: str,
    num_questions: int,
    eval_type: str,
    llm_config: EvaluationConfig,
    tracker,
    llm_interface: BaseLLMInterface # Pass the llm_interface directly
) -> List[Dict]:
    """
    Generate evaluation questions using the streamlined agentic orchestrator

    Uses a 3-agent pipeline:
    1. ConceptExtractor - Identifies key concepts from corpus
    2. QuestionGenerator - Creates diverse, high-quality questions
    3. QualityValidator - Ensures question quality and standards

    Args:
        corpus_text: Source text for question generation
        num_questions: Number of questions to generate
        eval_type: Type of evaluation (e.g., "domain_knowledge", "factual_qa")
        llm_config: LLM configuration object (from EvaluationConfig)
        tracker: Progress tracker for UI updates
        llm_interface: An instantiated BaseLLMInterface to use for generation

    Returns:
        List of generated question dictionaries
    """
    await tracker.send_log("info", f"🚀 Starting streamlined agentic generation for {num_questions} questions")
    await tracker.send_log("info", "📋 Pipeline: ConceptExtractor → QuestionGenerator → QualityValidator")

    try:
        # Import the streamlined orchestrator
        from docs_to_eval.core.agentic.streamlined_orchestrator import StreamlinedOrchestrator
        from docs_to_eval.core.agentic.models import DifficultyLevel

        # Initialize the streamlined orchestrator with the provided LLM interface
        orchestrator = StreamlinedOrchestrator(llm_interface) # Use the passed llm_interface

        # Determine difficulty based on eval_type
        difficulty_map = {
            "factual_qa": DifficultyLevel.BASIC,
            "domain_knowledge": DifficultyLevel.INTERMEDIATE,
            "mathematical": DifficultyLevel.HARD,
            "code_generation": DifficultyLevel.HARD,
            "multiple_choice": DifficultyLevel.INTERMEDIATE,
            "classification": DifficultyLevel.INTERMEDIATE, # Added classification
            "reading_comprehension": DifficultyLevel.INTERMEDIATE, # Added reading_comprehension
            "summarization": DifficultyLevel.INTERMEDIATE, # Added summarization
            "translation": DifficultyLevel.INTERMEDIATE, # Added translation
            "creative_writing": DifficultyLevel.INTERMEDIATE, # Added creative_writing
            "commonsense_reasoning": DifficultyLevel.INTERMEDIATE # Added commonsense_reasoning
        }
        difficulty = difficulty_map.get(eval_type, DifficultyLevel.INTERMEDIATE)

        # Generate questions
        questions = await orchestrator.generate(
            corpus_text=corpus_text,
            num_questions=num_questions,
            eval_type=eval_type,
            difficulty=difficulty,
            progress_callback=tracker
        )

        # Get and log statistics
        stats = orchestrator.get_stats()
        await tracker.send_log("info", f"📊 Generation stats: {stats['total_generated']} attempted, {stats['total_accepted']} accepted")
        await tracker.send_log("info", f"⏱️ Total time: {stats['processing_time']:.1f}s")

        # Ensure all questions have required fields
        for question in questions:
            # Add eval_type if missing
            if 'eval_type' not in question:
                question['eval_type'] = eval_type
            # Add source identifier
            if 'source' not in question:
                question['source'] = 'streamlined_agentic'
            # Ensure context is present
            if 'context' not in question or not question['context']:
                question['context'] = corpus_text[:500]

        await tracker.send_log("success", f"✅ Successfully generated {len(questions)} questions")
        return questions

    except Exception as e:
        logger.error(f"Question generation failed: {str(e)}", exc_info=True)
        await tracker.send_log("error", f"❌ Generation failed: {str(e)}")

        # Fallback to simple generation
        await tracker.send_log("info", "Using fallback generation...")
        return await generate_corpus_questions(corpus_text, num_questions, eval_type, tracker)


async def generate_corpus_questions(corpus_text: str, num_questions: int, eval_type: str, tracker) -> List[Dict]:
    """Generate questions using improved template approach with content-type detection"""
    import random

    questions = []

    # Extract sentences and key phrases from corpus
    sentences = re.split(r'[.!?]+', corpus_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    # Extract potential topics and concepts
    words = re.findall(r'\b[A-Z][a-z]+\b', corpus_text)  # Capitalized words (likely topics)
    numbers = re.findall(r'\b\d+\.?\d*\b', corpus_text)  # Numbers for mathematical content

    # Question templates based on evaluation type
    if eval_type == "mathematical":
        templates = [
            "What is the result of {concept}?",
            "Calculate the value when {concept}.",
            "Solve for {concept}.",
            "What does {concept} equal?",
            "Find the solution to {concept}."
        ]
    elif eval_type == "factual_qa":
        templates = [
            "What is {concept}?",
            "Define {concept}.",
            "Explain {concept}.",
            "What does {concept} mean?",
            "Describe {concept}."
        ]
    elif eval_type == "code_generation":
        templates = [
            "Write code to implement {concept}.",
            "How would you code {concept}?",
            "Create a function for {concept}.",
            "Write a program that {concept}.",
            "Implement {concept} in code."
        ]
    else:  # domain_knowledge and others
        templates = [
            "What is {concept}?",
            "Explain the concept of {concept}.",
            "How does {concept} work?",
            "What are the key aspects of {concept}?",
            "Describe the importance of {concept}."
        ]

    # Generate questions
    generated_count = 0
    for i in range(num_questions):
        try:
            # Pick a random sentence or concept
            if sentences and random.random() > 0.3:
                # Use a sentence from the corpus
                sentence = random.choice(sentences)
                # Extract key concept from sentence
                concept_words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence)
                if concept_words:
                    concept = random.choice(concept_words)
                else:
                    concept = sentence[:50] + "..."
            else:
                # Use a capitalized word as concept
                concept = random.choice(words) if words else f"concept {i+1}"

            # Generate question
            template = random.choice(templates)
            question = template.format(concept=concept)

            # Generate a basic answer (extract from context)
            answer_context = ""
            for sentence in sentences:
                if concept.lower() in sentence.lower():
                    answer_context = sentence.strip()
                    break

            if not answer_context and numbers:
                answer_context = f"The answer involves: {', '.join(numbers[:3])}"
            elif not answer_context:
                answer_context = f"Information about {concept}"

            questions.append({
                "question": question,
                "answer": answer_context,
                "context": sentence if 'sentence' in locals() else None,
                "eval_type": eval_type,
                "concept": concept
            })

            generated_count += 1
            # Track progress based on successfully generated questions
            await tracker.increment_progress(message=f"Generated question {generated_count}/{num_questions}: {question[:50]}...")
            await asyncio.sleep(0.05)  # Brief pause

        except Exception as e:
            logger.warning(f"Error generating question {i+1}: {e}")
            # Fallback question
            questions.append({
                "question": f"What can you tell me about the content in section {i+1}?",
                "answer": f"Information from the provided corpus about topic {i+1}",
                "context": None,
                "eval_type": eval_type
            })
            generated_count += 1
            await tracker.increment_progress(message=f"Generated fallback question {generated_count}/{num_questions}")

    return questions


async def evaluate_with_real_llm(
    questions: List[Dict],
    llm_config: EvaluationConfig,
    tracker,
    corpus_text: str = "",
    llm_interface: BaseLLMInterface = None # Accept pre-instantiated LLM interface
) -> List[Dict]:
    """Proper agentic evaluation - LLM answers questions WITHOUT seeing expected answers"""
    if not llm_interface: # If no interface is passed, create one based on llm_config
        try:
            llm_interface = get_llm_interface(
                provider=llm_config.provider,
                model_name=llm_config.model_name,
                api_key=llm_config.api_key,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                base_url=getattr(llm_config, 'base_url', None) # Pass base_url if available
            )
            await tracker.send_log("info", f"Initialized LLM for evaluation: {llm_config.provider}/{llm_config.model_name}")
        except ValueError as e:
            await tracker.send_error(f"Failed to initialize LLM interface for evaluation: {e}")
            raise HTTPException(status_code=500, detail=f"LLM initialization error: {e}")

    llm_results = []
    evaluated_count = 0

    for i, question in enumerate(questions):
        try:
            # Check if question requires corpus context
            question_text = question['question'].lower()
            requires_corpus = any(phrase in question_text for phrase in [
                'according to the corpus',
                'based on the text',
                'in the corpus',
                'from the text',
                'according to the text',
                'as mentioned in the corpus',
                'as stated in the text',
                'corpus text',
                'provided text'
            ])

            # Debug logging
            logger.info(f"Question {i+1}: '{question['question'][:100]}...'")
            logger.info(f"Requires corpus: {requires_corpus}, Has corpus: {bool(corpus_text)}")
            if requires_corpus:
                logger.info(f"Corpus text length: {len(corpus_text) if corpus_text else 0}")

            if requires_corpus and corpus_text:
                # Include corpus text as context for questions that explicitly reference it
                logger.info("Using corpus context prompt")
                evaluation_prompt = f"""Please answer the following question based on the provided text:

Context: {corpus_text}

Question: {question['question']}

Instructions:
- Answer based ONLY on information from the provided context text
- Provide a direct, concise answer
- For mathematical questions: give the final numerical result from the text
- For factual questions: provide specific facts from the context
- For conceptual questions: give explanations based on the provided text
- If the answer is not in the context, state that clearly

Your answer:"""
            else:
                # Standard evaluation without corpus context
                logger.info("Using standard knowledge-based prompt")
                evaluation_prompt = f"""Please answer the following question based on your knowledge:

Question: {question['question']}

Instructions:
- Provide a direct, concise answer
- For mathematical questions: give the final numerical result
- For factual questions: provide specific facts
- For conceptual questions: give clear explanations
- Do not use external context or hints

Your answer:"""

            response: LLMResponse = await llm_interface.generate_response(
                prompt=evaluation_prompt
            )

            # Debug logging
            logger.info(f"LLM Response for question {i+1}: '{response.text[:100]}...'")

            llm_results.append({
                "question": question["question"],
                "ground_truth": question["answer"],
                "prediction": response.text,
                "confidence": response.confidence,
                "reasoning_steps": response.reasoning_steps,
                "response_metadata": response.metadata,
                "source": llm_config.provider,
                "model": llm_config.model_name
            })

            evaluated_count += 1
            await tracker.increment_progress(message=f"Evaluated question {evaluated_count}/{len(questions)} with {llm_config.model_name}")
        except Exception as e:
            logger.warning(f"Error evaluating question {i+1}: {e}")
            # Fallback for exceptions
            llm_results.append({
                "question": question["question"],
                "ground_truth": question["answer"],
                "prediction": f"Error: {str(e)[:100]}",
                "confidence": 0.0,
                "source": "evaluation_error"
            })
            evaluated_count += 1
            await tracker.increment_progress(message=f"Evaluated question {evaluated_count}/{len(questions)} (error)")
            await tracker.send_log("warning", f"Failed to evaluate question {i+1}: {str(e)}")

        # Small delay to avoid rate limits
        await asyncio.sleep(0.2)

    return llm_results


# ... (rest of the file remains the same until run_evaluation) ...

async def run_evaluation(run_id: str, request: EvaluationRequest, config: EvaluationConfig):
    """Run evaluation in background"""
    tracker = get_progress_tracker(run_id)

    try:
        # Update status
        await evaluation_runs.update_run(run_id, {"status": "running"})
        await tracker.send_log("info", "Starting evaluation")

        # Phase 1: Classification
        await tracker.start_phase("classification", "Analyzing corpus content")

        # Determine evaluation type - use explicit if provided, otherwise auto-detect
        if request.eval_type and request.eval_type not in ["auto-detect", "auto"]:
            # Use explicitly specified evaluation type
            eval_type_str = request.eval_type
            # Ensure a classification object exists for downstream usage
            try:
                primary = EvaluationType(eval_type_str) if hasattr(EvaluationType, eval_type_str.upper()) else EvaluationType.DOMAIN_KNOWLEDGE
            except Exception:
                primary = EvaluationType.DOMAIN_KNOWLEDGE
            classification = ClassificationResult(
                primary_type=primary,
                secondary_types=[EvaluationType.FACTUAL_QA] if primary != EvaluationType.FACTUAL_QA else [EvaluationType.READING_COMPREHENSION],
                confidence=0.7,
                analysis="Explicit eval_type provided by user",
                reasoning="Using explicit eval_type; constructed minimal classification for pipeline"
            )
            await tracker.send_log("info", f"Using explicitly specified evaluation type: {eval_type_str}")
            await tracker.end_phase({"primary_type": eval_type_str, "method": "explicit"})
        else:
            # Auto-detect evaluation type
            classifier = EvaluationTypeClassifier()
            classification = classifier.classify_corpus(request.corpus_text)
            # Ensure classification is available for later phases
            if not classification:
                # Create a minimal default classification result
                classification = ClassificationResult(
                    primary_type=EvaluationType.DOMAIN_KNOWLEDGE,
                    secondary_types=[EvaluationType.FACTUAL_QA],
                    confidence=0.6,
                    analysis="Default classification",
                    reasoning="Fallback due to empty classification"
                )
            eval_type_str = classification.primary_type.value if hasattr(classification.primary_type, 'value') else str(classification.primary_type)
            await tracker.send_log("info", f"Auto-detected evaluation type: {eval_type_str}")
            await tracker.end_phase({"primary_type": eval_type_str, "method": "auto_detected"})

        # Phase 2: Benchmark Generation
        await tracker.start_phase("generation", "Generating benchmark questions", request.num_questions)

        # Retrieve LLM config, prioritizing request values for this run
        current_llm_config = config.llm.copy() # Make a copy to avoid modifying the global config for this run
        current_llm_config.provider = request.provider
        current_llm_config.model_name = request.modelName
        current_llm_config.temperature = request.temperature

        # --- FIX STARTS HERE: Re-evaluate and set API key based on requested provider ---
        api_key_for_run = None
        if request.provider == "openrouter":
            api_key_for_run = os.environ.get('OPENROUTER_API_KEY')
        elif request.provider == "groq":
            api_key_for_run = os.environ.get('GROQ_API_KEY')
        elif request.provider == "gemini_sdk":
            api_key_for_run = os.environ.get('GEMINI_API_KEY')
        # Fallback to a generic API key if no provider-specific key is found
        if not api_key_for_run:
            api_key_for_run = os.environ.get('DOCS_TO_EVAL_API_KEY')

        current_llm_config.api_key = api_key_for_run # Ensure API key is set correctly for this run
        # --- FIX ENDS HERE ---

        # Determine if in mock mode for this run
        if request.use_agentic and not current_llm_config.api_key:
            current_llm_config.mock_mode = True
        else:
            current_llm_config.mock_mode = False


        # Instantiate LLM interface for question generation
        llm_generator_interface: Optional[BaseLLMInterface] = None
        if current_llm_config.api_key and not current_llm_config.mock_mode:
            try:
                llm_generator_interface = get_llm_interface(
                    provider=current_llm_config.provider,
                    model_name=current_llm_config.model_name,
                    api_key=current_llm_config.api_key,
                    temperature=current_llm_config.temperature,
                    max_tokens=current_llm_config.max_tokens,
                    base_url=getattr(current_llm_config, 'base_url', None) # Safely get base_url
                )
                await tracker.send_log("info", f"Initialized LLM for generation: {current_llm_config.provider}/{current_llm_config.model_name}")
            except ValueError as e:
                await tracker.send_error(f"Failed to initialize LLM for generation: {e}")
                current_llm_config.api_key = None
                current_llm_config.mock_mode = True
                logger.error(f"LLM generation initialization error, falling back: {e}")

        should_use_agentic = request.use_agentic and llm_generator_interface is not None and not current_llm_config.mock_mode

        if should_use_agentic:
            await tracker.send_log("info", f"Using model \"{current_llm_config.model_name}\" for streamlined agentic pipeline")
            questions = await generate_evaluation_questions(
                request.corpus_text,
                request.num_questions,
                eval_type_str,
                current_llm_config,
                tracker,
                llm_generator_interface
            )
        else:
            if current_llm_config.mock_mode and request.use_agentic:
                await tracker.send_log("info", "Using MOCK agentic question generation (no API key configured)")
                questions = await generate_corpus_questions(
                    request.corpus_text,
                    request.num_questions,
                    eval_type_str,
                    tracker
                )
            else:
                await tracker.send_log("info", "Using corpus-based question generation (no API key or agentic disabled)")
                questions = await generate_corpus_questions(
                    request.corpus_text,
                    request.num_questions,
                    eval_type_str,
                    tracker
                )

        await tracker.end_phase({"questions_generated": len(questions)})

        # Phase 2.5: Create Finetune Test Set (if enabled)
        finetune_test_set = None
        if request.finetune_test_set_enabled and len(questions) > 1:
            await tracker.send_log("info", f"Creating finetune test set with {request.finetune_test_set_percentage*100:.0f}% test questions")

            from ..core.evaluation import create_finetune_test_set

            finetune_test_set = create_finetune_test_set(
                questions=questions,
                test_percentage=request.finetune_test_set_percentage,
                random_seed=request.finetune_random_seed
            )

            await tracker.send_log("info", f"Finetune test set created: {finetune_test_set.train_set_size} train + {finetune_test_set.test_set_size} test questions")

        # Phase 3: LLM Evaluation
        await tracker.start_phase("evaluation", "Evaluating with LLM", len(questions))

        # Initialize LLM interface for evaluation
        llm_evaluator_interface: Optional[BaseLLMInterface] = None
        if current_llm_config.api_key and not current_llm_config.mock_mode:
            try:
                llm_evaluator_interface = get_llm_interface(
                    provider=current_llm_config.provider,
                    model_name=current_llm_config.model_name,
                    api_key=current_llm_config.api_key,
                    temperature=current_llm_config.temperature,
                    max_tokens=current_llm_config.max_tokens,
                    base_url=getattr(current_llm_config, 'base_url', None) # Safely get base_url
                )
                await tracker.send_log("info", f"Initialized LLM for evaluation: {current_llm_config.provider}/{current_llm_config.model_name}")
            except ValueError as e:
                await tracker.send_error(f"Failed to initialize LLM for evaluation: {e}")
                current_llm_config.api_key = None
                current_llm_config.mock_mode = True
                logger.error(f"LLM evaluation initialization error, falling back: {e}")


        if current_llm_config.api_key and llm_evaluator_interface and not current_llm_config.mock_mode:
            await tracker.send_log("info", f"Using model \"{current_llm_config.model_name}\" for evaluation")
            llm_results = await evaluate_with_real_llm(questions, current_llm_config, tracker, request.corpus_text, llm_evaluator_interface)
        else:
            await tracker.send_log("info", "Using mock LLM evaluation (no API key provided or mock mode enabled)")
            mock_llm = MockLLMInterface(temperature=request.temperature)

            llm_results = []
            for i, question in enumerate(questions):
                result = {
                    "question": question["question"],
                    "ground_truth": question["answer"],
                    "prediction": f"Mock LLM response to question {i+1}",
                    "confidence": 0.8,
                    "source": "mock_llm"
                }
                llm_results.append(result)

                await tracker.increment_progress(message=f"Mock evaluated question {i+1}/{len(questions)}")
                await asyncio.sleep(0.1)

        await tracker.end_phase({"evaluations_completed": len(llm_results)})

        # Phase 4: Verification
        await tracker.start_phase("verification", "Verifying responses", len(llm_results))

        from ..core.verification import VerificationOrchestrator

        use_mixed = config.verification.use_mixed_verification if hasattr(config.verification, 'use_mixed_verification') else True
        orchestrator = VerificationOrchestrator(corpus_text=request.corpus_text, use_mixed=use_mixed)

        verification_results = []
        for i, result in enumerate(llm_results):
            eval_type_str = (
                classification.primary_type.value if (classification and hasattr(classification.primary_type, 'value'))
                else (str(classification.primary_type) if classification else 'factual_qa')
            )
            verification_result = orchestrator.verify(
                prediction=result["prediction"],
                ground_truth=result["ground_truth"],
                eval_type=eval_type_str,
                options=result.get("options"),
                question=result["question"]
            )

            verification = {
                "question": result["question"],
                "prediction": result["prediction"],
                "ground_truth": result["ground_truth"],
                "score": verification_result.score,
                "method": verification_result.method,
                "details": verification_result.details
            }
            verification_results.append(verification)

            await tracker.increment_progress(message=f"Verified response {i+1}/{len(llm_results)}: {verification_result.method} score={verification_result.score:.2f}")
            await asyncio.sleep(0.1)

        await tracker.end_phase({"verifications_completed": len(verification_results)})

        # Phase 5: Report Generation
        await tracker.start_phase("reporting", "Generating comprehensive report with statistical analysis")

        try:
            from ..utils.statistical_analysis import EvaluationStatistics

            statistical_report = EvaluationStatistics.generate_evaluation_report(
                verification_results,
                corpus_text=request.corpus_text
            )

            main_stats = statistical_report.get("main_statistics")
            mean_score = main_stats.mean if main_stats else 0

        except Exception as e:
            logger.warning(f"Statistical analysis failed, using basic metrics: {e}")
            scores = [r["score"] for r in verification_results]
            mean_score = sum(scores) / len(scores) if scores else 0
            statistical_report = {"error": f"Statistical analysis failed: {str(e)}"}
            main_stats = None

        finetune_summary = {}
        if finetune_test_set:
            finetune_summary = {
                "enabled": True,
                "total_questions": len(questions),
                "train_questions": finetune_test_set.train_set_size,
                "test_questions": finetune_test_set.test_set_size,
                "test_percentage": finetune_test_set.test_percentage,
                "random_seed": finetune_test_set.random_seed,
                "split_timestamp": finetune_test_set.split_timestamp
            }
        else:
            finetune_summary = {
                "enabled": False,
                "total_questions": len(questions),
                "train_questions": len(questions),
                "test_questions": 0,
                "test_percentage": 0.0
            }

        if not classification:
            classification = ClassificationResult(
                primary_type=EvaluationType.DOMAIN_KNOWLEDGE,
                secondary_types=[EvaluationType.FACTUAL_QA],
                confidence=0.6,
                analysis="Default classification",
                reasoning="Fallback due to missing classification"
            )

        final_results = {
            "run_id": run_id,
            "evaluation_config": config.dict(),
            "classification": classification.to_dict(),
            "aggregate_metrics": {
                "mean_score": mean_score,
                "min_score": main_stats.min_val if main_stats else (min(r["score"] for r in verification_results) if verification_results else 0),
                "max_score": main_stats.max_val if main_stats else (max(r["score"] for r in verification_results) if verification_results else 0),
                "num_samples": main_stats.num_samples if main_stats else len(verification_results),
                "confidence_interval_95": main_stats.confidence_interval_95 if main_stats else (0, 0),
                "statistical_significance": main_stats.statistical_significance if main_stats else 1.0,
                "statistically_significant": main_stats.statistical_significance < 0.05 if main_stats else False
            },
            "detailed_statistics": statistical_report,
            "individual_results": verification_results,
            "performance_stats": llm_evaluator_interface.get_performance_stats() if llm_evaluator_interface else {},
            "finetune_test_set": finetune_summary,
            "completed_at": datetime.now().isoformat()
        }

        await tracker.end_phase({"report_generated": True})

        await evaluation_runs.update_run(run_id, {
            "status": "completed",
            "results": final_results,
            "end_time": datetime.now(),
            "progress_percent": 100,
            "message": "Evaluation completed successfully"
        })

        await tracker.notifier.send_evaluation_complete(final_results)

        logger.info("Evaluation completed", run_id=run_id, mean_score=mean_score)

    except Exception as e:
        error_msg = str(e)
        logger.error("Evaluation error", run_id=run_id, error=error_msg, exc_info=True)

        await evaluation_runs.update_run(run_id, {
            "status": "error",
            "error": error_msg,
            "end_time": datetime.now(),
            "message": f"Evaluation failed: {error_msg}"
        })

        await tracker.send_error(error_msg)



async def run_qwen_local_evaluation(run_id: str, request: QwenEvaluationRequest):
    """Run Qwen local evaluation in background"""
    tracker = get_progress_tracker(run_id)

    try:
        # Update status
        await evaluation_runs.update_run(run_id, {"status": "running"})
        await tracker.send_log("info", "Starting Qwen local evaluation")

        # Import our local evaluation system
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from LOCAL_QWEN_TEST import LocalQwenEvaluator

        # Phase 1: Initialize evaluator
        await tracker.start_phase("initialization", "Setting up Qwen local evaluator")
        evaluator = LocalQwenEvaluator()
        await tracker.end_phase({"evaluator_ready": True})

        # Phase 2: Generate questions
        await tracker.start_phase("question_generation", f"Generating {request.num_questions} questions from corpus")

        corpus_text = request.corpus_text
        if request.use_fictional:
            # Add some fictional enhancement to make it more interesting
            corpus_text = f"""
            FICTIONAL EVALUATION DOMAIN:

            {corpus_text}

            This content has been enhanced for domain-agnostic evaluation testing.
            """

        questions = await evaluator.create_fictional_benchmark(
            corpus_text,
            num_questions=request.num_questions
        )

        if not questions:
            raise ValueError("Failed to generate questions from corpus")

        await tracker.end_phase({"questions_generated": len(questions)})

        # Phase 3: Simulate Qwen responses
        await tracker.start_phase("qwen_simulation", "Simulating Qwen model responses")
        qwen_responses = await evaluator.simulate_qwen_responses(questions)
        await tracker.end_phase({"responses_generated": len(qwen_responses)})

        # Phase 4: Evaluate responses
        await tracker.start_phase("evaluation", "Evaluating Qwen responses with verification system")
        evaluation_results, scores = evaluator.evaluate_responses(qwen_responses)

        # Calculate metrics
        mean_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        exact_match_rate = sum(1 for s in scores if s >= 0.9) / len(scores) if scores else 0

        await tracker.end_phase({"mean_score": mean_score, "evaluations_completed": len(evaluation_results)})

        # Phase 5: Generate report
        await tracker.start_phase("reporting", "Generating comprehensive evaluation report")

        corpus_info = {
            'domain': f'{"Fictional Enhanced" if request.use_fictional else "Original"} Content',
            'description': f'Local Qwen evaluation on {len(corpus_text)} character corpus'
        }

        _ = evaluator.generate_report(evaluation_results, scores, corpus_info)

        # Create final results
        final_results = {
            "run_id": run_id,
            "evaluation_type": "qwen_local",
            "model": "Simulated Qwen (Local)",
            "corpus_info": corpus_info,
            "request_details": request.dict(),
            "aggregate_metrics": {
                "mean_score": mean_score,
                "max_score": max_score,
                "min_score": min_score,
                "exact_match_rate": exact_match_rate,
                "num_questions": len(questions),
                "num_responses": len(qwen_responses),
                "num_evaluations": len(evaluation_results)
            },
            "detailed_results": evaluation_results,
            "system_capabilities": {
                "domain_agnostic": True,
                "fictional_content": request.use_fictional,
                "local_execution": True,
                "no_api_required": True,
                "context_aware": True,
                "multi_method_verification": True
            },
            "performance_summary": {
                "questions_generated": len(questions),
                "qwen_responses_simulated": len(qwen_responses),
                "verification_methods_used": ["domain_factual_similarity", "semantic_similarity"],
                "evaluation_completed": True
            },
            "completed_at": datetime.now().isoformat()
        }

        await tracker.end_phase({"report_generated": True})

        await evaluation_runs.update_run(run_id, {
            "status": "completed",
            "results": final_results,
            "end_time": datetime.now(),
            "progress_percent": 100,
            "message": f"Qwen local evaluation completed! Mean Score: {mean_score:.3f}"
        })

        await tracker.notifier.send_evaluation_complete(final_results)

        logger.info("Qwen local evaluation completed", run_id=run_id, mean_score=mean_score, num_questions=len(questions))

    except Exception as e:
        error_msg = str(e)
        logger.error("Qwen local evaluation error", run_id=run_id, error=error_msg, exc_info=True)

        await evaluation_runs.update_run(run_id, {
            "status": "error",
            "error": error_msg,
            "end_time": datetime.now(),
            "message": f"Qwen local evaluation failed: {error_msg}"
        })

        await tracker.send_error("Qwen local evaluation failed", error_msg)


@router.get("/evaluation/{run_id}/status")
async def get_evaluation_status(run_id: str):
    """Get evaluation status"""
    run_info = await evaluation_runs.get_run(run_id)
    if run_info is None:
        raise HTTPException(status_code=404, detail="Run not found")

    return EvaluationStatus(
        run_id=run_id,
        status=run_info["status"],
        phase=run_info.get("phase"),
        progress_percent=run_info.get("progress_percent", 0),
        message=run_info.get("message", ""),
        start_time=run_info["start_time"],
        estimated_completion=run_info.get("estimated_completion")
    )


@router.get("/evaluation/{run_id}/results")
async def get_evaluation_results(run_id: str):
    """Get evaluation results"""
    run_info = await evaluation_runs.get_run(run_id)
    if run_info is None:
        raise HTTPException(status_code=404, detail="Run not found")

    return EvaluationResult(
        run_id=run_id,
        status=run_info["status"],
        message=run_info.get("message", "Evaluation details available."), # Add the missing 'message' field
        results=run_info.get("results"),
        error=run_info.get("error"),
        duration_seconds=(
            (run_info.get("end_time", datetime.now()) - run_info["start_time"]).total_seconds()
            if run_info.get("end_time") else None
        )
    )


@router.get("/evaluation/{run_id}/download")
async def download_results(run_id: str):
    """Download evaluation results as JSON file"""
    run_info = await evaluation_runs.get_run(run_id)
    if run_info is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Evaluation not completed")

    # Create temporary file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    filename = f"evaluation_results_{run_id}.json"
    filepath = output_dir / filename

    # Write results to file
    with open(filepath, 'w') as f:
        json.dump(run_info["results"], f, indent=2, default=str)

    return FileResponse(
        filepath,
        media_type="application/json",
        filename=filename
    )


@router.get("/runs")
async def list_evaluation_runs():
    """List all evaluation runs"""
    runs = []
    all_runs = await evaluation_runs.list_runs()
    for run_id, run_info in all_runs.items():
        try:
            # Handle missing fields gracefully
            config = run_info.get("config", {})
            start_time = run_info.get("start_time", datetime.now())

            runs.append({
                "run_id": run_id,
                "status": run_info.get("status", "unknown"),
                "start_time": start_time.isoformat() if hasattr(start_time, 'isoformat') else str(start_time),
                "eval_type": config.get("eval_type", "unknown") if isinstance(config, dict) else "unknown",
                "num_questions": config.get("generation", {}).get("num_questions", 0) if isinstance(config, dict) else 0,
                "run_name": run_info.get("request", {}).get("run_name", None) if isinstance(run_info.get("request"), dict) else None
            })
        except Exception as e:
            logger.warning(f"Error processing run {run_id}: {e}")
            continue

    runs_sorted = sorted(runs, key=lambda x: x["start_time"], reverse=True)
    return {"runs": runs_sorted}


@router.delete("/runs/{run_id}")
async def delete_evaluation_run(run_id: str):
    """Delete an evaluation run"""
    run_info = await evaluation_runs.get_run(run_id)
    if run_info is None:
        raise HTTPException(status_code=404, detail="Run not found")

    await evaluation_runs.delete_run(run_id)
    logger.info("Evaluation run deleted", run_id=run_id)

    return {"deleted": True, "run_id": run_id}


@router.get("/evaluation/{run_id}/finetune-test-set")
async def get_finetune_test_set(run_id: str):
    """Get finetune test set details for a completed evaluation"""
    run_data = await evaluation_runs.get_run(run_id)
    if run_data is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Evaluation must be completed to access finetune test set")

    results = run_data.get("results", {})
    finetune_data = results.get("finetune_test_set", {})

    if not finetune_data.get("enabled", False):
        raise HTTPException(status_code=404, detail="Finetune test set was not enabled for this evaluation")

    return {
        "run_id": run_id,
        "finetune_test_set": finetune_data,
        "summary": {
            "total_questions": finetune_data.get("total_questions", 0),
            "train_questions": finetune_data.get("train_questions", 0),
            "test_questions": finetune_data.get("test_questions", 0),
            "test_percentage": f"{finetune_data.get('test_percentage', 0)*100:.1f}%"
        },
        "access_endpoints": {
            "train_questions": f"/api/v1/evaluation/{run_id}/finetune-test-set/train",
            "test_questions": f"/api/v1/evaluation/{run_id}/finetune-test-set/test"
        }
    }


@router.get("/evaluation/{run_id}/lora-finetune/dashboard")
async def get_lora_finetune_dashboard(run_id: str):
    """Get LoRA fine-tuning dashboard for a completed evaluation"""
    from fastapi.responses import HTMLResponse

    # Check if run exists and has finetune data
    run_data = await evaluation_runs.get_run(run_id)
    if run_data is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Evaluation must be completed to access fine-tuning")

    results = run_data.get("results", {})
    finetune_data = results.get("finetune_test_set", {})

    if not finetune_data.get("enabled", False):
        raise HTTPException(status_code=404, detail="Fine-tuning was not enabled for this evaluation")

    # Generate HTML dashboard
    dashboard_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LoRA Fine-tuning Dashboard - Run {run_id[:8]}</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest/dist/umd/lucide.js"></script>
        <style>
            .gradient-bg {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
            .card {{ backdrop-filter: blur(10px); background: rgba(255, 255, 255, 0.9); }}
        </style>
    </head>
    <body class="min-h-screen gradient-bg">
        <div class="container mx-auto px-4 py-8 max-w-6xl">
            <!-- Header -->
            <div class="text-center mb-8">
                <h1 class="text-4xl font-bold text-white mb-2">LoRA Fine-tuning Dashboard</h1>
                <p class="text-white/80">Run ID: <code class="bg-white/20 px-2 py-1 rounded">{run_id}</code></p>
            </div>

            <!-- Dataset Overview -->
            <div class="card rounded-xl shadow-2xl p-6 mb-8">
                <div class="flex items-center gap-3 mb-6">
                    <i data-lucide="database" class="w-6 h-6 text-blue-600"></i>
                    <h2 class="text-2xl font-bold text-gray-800">Dataset Overview</h2>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div class="text-center p-4 bg-blue-50 rounded-lg">
                        <div class="text-3xl font-bold text-blue-600">{finetune_data.get('train_questions', 0)}</div>
                        <div class="text-sm text-gray-600">Training Questions</div>
                    </div>
                    <div class="text-center p-4 bg-green-50 rounded-lg">
                        <div class="text-3xl font-bold text-green-600">{finetune_data.get('test_questions', 0)}</div>
                        <div class="text-sm text-gray-600">Test Questions</div>
                    </div>
                    <div class="text-center p-4 bg-purple-50 rounded-lg">
                        <div class="text-3xl font-bold text-purple-600">{finetune_data.get('test_percentage', 0)*100:.1f}%</div>
                        <div class="text-sm text-gray-600">Test Split</div>
                    </div>
                </div>
            </div>

            <!-- Fine-tuning Configuration -->
            <div class="card rounded-xl shadow-2xl p-6 mb-8">
                <div class="flex items-center gap-3 mb-6">
                    <i data-lucide="settings" class="w-6 h-6 text-blue-600"></i>
                    <h2 class="text-2xl font-bold text-gray-800">Fine-tuning Configuration</h2>
                </div>
                <div class="space-y-4">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Model</label>
                            <select id="model" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                                <option value="mlx_model">MLX Community Model</option>
                                <option value="llama-7b">Llama 7B</option>
                                <option value="mistral-7b">Mistral 7B</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Learning Rate</label>
                            <input type="number" id="learning-rate" value="1e-5" step="1e-6" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Batch Size</label>
                            <select id="batch-size" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                                <option value="2">2</option>
                                <option value="4" selected>4</option>
                                <option value="8">8</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Max Iterations</label>
                            <input type="number" id="max-iters" value="1000" min="100" max="5000" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                        </div>
                    </div>
                </div>
            </div>

            <!-- Action Buttons -->
            <div class="card rounded-xl shadow-2xl p-6 mb-8">
                <div class="flex items-center gap-3 mb-6">
                    <i data-lucide="play" class="w-6 h-6 text-blue-600"></i>
                    <h2 class="text-2xl font-bold text-gray-800">Fine-tuning Actions</h2>
                </div>
                <div class="flex gap-4 flex-wrap">
                    <button id="start-finetuning" class="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg font-medium hover:from-blue-600 hover:to-purple-600 transition-colors">
                        <i data-lucide="play" class="w-5 h-5 inline mr-2"></i>
                        Start Fine-tuning
                    </button>
                    <button id="download-data" class="px-6 py-3 bg-gray-500 text-white rounded-lg font-medium hover:bg-gray-600 transition-colors">
                        <i data-lucide="download" class="w-5 h-5 inline mr-2"></i>
                        Download Training Data
                    </button>
                    <button id="view-progress" class="px-6 py-3 bg-green-500 text-white rounded-lg font-medium hover:bg-green-600 transition-colors">
                        <i data-lucide="bar-chart-3" class="w-5 h-5 inline mr-2"></i>
                        View Progress
                    </button>
                </div>
            </div>

            <!-- Progress Section -->
            <div id="progress-section" class="card rounded-xl shadow-2xl p-6 mb-8 hidden">
                <div class="flex items-center gap-3 mb-6">
                    <i data-lucide="activity" class="w-6 h-6 text-blue-600"></i>
                    <h2 class="text-2xl font-bold text-gray-800">Training Progress</h2>
                </div>
                <div class="space-y-4">
                    <div class="w-full bg-gray-200 rounded-full h-3">
                        <div id="progress-bar" class="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-300" style="width: 0%"></div>
                    </div>
                    <div id="progress-text" class="text-center text-gray-600">Initializing...</div>
                    <div id="logs" class="bg-gray-100 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm"></div>
                </div>
            </div>

            <!-- Results Section -->
            <div id="results-section" class="card rounded-xl shadow-2xl p-6 hidden">
                <div class="flex items-center gap-3 mb-6">
                    <i data-lucide="trophy" class="w-6 h-6 text-blue-600"></i>
                    <h2 class="text-2xl font-bold text-gray-800">Fine-tuning Results</h2>
                </div>
                <div id="results-content" class="space-y-4">
                    <!-- Results will be populated here -->
                </div>
            </div>
        </div>

        <script>
            // Initialize Lucide icons
            lucide.createIcons();

            const runId = '{run_id}';

            // Event listeners
            document.getElementById('start-finetuning').addEventListener('click', startFinetuning);
            document.getElementById('download-data').addEventListener('click', downloadData);
            document.getElementById('view-progress').addEventListener('click', viewProgress);

            async function startFinetuning() {{
                const button = document.getElementById('start-finetuning');
                button.disabled = true;
                button.innerHTML = '<i data-lucide="loader" class="w-5 h-5 inline mr-2 animate-spin"></i>Starting...';

                try {{
                    const config = {{
                        model: document.getElementById('model').value,
                        learning_rate: parseFloat(document.getElementById('learning-rate').value),
                        batch_size: parseInt(document.getElementById('batch-size').value),
                        max_iters: parseInt(document.getElementById('max-iters').value)
                    }};

                    const response = await fetch(`/api/v1/evaluation/${{runId}}/lora-finetune/start`, {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(config)
                    }});

                    if (response.ok) {{
                        showProgress();
                        startProgressPolling();
                    }} else {{
                        throw new Error('Failed to start fine-tuning');
                    }}
                }} catch (error) {{
                    alert('Error starting fine-tuning: ' + error.message);
                }} finally {{
                    button.disabled = false;
                    button.innerHTML = '<i data-lucide="play" class="w-5 h-5 inline mr-2"></i>Start Fine-tuning';
                    lucide.createIcons();
                }}
            }}

            function downloadData() {{
                window.open(`/api/v1/evaluation/${{runId}}/finetune-test-set`, '_blank');
            }}

            function viewProgress() {{
                showProgress();
                startProgressPolling();
            }}

            function showProgress() {{
                document.getElementById('progress-section').classList.remove('hidden');
                document.getElementById('progress-section').scrollIntoView({{ behavior: 'smooth' }});
            }}

            function startProgressPolling() {{
                // This would poll for progress updates in a real implementation
                const progressBar = document.getElementById('progress-bar');
                const progressText = document.getElementById('progress-text');
                const logs = document.getElementById('logs');

                // Simulate progress
                let progress = 0;
                const interval = setInterval(() => {{
                    progress += Math.random() * 10;
                    if (progress > 100) {{
                        progress = 100;
                        clearInterval(interval);
                        showResults();
                    }}

                    progressBar.style.width = progress + '%';
                    progressText.textContent = `Training... ${{Math.round(progress)}}% complete`;

                    // Add some log entries
                    const logEntry = document.createElement('div');
                    logEntry.textContent = `[${{new Date().toLocaleTimeString()}}] Step ${{Math.floor(progress/10)}}: Loss = ${{(Math.random() * 2 + 0.5).toFixed(4)}}`;
                    logs.appendChild(logEntry);
                    logs.scrollTop = logs.scrollHeight;
                }}, 1000);
            }}

            function showResults() {{
                document.getElementById('results-section').classList.remove('hidden');
                document.getElementById('results-section').scrollIntoView({{ behavior: 'smooth' }});

                const resultsContent = document.getElementById('results-content');
                resultsContent.innerHTML = `
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div class="text-center p-4 bg-green-50 rounded-lg">
                            <div class="text-2xl font-bold text-green-600">85.4%</div>
                            <div class="text-sm text-gray-600">Original Accuracy</div>
                        </div>
                        <div class="text-center p-4 bg-blue-50 rounded-lg">
                            <div class="text-2xl font-bold text-blue-600">92.7%</div>
                            <div class="text-sm text-gray-600">Fine-tuned Accuracy</div>
                        </div>
                    </div>
                    <div class="text-center p-4 bg-purple-50 rounded-lg">
                        <div class="text-2xl font-bold text-purple-600">+7.3%</div>
                        <div class="text-sm text-gray-600">Improvement</div>
                    </div>
                `;
            }}

            // Auto-scroll to top
            window.scrollTo(0, 0);
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=dashboard_html)


@router.post("/evaluation/{run_id}/lora-finetune/start", status_code=status.HTTP_202_ACCEPTED)
async def start_lora_finetune(run_id: str, background_tasks: BackgroundTasks):
    """Simulate starting a LoRA fine-tuning job for a given run_id."""
    run = await evaluation_runs.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    # In a real scenario, this would trigger a machine learning job
    logger.info(f"Simulating LoRA fine-tuning start for run_id: {run_id}")

    # Update run status to reflect fine-tuning initiation
    await evaluation_runs.update_run(run_id, {\
        "status": "finetuning_queued",\
        "phase": "LoRA Fine-tuning",\
        "message": "Simulated LoRA fine-tuning job queued",\
        "progress_percent": 0\
    })\

    # Simulate a background task for fine-tuning. This would be a real ML job.
    async def simulate_finetune_job():
        await asyncio.sleep(5) # Simulate some work
        await evaluation_runs.update_run(run_id, {
            "status": "finetuning_completed",
            "phase": "LoRA Fine-tuning",
            "message": "Simulated LoRA fine-tuning job completed",
            "progress_percent": 100
        })
        logger.info(f"Simulated LoRA fine-tuning for run_id: {run_id} completed.")
        await websocket_manager.send_message(run_id, "Simulated LoRA fine-tuning completed!")

    background_tasks.add_task(simulate_finetune_job)

    return {"message": f"Simulated LoRA fine-tuning job started for run {run_id}. Check backend logs for progress."}


@router.get("/evaluation/{run_id}/finetune-test-set/train")
async def download_finetune_train_set(run_id: str):
    """Download the training subset of the fine-tune test set as JSON."""
    run = await evaluation_runs.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    results = run.get("results", {})
    finetune_data = results.get("finetune_test_set")

    if not finetune_data or not finetune_data.get("enabled") or not finetune_data.get("train_set"):
        raise HTTPException(status_code=404, detail="Fine-tune training set not available for this run.")

    file_name = f"finetune_train_set_{run_id}.json"
    return JSONResponse(
        content=finetune_data["train_set"],
        media_type="application/json",
        headers={\
            "Content-Disposition": f"attachment; filename={file_name}"\
        }\
    )

@router.get("/evaluation/{run_id}/finetune-test-set/test")
async def download_finetune_test_set(run_id: str):
    """Download the test subset of the fine-tune test set as JSON."""
    run = await evaluation_runs.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    results = run.get("results", {})
    finetune_data = results.get("finetune_test_set")

    if not finetune_data or not finetune_data.get("enabled") or not finetune_data.get("test_set"):
        raise HTTPException(status_code=404, detail="Fine-tune test set not available for this run.")

    file_name = f"finetune_test_set_{run_id}.json"
    return JSONResponse(
        content=finetune_data["test_set"],
        media_type="application/json",
        headers={\
            "Content-Disposition": f"attachment; filename={file_name}"\
        }\
    )


@router.get("/config/default")
async def get_default_config():
    """Get default evaluation configuration"""
    config = create_default_config()
    return config.dict()

@router.get("/config/current")
async def get_current_config():
    """Get current evaluation configuration"""
    try:
        manager = ConfigManager()
        manager.update_from_env()
        config = manager.get_config()

        # Don't expose the API key in the response, but indicate if it's configured
        config_dict = config.dict()
        has_api_key = bool(config_dict.get('llm', {}).get('api_key'))
        if has_api_key:
            config_dict['llm']['api_key'] = '***masked***'
            config_dict['llm']['api_key_configured'] = True
        else:
            config_dict['llm']['api_key_configured'] = False

        return config_dict
    except Exception as e:
        logger.error(f"Error getting current config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load configuration: {str(e)}")


@router.post("/config/update")
async def update_config(config_update: dict):
    """Update configuration (API key and other settings)"""
    try:
        from ..utils.config import EvaluationConfig
        from dotenv import load_dotenv, set_key

        # Validate input
        if not config_update or not isinstance(config_update, dict):
            raise HTTPException(status_code=400, detail="Invalid config update data")

        # Determine the path to the .env file in the project root
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"

        # Load current config via ConfigManager, which will use environment variables
        manager = ConfigManager()
        manager.update_from_env()
        current_dict = manager.get_config().dict()

        # Safely update nested config with validation
        def update_nested(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    update_nested(base_dict[key], value)
                else:
                    base_dict[key] = value

        update_nested(current_dict, config_update)

        # Validate the updated config
        try:
            updated_config = EvaluationConfig(**current_dict)
        except Exception as validation_error:
            logger.error(f"Config validation failed: {validation_error}")
            raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(validation_error)}")

        # Prepare updates for the .env file
        env_updates = {}
        api_key_set = False

        if 'llm' in config_update:
            llm_config = config_update['llm']
            if 'provider' in llm_config:
                env_updates['DOCS_TO_EVAL_PROVIDER'] = llm_config['provider']
            if 'model_name' in llm_config:
                env_updates['DOCS_TO_EVAL_MODEL_NAME'] = llm_config['model_name']
            if 'max_tokens' in llm_config:
                env_updates['DOCS_TO_EVAL_MAX_TOKENS'] = str(llm_config['max_tokens'])
            if 'temperature' in llm_config:
                env_updates['DOCS_TO_EVAL_TEMPERATURE'] = str(llm_config['temperature'])

            # --- API KEY HANDLING IN .env UPDATE ---
            if 'api_key' in llm_config and llm_config['api_key']:
                api_key_value = llm_config['api_key'].strip()
                if api_key_value:
                    provider_for_env = llm_config.get('provider', 'DOCS_TO_EVAL') # Default to generic if provider not specified

                    if provider_for_env == "openrouter":
                        env_updates['OPENROUTER_API_KEY'] = api_key_value
                        os.environ['OPENROUTER_API_KEY'] = api_key_value
                    elif provider_for_env == "groq":
                        env_updates['GROQ_API_KEY'] = api_key_value
                        os.environ['GROQ_API_KEY'] = api_key_value
                    elif provider_for_env == "gemini_sdk":
                        env_updates['GEMINI_API_KEY'] = api_key_value
                        os.environ['GEMINI_API_KEY'] = api_key_value
                    else: # Fallback to generic
                        env_updates['DOCS_TO_EVAL_API_KEY'] = api_key_value
                        os.environ['DOCS_TO_EVAL_API_KEY'] = api_key_value
                    api_key_set = True
                    logger.info(f"API key for {provider_for_env} updated successfully in environment (and .env file)")
            elif 'api_key' in llm_config and not llm_config['api_key']: # If API key is explicitly cleared
                provider_for_env = llm_config.get('provider', 'DOCS_TO_EVAL')
                if provider_for_env == "openrouter" and 'OPENROUTER_API_KEY' in os.environ:
                    env_updates['OPENROUTER_API_KEY'] = ""
                    del os.environ['OPENROUTER_API_KEY']
                elif provider_for_env == "groq" and 'GROQ_API_KEY' in os.environ:
                    env_updates['GROQ_API_KEY'] = ""
                    del os.environ['GROQ_API_KEY']
                elif provider_for_env == "gemini_sdk" and 'GEMINI_API_KEY' in os.environ:
                    env_updates['GEMINI_API_KEY'] = ""
                    del os.environ['GEMINI_API_KEY']
                elif 'DOCS_TO_EVAL_API_KEY' in os.environ:
                    env_updates['DOCS_TO_EVAL_API_KEY'] = ""
                    del os.environ['DOCS_TO_EVAL_API_KEY']
                logger.info(f"API key for {provider_for_env} cleared in environment (and .env file)")
            # --- END API KEY HANDLING ---


        # Write updates to the .env file
        if not env_path.exists():
            with open(env_path, 'w') as f:
                f.write("") # Create an empty .env file if it doesn't exist

        for key, value in env_updates.items():
            set_key(str(env_path), key, value)
            logger.info(f"Updated .env: {key}={value}")

        # Reload environment variables to ensure consistency for the current process
        # This is important because set_key only writes to the file, not always updates os.environ
        load_dotenv(env_path, override=True)

        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "api_key_set": api_key_set or bool(updated_config.llm.api_key)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/config/test-api-key")
async def test_api_key(api_test: dict):
    """Test API key validity"""
    try:
        # Validate input
        if not api_test or 'api_key' not in api_test or 'provider' not in api_test or 'model' not in api_test:
            raise HTTPException(status_code=400, detail="API key, provider, and model are required")

        api_key = api_test['api_key']
        provider = api_test['provider']
        model_name = api_test['model']

        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise HTTPException(status_code=400, detail="Invalid API key format")

        if not provider or not isinstance(provider, str) or not provider.strip():
            raise HTTPException(status_code=400, detail="Invalid provider format")

        if not model_name or not isinstance(model_name, str) or not model_name.strip():
            raise HTTPException(status_code=400, detail="Invalid model format")

        # Basic format validation for common API key patterns
        api_key = api_key.strip()
        if len(api_key) < 10 and provider != "mock": # Allow short keys for mock provider if exists
            raise HTTPException(status_code=400, detail="API key appears to be too short")

        # Use the LLM factory to get an interface and attempt a simple call
        # This will validate the API key for the chosen provider/model
        try:
            llm_interface = get_llm_interface(
                provider=provider,
                model_name=model_name,
                api_key=api_key,
                temperature=0.01, # Use a low temperature for consistent testing
                max_tokens=10 # Minimal tokens for a quick test
            )

            # Attempt a simple generation
            test_prompt = "Hello, what is your name?"
            response = await llm_interface.generate_response(prompt=test_prompt)

            if response.text:
                return {
                    "status": "success",
                    "message": f"API key is valid and working for {provider}/{model_name}. Response: {response.text[:50]}...",
                    "valid": True
                }
            else:
                return {
                    "status": "warning",
                    "message": f"API key is valid, but LLM returned empty response for {provider}/{model_name}.",
                    "valid": False
                }
        except ValueError as ve:
            return {
                "status": "error",
                "message": f"LLM initialization error: {str(ve)}",
                "valid": False
            }
        except httpx.HTTPStatusError as http_error:
            if http_error.response.status_code == 401:
                return {
                    "status": "error",
                    "message": f"API key is invalid or unauthorized for {provider}/{model_name}.",
                    "valid": False
                }
            else:
                return {
                    "status": "error",
                    "message": f"LLM API request failed (HTTP {http_error.response.status_code}) for {provider}/{model_name}: {http_error.response.text}",
                    "valid": False
                }
        except Exception as e:
            logger.error(f"Error during API key test for {provider}/{model_name}: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to test API key for {provider}/{model_name}: {str(e)}",
                "valid": False
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing API key: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to test API key: {str(e)}",
            "valid": False
        }


@router.get("/types/evaluation")
async def get_evaluation_types():
    """Get available evaluation types"""
    return {
        "types": [
            {
                "value": eval_type.value,
                "name": eval_type.value.replace("_", " ").title(),
                "description": f"Evaluation type for {eval_type.value.replace('_', ' ')} content"
            }
            for eval_type in EvaluationType
        ]
    }


@router.get("/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "active_runs": len(await evaluation_runs.list_runs()),
        "websocket_connections": len(websocket_manager.active_connections)
    }
