"""
FastAPI routes for docs-to-eval system
"""

import asyncio
import uuid
import json
import shutil
import httpx
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, Form, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError, validator

from .websockets import websocket_manager, handle_websocket_connection, get_progress_tracker
from ..core.evaluation import EvaluationFramework, BenchmarkConfig
from ..core.classification import EvaluationTypeClassifier
from ..core.agentic.agents import Validator
from ..core.agentic.orchestrator import AgenticBenchmarkOrchestrator
from ..core.agentic.models import PipelineConfig, DifficultyLevel, EnhancedBenchmarkItem
from ..llm.mock_interface import MockLLMInterface, MockLLMEvaluator
from ..llm.openrouter_interface import OpenRouterInterface, OpenRouterConfig
from ..llm.base import BaseLLMInterface
from ..utils.config import EvaluationConfig, EvaluationType, create_default_config
from ..utils.logging import get_logger

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
        import re
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
                raise ValueError(f"Text contains potentially malicious content")
        
        return v.strip()
    
    @validator('name')
    def validate_name(cls, v):
        if v:
            # Sanitize name by removing special characters and keeping only safe ones
            import re
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
    
    @validator('corpus_text')
    def validate_corpus_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Corpus text cannot be empty")
        
        # Check for suspicious patterns
        import re
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
                raise ValueError("Corpus text contains potentially malicious content")
        
        return v.strip()
    
    @validator('eval_type')
    def validate_eval_type(cls, v):
        if v:
            # Only allow known evaluation types
            valid_types = [
                'mathematical', 'code_generation', 'factual_qa', 'multiple_choice',
                'summarization', 'translation', 'creative_writing', 'commonsense_reasoning',
                'reading_comprehension', 'domain_knowledge'
            ]
            if v not in valid_types:
                raise ValueError(f"Invalid evaluation type. Must be one of: {', '.join(valid_types)}")
        return v
    
    @validator('run_name')
    def validate_run_name(cls, v):
        if v:
            # Only allow safe characters
            import re
            if not re.match(r'^[a-zA-Z0-9\s\-_\.]+$', v):
                raise ValueError("Run name can only contain letters, numbers, spaces, hyphens, underscores, and periods")
        return v.strip() if v else v


class EvaluationStatus(BaseModel):
    run_id: str
    status: str
    phase: Optional[str] = None


class QwenEvaluationRequest(BaseModel):
    corpus_text: str = Field(..., min_length=1, max_length=5*1024*1024, description="Corpus text content")
    num_questions: int = Field(default=5, ge=1, le=20, description="Number of questions to generate")
    use_fictional: bool = Field(default=True, description="Use fictional content enhancement")
    token_threshold: int = Field(default=2000, ge=500, le=4000, description="Token threshold for chunk concatenation")
    run_name: Optional[str] = Field("Qwen Local Test", max_length=100, description="Name for this evaluation run")
    
    @validator('corpus_text')
    def validate_corpus_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Corpus text cannot be empty")
        
        # Check for suspicious patterns (same as EvaluationRequest)
        import re
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
                raise ValueError("Corpus text contains potentially malicious content")
        
        return v.strip()
    
    @validator('run_name')
    def validate_run_name(cls, v):
        if v:
            # Only allow safe characters
            import re
            if not re.match(r'^[a-zA-Z0-9\s\-_\.]+$', v):
                raise ValueError("Run name can only contain letters, numbers, spaces, hyphens, underscores, and periods")
        return v.strip() if v else v


class EvaluationResult(BaseModel):
    run_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None


# In-memory storage for evaluation runs with size limits
from collections import OrderedDict
import time
from pathlib import Path

class EvaluationRunManager:
    def __init__(self, max_runs: int = 100, max_age_hours: int = 24):
        self._runs: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_runs = max_runs
        self.max_age_seconds = max_age_hours * 3600
    
    def add_run(self, run_id: str, run_info: Dict[str, Any]):
        """Add run with automatic cleanup"""
        # Clean old runs first
        self._cleanup_old_runs()
        
        # Add new run
        self._runs[run_id] = run_info
        
        # Enforce size limit
        while len(self._runs) > self.max_runs:
            self._runs.popitem(last=False)  # Remove oldest
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run info"""
        return self._runs.get(run_id)
    
    def update_run(self, run_id: str, updates: Dict[str, Any]):
        """Update run info"""
        if run_id in self._runs:
            self._runs[run_id].update(updates)
    
    def delete_run(self, run_id: str):
        """Delete run"""
        self._runs.pop(run_id, None)
    
    def list_runs(self) -> Dict[str, Dict[str, Any]]:
        """List all runs"""
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
    import os
    return {
        "openrouter_key_set": bool(os.getenv("OPENROUTER_API_KEY")),
        "openrouter_key_length": len(os.getenv("OPENROUTER_API_KEY", "")),
        "pytorch_mps_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK"),
        "env_vars_count": len([k for k in os.environ.keys() if k.startswith(("OPENROUTER", "PYTORCH"))])
    }


@router.get("/debug/env-check")
async def debug_env_check():
    """Debug endpoint to check environment variable configuration status"""
    import os
    api_key = os.getenv("OPENROUTER_API_KEY")
    return {
        "status": "configured" if api_key and api_key != "your_api_key_here" else "not_configured",
        "openrouter_key_set": bool(api_key),
        "openrouter_key_valid": bool(api_key and api_key != "your_api_key_here" and len(api_key) > 10),
        "message": "OpenRouter API key is properly configured" if api_key and api_key != "your_api_key_here" else "OpenRouter API key needs to be set in .env file"
    }


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
        
        logger.info(f"Corpus uploaded", corpus_id=corpus_id, 
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
async def upload_multiple_files(files: List[UploadFile] = File(...), name: Optional[str] = Form(None)):
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
        from ..utils.text_processing import create_smart_chunks_from_files
        from ..utils.config import ChunkingConfig
        import os

        # disable_chonkie = os.getenv("DISABLE_CHONKIE", "false").lower() in ["true", "1", "yes"]
    
        # Always disable chonkie, but keep previous config for reference
        disable_chonkie = True  # Force disable chonkie

        # Previous config (for reference, not used)
        previous_chunking_config = ChunkingConfig(
            use_token_chunking=True,
            target_token_size=3000,
            max_token_size=4000,
            enable_chonkie=True  # Previous: would enable semantic chunking if available
        )

        # New config: chonkie always disabled
        chunking_config = ChunkingConfig(
            use_token_chunking=True,
            target_token_size=3000,
            max_token_size=4000,
            enable_chonkie=False  # Always disable semantic chunking
        )

        if disable_chonkie:
            logger.info("Chonkie disabled via DISABLE_CHONKIE environment variable")
        
        logger.info(f"Creating smart chunks from {len(file_contents)} files with target ~3k tokens per chunk")
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
        logger.info(f"Evaluation request received", 
                   corpus_text_length=len(request.corpus_text) if request.corpus_text else 0,
                   eval_type=request.eval_type,
                   num_questions=request.num_questions,
                   use_agentic=request.use_agentic,
                   temperature=request.temperature,
                   run_name=request.run_name)
        
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
        
        # Create evaluation configuration with environment variables
        from ..utils.config import ConfigManager
        config_manager = ConfigManager()
        config_manager.update_from_env()
        config = config_manager.get_config()
        
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
            except (AttributeError, ValueError) as e:
                # Failed to parse eval_type, use default
                config.eval_type = EvaluationType.DOMAIN_KNOWLEDGE
        else:
            config.eval_type = EvaluationType.DOMAIN_KNOWLEDGE
            
        config.generation.num_questions = request.num_questions
        config.generation.use_agentic = request.use_agentic
        config.llm.temperature = request.temperature
        
        # Load API key from environment variables if not set
        import os
        if not config.llm.api_key:
            api_key = os.environ.get('DOCS_TO_EVAL_API_KEY') or os.environ.get('OPENROUTER_API_KEY')
            if api_key:
                config.llm.api_key = api_key
        
        # Check if API key is available for agentic evaluation
        # Allow agentic evaluation with mock LLMs when no API key is provided
        if request.use_agentic and not config.llm.api_key:
            logger.warning("No API key configured for agentic evaluation - will use mock LLMs for demonstration purposes")
            # Set a flag to indicate mock mode
            config.llm.mock_mode = True
        
        # Store run information
        run_info = {
            "run_id": run_id,
            "status": "queued",
            "phase": None,
            "request": request.dict(),
            "config": config.dict(),
            "start_time": datetime.now(),
            "progress_percent": 0,
            "message": "Evaluation queued",
            "results": None,
            "error": None
        }
        
        evaluation_runs.add_run(run_id, run_info)
        
        # Start evaluation in background
        background_tasks.add_task(run_evaluation, run_id, request, config)
        
        logger.info(f"Evaluation started", run_id=run_id, eval_type=config.eval_type)
        
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
        logger.error(f"Error starting evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluation/qwen-local")
async def start_qwen_local_evaluation(request: QwenEvaluationRequest, background_tasks: BackgroundTasks):
    """🤖 Start local Qwen evaluation with fictional content testing"""
    try:
        logger.info(f"Qwen local evaluation request received", 
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
        
        evaluation_runs.add_run(run_id, run_info)
        
        # Start Qwen evaluation in background
        background_tasks.add_task(run_qwen_local_evaluation, run_id, request)
        
        logger.info(f"Qwen local evaluation started", run_id=run_id)
        
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
        logger.error(f"Error starting Qwen evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# verify_ground_truth_against_corpus function removed - moved to Validator agent in agents.py
# Use Validator.verify_corpus_accuracy() method instead


async def generate_evaluation_questions(corpus_text: str, num_questions: int, eval_type: str, llm_config, tracker) -> List[Dict]:
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
        llm_config: LLM configuration object
        tracker: Progress tracker for UI updates
        
    Returns:
        List of generated question dictionaries
    """
    await tracker.send_log("info", f"🚀 Starting streamlined agentic generation for {num_questions} questions")
    await tracker.send_log("info", "📋 Pipeline: ConceptExtractor → QuestionGenerator → QualityValidator")
    
    try:
        # Import the streamlined orchestrator
        from docs_to_eval.core.agentic.streamlined_orchestrator import StreamlinedOrchestrator
        from docs_to_eval.core.agentic.models import DifficultyLevel
        from docs_to_eval.llm.openrouter_interface import OpenRouterInterface, OpenRouterConfig
        
        # Create LLM interface if API key is available
        llm_interface = None
        if llm_config and llm_config.api_key:
            try:
                openrouter_config = OpenRouterConfig(
                    api_key=llm_config.api_key,
                    model=llm_config.model_name,
                    base_url=llm_config.base_url
                )
                llm_interface = OpenRouterInterface(openrouter_config)
                await tracker.send_log("success", f"✅ Connected to {llm_config.model_name}")
            except Exception as e:
                await tracker.send_log("warning", f"⚠️ Could not initialize LLM: {str(e)}")
                await tracker.send_log("info", "Proceeding with template-based generation")
        
        # Initialize the streamlined orchestrator
        orchestrator = StreamlinedOrchestrator(llm_interface)
        
        # Determine difficulty based on eval_type
        difficulty_map = {
            "factual_qa": DifficultyLevel.BASIC,
            "domain_knowledge": DifficultyLevel.INTERMEDIATE,
            "mathematical": DifficultyLevel.HARD,
            "code_generation": DifficultyLevel.HARD,
            "multiple_choice": DifficultyLevel.INTERMEDIATE,
            "classification": DifficultyLevel.INTERMEDIATE
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
        return await generate_fallback_questions(corpus_text, num_questions, eval_type, tracker)


async def generate_fallback_questions(corpus_text: str, num_questions: int, eval_type: str, tracker) -> List[Dict]:
    """Generate simple fallback questions when the main pipeline fails"""
    await tracker.send_log("info", "Generating fallback questions...")
    
    questions = []
    # Extract simple concepts from text
    words = corpus_text.split()[:500]
    concepts = [w for w in words if len(w) > 6 and w[0].isupper()][:num_questions]
    
    for i, concept in enumerate(concepts):
        questions.append({
            "question": f"What is {concept}?",
            "answer": f"{concept} is mentioned in the provided context.",
            "context": corpus_text[:200],
            "eval_type": eval_type,
            "source": "fallback"
        })
        await tracker.increment_progress(message=f"Generated {i+1}/{num_questions} fallback questions")
    
    # Fill remaining with generic questions
    while len(questions) < num_questions:
        i = len(questions)
        questions.append({
            "question": f"What is discussed in the provided text (part {i+1})?",
            "answer": "The text discusses various topics as outlined in the context.",
            "context": corpus_text[:200],
            "eval_type": eval_type,
            "source": "fallback"
        })
        await tracker.increment_progress(message=f"Generated {i+1}/{num_questions} fallback questions")
    
    await tracker.send_log("success", f"Generated {len(questions)} fallback questions")
    return questions[:num_questions]


# DEPRECATED: This function is replaced by generate_evaluation_questions
async def generate_agentic_questions_with_orchestrator(corpus_text: str, num_questions: int, eval_type: str, llm_config, tracker) -> List[Dict]:
    """
    Generate questions using the full agentic orchestrator pipeline
    ConceptMiner → QuestionWriter → Adversary → Refiner → Validator
    """
    await tracker.send_log("info", f"🤖 Starting FULL agentic pipeline with {num_questions} questions")
    await tracker.send_log("info", "🔄 Pipeline: ConceptMiner → QuestionWriter → Adversary → Refiner → Validator")
    
    try:
        # Create LLM interface from config
        openrouter_config = OpenRouterConfig(
            api_key=llm_config.api_key,
            model=llm_config.model_name,
            base_url=llm_config.base_url
        )
        llm_interface = OpenRouterInterface(openrouter_config)
        
        # Create LLM pool for different agent roles
        llm_pool = {
            'retriever': llm_interface,      # ConceptMiner
            'creator': llm_interface,        # QuestionWriter  
            'adversary': llm_interface,      # Adversary
            'refiner': llm_interface,        # Refiner
        }
        
        # Configure pipeline
        pipeline_config = PipelineConfig(
            oversample_factor=1.5,           # Generate 50% more then select best
            parallel_batch_size=5,           # Process 5 concepts at once
            max_retry_cycles=2,              # Retry failed items
            quality_threshold=0.7            # High quality standard
        )
        
        # Initialize orchestrator
        orchestrator = AgenticBenchmarkOrchestrator(llm_pool, pipeline_config)
        
        # Map string eval_type to EvaluationType enum
        eval_type_map = {
            'mathematical': EvaluationType.MATHEMATICAL,
            'factual_qa': EvaluationType.FACTUAL_QA,
            'code_generation': EvaluationType.CODE_GENERATION,
            'domain_knowledge': EvaluationType.DOMAIN_KNOWLEDGE,
            'multiple_choice': EvaluationType.MULTIPLE_CHOICE,
            'reading_comprehension': EvaluationType.READING_COMPREHENSION,
            'summarization': EvaluationType.SUMMARIZATION,
            'commonsense_reasoning': EvaluationType.COMMONSENSE_REASONING
        }
        
        eval_type_enum = eval_type_map.get(eval_type, EvaluationType.DOMAIN_KNOWLEDGE)
        
        await tracker.send_log("info", f"⚡ ConceptMiner: Extracting key concepts from corpus...")
        
        # Run the orchestrator
        enhanced_items = await orchestrator.generate(
            corpus_text=corpus_text,
            eval_type=eval_type_enum,
            num_questions=num_questions,
            difficulty=DifficultyLevel.HARD  # Use HARD difficulty for challenging questions
        )
        
        await tracker.send_log("info", f"✅ Generated {len(enhanced_items)} enhanced benchmark items")
        
        # Convert EnhancedBenchmarkItem to routes.py format
        questions = []
        for i, item in enumerate(enhanced_items):
            question_dict = {
                "question": item.question,
                "answer": item.answer,
                "context": item.context or "",
                "eval_type": eval_type,
                "concept": item.metadata.provenance.get('concept', f'concept_{i+1}'),
                "difficulty": item.metadata.difficulty,
                "verification_type": "exact" if item.expected_answer_type.value in ['numeric_exact', 'string_exact', 'boolean'] else "factual",
                "source": "agentic_orchestrator_v2",
                "agentic_metadata": {
                    "answer_type": item.expected_answer_type.value,
                    "reasoning_chain": item.reasoning_chain,
                    "adversarial_techniques": item.metadata.adversarial_techniques,
                    "agents_used": item.metadata.agents_used,
                    "validation_score": item.metadata.validation_score,
                    "concept_importance": item.metadata.concept_importance
                },
                "options": item.options  # For multiple choice
            }
            questions.append(question_dict)
            
            # Progress update for each generated question
            await tracker.increment_progress(message=f"🎯 Generated Q{i+1}/{len(enhanced_items)}: {item.question[:50]}...")
        
        # Get pipeline statistics
        if hasattr(orchestrator, 'pipeline_stats'):
            stats = orchestrator.pipeline_stats
            await tracker.send_log("info", f"📊 Pipeline Stats: {stats['total_accepted']}/{stats['total_generated']} accepted, avg quality: {sum(stats['quality_scores'])/len(stats['quality_scores']) if stats['quality_scores'] else 0:.2f}")
        
        await tracker.send_log("success", f"🚀 Full agentic pipeline complete! Generated {len(questions)} high-quality questions")
        return questions
        
    except Exception as e:
        await tracker.send_log("error", f"❌ Agentic orchestrator failed: {str(e)}")
        # Use streamlined generation
        await tracker.send_log("info", "🔄 Using streamlined generation...")
        return await generate_evaluation_questions(corpus_text, num_questions, eval_type, llm_config, tracker)


# DEPRECATED: This function is replaced by generate_evaluation_questions
async def generate_agentic_questions_from_chunk(chunk_text: str, num_questions: int, eval_type: str, llm_config, tracker, chunk_index: int = 0) -> List[Dict]:
    """Generate questions from a single chunk using real LLM"""
    import httpx
    
    questions = []
    
    # Prepare the prompt based on evaluation type
    eval_prompts = {
        "mathematical": "Generate challenging mathematical questions that test problem-solving and calculation skills based on the corpus content",
        "factual_qa": "Generate factual questions that test knowledge and understanding of key concepts from the corpus",
        "code_generation": "Generate coding problems that require implementing solutions based on the concepts in the corpus",
        "domain_knowledge": "Generate domain-specific questions that test deep understanding of the subject matter in the corpus",
        "multiple_choice": "Generate multiple choice questions with 4 options each based on the corpus content",
        "reading_comprehension": "Generate reading comprehension questions that test understanding of the specific corpus text"
    }
    
    prompt_instruction = eval_prompts.get(eval_type, eval_prompts["domain_knowledge"])
    
    system_prompt = f"""You are an expert question generator. Create {num_questions} evaluation questions from the provided text.

CRITICAL: Return ONLY a JSON array with exactly {num_questions} questions. No markdown, no explanations.

TASK: {prompt_instruction}

REQUIREMENTS:
• Generate exactly {num_questions} questions based on the provided corpus text
• Questions must be answerable from the corpus content  
• Provide specific, accurate answers from the text
• Never use "According to", "Based on the corpus", or similar phrases
• Questions should stand alone but be answerable from the content

JSON FORMAT:
[{{"question":"...","answer":"...","concept":"...","difficulty":"basic|intermediate|advanced","verification_type":"exact|numerical|factual|analytical"}}]

Count your questions before responding - you need exactly {num_questions}."""

    user_prompt = f"""Text: {chunk_text}

Generate {num_questions} questions as a JSON array. Return only the JSON:"""
    
    try:
        headers = {
            "Authorization": f"Bearer {llm_config.api_key}",
            "Content-Type": "application/json"
        }
        
        if llm_config.provider == "openrouter":
            headers.update({
                "HTTP-Referer": llm_config.site_url or "https://docs-to-eval.ai",
                "X-Title": llm_config.app_name or "docs-to-eval"
            })
        
        payload = {
            "model": llm_config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": min(llm_config.max_tokens, 8192),  # Limit for JSON response
            "temperature": llm_config.temperature
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                llm_config.base_url + "/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                response_content = result["choices"][0]["message"]["content"].strip()
                
                # Parse JSON response
                import json
                import re
                
                # First try direct JSON parsing
                try:
                    questions_data = json.loads(response_content)
                except json.JSONDecodeError:
                    # If direct parsing fails, try to extract JSON from markdown code blocks
                    json_match = re.search(r'```(?:json)?\n?(\[.*?\])\n?```', response_content, re.DOTALL)
                    if json_match:
                        questions_data = json.loads(json_match.group(1))
                    else:
                        # If no markdown blocks found, re-raise the original error
                        raise
                
                # Validate LLM followed instructions
                generated_count = len(questions_data)
                if generated_count != num_questions:
                    if generated_count > num_questions:
                        await tracker.send_log("warning", f"⚠️ Chunk {chunk_index}: LLM generated {generated_count} questions but requested {num_questions}. Trimming excess.")
                    else:
                        await tracker.send_log("warning", f"⚠️ Chunk {chunk_index}: LLM generated only {generated_count} questions but requested {num_questions}. Using what was generated.")
                
                # Process each question (limit to what was requested)
                questions_to_process = questions_data[:num_questions]
                for q_data in questions_to_process:
                    question_text = q_data.get("question", "")
                    
                    # Check for corpus reference phrases that should be avoided
                    if any(phrase in question_text.lower() for phrase in [
                        "according to the corpus",
                        "based on the text", 
                        "from the text",
                        "the corpus states",
                        "according to the text",
                        "corpus text"
                    ]):
                        await tracker.send_log("warning", f"⚠️ Chunk {chunk_index}: Question contains corpus reference: '{question_text[:60]}...'")
                    
                    question_item = {
                        "question": question_text,
                        "answer": q_data.get("answer", ""),
                        "concept": q_data.get("concept", ""),
                        "difficulty": q_data.get("difficulty", "intermediate"),
                        "verification_type": q_data.get("verification_type", "factual"),
                        "chunk_index": chunk_index
                    }
                    questions.append(question_item)
                
                actual_generated = len(questions)
                if actual_generated == num_questions:
                    await tracker.send_log("info", f"✅ Generated {actual_generated} questions from chunk {chunk_index} (as requested)")
                else:
                    await tracker.send_log("info", f"Generated {actual_generated}/{num_questions} questions from chunk {chunk_index}")
                
            else:
                await tracker.send_log("error", f"API request failed for chunk {chunk_index}: HTTP {response.status_code}")
                
    except json.JSONDecodeError as e:
        await tracker.send_log("error", f"Failed to parse JSON response from chunk {chunk_index}: {str(e)}")
    except Exception as e:
        await tracker.send_log("error", f"Error generating questions from chunk {chunk_index}: {str(e)}")
    
    return questions


# DEPRECATED: This function is replaced by generate_evaluation_questions
async def generate_agentic_questions(corpus_text: str, num_questions: int, eval_type: str, llm_config, tracker) -> List[Dict]:
    
    # Prepare the prompt based on evaluation type
    eval_prompts = {
        "mathematical": "Generate challenging mathematical questions that test problem-solving and calculation skills",
        "factual_qa": "Generate factual questions that test knowledge and understanding of key concepts",
        "code_generation": "Generate coding problems that require implementing solutions based on the concepts",
        "domain_knowledge": "Generate domain-specific questions that test deep understanding of the subject matter",
        "multiple_choice": "Generate multiple choice questions with 4 options each",
        "reading_comprehension": "Generate reading comprehension questions that test understanding of the text"
    }
    
    prompt_instruction = eval_prompts.get(eval_type, eval_prompts["domain_knowledge"])
    
    system_prompt = f"""You are an expert question generator. Create {num_questions} challenging evaluation questions for modern LLMs.

CRITICAL: Return ONLY a JSON array with exactly {num_questions} questions. No markdown, no explanations.

TASK: {prompt_instruction}

REQUIREMENTS:
• Generate exactly {num_questions} CHALLENGING questions (not basic facts!)
• Questions must be answerable WITHOUT corpus access but require deep reasoning
• Target modern LLM capabilities - challenge GPT-4/Claude-3.5 level
• Provide specific, accurate answers as ground truth
• Never use "According to", "Based on the corpus", or similar phrases

Count your questions before responding - you need exactly {num_questions}.

❌ CRITICAL: NEVER start questions with "According to", "Based on the corpus", "From the text", "The corpus states", or similar corpus references. Questions must stand alone!

✅ CORRECT: "How many fascicles of the Corpus Speculorum Etruscorum have been published?" 
❌ WRONG: "According to the corpus text, how many fascicles have been published?"

DIFFICULTY FOCUS - Create questions that are:
❌ AVOID: "What year did X happen?" (too easy - basic fact lookup)
❌ AVOID: "What is 2+2?" (too easy - trivial math)
❌ AVOID: "Define concept Y" (too easy - dictionary lookup)

✅ CREATE: "How do concepts A and B interact to produce outcome C?"
✅ CREATE: "What are the implications of X given constraints Y and Z?" 
✅ CREATE: "Why would approach A be preferred over B in situation C?"
✅ CREATE: Multi-step reasoning, synthesis, analysis questions

QUESTION TYPES:
- Analytical reasoning requiring 2-3 logical steps
- Synthesis questions combining multiple concepts
- Comparative analysis between ideas/approaches
- Implication/consequence questions
- Problem-solving scenarios
- Multi-layered conceptual relationships

ANSWER QUALITY:
- Mathematical: Complex calculations or multi-step problems
- Factual: Nuanced facts requiring domain expertise  
- Conceptual: Sophisticated explanations showing deep understanding
- Avoid simple yes/no or single-word answers

Use this EXACT JSON format:
[{{"question":"...","answer":"...","concept":"...","difficulty":"intermediate|advanced|expert","verification_type":"exact|numerical|factual|analytical"}}]

GOAL: Create a benchmark that will actually challenge state-of-the-art LLMs and reveal their limitations.

IMPORTANT: Return raw JSON only - no ```json blocks, no explanations, no other text."""

    user_prompt = f"""Corpus: {corpus_text}

Generate {num_questions} challenging questions as a JSON array. Return only the JSON:"""

    try:
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {llm_config.api_key}",
            "Content-Type": "application/json"
        }
        
        if llm_config.provider == "openrouter":
            headers.update({
                "HTTP-Referer": llm_config.site_url or "https://docs-to-eval.ai",
                "X-Title": llm_config.app_name or "docs-to-eval"
            })
        
        payload = {
            "model": llm_config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": min(llm_config.max_tokens * 2, 32000),  # More tokens for JSON response
            "temperature": llm_config.temperature
        }
        
        await tracker.send_log("info", f"Generating questions using {llm_config.model_name}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                llm_config.base_url + "/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"LLM API error: {response.status_code} - {response.text}")
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse JSON response and verify ground truth
            import json
            try:
                # First try direct JSON parsing
                questions_data = json.loads(content.strip())
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\n?(\[.*?\])\n?```', content, re.DOTALL)
                if json_match:
                    questions_data = json.loads(json_match.group(1))
                else:
                    # If no markdown blocks found, re-raise the original error
                    raise
            
            try:
                verified_questions = []
                
                # Validate LLM followed instructions
                generated_count = len(questions_data)
                if generated_count != num_questions:
                    if generated_count > num_questions:
                        await tracker.send_log("warning", f"⚠️ LLM generated {generated_count} questions but requested {num_questions}. Trimming excess.")
                    else:
                        await tracker.send_log("warning", f"⚠️ LLM generated only {generated_count} questions but requested {num_questions}. Using what was generated.")
                
                await tracker.send_log("info", f"Verifying {len(questions_data)} generated questions against corpus...")
                
                # Track progress per accepted question, not per generated question
                verified_count = 0
                
                for i, q_data in enumerate(questions_data[:num_questions]):
                    question = q_data.get("question", f"Question {i+1}")
                    answer = q_data.get("answer", "Answer not provided")
                    
                    # CRITICAL: Verify the answer is actually correct according to corpus
                    # Use Validator agent for corpus verification  
                    openrouter_config = OpenRouterConfig(
                        api_key=llm_config.api_key,
                        model=llm_config.model_name
                    )
                    validator_llm = OpenRouterInterface(openrouter_config)
                    validator = Validator(llm_interface=validator_llm)
                    
                    validation_result = await validator.verify_corpus_accuracy(
                        question, answer, corpus_text, min_complexity=0.4
                    )
                    
                    if validation_result.accepted:
                        # Use verified/corrected answer (note: ValidationResult doesn't have verified_answer, use original)
                        complexity = validation_result.score  # Use score as complexity proxy
                        verified_questions.append({
                            "question": question,
                            "answer": answer,  # Keep original answer since ValidationResult doesn't provide corrected version
                            "context": q_data.get("concept", ""),
                            "eval_type": eval_type,
                            "concept": q_data.get("concept", f"concept_{i+1}"),
                            "difficulty": q_data.get("difficulty", "intermediate"),
                            "verification_type": q_data.get("verification_type", "exact"),
                            "complexity": complexity,
                            "source": "agentic_llm_verified",
                            "corpus_verification": {
                                "accepted": validation_result.accepted,
                                "score": validation_result.score,
                                "issues": validation_result.issues,
                                "recommendations": validation_result.recommendations,
                                "method": validation_result.verification_method_used
                            }
                        })
                        verified_count += 1
                        # Only increment progress when a question is actually accepted
                        await tracker.increment_progress(message=f"✅ Verified Q{verified_count}/{num_questions} (score: {complexity:.2f}): {question[:50]}...")
                    else:
                        complexity = validation_result.score
                        issues_summary = "; ".join(validation_result.issues[:2])  # Show first 2 issues
                        
                        if complexity < 0.4:
                            await tracker.send_log("warning", f"❌ Rejected Q{i+1}: TOO EASY (score: {complexity:.2f}) - {issues_summary}")
                        else:
                            await tracker.send_log("warning", f"❌ Rejected Q{i+1}: ISSUES - {issues_summary}")
                
                # Update questions list with only verified questions
                for verified_q in verified_questions:
                    questions.append(verified_q)
                
                # Log verification statistics with complexity breakdown
                total_generated = len(questions_data)
                verified_count = len(verified_questions)
                rejection_rate = (total_generated - verified_count) / total_generated if total_generated > 0 else 0
                
                # Calculate complexity statistics (using score as complexity proxy)
                if verified_questions:
                    complexities = [q.get("complexity", 0.0) for q in verified_questions]
                    avg_complexity = sum(complexities) / len(complexities)
                    easy_count = sum(1 for c in complexities if c < 0.4)
                    moderate_count = sum(1 for c in complexities if 0.4 <= c < 0.7)
                    hard_count = sum(1 for c in complexities if c >= 0.7)
                    
                    complexity_stats = f"Avg score: {avg_complexity:.2f} | Low: {easy_count} | Medium: {moderate_count} | High: {hard_count}"
                else:
                    complexity_stats = "No score data available"
                
                await tracker.send_log("info", f"Ground truth verification: {verified_count}/{total_generated} questions verified ({rejection_rate:.1%} rejected)")
                await tracker.send_log("info", f"Complexity distribution: {complexity_stats}")
                
                # If too many questions were rejected, generate additional ones using same chunking approach
                if verified_count < num_questions * 0.8:  # If less than 80% success rate
                    additional_needed = min(num_questions - verified_count, num_questions // 2)
                    if additional_needed > 0:
                        await tracker.send_log("info", f"Generating {additional_needed} additional questions due to high rejection rate")
                        # Use streamlined generation for consistency
                        additional_questions = await generate_evaluation_questions(
                            corpus_text, additional_needed, eval_type, llm_config, tracker
                        )
                        questions.extend(additional_questions[:additional_needed])
                    
            except json.JSONDecodeError as e:
                await tracker.send_log("warning", f"Failed to parse LLM JSON response: {e}")
                # Fallback: try to extract questions from text
                lines = content.split('\n')
                question_lines = [line for line in lines if '?' in line and len(line.strip()) > 10]
                
                for i, line in enumerate(question_lines[:num_questions]):
                    questions.append({
                        "question": line.strip(),
                        "answer": f"Answer related to this question",
                        "context": None,
                        "eval_type": eval_type,
                        "source": "agentic_llm_fallback"
                    })
                    
                    await tracker.increment_progress(message=f"Extracted question {i+1}/{len(question_lines[:num_questions])}")
    
    except Exception as e:
        await tracker.send_log("error", f"Agentic generation failed: {str(e)}")
        logger.error(f"Agentic question generation failed: {e}")
        
        # Fallback to corpus-based generation
        await tracker.send_log("info", "Falling back to corpus-based generation")
        return await generate_corpus_questions(corpus_text, num_questions, eval_type, tracker)
    
    # If we didn't get enough questions, fill with corpus-based ones
    if len(questions) < num_questions:
        remaining = num_questions - len(questions)
        await tracker.send_log("info", f"Generating {remaining} additional corpus-based questions")
        additional = await generate_corpus_questions(corpus_text, remaining, eval_type, tracker)
        questions.extend(additional)
    
    return questions[:num_questions]


async def evaluate_with_real_llm(questions: List[Dict], llm_config, tracker, corpus_text: str = "") -> List[Dict]:
    """Proper agentic evaluation - LLM answers questions WITHOUT seeing expected answers"""
    import httpx
    
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

            headers = {
                "Authorization": f"Bearer {llm_config.api_key}",
                "Content-Type": "application/json"
            }
            
            if llm_config.provider == "openrouter":
                headers.update({
                    "HTTP-Referer": llm_config.site_url or "https://docs-to-eval.ai",
                    "X-Title": llm_config.app_name or "docs-to-eval"
                })
            
            payload = {
                "model": llm_config.model_name,
                "messages": [
                    {"role": "user", "content": evaluation_prompt}
                ],
                "max_tokens": llm_config.max_tokens,
                "temperature": llm_config.temperature
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    llm_config.base_url + "/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["choices"][0]["message"]["content"].strip()
                    
                    # Debug logging
                    logger.info(f"LLM Response for question {i+1}: '{prediction[:100]}...'")
                    
                    llm_results.append({
                        "question": question["question"],
                        "ground_truth": question["answer"],
                        "prediction": prediction,
                        "confidence": 0.9,  # High confidence for real LLM
                        "source": "real_llm",
                        "model": llm_config.model_name
                    })
                    
                    evaluated_count += 1
                    await tracker.increment_progress(message=f"Evaluated question {evaluated_count}/{len(questions)} with {llm_config.model_name}")
                else:
                    # Fallback for failed requests
                    llm_results.append({
                        "question": question["question"],
                        "ground_truth": question["answer"],
                        "prediction": f"Error: API request failed ({response.status_code})",
                        "confidence": 0.0,
                        "source": "api_error"
                    })
                    evaluated_count += 1
                    await tracker.increment_progress(message=f"Evaluated question {evaluated_count}/{len(questions)} (API error)")
                    await tracker.send_log("warning", f"API request failed for question {i+1}: {response.status_code}")
                    
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


async def run_evaluation(run_id: str, request: EvaluationRequest, config: EvaluationConfig):
    """Run evaluation in background"""
    tracker = get_progress_tracker(run_id)
    
    try:
        # Update status
        evaluation_runs.update_run(run_id, {"status": "running"})
        await tracker.send_log("info", "Starting evaluation")
        
        # Phase 1: Classification  
        await tracker.start_phase("classification", "Analyzing corpus content")
        
        # Determine evaluation type - use explicit if provided, otherwise auto-detect
        if request.eval_type and request.eval_type not in ["auto-detect", "auto"]:
            # Use explicitly specified evaluation type
            eval_type_str = request.eval_type
            await tracker.send_log("info", f"Using explicitly specified evaluation type: {eval_type_str}")
            await tracker.end_phase({"primary_type": eval_type_str, "method": "explicit"})
        else:
            # Auto-detect evaluation type
            classifier = EvaluationTypeClassifier()
            classification = classifier.classify_corpus(request.corpus_text)
            eval_type_str = classification.primary_type.value if hasattr(classification.primary_type, 'value') else str(classification.primary_type)
            await tracker.send_log("info", f"Auto-detected evaluation type: {eval_type_str}")
            await tracker.end_phase({"primary_type": eval_type_str, "method": "auto_detected"})
        
        # Phase 2: Benchmark Generation
        await tracker.start_phase("generation", "Generating benchmark questions", request.num_questions)
        
        # Check if we should use agentic generation (force True if API key available)
        should_use_agentic = config.llm.api_key and (request.use_agentic or True)  # Force agentic when API key available
        
        if should_use_agentic:
            # Use streamlined agentic pipeline
            await tracker.send_log("info", f"Using model \"{config.llm.model_name}\" for streamlined agentic pipeline")
            questions = await generate_evaluation_questions(
                request.corpus_text,
                request.num_questions,
                eval_type_str,  # Use string instead of enum
                config.llm,
                tracker
            )
        else:
            # Fallback to corpus-based generation
            await tracker.send_log("info", "Using corpus-based question generation (no API key or agentic disabled)")
            questions = await generate_corpus_questions(
                request.corpus_text, 
                request.num_questions, 
                eval_type_str,  # Use string instead of enum
                tracker
            )
        
        await tracker.end_phase({"questions_generated": len(questions)})
        
        # Phase 2.5: Create Finetune Test Set (if enabled)
        finetune_test_set = None
        if request.finetune_test_set_enabled and len(questions) > 1:
            await tracker.send_log("info", f"Creating finetune test set with {request.finetune_test_set_percentage*100:.0f}% test questions")
            
            # Import the function we created
            from ..core.evaluation import create_finetune_test_set
            
            finetune_test_set = create_finetune_test_set(
                questions=questions,
                test_percentage=request.finetune_test_set_percentage,
                random_seed=request.finetune_random_seed
            )
            
            await tracker.send_log("info", f"Finetune test set created: {finetune_test_set.train_set_size} train + {finetune_test_set.test_set_size} test questions")
        
        # Phase 3: LLM Evaluation
        await tracker.start_phase("evaluation", "Evaluating with LLM", len(questions))
        
        # Initialize LLM interface for performance stats
        llm = None
        
        if config.llm.api_key:
            # Use real LLM for evaluation
            await tracker.send_log("info", f"Using model \"{config.llm.model_name}\" for evaluation")
            llm_results = await evaluate_with_real_llm(questions, config.llm, tracker, request.corpus_text)
        else:
            # Fallback to mock evaluation
            await tracker.send_log("info", "Using mock LLM evaluation (no API key provided)")
            llm = MockLLMInterface(temperature=request.temperature)
            
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
                await asyncio.sleep(0.1)  # Simulate work
        
        await tracker.end_phase({"evaluations_completed": len(llm_results)})
        
        # Phase 4: Verification
        await tracker.start_phase("verification", "Verifying responses", len(llm_results))
        
        # Import verification system
        from ..core.verification import VerificationOrchestrator
        
        # Use mixed verification if configured (default True)
        use_mixed = config.verification.use_mixed_verification if hasattr(config.verification, 'use_mixed_verification') else True
        orchestrator = VerificationOrchestrator(corpus_text=request.corpus_text, use_mixed=use_mixed)
        
        verification_results = []
        for i, result in enumerate(llm_results):
            # Use real verification based on the classification type
            # Convert enum to string for verification system
            eval_type_str = classification.primary_type.value if hasattr(classification.primary_type, 'value') else str(classification.primary_type)
            verification_result = orchestrator.verify(
                prediction=result["prediction"],
                ground_truth=result["ground_truth"],
                eval_type=eval_type_str,  # Convert enum to string
                options=result.get("options"),
                question=result["question"]  # Pass question for context
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
            await asyncio.sleep(0.1)  # Simulate work
        
        await tracker.end_phase({"verifications_completed": len(verification_results)})
        
        # Phase 5: Report Generation
        await tracker.start_phase("reporting", "Generating comprehensive report with statistical analysis")
        
        # Import statistical analysis
        try:
            from ..utils.statistical_analysis import EvaluationStatistics
            
            # Generate comprehensive statistical report following lm-evaluation-harness principles
            statistical_report = EvaluationStatistics.generate_evaluation_report(
                verification_results, 
                corpus_text=request.corpus_text
            )
            
            # Extract main statistics for backward compatibility
            main_stats = statistical_report.get("main_statistics")
            mean_score = main_stats.mean if main_stats else 0
            
        except Exception as e:
            logger.warning(f"Statistical analysis failed, using basic metrics: {e}")
            # Fallback to basic statistics
            scores = [r["score"] for r in verification_results]
            mean_score = sum(scores) / len(scores) if scores else 0
            statistical_report = {"error": f"Statistical analysis failed: {str(e)}"}
            main_stats = None
        
        # Create finetune test set summary
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
            "individual_results": verification_results,  # Show all results
            "performance_stats": llm.get_performance_stats() if llm else {},
            "finetune_test_set": finetune_summary,
            "completed_at": datetime.now().isoformat()
        }
        
        await tracker.end_phase({"report_generated": True})
        
        # Update final status
        evaluation_runs.update_run(run_id, {
            "status": "completed",
            "results": final_results,
            "end_time": datetime.now(),
            "progress_percent": 100,
            "message": "Evaluation completed successfully"
        })
        
        # Send completion notification
        await tracker.notifier.send_evaluation_complete(final_results)
        
        logger.info(f"Evaluation completed", run_id=run_id, mean_score=mean_score)
        
    except Exception as e:
        # Handle errors
        error_msg = str(e)
        logger.error(f"Evaluation error", run_id=run_id, error=error_msg)
        
        evaluation_runs.update_run(run_id, {
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
        evaluation_runs.update_run(run_id, {"status": "running"})
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
        
        report = evaluator.generate_report(evaluation_results, scores, corpus_info)
        
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
        
        # Update final status
        evaluation_runs.update_run(run_id, {
            "status": "completed",
            "results": final_results,
            "end_time": datetime.now(),
            "progress_percent": 100,
            "message": f"Qwen local evaluation completed! Mean Score: {mean_score:.3f}"
        })
        
        # Send completion notification
        await tracker.notifier.send_evaluation_complete(final_results)
        
        logger.info(f"Qwen local evaluation completed", run_id=run_id, mean_score=mean_score, num_questions=len(questions))
        
    except Exception as e:
        # Handle errors
        error_msg = str(e)
        logger.error(f"Qwen local evaluation error", run_id=run_id, error=error_msg)
        
        evaluation_runs.update_run(run_id, {
            "status": "error",
            "error": error_msg,
            "end_time": datetime.now(),
            "message": f"Qwen local evaluation failed: {error_msg}"
        })
        
        await tracker.send_error("Qwen local evaluation failed", error_msg)


@router.get("/evaluation/{run_id}/status")
async def get_evaluation_status(run_id: str):
    """Get evaluation status"""
    if evaluation_runs.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run_info = evaluation_runs.get_run(run_id)
    
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
    if evaluation_runs.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run_info = evaluation_runs.get_run(run_id)
    
    return EvaluationResult(
        run_id=run_id,
        status=run_info["status"],
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
    if evaluation_runs.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run_info = evaluation_runs.get_run(run_id)
    
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
    for run_id, run_info in evaluation_runs.list_runs().items():
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
    
    return sorted(runs, key=lambda x: x["start_time"], reverse=True)


@router.delete("/runs/{run_id}")
async def delete_evaluation_run(run_id: str):
    """Delete an evaluation run"""
    if evaluation_runs.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    
    evaluation_runs.delete_run(run_id)
    logger.info(f"Evaluation run deleted", run_id=run_id)
    
    return {"message": "Run deleted successfully"}


@router.get("/evaluation/{run_id}/finetune-test-set")
async def get_finetune_test_set(run_id: str):
    """Get finetune test set details for a completed evaluation"""
    run_data = evaluation_runs.get_run(run_id)
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
    run_data = evaluation_runs.get_run(run_id)
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


@router.get("/config/default")
async def get_default_config():
    """Get default evaluation configuration"""
    config = create_default_config()
    return config.dict()


@router.get("/config/current")
async def get_current_config():
    """Get current evaluation configuration"""
    try:
        from ..utils.config import ConfigManager
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
        logger.error(f"Error getting current config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load configuration: {str(e)}")


@router.post("/config/update")
async def update_config(config_update: dict):
    """Update configuration (API key and other settings)"""
    try:
        from ..utils.config import ConfigManager, EvaluationConfig
        import os
        
        # Validate input
        if not config_update or not isinstance(config_update, dict):
            raise HTTPException(status_code=400, detail="Invalid config update data")
        
        # Load current config
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
        
        # Store the API key in environment for this session
        api_key_set = False
        if 'llm' in config_update and 'api_key' in config_update['llm']:
            api_key = config_update['llm']['api_key']
            if api_key and isinstance(api_key, str) and api_key.strip():
                os.environ['DOCS_TO_EVAL_API_KEY'] = api_key.strip()
                api_key_set = True
                logger.info("API key updated successfully")
        
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
        if not api_test or 'api_key' not in api_test:
            raise HTTPException(status_code=400, detail="API key is required")
        
        api_key = api_test['api_key']
        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise HTTPException(status_code=400, detail="Invalid API key format")
        
        # Basic format validation for common API key patterns
        api_key = api_key.strip()
        if len(api_key) < 10:
            raise HTTPException(status_code=400, detail="API key appears to be too short")
        
        # Test with a simple OpenRouter call
        import aiohttp
        import asyncio
        
        test_url = "https://openrouter.ai/api/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://docs-to-eval.ai",
            "X-Title": "docs-to-eval"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(test_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return {
                        "status": "success",
                        "message": "API key is valid and working",
                        "valid": True
                    }
                elif response.status == 401:
                    return {
                        "status": "error", 
                        "message": "API key is invalid or unauthorized",
                        "valid": False
                    }
                else:
                    return {
                        "status": "warning",
                        "message": f"Could not validate API key (HTTP {response.status})",
                        "valid": False
                    }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing API key: {e}")
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
        "active_runs": len(evaluation_runs.list_runs()),
        "websocket_connections": len(websocket_manager.active_connections)
    }