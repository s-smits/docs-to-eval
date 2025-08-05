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
from ..llm.mock_interface import MockLLMInterface, MockLLMEvaluator
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
        
        # Create chunking config optimized for ~3k tokens per chunk
        chunking_config = ChunkingConfig(
            use_token_chunking=True,
            target_token_size=3000,
            max_token_size=4000,
            enable_chonkie=True  # Use semantic chunking if available
        )
        
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
        if request.use_agentic and not config.llm.api_key:
            raise HTTPException(
                status_code=400, 
                detail="API key is required for agentic evaluation. Please set OPENROUTER_API_KEY environment variable or configure it in Settings."
            )
        
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


async def verify_ground_truth_against_corpus(question: str, proposed_answer: str, corpus_text: str, llm_config, tracker) -> Dict[str, Any]:
    """Verify that the proposed answer is actually correct according to the corpus"""
    import httpx
    import json
    
    verification_prompt = f"""You are a fact-checker and complexity assessor.

CRITICAL: You must respond with ONLY a valid JSON object. Do not use markdown code blocks, explanations, or any other formatting.

TASK:
1. Verify if the proposed answer is factually correct according to the source text
2. Assess the complexity/difficulty of the question for modern LLMs

SOURCE TEXT: {corpus_text[:50000]}

QUESTION: {question}
PROPOSED ANSWER: {proposed_answer}

QUESTION RULE: The question must NOT start with phrases like "Based on the corpus", "According to the text", or similar. You must act as if the next LLM answering this question will NOT have access to the corpus or source text. The question must stand alone and make sense without referencing the corpus.

Use this EXACT JSON format:
{{"is_correct":true,"verified_answer":"...","confidence":0.8,"complexity":0.7,"reasoning":"...","evidence":"...","complexity_analysis":"..."}}

COMPLEXITY SCORING (0.0-1.0):
- 0.0-0.3: TOO EASY - Simple facts, basic math, obvious answers (reject these)
- 0.4-0.6: MODERATE - Requires some reasoning, domain knowledge, multi-step thinking
- 0.7-1.0: CHALLENGING - Complex reasoning, obscure facts, multi-layered analysis

RULES:
1. Answer correct ONLY if directly supported by source text
2. REJECT questions with complexity < 0.4 (too easy for modern LLMs)
3. Prefer questions requiring reasoning, synthesis, or domain expertise
4. Consider: Would GPT-4/Claude-3.5 find this challenging?

IMPORTANT: Return raw JSON only - no ```json blocks, no explanations, no other text."""

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
                {"role": "user", "content": verification_prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.1  # Low temperature for factual verification
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                llm_config.base_url + "/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                verification_content = result["choices"][0]["message"]["content"].strip()
                
                try:
                    # First try direct JSON parsing
                    verification_result = json.loads(verification_content)
                except json.JSONDecodeError:
                    # If direct parsing fails, try to extract JSON from markdown code blocks
                    import re
                    json_match = re.search(r'```(?:json)?\n?(\{.*?\})\n?```', verification_content, re.DOTALL)
                    if json_match:
                        verification_result = json.loads(json_match.group(1))
                    else:
                        # If no markdown blocks found, re-raise the original error
                        raise
                
                try:
                    
                    # Ensure all required fields are present
                    complexity = verification_result.get("complexity", 0.0)
                    is_correct = verification_result.get("is_correct", False)
                    
                    # CRITICAL: Reject questions that are too easy for modern LLMs
                    if complexity < 0.4:
                        is_correct = False  # Override - reject easy questions regardless of correctness
                    
                    return {
                        "is_correct": is_correct,
                        "verified_answer": verification_result.get("verified_answer", proposed_answer),
                        "confidence": verification_result.get("confidence", 0.0),
                        "complexity": complexity,
                        "reasoning": verification_result.get("reasoning", "Verification failed"),
                        "evidence": verification_result.get("evidence", "No evidence found"),
                        "complexity_analysis": verification_result.get("complexity_analysis", "No complexity analysis"),
                        "verification_method": "llm_corpus_check_with_complexity"
                    }
                    
                except json.JSONDecodeError:
                    return {
                        "is_correct": False,
                        "verified_answer": proposed_answer,
                        "confidence": 0.0,
                        "complexity": 0.0,
                        "reasoning": "Failed to parse verification response",
                        "evidence": "",
                        "complexity_analysis": "Parse error - cannot assess complexity",
                        "verification_method": "failed_parse"
                    }
            else:
                return {
                    "is_correct": False,
                    "verified_answer": proposed_answer,
                    "confidence": 0.0,
                    "complexity": 0.0,
                    "reasoning": f"Verification API failed: {response.status_code}",
                    "evidence": "",
                    "complexity_analysis": "API error - cannot assess complexity",
                    "verification_method": "api_error"
                }
                
    except Exception as e:
        await tracker.send_log("error", f"Ground truth verification error: {str(e)}")
        return {
            "is_correct": False,
            "verified_answer": proposed_answer,
            "confidence": 0.0,
            "complexity": 0.0,
            "reasoning": f"Verification exception: {str(e)}",
            "evidence": "",
            "complexity_analysis": "Exception occurred - cannot assess complexity",
            "verification_method": "exception"
        }


async def generate_agentic_questions_with_chunking(corpus_text: str, num_questions: int, eval_type: str, llm_config, tracker) -> List[Dict]:
    """Generate questions using real LLM with smart chunking for optimal context"""
    import httpx
    from ..utils.text_processing import create_smart_chunks
    from ..utils.config import ChunkingConfig
    
    # Create smart chunks optimized for ~3k tokens each
    chunking_config = ChunkingConfig(
        use_token_chunking=True,
        target_token_size=3000,
        max_token_size=4000,
        enable_chonkie=True
    )
    
    await tracker.send_log("info", f"Creating smart chunks from corpus ({len(corpus_text)} chars) with ~3k token targets...")
    chunks = create_smart_chunks(corpus_text, chunking_config)
    await tracker.send_log("info", f"Created {len(chunks)} smart chunks for question generation")
    
    # Smart chunk selection strategy
    if len(chunks) > num_questions:
        # If we have more chunks than questions, randomly sample chunks for variety
        import random
        selected_chunks = random.sample(chunks, min(num_questions, len(chunks)))
        questions_per_chunk = 1  # One question per selected chunk
        await tracker.send_log("info", f"Randomly selected {len(selected_chunks)} chunks from {len(chunks)} total for variety")
    elif len(chunks) <= num_questions:
        # If we have fewer chunks than questions, use all chunks
        selected_chunks = chunks
        questions_per_chunk = max(1, num_questions // len(chunks))
        await tracker.send_log("info", f"Using all {len(chunks)} chunks with ~{questions_per_chunk} questions per chunk")
    
    all_questions = []
    remaining_questions = num_questions
    
    for i, chunk in enumerate(selected_chunks):
        if remaining_questions <= 0:
            break
            
        # Calculate questions for this chunk
        if len(selected_chunks) > num_questions:
            # One question per chunk when we have more chunks than questions
            chunk_questions = 1
        else:
            # Distribute questions evenly, with last chunk getting remainder
            chunk_questions = min(questions_per_chunk, remaining_questions)
            if i == len(selected_chunks) - 1:  # Last chunk gets any remaining questions
                chunk_questions = remaining_questions
            
        await tracker.send_log("info", f"Generating {chunk_questions} questions from chunk {i+1}/{len(chunks)} ({chunk.get('token_count', 'N/A')} tokens)")
        
        # Use the existing agentic generation with this chunk
        chunk_questions_data = await generate_agentic_questions_from_chunk(
            chunk['text'], 
            chunk_questions, 
            eval_type, 
            llm_config, 
            tracker, 
            chunk_index=i
        )
        
        all_questions.extend(chunk_questions_data)
        remaining_questions -= len(chunk_questions_data)
    
    await tracker.send_log("info", f"Generated {len(all_questions)} total questions from {len(chunks)} chunks")
    return all_questions[:num_questions]  # Ensure we don't exceed the requested number


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
    
    system_prompt = f"""You are an expert question generator specializing in creating evaluation questions from corpus text.

CRITICAL: You must respond with ONLY a valid JSON array. Do not use markdown code blocks, explanations, or any other formatting.

REQUIREMENTS:
1. {prompt_instruction}
2. Generate exactly {num_questions} questions based DIRECTLY on the provided corpus text
3. Questions must be answerable from the corpus content with specific, accurate answers
4. Create questions that test understanding of the actual content, not general knowledge
5. Provide EXACT answers that can be found in or directly derived from the corpus

QUESTION QUALITY:
- Focus on key concepts, relationships, and important details from the corpus
- Create questions that require understanding, not just memory
- For mathematical content: use actual numbers/calculations from the corpus
- For factual content: ask about specific information present in the text
- For conceptual content: test understanding of relationships and implications
- Avoid questions that could be answered without reading the corpus

ANSWER QUALITY:
- Mathematical: Use exact numbers and calculations from the corpus
- Factual: Provide specific facts directly from the text
- Conceptual: Give explanations based on the corpus content
- Keep answers concise but complete

Use this EXACT JSON format:
[{{"question":"...","answer":"...","concept":"...","difficulty":"basic|intermediate|advanced","verification_type":"exact|numerical|factual|analytical"}}]

IMPORTANT: Return raw JSON only - no ```json blocks, no explanations, no other text."""

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
                
                # Process each question
                for q_data in questions_data[:num_questions]:
                    question_item = {
                        "question": q_data.get("question", ""),
                        "answer": q_data.get("answer", ""),
                        "concept": q_data.get("concept", ""),
                        "difficulty": q_data.get("difficulty", "intermediate"),
                        "verification_type": q_data.get("verification_type", "factual"),
                        "chunk_index": chunk_index
                    }
                    questions.append(question_item)
                
                await tracker.send_log("info", f"Generated {len(questions)} questions from chunk {chunk_index}")
                
            else:
                await tracker.send_log("error", f"API request failed for chunk {chunk_index}: HTTP {response.status_code}")
                
    except json.JSONDecodeError as e:
        await tracker.send_log("error", f"Failed to parse JSON response from chunk {chunk_index}: {str(e)}")
    except Exception as e:
        await tracker.send_log("error", f"Error generating questions from chunk {chunk_index}: {str(e)}")
    
    return questions


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
    
    system_prompt = f"""You are an expert question generator specializing in CHALLENGING evaluation questions for modern LLMs.

CRITICAL: You must respond with ONLY a valid JSON array. Do not use markdown code blocks, explanations, or any other formatting.

REQUIREMENTS:
1. {prompt_instruction}
2. Generate exactly {num_questions} CHALLENGING questions (not basic facts!)
3. Questions must be answerable WITHOUT corpus access but require deep reasoning
4. Target modern LLM capabilities - make questions that will challenge GPT-4/Claude-3.5
5. Provide EXACT, SPECIFIC answers as ground truth

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

    user_prompt = f"""Corpus: {corpus_text[:120000]}

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
                
                await tracker.send_log("info", f"Verifying {len(questions_data)} generated questions against corpus...")
                
                # Track progress per accepted question, not per generated question
                verified_count = 0
                
                for i, q_data in enumerate(questions_data[:num_questions]):
                    question = q_data.get("question", f"Question {i+1}")
                    answer = q_data.get("answer", "Answer not provided")
                    
                    # CRITICAL: Verify the answer is actually correct according to corpus
                    verification_result = await verify_ground_truth_against_corpus(
                        question, answer, corpus_text, llm_config, tracker
                    )
                    
                    if verification_result["is_correct"]:
                        # Use verified/corrected answer
                        complexity = verification_result.get("complexity", 0.0)
                        verified_questions.append({
                            "question": question,
                            "answer": verification_result["verified_answer"],
                            "context": q_data.get("concept", ""),
                            "eval_type": eval_type,
                            "concept": q_data.get("concept", f"concept_{i+1}"),
                            "difficulty": q_data.get("difficulty", "intermediate"),
                            "verification_type": q_data.get("verification_type", "exact"),
                            "complexity": complexity,
                            "source": "agentic_llm_verified",
                            "corpus_verification": verification_result
                        })
                        verified_count += 1
                        # Only increment progress when a question is actually accepted
                        await tracker.increment_progress(message=f"✅ Verified Q{verified_count}/{num_questions} (complexity: {complexity:.2f}): {question[:50]}...")
                    else:
                        complexity = verification_result.get("complexity", 0.0)
                        reasoning = verification_result.get("reasoning", "Unknown reason")
                        
                        if complexity < 0.4:
                            await tracker.send_log("warning", f"❌ Rejected Q{i+1}: TOO EASY (complexity: {complexity:.2f}) - {reasoning}")
                        else:
                            await tracker.send_log("warning", f"❌ Rejected Q{i+1}: INCORRECT - {reasoning}")
                
                # Update questions list with only verified questions
                for verified_q in verified_questions:
                    questions.append(verified_q)
                
                # Log verification statistics with complexity breakdown
                total_generated = len(questions_data)
                verified_count = len(verified_questions)
                rejection_rate = (total_generated - verified_count) / total_generated if total_generated > 0 else 0
                
                # Calculate complexity statistics
                if verified_questions:
                    complexities = [q.get("complexity", 0.0) for q in verified_questions]
                    avg_complexity = sum(complexities) / len(complexities)
                    easy_count = sum(1 for c in complexities if c < 0.4)
                    moderate_count = sum(1 for c in complexities if 0.4 <= c < 0.7)
                    hard_count = sum(1 for c in complexities if c >= 0.7)
                    
                    complexity_stats = f"Avg complexity: {avg_complexity:.2f} | Easy: {easy_count} | Moderate: {moderate_count} | Hard: {hard_count}"
                else:
                    complexity_stats = "No complexity data available"
                
                await tracker.send_log("info", f"Ground truth verification: {verified_count}/{total_generated} questions verified ({rejection_rate:.1%} rejected)")
                await tracker.send_log("info", f"Complexity distribution: {complexity_stats}")
                
                # If too many questions were rejected, generate additional ones
                if verified_count < num_questions * 0.8:  # If less than 80% success rate
                    additional_needed = min(num_questions - verified_count, num_questions // 2)
                    if additional_needed > 0:
                        await tracker.send_log("info", f"Generating {additional_needed} additional questions due to high rejection rate")
                        # Recursive call with adjusted prompt for better quality
                        additional_questions = await generate_agentic_questions(
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
                        "answer": f"Based on the corpus content related to this question",
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


async def evaluate_with_real_llm(questions: List[Dict], llm_config, tracker) -> List[Dict]:
    """Proper agentic evaluation - LLM answers questions WITHOUT seeing expected answers"""
    import httpx
    
    llm_results = []
    evaluated_count = 0
    
    for i, question in enumerate(questions):
        try:
            # CRITICAL: LLM answers question WITHOUT context or expected answer
            # This is the proper evaluation setup - blind testing
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
                answer_context = f"Based on the corpus content about {concept}"
            
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
        classifier = EvaluationTypeClassifier()
        classification = classifier.classify_corpus(request.corpus_text)
        await tracker.end_phase({"primary_type": classification.primary_type})
        
        # Phase 2: Benchmark Generation
        await tracker.start_phase("generation", "Generating benchmark questions", request.num_questions)
        
        # Convert enum to string for all generation functions
        eval_type_str = classification.primary_type.value if hasattr(classification.primary_type, 'value') else str(classification.primary_type)
        
        # Check if we should use agentic generation (force True if API key available)
        should_use_agentic = config.llm.api_key and (request.use_agentic or True)  # Force agentic when API key available
        
        if should_use_agentic:
            # Use real LLM for agentic question generation with smart chunking
            await tracker.send_log("info", f"Using model \"{config.llm.model_name}\" for agentic question generation with smart chunking")
            questions = await generate_agentic_questions_with_chunking(
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
            llm_results = await evaluate_with_real_llm(questions, config.llm, tracker)
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
        orchestrator = VerificationOrchestrator(corpus_text=request.corpus_text)
        
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


@router.get("/config/default")
async def get_default_config():
    """Get default evaluation configuration"""
    config = create_default_config()
    return config.dict()


@router.get("/config/current")
async def get_current_config():
    """Get current evaluation configuration"""
    from ..utils.config import ConfigManager
    manager = ConfigManager()
    manager.update_from_env()
    config = manager.get_config()
    
    # Don't expose the API key in the response
    config_dict = config.dict()
    if config_dict.get('llm', {}).get('api_key'):
        config_dict['llm']['api_key'] = '***masked***'
    
    return config_dict


@router.post("/config/update")
async def update_config(config_update: dict):
    """Update configuration (API key and other settings)"""
    try:
        from ..utils.config import ConfigManager, EvaluationConfig
        
        # Load current config
        manager = ConfigManager()
        manager.update_from_env()
        
        # Update with provided values
        current_dict = manager.get_config().dict()
        
        # Safely update nested config
        def update_nested(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    update_nested(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        update_nested(current_dict, config_update)
        
        # Validate the updated config
        updated_config = EvaluationConfig(**current_dict)
        
        # Store the API key in environment for this session
        if 'llm' in config_update and 'api_key' in config_update['llm']:
            import os
            os.environ['DOCS_TO_EVAL_API_KEY'] = config_update['llm']['api_key']
        
        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "api_key_set": bool(updated_config.llm.api_key)
        }
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/config/test-api-key")
async def test_api_key(api_test: dict):
    """Test API key validity"""
    try:
        api_key = api_test.get('api_key')
        provider = api_test.get('provider', 'openrouter')
        model = api_test.get('model', 'openai/gpt-3.5-turbo')
        
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Test the API key with a simple request
        import httpx
        
        if provider == 'openrouter':
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://docs-to-eval.ai",
                "X-Title": "docs-to-eval",
                "Content-Type": "application/json"
            }
            
            test_payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=test_payload
                )
                
                if response.status_code == 200:
                    return {"status": "success", "message": "API key is valid and working"}
                elif response.status_code == 401:
                    return {"status": "error", "message": "Invalid API key - check your key and try again"}
                elif response.status_code == 402:
                    return {"status": "error", "message": "Payment required - insufficient credits in your OpenRouter account. Add credits at openrouter.ai/credits"}
                elif response.status_code == 429:
                    return {"status": "error", "message": "Rate limit exceeded - too many requests. Try again in a few minutes"}
                elif response.status_code == 403:
                    return {"status": "error", "message": "Access forbidden - check your API key permissions"}
                else:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('error', {}).get('message', f'HTTP {response.status_code}')
                        return {"status": "error", "message": f"API error: {error_msg}"}
                    except (json.JSONDecodeError, AttributeError) as e:
                        # Failed to parse error response
                        return {"status": "error", "message": f"API test failed with status {response.status_code}"}
        
        else:
            return {"status": "error", "message": f"Provider {provider} not supported for testing"}
            
    except Exception as e:
        logger.error(f"Error testing API key: {e}")
        return {"status": "error", "message": f"API test failed: {str(e)}"}


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