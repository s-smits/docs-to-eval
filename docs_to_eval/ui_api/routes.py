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
from pydantic import BaseModel, Field, ValidationError, ConfigDict

from .websockets import websocket_manager, handle_websocket_connection, get_progress_tracker
from ..core.evaluation import EvaluationFramework, BenchmarkConfig
from ..core.classification import EvaluationTypeClassifier
from ..llm.mock_interface import MockLLMInterface, MockLLMEvaluator
from ..utils.config import EvaluationConfig, EvaluationType, create_default_config
from ..utils.logging import get_logger
from ..utils.text_processing import predict_optimal_questions, create_smart_chunks

# Create router
router = APIRouter()
logger = get_logger("api_routes")


# Pydantic models for API
class CorpusUploadRequest(BaseModel):
    text: str
    name: Optional[str] = "corpus"
    description: Optional[str] = ""


class EvaluationRequest(BaseModel):
    corpus_text: str
    eval_type: Optional[str] = None  # Accept string, convert later
    num_questions: int = Field(default=20, ge=1, le=200)
    use_agentic: bool = True
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_concurrent: int = Field(
        default=5, 
        ge=1, 
        le=20, 
        description="Max concurrent LLM requests (5=balanced, 1=conservative, 10+=aggressive)"
    )
    run_name: Optional[str] = None


class EvaluationStatus(BaseModel):
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        # For backwards compatibility, make dict() work like model_dump(mode='json')
        use_enum_values=True
    )
    
    run_id: str
    status: str
    phase: Optional[str] = None
    progress_percent: float = 0
    message: str = ""
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    
    def dict(self, **kwargs):
        """Override dict method to use proper JSON serialization"""
        return self.model_dump(mode='json', **kwargs)


class EvaluationResult(BaseModel):
    run_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None


class TextAnalysisRequest(BaseModel):
    text: str
    eval_type_hint: Optional[str] = None


class TextAnalysisResponse(BaseModel):
    text_stats: Dict[str, Any]
    suggested_questions: int
    min_questions: int  
    max_questions: int
    reasoning: str
    chunk_info: Dict[str, Any]
    eval_type_suggestions: List[str]


# Persistent storage for evaluation runs
EVALUATION_RUNS_FILE = Path("cache/evaluation_runs.json")
evaluation_runs: Dict[str, Dict[str, Any]] = {}

def load_evaluation_runs():
    """Load evaluation runs from disk"""
    global evaluation_runs
    try:
        if EVALUATION_RUNS_FILE.exists():
            with open(EVALUATION_RUNS_FILE, 'r') as f:
                data = json.load(f)
                # Convert datetime strings back to datetime objects where needed
                for run_id, run_data in data.items():
                    if 'start_time' in run_data and isinstance(run_data['start_time'], str):
                        run_data['start_time'] = datetime.fromisoformat(run_data['start_time'])
                    if 'end_time' in run_data and isinstance(run_data['end_time'], str):
                        run_data['end_time'] = datetime.fromisoformat(run_data['end_time'])
                    if 'estimated_completion' in run_data and isinstance(run_data['estimated_completion'], str):
                        run_data['estimated_completion'] = datetime.fromisoformat(run_data['estimated_completion'])
                evaluation_runs = data
                logger.info(f"Loaded {len(evaluation_runs)} evaluation runs from disk")
    except Exception as e:
        logger.error(f"Failed to load evaluation runs from disk: {e}")
        evaluation_runs = {}

def save_evaluation_runs():
    """Save evaluation runs to disk"""
    try:
        # Ensure cache directory exists
        EVALUATION_RUNS_FILE.parent.mkdir(exist_ok=True)
        
        # Convert datetime objects and other non-serializable objects for JSON
        serializable_data = {}
        for run_id, run_data in evaluation_runs.items():
            serializable_run = {}
            for key, value in run_data.items():
                if key in ['start_time', 'end_time', 'estimated_completion'] and isinstance(value, datetime):
                    serializable_run[key] = value.isoformat()
                elif key == 'results' and value is not None:
                    # Skip complex results objects to avoid serialization issues
                    # Results are already saved to output files, so this is just status tracking
                    serializable_run[key] = {"status": "completed", "saved_to_file": True}
                elif isinstance(value, (str, int, float, bool, type(None))):
                    serializable_run[key] = value
                elif isinstance(value, (list, dict)):
                    try:
                        # Test if it's JSON serializable
                        json.dumps(value)
                        serializable_run[key] = value
                    except (TypeError, ValueError):
                        # Skip non-serializable complex objects
                        serializable_run[key] = f"<{type(value).__name__} object>"
                else:
                    # Convert other objects to string representation
                    serializable_run[key] = str(value)
            
            serializable_data[run_id] = serializable_run
        
        with open(EVALUATION_RUNS_FILE, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save evaluation runs to disk: {e}")

# Load existing evaluation runs at startup
load_evaluation_runs()


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
    """Upload corpus from file"""
    try:
        # Read file content
        content = await file.read()
        
        # Detect encoding and decode
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('latin-1')
        
        # Create request
        request = CorpusUploadRequest(
            text=text,
            name=name or file.filename or "uploaded_corpus",
            description=f"Uploaded from file: {file.filename}"
        )
        
        return await upload_corpus_text(request)
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/corpus/upload-multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...), name: Optional[str] = Form(None)):
    """Upload multiple files as a single corpus"""
    try:
        combined_text = ""
        file_names = []
        total_size = 0
        
        # Supported text file extensions
        supported_extensions = {'.txt', '.md', '.py', '.js', '.json', '.csv', '.html', '.xml', '.yml', '.yaml', '.cfg', '.ini', '.log'}
        
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
                
                # Add file separator and content
                if combined_text:
                    combined_text += f"\n\n{'='*60}\n"
                combined_text += f"FILE: {file.filename}\n{'='*60}\n\n"
                combined_text += text
                file_names.append(file.filename)
                
            except Exception as e:
                logger.warning(f"Error processing file {file.filename}: {e}")
                continue
        
        if not combined_text:
            raise HTTPException(status_code=400, detail="No valid text files found")
        
        # Create request
        corpus_name = name or f"Multiple files ({len(file_names)} files)"
        request = CorpusUploadRequest(
            text=combined_text,
            name=corpus_name,
            description=f"Combined corpus from {len(file_names)} files: {', '.join(file_names[:5])}" + 
                       (f" and {len(file_names)-5} more" if len(file_names) > 5 else "")
        )
        
        result = await upload_corpus_text(request)
        result["files_processed"] = len(file_names)
        result["file_names"] = file_names
        
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
        
        # Validate required fields
        if not request.corpus_text or not request.corpus_text.strip():
            raise HTTPException(status_code=422, detail="corpus_text is required and cannot be empty")
        # Generate run ID
        run_id = str(uuid.uuid4())
        
        # Create evaluation configuration
        config = create_default_config()
        
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
            except:
                config.eval_type = EvaluationType.DOMAIN_KNOWLEDGE
        else:
            config.eval_type = EvaluationType.DOMAIN_KNOWLEDGE
            
        config.generation.num_questions = request.num_questions
        config.generation.use_agentic = request.use_agentic
        config.llm.temperature = request.temperature
        config.llm.max_concurrent = request.max_concurrent
        
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
        
        evaluation_runs[run_id] = run_info
        save_evaluation_runs()  # Persist to disk
        
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


async def generate_agentic_questions(corpus_text: str, num_questions: int, eval_type: str, llm_config, tracker) -> List[Dict]:
    """Generate advanced questions using sophisticated stress-testing approach"""
    import httpx
    
    questions = []
    
    # Import the advanced question generation system
    try:
        from ..core.advanced_question_generation import AdvancedQuestionGenerator, create_advanced_question_prompt
        
        # First try advanced question generation for sophisticated stress-testing
        await tracker.send_log("info", "Using advanced question generation for stress-testing LLMs")
        
        generator = AdvancedQuestionGenerator(corpus_text, domain="historical")
        # Generate at most half the questions using advanced method, but respect the total limit
        max_advanced = min(num_questions // 2, 10, num_questions)
        advanced_questions = generator.generate_advanced_question_set(max_advanced)
        
        # Convert to expected format
        for q in advanced_questions:
            if len(questions) >= num_questions:  # Respect the limit
                break
            questions.append({
                "question": q["question"],
                "answer": q["answer"],
                "context": q.get("concept", ""),
                "eval_type": eval_type,
                "concept": q.get("concept", ""),
                "difficulty": q.get("difficulty", "advanced"),
                "complexity_layer": q.get("complexity_layer", "synthesis"),
                "source": "advanced_generation"
            })
        
        await tracker.send_log("info", f"Generated {len(questions)} sophisticated stress-testing questions")
        
    except Exception as e:
        await tracker.send_log("warning", f"Advanced generation failed: {e}, falling back to LLM generation")
        advanced_questions = []
    
    # Fill remaining questions with LLM-generated questions using advanced prompts
    remaining_questions = max(0, num_questions - len(questions))
    if remaining_questions > 0:
        try:
            # Use the advanced prompt system for sophisticated question generation
            from ..core.advanced_question_generation import create_advanced_question_prompt
            
            system_prompt = create_advanced_question_prompt(corpus_text, remaining_questions, eval_type)
            
            user_prompt = f"""Based on this corpus, generate {remaining_questions} advanced evaluation questions that stress-test LLM capabilities:

---CORPUS START---
{corpus_text[:120000]}  
---CORPUS END---

Generate questions following the advanced complexity requirements specified in the instructions."""
            
        except Exception:
            # Fallback to original prompt system
            eval_prompts = {
                "mathematical": "Generate challenging mathematical questions that require multi-step reasoning and domain expertise",
                "factual_qa": "Generate questions requiring synthesis across sources and deep domain knowledge",
                "code_generation": "Generate complex coding problems that require understanding of underlying principles",
                "domain_knowledge": "Generate sophisticated questions that stress-test deep understanding and reasoning",
                "multiple_choice": "Generate challenging multiple choice questions with plausible distractors",
                "reading_comprehension": "Generate complex comprehension questions requiring inference and analysis"
            }
            
            prompt_instruction = eval_prompts.get(eval_type, eval_prompts["domain_knowledge"])
            
            system_prompt = f"""You are an expert creating ADVANCED evaluation questions designed to stress-test LLMs beyond surface-level knowledge.

OBJECTIVE: Generate questions that require synthesis, inference, reasoning under ambiguity, and domain expertise integration.

Instructions:
1. {prompt_instruction}
2. Generate exactly {remaining_questions} questions
3. Force models to make connections between disparate information
4. Include plausible assumptions or "what-if" scenarios  
5. Require both calculation AND interpretation for mathematical questions
6. Test reasoning about methodology, causation, or broader implications
7. For mathematical questions, provide FINAL NUMERICAL ANSWER only
8. Questions should be genuinely challenging for advanced AI systems

Format your response as a JSON array with this structure:
[
  {{
    "question": "Complex, multi-layered question requiring synthesis and inference",
    "answer": "Expected reasoning approach and key conclusions",
    "concept": "Primary concept being stress-tested", 
    "difficulty": "advanced|expert|research",
    "complexity_layer": "synthesis|inference|ambiguity|extrapolation"
  }}
]

Only return the JSON array, no other text."""
            
            user_prompt = f"""Based on this corpus, generate {remaining_questions} advanced stress-testing questions:

---CORPUS START---
{corpus_text[:120000]}  
---CORPUS END---

Generate sophisticated questions that challenge advanced reasoning capabilities."""

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
            
            await tracker.send_log("info", f"Generating sophisticated questions using {llm_config.model_name}")
            
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
                
                # Parse JSON response
                import json
                try:
                    questions_data = json.loads(content.strip())
                    
                    for i, q_data in enumerate(questions_data[:remaining_questions]):
                        if len(questions) >= num_questions:  # Respect the limit
                            break
                        questions.append({
                            "question": q_data.get("question", f"Advanced Question {len(questions)+1}"),
                            "answer": q_data.get("answer", "Answer not provided"),
                            "context": q_data.get("concept", ""),
                            "eval_type": eval_type,
                            "concept": q_data.get("concept", f"concept_{len(questions)+1}"),
                            "difficulty": q_data.get("difficulty", "advanced"),
                            "complexity_layer": q_data.get("complexity_layer", "synthesis"),
                            "source": "advanced_agentic_llm"
                        })
                        
                        await tracker.increment_progress(message=f"Generated advanced question {len(questions)}: {q_data.get('question', '')[:50]}...")
                        
                except json.JSONDecodeError as e:
                    await tracker.send_log("warning", f"Failed to parse LLM JSON response: {e}")
                    # Fallback: try to extract questions from text
                    lines = content.split('\n')
                    question_lines = [line for line in lines if '?' in line and len(line.strip()) > 10]
                    
                    for i, line in enumerate(question_lines[:remaining_questions]):
                        if len(questions) >= num_questions:  # Respect the limit
                            break
                        questions.append({
                            "question": line.strip(),
                            "answer": f"Based on the corpus content related to this question",
                            "context": None,
                            "eval_type": eval_type,
                            "source": "advanced_llm_fallback"
                        })
                        
                        await tracker.increment_progress(message=f"Extracted question {len(questions)}")
        
        except Exception as e:
            await tracker.send_log("error", f"Advanced LLM generation failed: {str(e)}")
            logger.error(f"Advanced LLM question generation failed: {e}")
    
    # If we still don't have enough questions, use corpus-based generation
    if len(questions) < num_questions:
        remaining = num_questions - len(questions)
        await tracker.send_log("info", f"Generating {remaining} additional corpus-based questions")
        additional = await generate_corpus_questions(corpus_text, remaining, eval_type, tracker)
        questions.extend(additional[:remaining])  # Only take what we need
    
    return questions[:num_questions]


async def evaluate_single_question(question: Dict, question_index: int, llm_config, tracker, semaphore: asyncio.Semaphore, shared_client: httpx.AsyncClient) -> Dict:
    """Evaluate a single question with rate limiting"""
    async with semaphore:  # Limit concurrent requests
        try:
            # Prepare evaluation prompt with context if available
            if question.get('context'):
                evaluation_prompt = f"""Context: {question['context']}

Based on the context above, please answer the following question:

Question: {question['question']}

Provide a clear, accurate answer. If the question is mathematical, show your work. If it's factual, provide specific details."""
            else:
                evaluation_prompt = f"""Please answer the following question based on your knowledge:

Question: {question['question']}

Provide a clear, accurate answer. If the question is mathematical, show your work. If it's factual, provide specific details."""

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
            
            response = await shared_client.post(
                llm_config.base_url + "/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result["choices"][0]["message"]["content"].strip()
                
                result_dict = {
                    "question": question["question"],
                    "ground_truth": question["answer"],
                    "prediction": prediction,
                    "confidence": 0.9,  # High confidence for real LLM
                    "source": "real_llm",
                    "model": llm_config.model_name
                }
                
                await tracker.increment_progress(message=f"✓ Completed question {question_index+1} with {llm_config.model_name}")
                return result_dict
            else:
                # Fallback for failed requests
                result_dict = {
                    "question": question["question"],
                    "ground_truth": question["answer"],
                    "prediction": f"Error: API request failed ({response.status_code})",
                    "confidence": 0.0,
                    "source": "api_error"
                }
                await tracker.send_log("warning", f"API request failed for question {question_index+1}: {response.status_code}")
                return result_dict
                
        except Exception as e:
            logger.warning(f"Error evaluating question {question_index+1}: {e}")
            # Fallback for exceptions
            result_dict = {
                "question": question["question"],
                "ground_truth": question["answer"],
                "prediction": f"Error: {str(e)[:100]}",
                "confidence": 0.0,
                "source": "evaluation_error"
            }
            await tracker.send_log("warning", f"Failed to evaluate question {question_index+1}: {str(e)}")
            return result_dict


async def evaluate_with_real_llm(questions: List[Dict], llm_config, tracker) -> List[Dict]:
    """Evaluate questions using real LLM with optimized concurrent processing"""
    
    # Configure concurrency - balance speed with rate limits
    max_concurrent_requests = getattr(llm_config, 'max_concurrent', 5)  # Use configured value, default 5
    
    await tracker.send_log("info", f"Starting concurrent evaluation with {max_concurrent_requests} parallel requests")
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    # Use optimized connection pooling - only what we need
    timeout = httpx.Timeout(30.0, connect=10.0)
    # Sensible connection limits: keepalive = concurrent, max = concurrent + 2 (buffer)
    limits = httpx.Limits(
        max_keepalive_connections=max_concurrent_requests, 
        max_connections=max_concurrent_requests + 2
    )
    
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        # Create tasks for all questions
        tasks = [
            evaluate_single_question(question, i, llm_config, tracker, semaphore, client)
            for i, question in enumerate(questions)
        ]
        
        # Execute all tasks concurrently
        try:
            llm_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions that occurred
            final_results = []
            for i, result in enumerate(llm_results):
                if isinstance(result, Exception):
                    # Convert exception to error result
                    error_result = {
                        "question": questions[i]["question"],
                        "ground_truth": questions[i]["answer"],
                        "prediction": f"Error: {str(result)[:100]}",
                        "confidence": 0.0,
                        "source": "concurrent_error"
                    }
                    final_results.append(error_result)
                    await tracker.send_log("error", f"Concurrent evaluation error for question {i+1}: {str(result)}")
                else:
                    final_results.append(result)
            
            await tracker.send_log("info", f"Completed concurrent evaluation of {len(final_results)} questions")
            return final_results
            
        except Exception as e:
            await tracker.send_log("error", f"Critical error in concurrent evaluation: {str(e)}")
            # Fallback to empty results
            return []


def _detect_content_type(corpus_text: str) -> str:
    """Detect actual content type to prevent template-content mismatches"""
    
    # Mathematical content indicators
    math_patterns = [
        r'\b\d+\s*[+\-*/=]\s*\d+',  # Simple equations
        r'\$[^$]+\$',  # LaTeX math
        r'\\[a-zA-Z]+\{[^}]*\}',  # LaTeX commands
        r'\b(equation|formula|calculate|solve|derivative|integral)\b',
        r'\b\d+\.\d+\b.*\b\d+\.\d+\b'  # Multiple decimal numbers
    ]
    
    # Code content indicators  
    code_patterns = [
        r'def\s+\w+\(',  # Python functions
        r'function\s+\w+\(',  # JavaScript functions
        r'\bclass\s+\w+\b',  # Class definitions
        r'import\s+\w+',  # Import statements
        r'#include\s*<',  # C/C++ includes
    ]
    
    math_score = sum(1 for pattern in math_patterns if re.search(pattern, corpus_text, re.IGNORECASE))
    code_score = sum(1 for pattern in code_patterns if re.search(pattern, corpus_text, re.IGNORECASE))
    
    if math_score >= 2:
        return "mathematical"
    elif code_score >= 2:
        return "code_generation"
    else:
        return "factual_qa"  # Safe default

def _extract_semantic_concepts(corpus_text: str, eval_type: str) -> List[str]:
    """Extract concepts appropriate for the evaluation type"""
    
    if eval_type == "mathematical":
        # Extract mathematical entities: variables, equations, numbers
        concepts = []
        # Mathematical variables (single letters, often in equations)
        variables = re.findall(r'\b[a-zA-Z]\b(?=\s*[=+\-*/])', corpus_text)
        concepts.extend(variables)
        
        # Mathematical expressions in text
        expressions = re.findall(r'\b\w+\s*=\s*[^.,;]+', corpus_text)
        concepts.extend([expr.strip() for expr in expressions[:5]])
        
        # Numbers with units or context
        numbers_in_context = re.findall(r'\b\d+(?:\.\d+)?\s*(?:grams?|meters?|seconds?|degrees?|percent)', corpus_text, re.IGNORECASE)
        concepts.extend(numbers_in_context[:3])
        
        return concepts[:10] if concepts else ["x", "the equation", "the value"]
    
    elif eval_type == "code_generation":
        # Extract programming concepts
        concepts = []
        # Function names
        functions = re.findall(r'(?:def|function)\s+(\w+)', corpus_text)
        concepts.extend(functions)
        
        # Class names
        classes = re.findall(r'class\s+(\w+)', corpus_text)
        concepts.extend(classes)
        
        # Common programming terms
        prog_terms = re.findall(r'\b(algorithm|function|method|class|variable|array|list|dictionary)\b', corpus_text, re.IGNORECASE)
        concepts.extend(list(set(prog_terms)))
        
        return concepts[:10] if concepts else ["a function", "an algorithm", "a data structure"]
    
    else:  # factual_qa and domain_knowledge
        # Extract proper nouns and key domain terms
        concepts = []
        
        # Proper nouns (capitalized words, likely important entities)
        proper_nouns = re.findall(r'\b[A-Z][a-z]{2,}\b', corpus_text)
        # Filter out common English words
        common_words = {'The', 'This', 'That', 'There', 'These', 'Those', 'When', 'Where', 'Why', 'How', 'What', 'Which', 'Who'}
        proper_nouns = [word for word in proper_nouns if word not in common_words]
        concepts.extend(list(set(proper_nouns))[:15])
        
        # Multi-word entities (capitalize phrases)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', corpus_text)
        concepts.extend(entities[:10])
        
        return concepts[:20] if concepts else ["the main topic", "the subject matter", "the concept"]

async def generate_corpus_questions(corpus_text: str, num_questions: int, eval_type: str, tracker) -> List[Dict]:
    """Generate questions using improved template approach with content-type detection"""
    import random
    
    questions = []
    
    # CRITICAL FIX: Detect actual content type to prevent mismatches
    detected_type = _detect_content_type(corpus_text)
    if detected_type != eval_type:
        await tracker.send_log("warning", f"Content type mismatch detected. Requested: {eval_type}, Detected: {detected_type}. Using detected type.")
        eval_type = detected_type
    
    # Extract sentences for context
    sentences = re.split(r'[.!?]+', corpus_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    # Extract semantic concepts appropriate for the evaluation type
    concepts = _extract_semantic_concepts(corpus_text, eval_type)
    
    # Define templates with better specificity
    if eval_type == "mathematical":
        templates = [
            "What is the value of {concept}?",
            "Calculate {concept}.",
            "If {concept}, what is the result?",
            "Solve for the value in: {concept}",
            "What does {concept} equal?"
        ]
    elif eval_type == "factual_qa":
        templates = [
            "What is {concept}?",
            "Define {concept}.",
            "Explain {concept}.",
            "What does {concept} refer to?",
            "Describe the significance of {concept}."
        ]
    elif eval_type == "code_generation":
        templates = [
            "Write code to implement {concept}.",
            "How would you implement {concept}?",
            "Create {concept} in code.",
            "Write a program for {concept}.",
            "Implement {concept} using appropriate data structures."
        ]
    else:  # domain_knowledge and others
        templates = [
            "What is {concept}?",
            "Explain the concept of {concept}.",
            "What is the significance of {concept}?",
            "How does {concept} relate to the main topic?",
            "Describe the role of {concept}."
        ]
    
    # Generate questions with improved logic
    for i in range(num_questions):
        if len(questions) >= num_questions:
            break
            
        try:
            # Select appropriate concept
            if concepts:
                concept = random.choice(concepts)
                # Remove used concept to avoid duplicates
                if concept in concepts:
                    concepts.remove(concept)
                if not concepts:  # Refresh if we run out
                    concepts = _extract_semantic_concepts(corpus_text, eval_type)
            else:
                concept = f"topic {i+1}"
            
            # Generate question with validation
            template = random.choice(templates)
            question = template.format(concept=concept)
            
            # Generate contextual answer
            answer_context = ""
            for sentence in sentences:
                if any(word.lower() in sentence.lower() for word in concept.split()[:2]):
                    answer_context = sentence.strip()
                    break
            
            if not answer_context:
                if eval_type == "mathematical":
                    answer_context = f"The mathematical relationship involving {concept}"
                elif eval_type == "factual_qa":
                    answer_context = f"Based on the corpus content related to {concept}"
                else:
                    answer_context = f"Information about {concept} from the provided text"
            
            questions.append({
                "question": question,
                "answer": answer_context,
                "context": None,  # Will be added later if needed
                "eval_type": eval_type,
                "concept": concept,
                "generation_mode": "template_fallback",  # Flag this as fallback
                "quality_warning": "Generated using emergency template fallback due to agentic pipeline failure"
            })
            
            await tracker.increment_progress(message=f"Generated question {len(questions)}: {question[:50]}...")
            await asyncio.sleep(0.05)
            
        except Exception as e:
            logger.warning(f"Error generating question {i+1}: {e}")
            # Final fallback with clear labeling
            if len(questions) < num_questions:
                questions.append({
                    "question": f"What information can you provide about the content in section {i+1}?",
                    "answer": f"Relevant information from the corpus about section {i+1}",
                    "context": None,
                    "eval_type": "factual_qa",  # Safe default
                    "concept": f"section_{i+1}",
                    "generation_mode": "emergency_fallback",
                    "quality_warning": "Emergency fallback question due to generation failure"
                })
    
    return questions


async def run_evaluation(run_id: str, request: EvaluationRequest, config: EvaluationConfig):
    """Run evaluation in background"""
    tracker = get_progress_tracker(run_id)
    
    try:
        # Update status
        evaluation_runs[run_id]["status"] = "running"
        save_evaluation_runs()  # Persist to disk
        await tracker.send_log("info", "Starting evaluation")
        
        # Phase 1: Classification
        await tracker.start_phase("classification", "Analyzing corpus content")
        classifier = EvaluationTypeClassifier()
        classification = classifier.classify_corpus(request.corpus_text)
        await tracker.end_phase({"primary_type": classification.primary_type})
        
        # Phase 2: Benchmark Generation
        await tracker.start_phase("generation", "Generating benchmark questions", request.num_questions)
        
        # Check if we should use agentic generation
        if request.use_agentic and config.llm.api_key:
            # Use real LLM for agentic question generation
            await tracker.send_log("info", f"Using model \"{config.llm.model_name}\" for agentic question generation")
            questions = await generate_agentic_questions(
                request.corpus_text,
                request.num_questions,
                classification.primary_type,
                config.llm,
                tracker
            )
        else:
            # Fallback to corpus-based generation
            await tracker.send_log("info", "Using corpus-based question generation (no API key or agentic disabled)")
            questions = await generate_corpus_questions(
                request.corpus_text, 
                request.num_questions, 
                classification.primary_type,
                tracker
            )
        
        await tracker.end_phase({"questions_generated": len(questions)})
        
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
                
                await tracker.increment_progress(message=f"Mock evaluated question {i+1}")
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
            verification_result = orchestrator.verify(
                prediction=result["prediction"],
                ground_truth=result["ground_truth"],
                eval_type=classification.primary_type,  # Use the classified type
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
            
            await tracker.increment_progress(message=f"Verified response {i+1}: {verification_result.method} score={verification_result.score:.2f}")
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
                corpus_text=request.corpus_text,
                eval_type=classification.primary_type  # CRITICAL FIX: Use task-specific baseline
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
        
        # CRITICAL FIX: Extract and surface quality warnings
        quality_warnings = []
        generation_modes = []
        
        # Check questions for quality warnings
        for question in questions:
            if isinstance(question, dict):
                if warning := question.get("quality_warning"):
                    quality_warnings.append(warning)
                if mode := question.get("generation_mode"):
                    generation_modes.append(mode)
        
        # Check metadata in verification results for agentic fallback warnings
        for result in verification_results:
            if "details" in result and isinstance(result["details"], dict):
                metadata = result["details"].get("metadata", {})
                if isinstance(metadata, dict):
                    provenance = metadata.get("provenance", {})
                    if isinstance(provenance, dict):
                        if warning := provenance.get("quality_warning"):
                            quality_warnings.append(warning)
                        if mode := provenance.get("generation_mode"):
                            generation_modes.append(mode)
        
        # Determine overall quality status
        has_fallback = any(mode in ['template_fallback', 'emergency_fallback', 'agentic_fallback'] for mode in generation_modes)
        quality_degraded = len(quality_warnings) > 0
        
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
            "quality_status": {
                "has_fallback_questions": has_fallback,
                "quality_degraded": quality_degraded,
                "warnings": list(set(quality_warnings)),  # Remove duplicates
                "generation_modes": list(set(generation_modes)),
                "total_warnings": len(quality_warnings),
                "fallback_percentage": (sum(1 for mode in generation_modes if mode in ['template_fallback', 'emergency_fallback', 'agentic_fallback']) / max(1, len(questions))) * 100
            },
            "detailed_statistics": statistical_report,
            "individual_results": verification_results[:10],  # First 10 only
            "performance_stats": llm.get_performance_stats() if llm else {},
            "completed_at": datetime.now().isoformat()
        }
        
        await tracker.end_phase({"report_generated": True})
        
        # Update final status
        evaluation_runs[run_id].update({
            "status": "completed",
            "results": final_results,
            "end_time": datetime.now(),
            "progress_percent": 100,
            "message": "Evaluation completed successfully"
        })
        save_evaluation_runs()  # Persist to disk
        
        # Send completion notification with quality warnings
        await tracker.notifier.send_evaluation_complete(final_results)
        
        # Log completion with quality status
        if quality_degraded:
            logger.warning(f"Evaluation completed with quality degradation", 
                         run_id=run_id, mean_score=mean_score, 
                         fallback_percentage=final_results["quality_status"]["fallback_percentage"],
                         total_warnings=len(quality_warnings))
        else:
            logger.info(f"Evaluation completed successfully", run_id=run_id, mean_score=mean_score)
        
    except Exception as e:
        # Handle errors
        error_msg = str(e)
        logger.error(f"Evaluation error", run_id=run_id, error=error_msg)
        
        evaluation_runs[run_id].update({
            "status": "error",
            "error": error_msg,
            "end_time": datetime.now(),
            "message": f"Evaluation failed: {error_msg}"
        })
        save_evaluation_runs()  # Persist to disk
        
        await tracker.send_error(error_msg)


@router.get("/evaluation/{run_id}/status")
async def get_evaluation_status(run_id: str):
    """Get evaluation status with optimized polling recommendations"""
    if run_id not in evaluation_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run_info = evaluation_runs[run_id]
    
    # Add polling recommendations based on status
    polling_interval = 1.0  # Default 1 second
    if run_info["status"] == "running":
        # Reduce polling frequency during processing
        polling_interval = 2.0 if run_info.get("progress_percent", 0) < 90 else 0.5
    elif run_info["status"] in ["completed", "failed", "cancelled"]:
        # No need for frequent polling when done
        polling_interval = 10.0
    
    response = EvaluationStatus(
        run_id=run_id,
        status=run_info["status"],
        phase=run_info.get("phase"),
        progress_percent=run_info.get("progress_percent", 0),
        message=run_info.get("message", ""),
        start_time=run_info["start_time"],
        estimated_completion=run_info.get("estimated_completion")
    )
    
    # Add recommended polling interval as header
    return JSONResponse(
        content=response.dict(),
        headers={"X-Recommended-Poll-Interval": str(polling_interval)}
    )


@router.get("/evaluation/{run_id}/results")
async def get_evaluation_results(run_id: str):
    """Get evaluation results"""
    if run_id not in evaluation_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run_info = evaluation_runs[run_id]
    
    return EvaluationResult(
        run_id=run_id,
        status=run_info["status"],
        results=run_info.get("results"),
        error=run_info.get("error"),
        duration_seconds=(
            ((run_info.get("end_time") if isinstance(run_info.get("end_time"), datetime) else datetime.fromisoformat(run_info.get("end_time"))) - 
             (run_info["start_time"] if isinstance(run_info["start_time"], datetime) else datetime.fromisoformat(run_info["start_time"]))).total_seconds()
            if run_info.get("end_time") else None
        )
    )


@router.get("/evaluation/{run_id}/download")
async def download_results(run_id: str):
    """Download evaluation results as JSON file"""
    if run_id not in evaluation_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run_info = evaluation_runs[run_id]
    
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
    for run_id, run_info in evaluation_runs.items():
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
    if run_id not in evaluation_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    del evaluation_runs[run_id]
    logger.info(f"Evaluation run deleted", run_id=run_id)
    
    return {"message": "Run deleted successfully"}


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
                    except:
                        return {"status": "error", "message": f"API test failed with status {response.status_code}"}
        
        else:
            return {"status": "error", "message": f"Provider {provider} not supported for testing"}
            
    except Exception as e:
        logger.error(f"Error testing API key: {e}")
        return {"status": "error", "message": f"API test failed: {str(e)}"}


@router.post("/analysis/text")
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze text corpus and provide intelligent suggestions for evaluation setup.
    
    Returns question count suggestions, chunking information, and evaluation type recommendations.
    """
    try:
        # Get question predictions
        question_prediction = predict_optimal_questions(
            request.text, 
            eval_type=request.eval_type_hint
        )
        
        # Get chunking information with adaptive sizing for 2k-4k range
        optimal_chunk_size = 3000  # Default 3k
        text_length = len(request.text)
        
        # Adaptive chunk sizing based on text length and complexity
        if text_length > 20000:
            optimal_chunk_size = 4000  # Larger chunks for very long texts
        elif text_length < 5000:
            optimal_chunk_size = 2000  # Smaller chunks for shorter texts
        
        chunks = create_smart_chunks(request.text, target_chunk_size=optimal_chunk_size, overlap_percent=5.0)
        
        # Collect chunking method statistics
        chunk_methods = {}
        for chunk in chunks:
            method = chunk.get("method", "unknown")
            chunk_methods[method] = chunk_methods.get(method, 0) + 1
        
        primary_method = max(chunk_methods.items(), key=lambda x: x[1])[0] if chunk_methods else "unknown"
        
        chunk_info = {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(len(chunk["text"]) for chunk in chunks) // max(1, len(chunks)),
            "target_chunk_size": optimal_chunk_size,
            "total_processed_chars": sum(len(chunk["text"]) for chunk in chunks),
            "overlap_efficiency": round((question_prediction["text_stats"]["char_count"] / 
                                      max(1, sum(len(chunk["text"]) for chunk in chunks))) * 100, 1),
            "chunking_method": primary_method,
            "method_distribution": chunk_methods,
            "avg_semantic_score": round(sum(chunk.get("semantic_score", 1.0) for chunk in chunks) / max(1, len(chunks)), 3),
            "size_range": f"{min(len(chunk['text']) for chunk in chunks) if chunks else 0}-{max(len(chunk['text']) for chunk in chunks) if chunks else 0} chars",
            "chonkie_features": {
                "uses_advanced_chunking": primary_method.startswith("chonkie_"),
                "semantic_boundaries": primary_method in ["chonkie_semantic", "chonkie_late"],
                "structure_aware": primary_method == "chonkie_recursive",
                "global_context_preserved": primary_method == "chonkie_late",
                "optimal_for_llm": optimal_chunk_size >= 2000 and optimal_chunk_size <= 4000
            },
            "quality_metrics": {
                "coherence_score": round(sum(chunk.get("semantic_coherence", chunk.get("semantic_score", 1.0)) for chunk in chunks) / max(1, len(chunks)), 3),
                "size_variance": round(
                    (sum((len(chunk["text"]) - sum(len(c["text"]) for c in chunks) / len(chunks))**2 for chunk in chunks) / max(1, len(chunks)))**0.5, 1
                ) if chunks else 0,
                "overlap_quality": "good" if 3 <= overlap_percent <= 7 else "suboptimal"
            }
        }
        
        # Suggest evaluation types based on content analysis
        eval_suggestions = []
        text_stats = question_prediction["text_stats"]
        
        if text_stats["has_math"]:
            eval_suggestions.append("mathematical")
        if text_stats["has_code"]:  
            eval_suggestions.append("code_generation")
        if text_stats["entity_count"] > 10:
            eval_suggestions.extend(["factual_qa", "domain_knowledge"])
        if text_stats["vocabulary_richness"] < 0.4:
            eval_suggestions.append("reading_comprehension")
        if not eval_suggestions:
            eval_suggestions.append("domain_knowledge")  # Safe default
            
        # Remove duplicates and limit to top 3
        eval_suggestions = list(dict.fromkeys(eval_suggestions))[:3]
        
        return TextAnalysisResponse(
            text_stats=text_stats,
            suggested_questions=question_prediction["suggested"],
            min_questions=question_prediction["min"],
            max_questions=question_prediction["max"],
            reasoning=question_prediction["reasoning"],
            chunk_info=chunk_info,
            eval_type_suggestions=eval_suggestions
        )
        
    except Exception as e:
        logger.error(f"Text analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


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
        "active_runs": len(evaluation_runs),
        "websocket_connections": len(websocket_manager.active_connections)
    }