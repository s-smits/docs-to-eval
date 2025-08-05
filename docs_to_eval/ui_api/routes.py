"""
FastAPI routes for docs-to-eval system
"""

import asyncio
import uuid
import json
import shutil
<<<<<<< HEAD
import httpx
import re
=======
>>>>>>> parent of 6898d60 (increase concurrency and benchmark complexity and math)
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, Form, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

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
    text: str
    name: Optional[str] = "corpus"
    description: Optional[str] = ""


class EvaluationRequest(BaseModel):
    corpus_text: str
    eval_type: Optional[str] = None  # Accept string, convert later
    num_questions: int = Field(default=20, ge=1, le=200)
    use_agentic: bool = True
    temperature: float = Field(default=0.7, ge=0, le=2)
    run_name: Optional[str] = None


class EvaluationStatus(BaseModel):
    run_id: str
    status: str
    phase: Optional[str] = None
    progress_percent: float = 0
    message: str = ""
    start_time: datetime
    estimated_completion: Optional[datetime] = None


class EvaluationResult(BaseModel):
    run_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None


# In-memory storage for evaluation runs (in production, use a database)
evaluation_runs: Dict[str, Dict[str, Any]] = {}


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
    """Generate questions using real LLM (agentic approach)"""
    import httpx
    
    questions = []
    
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
    
    system_prompt = f"""You are an expert educational assessment creator. Your task is to generate high-quality evaluation questions based on the provided corpus.

Instructions:
1. {prompt_instruction}
2. Generate exactly {num_questions} questions
3. Each question should be clear, specific, and directly based on the corpus content
4. For mathematical questions, provide the FINAL NUMERICAL ANSWER only (e.g., "574.56 cubic cm" or "22.9%")
5. For factual questions, provide concise direct answers (e.g., "54 years" or "Etruscan bronze artifact")
6. Questions should vary in difficulty from basic to advanced
7. Focus on the most important concepts from the corpus

Format your response as a JSON array with this structure:
[
  {{
    "question": "Your question here",
    "answer": "Final answer only - no calculations or explanations",
    "concept": "Main concept being tested",
    "difficulty": "basic|intermediate|advanced"
  }}
]

IMPORTANT: Keep answers simple and focused. For math problems, only give the final number/result, not the calculation steps.

Only return the JSON array, no other text."""

    user_prompt = f"""Based on this corpus, generate {num_questions} evaluation questions:

---CORPUS START---
{corpus_text[:120000]}  
---CORPUS END---

Generate the questions as specified in the instructions."""

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
            
            # Parse JSON response
            import json
            try:
                questions_data = json.loads(content.strip())
                
                for i, q_data in enumerate(questions_data[:num_questions]):
                    questions.append({
                        "question": q_data.get("question", f"Question {i+1}"),
                        "answer": q_data.get("answer", "Answer not provided"),
                        "context": q_data.get("concept", ""),
                        "eval_type": eval_type,
                        "concept": q_data.get("concept", f"concept_{i+1}"),
                        "difficulty": q_data.get("difficulty", "intermediate"),
                        "source": "agentic_llm"
                    })
                    
                    await tracker.increment_progress(message=f"Generated agentic question {i+1}: {q_data.get('question', '')[:50]}...")
                    
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
                    
                    await tracker.increment_progress(message=f"Extracted question {i+1}")
    
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
    """Evaluate questions using real LLM"""
    import httpx
    
    llm_results = []
    
    for i, question in enumerate(questions):
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
                    
                    await tracker.increment_progress(message=f"Evaluated question {i+1} with {llm_config.model_name}")
                else:
                    # Fallback for failed requests
                    llm_results.append({
                        "question": question["question"],
                        "ground_truth": question["answer"],
                        "prediction": f"Error: API request failed ({response.status_code})",
                        "confidence": 0.0,
                        "source": "api_error"
                    })
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
            await tracker.send_log("warning", f"Failed to evaluate question {i+1}: {str(e)}")
        
        # Small delay to avoid rate limits
        await asyncio.sleep(0.2)
    
    return llm_results


async def generate_corpus_questions(corpus_text: str, num_questions: int, eval_type: str, tracker) -> List[Dict]:
<<<<<<< HEAD
    """Generate questions using improved template approach with content-type detection"""
=======
    """Generate questions based on corpus content"""
    import re
>>>>>>> parent of 6898d60 (increase concurrency and benchmark complexity and math)
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
            
            await tracker.increment_progress(message=f"Generated question {i+1}: {question[:50]}...")
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
    
    return questions


async def run_evaluation(run_id: str, request: EvaluationRequest, config: EvaluationConfig):
    """Run evaluation in background"""
    tracker = get_progress_tracker(run_id)
    
    try:
        # Update status
        evaluation_runs[run_id]["status"] = "running"
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
        
        # Send completion notification
        await tracker.notifier.send_evaluation_complete(final_results)
        
        logger.info(f"Evaluation completed", run_id=run_id, mean_score=mean_score)
        
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
        
        await tracker.send_error(error_msg)


@router.get("/evaluation/{run_id}/status")
async def get_evaluation_status(run_id: str):
    """Get evaluation status"""
    if run_id not in evaluation_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run_info = evaluation_runs[run_id]
    
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
    if run_id not in evaluation_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run_info = evaluation_runs[run_id]
    
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