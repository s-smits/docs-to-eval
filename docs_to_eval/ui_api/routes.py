"""
FastAPI routes for docs-to-eval system
"""

import asyncio
import uuid
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, Form, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .websockets import websocket_manager, handle_websocket_connection, get_progress_tracker
from ..core.evaluation import EvaluationFramework, EvaluationType, BenchmarkConfig
from ..core.classification import EvaluationTypeClassifier
from ..llm.mock_interface import MockLLMInterface, MockLLMEvaluator
from ..utils.config import EvaluationConfig, create_default_config
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
    eval_type: Optional[EvaluationType] = None
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
            "stats": corpus_info["stats"]
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


@router.post("/evaluation/start")
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Start a new evaluation run"""
    try:
        # Generate run ID
        run_id = str(uuid.uuid4())
        
        # Create evaluation configuration
        config = create_default_config()
        config.eval_type = request.eval_type or EvaluationType.DOMAIN_KNOWLEDGE
        config.generation.num_questions = request.num_questions
        config.generation.use_agentic = request.use_agentic
        config.llm.temperature = request.temperature
        
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
        
    except Exception as e:
        logger.error(f"Error starting evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        
        # Create framework and generate benchmark
        framework = EvaluationFramework()
        benchmark_config = framework.create_benchmark_from_corpus(request.corpus_text, request.num_questions)
        
        # Generate questions (simplified for demo)
        questions = []
        for i in range(request.num_questions):
            questions.append({
                "question": f"Sample question {i+1} about the corpus content",
                "answer": f"Sample answer {i+1}",
                "context": None,
                "eval_type": classification.primary_type
            })
            
            await tracker.increment_progress(message=f"Generated question {i+1}")
            await asyncio.sleep(0.1)  # Simulate work
        
        await tracker.end_phase({"questions_generated": len(questions)})
        
        # Phase 3: LLM Evaluation
        await tracker.start_phase("evaluation", "Evaluating with LLM", len(questions))
        
        llm = MockLLMInterface(temperature=request.temperature)
        evaluator = MockLLMEvaluator(llm)
        
        llm_results = []
        for i, question in enumerate(questions):
            result = {
                "question": question["question"],
                "ground_truth": question["answer"],
                "prediction": f"Mock LLM response to question {i+1}",
                "confidence": 0.8
            }
            llm_results.append(result)
            
            await tracker.increment_progress(message=f"Evaluated question {i+1}")
            await asyncio.sleep(0.1)  # Simulate work
        
        await tracker.end_phase({"evaluations_completed": len(llm_results)})
        
        # Phase 4: Verification
        await tracker.start_phase("verification", "Verifying responses", len(llm_results))
        
        verification_results = []
        for i, result in enumerate(llm_results):
            verification = {
                "question": result["question"],
                "prediction": result["prediction"],
                "ground_truth": result["ground_truth"],
                "score": 0.7,  # Mock score
                "method": "exact_match"
            }
            verification_results.append(verification)
            
            await tracker.increment_progress(message=f"Verified response {i+1}")
            await asyncio.sleep(0.1)  # Simulate work
        
        await tracker.end_phase({"verifications_completed": len(verification_results)})
        
        # Phase 5: Report Generation
        await tracker.start_phase("reporting", "Generating comprehensive report")
        
        # Calculate aggregate metrics
        total_score = sum(r["score"] for r in verification_results)
        mean_score = total_score / len(verification_results) if verification_results else 0
        
        final_results = {
            "run_id": run_id,
            "evaluation_config": config.dict(),
            "classification": classification.to_dict(),
            "aggregate_metrics": {
                "mean_score": mean_score,
                "min_score": min(r["score"] for r in verification_results) if verification_results else 0,
                "max_score": max(r["score"] for r in verification_results) if verification_results else 0,
                "num_samples": len(verification_results)
            },
            "individual_results": verification_results[:10],  # First 10 only
            "performance_stats": llm.get_performance_stats(),
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
        runs.append({
            "run_id": run_id,
            "status": run_info["status"],
            "start_time": run_info["start_time"],
            "eval_type": run_info["config"]["eval_type"],
            "num_questions": run_info["config"]["generation"]["num_questions"]
        })
    
    return {"runs": sorted(runs, key=lambda x: x["start_time"], reverse=True)}


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