"""
LoRA Fine-tuning Integration for docs-to-eval system
Integrates the MLX-based LoRA fine-tuning with the evaluation pipeline
"""

import json
import tempfile
import asyncio
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field

from ..evaluation import FinetuneTestSet, BenchmarkItem


@dataclass
class LoRAFinetuningConfig:
    """Configuration for LoRA fine-tuning"""
    model_path: str = "mlx_model"  # Path to base model
    lora_layers: int = 16  # Number of layers to fine-tune
    batch_size: int = 4
    learning_rate: float = 1e-5
    max_iters: int = 1000
    steps_per_report: int = 10
    steps_per_eval: int = 200
    save_every: int = 100
    temperature: float = 0.8
    max_tokens: int = 100
    adapter_file: str = "adapters.npz"
    seed: int = 42


class LoRAFinetuningRequest(BaseModel):
    """Request model for LoRA fine-tuning"""
    run_id: str
    config: Dict[str, Any] = Field(default_factory=dict)
    model_path: Optional[str] = "mlx_model"
    custom_training_data: Optional[List[Dict[str, str]]] = None


class LoRAFinetuningResult(BaseModel):
    """Result of LoRA fine-tuning process"""
    run_id: str
    status: str  # "running", "completed", "failed"
    adapter_path: Optional[str] = None
    training_metrics: Dict[str, Any] = Field(default_factory=dict)
    final_loss: Optional[float] = None
    training_time: Optional[float] = None
    error_message: Optional[str] = None


class ModelComparisonResult(BaseModel):
    """Result of comparing original vs finetuned model"""
    run_id: str
    original_accuracy: float
    finetuned_accuracy: float
    improvement: float  # Percentage improvement
    test_questions_count: int
    detailed_results: List[Dict[str, Any]] = Field(default_factory=list)


class LoRAFinetuningOrchestrator:
    """
    Main orchestrator for LoRA fine-tuning integration
    
    This class handles:
    1. Data preparation from finetune test sets
    2. LoRA fine-tuning execution
    3. Model comparison and evaluation
    4. Result management
    """
    
    def __init__(self, base_work_dir: Optional[Path] = None):
        self.base_work_dir = base_work_dir or Path(tempfile.gettempdir()) / "lora_finetuning"
        self.base_work_dir.mkdir(exist_ok=True)
        self.lora_script_path = Path(__file__).parent / "lora.py"
        
    def create_working_directory(self, run_id: str) -> Path:
        """Create a working directory for this fine-tuning run"""
        # Sanitize run_id to prevent path traversal attacks
        safe_run_id = self._sanitize_path_component(run_id)
        work_dir = self.base_work_dir / f"run_{safe_run_id}"
        work_dir.mkdir(exist_ok=True)
        (work_dir / "data").mkdir(exist_ok=True)
        return work_dir
    
    def _sanitize_path_component(self, component: str) -> str:
        """Sanitize a path component to prevent directory traversal"""
        # Only allow alphanumeric characters, hyphens, and underscores
        import re
        safe_component = re.sub(r'[^a-zA-Z0-9\-_]', '_', component)
        # Limit length
        safe_component = safe_component[:50]
        # Ensure it's not empty
        if not safe_component:
            safe_component = "unknown"
        return safe_component
    
    def prepare_training_data(self, 
                            finetune_test_set: FinetuneTestSet, 
                            work_dir: Path) -> Tuple[Path, Path, Path]:
        """
        Prepare training data in the format expected by LoRA script
        
        Args:
            finetune_test_set: The test set containing train/test split
            work_dir: Working directory for this run
            
        Returns:
            Tuple of (train_file, valid_file, test_file) paths
        """
        data_dir = work_dir / "data"
        # Ensure data directory exists
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare training data (80% of train set for training, 20% for validation)
        train_questions = finetune_test_set.train_questions
        
        if len(train_questions) == 0:
            raise ValueError("No training questions available in finetune_test_set")
            
        # Ensure we have at least some data for validation
        if len(train_questions) < 2:
            # If only 1 question, use it for both train and validation
            train_data = train_questions
            valid_data = train_questions
        else:
            train_size = max(1, int(len(train_questions) * 0.8))
            train_data = train_questions[:train_size]
            valid_data = train_questions[train_size:] if train_size < len(train_questions) else train_questions[-1:]
        
        test_data = finetune_test_set.test_questions
        
        # Convert to LoRA training format
        def format_question_for_training(item: Union[BenchmarkItem, Dict[str, Any]]) -> Dict[str, str]:
            """Format a benchmark item for LoRA training"""
            # Handle both BenchmarkItem objects and dictionaries
            if isinstance(item, dict):
                question = item.get("question", "")
                answer = item.get("answer", "")
            else:
                question = getattr(item, 'question', "")
                answer = getattr(item, 'answer', "")
            
            # Clean and validate data
            question = str(question).strip()
            answer = str(answer).strip()
            
            if not question or not answer:
                raise ValueError(f"Empty question or answer found: question='{question}', answer='{answer}'")
            
            # Create instruction-following format
            prompt = f"Question: {question}\nAnswer:"
            completion = f" {answer}"
            return {"text": prompt + completion}
        
        # Write training files
        train_file = data_dir / "train.jsonl"
        valid_file = data_dir / "valid.jsonl"
        test_file = data_dir / "test.jsonl"
        
        # Write training data
        try:
            with open(train_file, 'w', encoding='utf-8') as f:
                for i, item in enumerate(train_data):
                    try:
                        formatted_item = format_question_for_training(item)
                        f.write(json.dumps(formatted_item, ensure_ascii=False) + "\n")
                    except Exception as e:
                        raise ValueError(f"Error formatting training item {i}: {str(e)}")
            
            with open(valid_file, 'w', encoding='utf-8') as f:
                for i, item in enumerate(valid_data):
                    try:
                        formatted_item = format_question_for_training(item)
                        f.write(json.dumps(formatted_item, ensure_ascii=False) + "\n")
                    except Exception as e:
                        raise ValueError(f"Error formatting validation item {i}: {str(e)}")
            
            with open(test_file, 'w', encoding='utf-8') as f:
                for i, item in enumerate(test_data):
                    try:
                        formatted_item = format_question_for_training(item)
                        f.write(json.dumps(formatted_item, ensure_ascii=False) + "\n")
                    except Exception as e:
                        raise ValueError(f"Error formatting test item {i}: {str(e)}")
            
            print("✅ Prepared training data:")
            print(f"   📝 Training samples: {len(train_data)} → {train_file}")
            print(f"   🔍 Validation samples: {len(valid_data)} → {valid_file}")
            print(f"   🧪 Test samples: {len(test_data)} → {test_file}")
            
        except Exception as e:
            raise ValueError(f"Failed to write training data files: {str(e)}")
        
        return train_file, valid_file, test_file
    
    async def run_lora_finetuning(self, 
                                finetune_test_set: FinetuneTestSet,
                                config: LoRAFinetuningConfig,
                                run_id: str,
                                progress_callback: Optional[callable] = None) -> LoRAFinetuningResult:
        """
        Run LoRA fine-tuning process
        
        Args:
            finetune_test_set: Test set with train/test split
            config: LoRA fine-tuning configuration
            run_id: Unique run identifier
            progress_callback: Optional callback for progress updates
            
        Returns:
            LoRAFinetuningResult with training results
        """
        try:
            # Create working directory
            work_dir = self.create_working_directory(run_id)
            
            if progress_callback:
                await progress_callback("info", "Preparing training data...")
            
            # Prepare training data
            train_file, valid_file, test_file = self.prepare_training_data(
                finetune_test_set, work_dir
            )
            
            if progress_callback:
                await progress_callback("info", f"Training data prepared: {len(finetune_test_set.train_questions)} training examples")
                await progress_callback("info", f"Test questions available: {len(finetune_test_set.test_questions)} test examples")
            
            # Prepare LoRA training command
            adapter_file = work_dir / config.adapter_file
            
            cmd = [
                "python", str(self.lora_script_path),
                "--train",
                "--model", config.model_path,
                "--data", str(work_dir / "data"),
                "--lora-layers", str(config.lora_layers),
                "--batch-size", str(config.batch_size),
                "--learning-rate", str(config.learning_rate),
                "--iters", str(config.max_iters),
                "--steps-per-report", str(config.steps_per_report),
                "--steps-per-eval", str(config.steps_per_eval),
                "--save-every", str(config.save_every),
                "--adapter-file", str(adapter_file),
                "--seed", str(config.seed)
            ]
            
            if progress_callback:
                await progress_callback("info", f"Starting LoRA fine-tuning with {config.max_iters} iterations...")
            
            # Run LoRA training
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = f"LoRA training failed: {stderr.decode()}"
                if progress_callback:
                    await progress_callback("error", error_msg)
                return LoRAFinetuningResult(
                    run_id=run_id,
                    status="failed",
                    error_message=error_msg
                )
            
            # Parse training metrics from stdout
            training_output = stdout.decode()
            final_loss = self._extract_final_loss(training_output)
            
            if progress_callback:
                await progress_callback("info", f"LoRA fine-tuning completed! Final loss: {final_loss:.4f}")
            
            return LoRAFinetuningResult(
                run_id=run_id,
                status="completed",
                adapter_path=str(adapter_file),
                final_loss=final_loss,
                training_metrics={"stdout": training_output}
            )
            
        except Exception as e:
            error_msg = f"LoRA fine-tuning error: {str(e)}"
            if progress_callback:
                await progress_callback("error", error_msg)
            return LoRAFinetuningResult(
                run_id=run_id,
                status="failed",
                error_message=error_msg
            )
    
    def _extract_final_loss(self, training_output: str) -> Optional[float]:
        """Extract final validation loss from training output"""
        lines = training_output.strip().split('\n')
        for line in reversed(lines):
            if "Val loss" in line:
                try:
                    # Extract loss value from line like "Iter 1000: Val loss 2.345, Val took 1.23s"
                    parts = line.split("Val loss")[1].split(",")[0].strip()
                    return float(parts)
                except (ValueError, IndexError, AttributeError):
                    # Failed to parse loss value from this line, try next
                    continue
        return None
    
    async def evaluate_model_on_questions(self,
                                        questions: List[Union[BenchmarkItem, Dict[str, Any]]],
                                        model_path: str,
                                        adapter_path: Optional[str] = None,
                                        config: Optional[LoRAFinetuningConfig] = None) -> List[Dict[str, Any]]:
        """
        Evaluate a model (with optional LoRA adapter) on test questions
        
        Args:
            questions: List of questions to evaluate
            model_path: Path to base model
            adapter_path: Optional path to LoRA adapter
            config: LoRA configuration
            
        Returns:
            List of evaluation results
        """
        if config is None:
            config = LoRAFinetuningConfig()
        
        # Validate model_path to prevent directory traversal
        model_path = Path(model_path).resolve()
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Validate adapter_path if provided
        if adapter_path:
            adapter_path = Path(adapter_path).resolve()
            if not adapter_path.exists():
                raise ValueError(f"Adapter path does not exist: {adapter_path}")
        
        results = []
        
        for i, question_item in enumerate(questions):
            try:
                # Handle both BenchmarkItem objects and dictionaries
                if isinstance(question_item, dict):
                    question_id = question_item.get("id", f"q_{i}")
                    question_text = question_item.get("question", "")
                    expected_answer = question_item.get("answer", "")
                else:
                    question_id = question_item.metadata.get("id", f"q_{i}") if hasattr(question_item, 'metadata') else f"q_{i}"
                    question_text = question_item.question if hasattr(question_item, 'question') else ""
                    expected_answer = question_item.answer if hasattr(question_item, 'answer') else ""
                
                # Sanitize question text to prevent command injection
                # Remove any control characters, newlines, and limit length
                question_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', question_text)
                question_text = question_text.replace('\n', ' ').replace('\r', ' ')
                question_text = ' '.join(question_text.split())[:1000]  # Normalize whitespace
                
                # Create safe prompt string
                prompt_text = f"Question: {question_text}\nAnswer:"
                
                # Prepare generation command - subprocess.exec does NOT use shell, so it's safe
                cmd = [
                    "python", str(self.lora_script_path),
                    "--model", str(model_path),
                    "--prompt", prompt_text,  # Safe when using subprocess.exec without shell
                    "--max-tokens", str(int(config.max_tokens)),
                    "--temp", str(float(config.temperature))
                ]
                
                if adapter_path:
                    cmd.extend(["--adapter-file", str(adapter_path)])
                
                # Run generation with timeout
                # subprocess.create_subprocess_exec does NOT use shell, preventing injection
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    # Run process with timeout to prevent hanging
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=60.0  # 60 second timeout
                    )
                    
                    if process.returncode == 0:
                        generated_text = stdout.decode().strip()
                        # Extract answer from generated text
                        prediction = self._extract_answer_from_generation(generated_text)
                        
                        # Simple accuracy check
                        is_correct = self._check_answer_correctness(
                            prediction, expected_answer
                        )
                        
                        results.append({
                            "question_id": question_id,
                            "question": question_text,
                            "expected_answer": expected_answer,
                            "predicted_answer": prediction,
                            "is_correct": is_correct,
                            "raw_generation": generated_text
                        })
                    else:
                        error_msg = stderr.decode() if stderr else "Unknown error"
                        results.append({
                            "question_id": question_id,
                            "question": question_text,
                            "expected_answer": expected_answer,
                            "predicted_answer": "ERROR",
                            "is_correct": False,
                            "error": error_msg[:500]  # Limit error message length
                        })
                        
                except asyncio.TimeoutError:
                    # Handle timeout
                    process.kill()
                    await process.wait()
                    results.append({
                        "question_id": question_id,
                        "question": question_text,
                        "expected_answer": expected_answer,
                        "predicted_answer": "TIMEOUT",
                        "is_correct": False,
                        "error": "Generation timeout after 60 seconds"
                    })
                        
            except Exception as e:
                # Handle both BenchmarkItem objects and dictionaries for error case
                if isinstance(question_item, dict):
                    error_question_id = question_item.get("id", f"q_{i}")
                    error_question_text = question_item["question"]
                    error_expected_answer = question_item["answer"]
                else:
                    error_question_id = question_item.metadata.get("id", f"q_{i}")
                    error_question_text = question_item.question
                    error_expected_answer = question_item.answer
                
                results.append({
                    "question_id": error_question_id,
                    "question": error_question_text,
                    "expected_answer": error_expected_answer,
                    "predicted_answer": "ERROR",
                    "is_correct": False,
                    "error": str(e)
                })
        
        return results
    
    def _extract_answer_from_generation(self, generated_text: str) -> str:
        """Extract the answer portion from generated text"""
        # Look for the answer after "Answer:" prompt
        if "Answer:" in generated_text:
            answer_part = generated_text.split("Answer:")[-1].strip()
            # Remove the separator line if present
            if "=" in answer_part:
                answer_part = answer_part.split("=")[0].strip()
            return answer_part
        return generated_text.strip()
    
    def _check_answer_correctness(self, prediction: str, ground_truth: str) -> bool:
        """Simple answer correctness check"""
        pred_clean = prediction.lower().strip()
        truth_clean = ground_truth.lower().strip()
        
        # Exact match
        if pred_clean == truth_clean:
            return True
        
        # Check if ground truth is contained in prediction
        if truth_clean in pred_clean:
            return True
        
        # For numerical answers, try to extract numbers
        import re
        pred_numbers = re.findall(r'-?\d+\.?\d*', pred_clean)
        truth_numbers = re.findall(r'-?\d+\.?\d*', truth_clean)
        
        if pred_numbers and truth_numbers:
            try:
                return float(pred_numbers[0]) == float(truth_numbers[0])
            except (ValueError, IndexError):
                # Failed to convert to float, skip numerical comparison
                pass
        
        return False
    
    async def compare_models(self,
                           finetune_test_set: FinetuneTestSet,
                           model_path: str,
                           adapter_path: str,
                           run_id: str,
                           config: Optional[LoRAFinetuningConfig] = None,
                           progress_callback: Optional[callable] = None) -> ModelComparisonResult:
        """
        Compare original model vs finetuned model on test questions
        
        Args:
            finetune_test_set: Test set with questions
            model_path: Path to base model
            adapter_path: Path to LoRA adapter
            run_id: Unique run identifier
            config: LoRA configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            ModelComparisonResult with comparison metrics
        """
        if config is None:
            config = LoRAFinetuningConfig()
        
        test_questions = finetune_test_set.test_questions
        
        if progress_callback:
            await progress_callback("info", f"Evaluating original model on {len(test_questions)} test questions...")
        
        # Evaluate original model
        original_results = await self.evaluate_model_on_questions(
            test_questions, model_path, adapter_path=None, config=config
        )
        
        if progress_callback:
            await progress_callback("info", f"Evaluating finetuned model on {len(test_questions)} test questions...")
        
        # Evaluate finetuned model
        finetuned_results = await self.evaluate_model_on_questions(
            test_questions, model_path, adapter_path=adapter_path, config=config
        )
        
        # Calculate accuracies
        original_correct = sum(1 for r in original_results if r["is_correct"])
        finetuned_correct = sum(1 for r in finetuned_results if r["is_correct"])
        
        original_accuracy = original_correct / len(test_questions)
        finetuned_accuracy = finetuned_correct / len(test_questions)
        
        improvement = ((finetuned_accuracy - original_accuracy) / original_accuracy * 100) if original_accuracy > 0 else 0
        
        # Combine results for detailed comparison  
        detailed_results = []
        for orig, fine in zip(original_results, finetuned_results):
            detailed_results.append({
                "question_id": orig["question_id"],
                "question": orig["question"],
                "expected_answer": orig["expected_answer"],
                "original_prediction": orig["predicted_answer"],
                "finetuned_prediction": fine["predicted_answer"],
                "original_correct": orig["is_correct"],
                "finetuned_correct": fine["is_correct"],
                "improvement": fine["is_correct"] and not orig["is_correct"]
            })
        
        if progress_callback:
            await progress_callback("info", f"Model comparison completed! Original: {original_accuracy:.1%}, Finetuned: {finetuned_accuracy:.1%}, Improvement: {improvement:+.1f}%")
        
        return ModelComparisonResult(
            run_id=run_id,
            original_accuracy=original_accuracy,
            finetuned_accuracy=finetuned_accuracy,
            improvement=improvement,
            test_questions_count=len(test_questions),
            detailed_results=detailed_results
        )


# Global orchestrator instance
lora_orchestrator = LoRAFinetuningOrchestrator()