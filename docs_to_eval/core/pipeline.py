"""
Core Pipeline class for unified orchestration of the docs-to-eval system.
This centralizes the end-to-end evaluation logic and serves as the single
entry point for all evaluation workflows.
"""

import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..utils.config import EvaluationConfig
from ..utils.logging import setup_logging, evaluation_context
from .classification import EvaluationTypeClassifier
from .evaluation import EvaluationFramework
from .benchmarks import BenchmarkGeneratorFactory
from .verification import VerificationOrchestrator
from ..llm.mock_interface import MockLLMInterface, MockLLMEvaluator


class EvaluationPipeline:
    """
    Unified pipeline orchestrator for the docs-to-eval system.
    
    This class encapsulates the complete evaluation workflow:
    1. Corpus classification
    2. Benchmark generation (standard or agentic)
    3. LLM evaluation
    4. Response verification
    5. Results aggregation
    
    The pipeline is designed to be:
    - Configuration-driven (uses EvaluationConfig)
    - Async-native for scalability
    - UI-agnostic (can be called from CLI, API, or tests)
    - Strategy-pattern enabled (swaps generators based on config)
    """
    
    def __init__(self, config: EvaluationConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.run_id = str(uuid.uuid4())[:8]
        
        # Initialize core components
        self.classifier = EvaluationTypeClassifier()
        self.framework = EvaluationFramework()
        self.generator_factory = BenchmarkGeneratorFactory()
        self.verifier = VerificationOrchestrator()
        
        # Initialize LLM interface (will be configurable in future)
        self.llm = MockLLMInterface(temperature=config.llm.temperature)
        self.evaluator = MockLLMEvaluator(self.llm)
        
        # Setup logging
        setup_logging(config.system.log_level)
        
    async def run_async(self, corpus_text: str, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline asynchronously.
        
        Args:
            corpus_text: The input text corpus to evaluate
            output_path: Optional path to save results (defaults to config.system.output_dir)
            
        Returns:
            Dictionary containing complete evaluation results
        """
        
        if output_path is None:
            output_path = Path(self.config.system.output_dir)
        output_path.mkdir(exist_ok=True)
        
        with evaluation_context(self.run_id) as eval_logger:
            eval_logger.logger.info(f"Starting evaluation pipeline (Run ID: {self.run_id})")
            
            # Phase 1: Corpus Classification
            eval_logger.start_phase("classification")
            classification = await self._classify_corpus(corpus_text, eval_logger)
            eval_logger.end_phase() 
            
            # Phase 2: Benchmark Generation
            eval_logger.start_phase("generation")
            questions = await self._generate_benchmark(corpus_text, classification, eval_logger)
            eval_logger.end_phase()
            
            # Phase 3: LLM Evaluation
            eval_logger.start_phase("evaluation")
            llm_results = await self._evaluate_with_llm(questions, eval_logger)
            eval_logger.end_phase()
            
            # Phase 4: Response Verification
            eval_logger.start_phase("verification")
            verification_results = await self._verify_responses(llm_results, eval_logger)
            eval_logger.end_phase()
            
            # Phase 5: Results Aggregation
            results = self._aggregate_results(classification, verification_results, eval_logger)
            
            # Save results
            await self._save_results(results, output_path)
            
            eval_logger.logger.info(f"Evaluation pipeline completed successfully")
            return results
    
    async def _classify_corpus(self, corpus_text: str, logger) -> Any:
        """Classify the corpus to determine evaluation strategy."""
        logger.logger.info("Analyzing corpus for evaluation type...")
        
        # Use existing classifier or override from config
        if self.config.eval_type:
            # Use configured eval type with basic analysis
            classification = self.classifier.classify_corpus(corpus_text)
            classification.primary_type = self.config.eval_type
        else:
            # Auto-classify
            classification = self.classifier.classify_with_examples(corpus_text, num_examples=3)
        
        logger.log_classification(classification.to_dict())
        return classification
    
    async def _generate_benchmark(self, corpus_text: str, classification: Any, logger) -> List[Dict[str, Any]]:
        """Generate benchmark questions using the appropriate strategy."""
        logger.info(f"Generating {self.config.generation.num_questions} questions...")
        
        # Create generator based on configuration strategy
        generator = self.generator_factory.create_generator(
            eval_type=classification.primary_type,
            use_agentic=self.config.generation.use_agentic,
            config=self.config
        )
        
        # Generate questions
        questions = []
        if hasattr(generator, 'generate_async'):
            # Use async generation if available (agentic)
            benchmark_items = await generator.generate_async(
                corpus_text=corpus_text,
                num_questions=self.config.generation.num_questions
            )
            questions = [item.to_dict() for item in benchmark_items]
        else:
            # Fall back to sync generation
            questions = generator.generate_questions(
                corpus_text=corpus_text,
                num_questions=self.config.generation.num_questions
            )
        
        logger.log_benchmark_generation(len(questions), self.config.dict())
        return questions
    
    async def _evaluate_with_llm(self, questions: List[Dict[str, Any]], logger) -> List[Dict[str, Any]]:
        """Evaluate questions using the configured LLM."""
        logger.info(f"Evaluating {len(questions)} questions with LLM...")
        
        results = []
        for i, question in enumerate(questions):
            # Mock evaluation for now - will be replaced with real LLM calls
            result = {
                "question": question.get("question", question.get("text", "")),
                "ground_truth": question.get("answer", question.get("correct_answer", "")),
                "prediction": f"Mock LLM response {i+1}",
                "confidence": 0.8,
                "eval_type": question.get("eval_type", "unknown")
            }
            results.append(result) 
            
            # Small delay for async behavior simulation
            await asyncio.sleep(0.001)
        
        return results
    
    async def _verify_responses(self, llm_results: List[Dict[str, Any]], logger) -> List[Dict[str, Any]]:
        """Verify LLM responses using the configured verification method."""
        logger.info(f"Verifying {len(llm_results)} responses...")
        
        verification_results = []
        for result in llm_results:
            verification = {
                "question": result["question"],
                "prediction": result["prediction"], 
                "ground_truth": result["ground_truth"],
                "score": 0.7,  # Mock score - will use real verification
                "method": self.config.verification.method.value,
                "eval_type": result.get("eval_type", "unknown")
            }
            verification_results.append(verification)
            await asyncio.sleep(0.001)
        
        mean_score = sum(r["score"] for r in verification_results) / len(verification_results)
        logger.log_verification_results({"mean_score": mean_score})
        
        return verification_results
    
    def _aggregate_results(self, classification: Any, verification_results: List[Dict[str, Any]], logger) -> Dict[str, Any]:
        """Aggregate all results into final output format."""
        logger.info("Aggregating final results...")
        
        scores = [r["score"] for r in verification_results]
        
        return {
            "run_id": self.run_id,
            "config": self.config.dict(),
            "classification": classification.to_dict() if hasattr(classification, 'to_dict') else str(classification),
            "aggregate_metrics": {
                "mean_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "num_samples": len(verification_results),
                "std_dev": self._calculate_std_dev(scores)
            },
            "individual_results": verification_results[:self.config.reporting.max_individual_results],
            "performance_stats": self.llm.get_performance_stats() if hasattr(self.llm, 'get_performance_stats') else {},
            "completed_at": datetime.now().isoformat()
        }
    
    def _calculate_std_dev(self, scores: List[float]) -> float:
        """Calculate standard deviation of scores."""
        if not scores:
            return 0.0
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return variance ** 0.5
    
    async def _save_results(self, results: Dict[str, Any], output_path: Path):
        """Save results to output directory."""
        import json
        
        results_file = output_path / f"evaluation_results_{self.run_id}.json"
        
        # Ensure JSON serializable
        json_results = json.loads(json.dumps(results, default=str))
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)


class PipelineFactory:
    """Factory for creating configured pipeline instances."""
    
    @staticmethod
    def create_pipeline(config: EvaluationConfig) -> EvaluationPipeline:
        """Create a pipeline instance with the given configuration."""
        return EvaluationPipeline(config)
    
    @staticmethod
    def create_default_pipeline() -> EvaluationPipeline:
        """Create a pipeline with default configuration."""
        from ..utils.config import create_default_config
        return EvaluationPipeline(create_default_config())