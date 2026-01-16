"""
Agentic Benchmark Generator
Integrates the agentic orchestrator with the existing benchmark generation framework
"""

import asyncio
import time
from typing import List, Dict, Any, Optional

from .orchestrator import AgenticBenchmarkOrchestrator
from .models import PipelineConfig, DifficultyLevel
from ..benchmarks import BenchmarkGenerator
from ..evaluation import EvaluationType
from ...llm.base import BaseLLMInterface


class AgenticBenchmarkGenerator(BenchmarkGenerator):
    """
    Agentic benchmark generator that integrates with existing framework
    Uses specialized agents orchestrated through intelligent pipeline
    """
    
    def __init__(self, eval_type: EvaluationType, llm_pool: Optional[Dict[str, BaseLLMInterface]] = None, 
                 config: Optional[PipelineConfig] = None):
        """
        Initialize agentic generator
        
        Args:
            eval_type: Type of evaluation to generate
            llm_pool: Dictionary of LLM interfaces for different agent roles
            config: Pipeline configuration
        """
        super().__init__(eval_type, llm=None)  # Don't use the single LLM from parent
        
        # Set up LLM pool with fallbacks
        self.llm_pool = llm_pool or self._create_default_llm_pool()
        
        # Configure pipeline based on eval type
        self.config = config or self._create_default_config(eval_type)

        # Initialize orchestrator
        self.orchestrator = AgenticBenchmarkOrchestrator(self.llm_pool, self.config)
        
        # Track generation metadata
        self.generation_metadata = {
            'source': 'agentic_v2',
            'eval_type': eval_type,
            'agent_versions': {},
            'pipeline_config': self.config.model_dump()
        }
    
    def generate_benchmark(self, corpus_text: str, num_questions: int = 50) -> List[Dict[str, Any]]:
        """
        Generate benchmark using agentic pipeline (sync wrapper for async method)
        
        Args:
            corpus_text: Source text for benchmark generation
            num_questions: Number of questions to generate
            
        Returns:
            List of benchmark items compatible with existing framework
        """
        # Run async generation in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            enhanced_items = loop.run_until_complete(
                self.generate_benchmark_async(corpus_text, num_questions)
            )
            
            # Convert to standard format for compatibility
            standard_items = []
            for item in enhanced_items:
                standard_item = item.to_standard_benchmark_item()
                standard_items.append(standard_item)
            
            return standard_items
            
        except Exception:
            # Fallback to basic generation if agentic pipeline fails
            return self._fallback_to_basic_generation(corpus_text, num_questions)
    
    async def generate_benchmark_async(self, corpus_text: str, num_questions: int = 50) -> List[Any]:
        """
        Async version of benchmark generation using agentic pipeline
        
        Args:
            corpus_text: Source text for benchmark generation
            num_questions: Number of questions to generate
            
        Returns:
            List of EnhancedBenchmarkItem objects
        """
        start_time = time.time()
        
        # Validate pipeline health before generation
        health_status = await self.orchestrator.validate_pipeline_health()
        if not health_status['healthy']:
            raise RuntimeError(f"Pipeline unhealthy: {health_status['issues']}")
        
        # Determine difficulty based on eval type and configuration
        difficulty = self._determine_target_difficulty()
        
        # Run agentic generation
        enhanced_items = await self.orchestrator.generate(
            corpus_text=corpus_text,
            eval_type=self.eval_type,
            num_questions=num_questions,
            difficulty=difficulty
        )
        
        # Update generation metadata
        generation_time = time.time() - start_time
        self.generation_metadata.update({
            'generation_time': generation_time,
            'items_generated': len(enhanced_items),
            'pipeline_metrics': self.orchestrator.get_pipeline_metrics(),
            'timestamp': time.time()
        })
        
        return enhanced_items
    
    def _create_default_llm_pool(self) -> Dict[str, BaseLLMInterface]:
        """Create default LLM pool using mock interfaces"""
        from ...llm.mock_interface import MockLLMInterface
        
        # Use mock interfaces as fallback
        mock_llm = MockLLMInterface()
        
        return {
            'retriever': mock_llm,
            'creator': mock_llm,
            'adversary': mock_llm,
            'refiner': mock_llm
        }
    
    def _create_default_config(self, eval_type: EvaluationType) -> PipelineConfig:
        """Create appropriate configuration based on evaluation type"""
        
        # Adjust configuration based on eval type characteristics
        difficulty_map = {
            EvaluationType.MATHEMATICAL: DifficultyLevel.HARD,
            EvaluationType.CODE_GENERATION: DifficultyLevel.HARD,
            EvaluationType.FACTUAL_QA: DifficultyLevel.INTERMEDIATE,
            EvaluationType.DOMAIN_KNOWLEDGE: DifficultyLevel.HARD,
            EvaluationType.MULTIPLE_CHOICE: DifficultyLevel.INTERMEDIATE,
            EvaluationType.READING_COMPREHENSION: DifficultyLevel.INTERMEDIATE
        }
        
        target_difficulty = difficulty_map.get(eval_type, DifficultyLevel.HARD)
        
        # Validation thresholds by eval type
        validation_thresholds = {
            EvaluationType.MATHEMATICAL: 0.8,  # High precision required
            EvaluationType.CODE_GENERATION: 0.7,
            EvaluationType.FACTUAL_QA: 0.7,
            EvaluationType.DOMAIN_KNOWLEDGE: 0.6,
            EvaluationType.MULTIPLE_CHOICE: 0.7,
            EvaluationType.READING_COMPREHENSION: 0.5  # More subjective
        }
        
        min_score = validation_thresholds.get(eval_type, 0.6)
        
        return PipelineConfig(
            difficulty=target_difficulty,
            min_validation_score=min_score,
            oversample_factor=2.5,  # Generate extra candidates for selection
            parallel_batch_size=3,  # Conservative for stability
            max_retry_cycles=2,
            enforce_deterministic_split=True
        )
    
    def _determine_target_difficulty(self) -> DifficultyLevel:
        """Determine target difficulty based on eval type and configuration"""
        
        # Override from config if specified
        if hasattr(self.config, 'difficulty'):
            return self.config.difficulty
        
        # Default mapping for eval types that should be challenging
        challenging_types = {
            EvaluationType.MATHEMATICAL,
            EvaluationType.CODE_GENERATION,
            EvaluationType.DOMAIN_KNOWLEDGE,
            EvaluationType.COMMONSENSE_REASONING
        }
        
        if self.eval_type in challenging_types:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.INTERMEDIATE
    
    def _fallback_to_basic_generation(self, corpus_text: str, num_questions: int) -> List[Dict[str, Any]]:
        """Fallback to basic generation when agentic pipeline fails"""
        
        # Import appropriate basic generator
        from ..benchmarks import BenchmarkGeneratorFactory
        
        basic_generator = BenchmarkGeneratorFactory.create_generator(self.eval_type)
        basic_items = basic_generator.generate_benchmark(corpus_text, num_questions)
        
        # Add metadata indicating fallback was used
        for item in basic_items:
            if 'metadata' not in item:
                item['metadata'] = {}
            item['metadata'].update({
                'source': 'agentic_fallback',
                'fallback_reason': 'agentic_pipeline_failed',
                'original_source': 'basic_generator'
            })
        
        return basic_items
    
    def get_generation_report(self) -> Dict[str, Any]:
        """Get comprehensive generation report"""
        
        pipeline_metrics = self.orchestrator.get_pipeline_metrics()
        
        return {
            'generator_type': 'agentic',
            'eval_type': self.eval_type,
            'configuration': self.config.model_dump(),
            'generation_metadata': self.generation_metadata,
            'pipeline_metrics': pipeline_metrics,
            'agent_performance': pipeline_metrics.get('agent_stats', {}),
            'quality_statistics': {
                'avg_quality_score': pipeline_metrics.get('avg_quality_score'),
                'acceptance_rate': pipeline_metrics.get('acceptance_rate'),
                'retry_rate': pipeline_metrics.get('retry_cycles', 0) / max(1, pipeline_metrics.get('total_generated', 1))
            }
        }
    
    def validate_deterministic_split(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate that deterministic/non-deterministic split is correct
        
        Args:
            items: Generated benchmark items
            
        Returns:
            Validation report
        """
        
        deterministic_count = 0
        non_deterministic_count = 0
        misclassified = []
        
        for i, item in enumerate(items):
            metadata = item.get('metadata', {})
            marked_deterministic = metadata.get('deterministic', True)
            
            # Check if classification matches eval type expectations
            expected_deterministic = self._should_be_deterministic(item)
            
            if marked_deterministic:
                deterministic_count += 1
            else:
                non_deterministic_count += 1
            
            if marked_deterministic != expected_deterministic:
                misclassified.append({
                    'index': i,
                    'question': item.get('question', '')[:50] + '...',
                    'marked_as': 'deterministic' if marked_deterministic else 'non_deterministic',
                    'should_be': 'deterministic' if expected_deterministic else 'non_deterministic'
                })
        
        return {
            'total_items': len(items),
            'deterministic_count': deterministic_count,
            'non_deterministic_count': non_deterministic_count,
            'deterministic_ratio': deterministic_count / len(items) if items else 0,
            'misclassified_count': len(misclassified),
            'misclassified_items': misclassified,
            'classification_accuracy': 1.0 - (len(misclassified) / len(items)) if items else 1.0
        }
    
    def _should_be_deterministic(self, item: Dict[str, Any]) -> bool:
        """Check if an item should be classified as deterministic"""
        
        from ..evaluation import is_deterministic
        
        eval_type = item.get('eval_type', self.eval_type)
        
        # Check eval type
        if is_deterministic(eval_type):
            return True
        
        # Check answer characteristics
        answer = item.get('answer', '').strip()
        
        # Short factual answers tend to be deterministic
        if len(answer.split()) <= 3 and eval_type in [EvaluationType.FACTUAL_QA, EvaluationType.DOMAIN_KNOWLEDGE]:
            return True
        
        # Multiple choice is deterministic
        if item.get('options') and len(item['options']) > 1:
            return True
        
        # Numeric answers are deterministic
        import re
        if re.search(r'^\d+\.?\d*$', answer):
            return True
        
        return False
    
    def export_schema(self) -> Dict[str, Any]:
        """Export the schema used by this generator"""
        
        from .models import export_schemas
        
        return {
            'generator_type': 'agentic',
            'eval_type': self.eval_type,
            'schemas': export_schemas(),
            'config_schema': PipelineConfig.schema(),
            'version': 'v2.0'
        }