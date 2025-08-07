"""
Agentic Benchmark Orchestrator
Coordinates the full pipeline of specialized agents for intelligent benchmark generation
"""

import asyncio
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import logging

from .models import (
    BenchmarkDraft,
    BenchmarkCandidate,
    EnhancedBenchmarkItem,
    PipelineConfig,
    DifficultyLevel,
    BenchmarkMetadata,
    create_enhanced_metadata
)
from .agents import (
    ConceptMiner,
    QuestionWriter,
    Adversary,
    Refiner,
    Validator
)
from ..evaluation import EvaluationType, is_deterministic
from ...llm.base import BaseLLMInterface


logger = logging.getLogger(__name__)


class AgenticBenchmarkOrchestrator:
    """
    High-level orchestrator for agentic benchmark generation
    Coordinates all agents in the intelligent pipeline workflow
    """
    
    def __init__(self, llm_pool: Dict[str, BaseLLMInterface], config: Optional[PipelineConfig] = None):
        """
        Initialize orchestrator with LLM pool and configuration
        
        Args:
            llm_pool: Dictionary of LLM interfaces for different roles
                     Expected keys: 'retriever', 'creator', 'adversary', 'refiner'
            config: Pipeline configuration
        """
        self.llm_pool = llm_pool
        self.config = config or PipelineConfig()
        
        # Initialize agents with appropriate LLM interfaces and configs
        self.concept_miner = ConceptMiner(
            llm_interface=llm_pool.get('retriever'),
            config=self.config.agent_configs.get('concept_miner')
        )
        
        self.question_writer = QuestionWriter(
            llm_interface=llm_pool.get('creator'),
            config=self.config.agent_configs.get('question_writer')
        )
        
        self.adversary = Adversary(
            llm_interface=llm_pool.get('adversary'),
            config=self.config.agent_configs.get('adversary')
        )
        
        self.refiner = Refiner(
            llm_interface=llm_pool.get('refiner'),
            config=self.config.agent_configs.get('refiner')
        )
        
        self.validator = Validator(
            config=self.config.agent_configs.get('validator')
        )
        
        # Pipeline metrics
        self.pipeline_stats = {
            'total_generated': 0,
            'total_accepted': 0,
            'total_processing_time': 0.0,
            'agent_stats': {},
            'quality_scores': [],
            'retry_cycles': 0
        }
    
    async def generate(
        self,
        corpus_text: str,
        eval_type: EvaluationType,
        num_questions: int = 50,
        difficulty: DifficultyLevel = DifficultyLevel.HARD
    ) -> List[EnhancedBenchmarkItem]:
        """
        Main generation method - orchestrates the full agentic pipeline
        
        Args:
            corpus_text: Source text for benchmark generation
            eval_type: Type of evaluation to generate
            num_questions: Target number of questions
            difficulty: Target difficulty level
            
        Returns:
            List of high-quality benchmark items
        """
        start_time = time.time()
        logger.info(f"Starting agentic generation: {num_questions} questions, {eval_type}, {difficulty}")
        
        try:
            # Step 0: Mine concepts once (shared across all questions)
            logger.info("Step 0: Mining key concepts from corpus")
            try:
                concept_extraction = await self.concept_miner.produce(
                    corpus_text, 
                    k=int(num_questions * self.config.oversample_factor)
                )
                logger.info("ConceptMiner completed successfully")
            except Exception as e:
                logger.error(f"ConceptMiner failed: {str(e)}", exc_info=True)
                raise
            
            if not concept_extraction.key_concepts:
                logger.warning("No concepts extracted, falling back to simple generation")
                return await self._fallback_generation(corpus_text, eval_type, num_questions, difficulty)
            
            # Check if we got fallback concepts (indicating ConceptMiner failure)
            if any(concept.startswith("fallback_concept_") for concept in concept_extraction.key_concepts):
                logger.warning("ConceptMiner returned fallback concepts - continuing but quality may be reduced")
            
            logger.info(f"Successfully extracted {len(concept_extraction.key_concepts)} concepts: {concept_extraction.key_concepts[:5]}...")
            
            # Step 1: Parallel pipeline generation with oversampling
            target_concepts = concept_extraction.key_concepts[:int(num_questions * self.config.oversample_factor)]
            logger.info(f"Processing {len(target_concepts)} concepts in parallel batches")
            
            generated_items = []
            
            # Process concepts in parallel batches to manage resource usage
            batch_size = self.config.parallel_batch_size
            for i in range(0, len(target_concepts), batch_size):
                batch_concepts = target_concepts[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_concepts)} concepts")
                
                # Create tasks for this batch
                batch_tasks = [
                    self._pipeline_single_concept(concept, corpus_text, eval_type, difficulty, concept_extraction)
                    for concept in batch_concepts
                ]
                
                # Execute batch with proper error handling
                batch_results = await self._execute_batch_with_timeout(batch_tasks)
                
                # Collect successful results
                for result in batch_results:
                    if result is not None:
                        generated_items.append(result)
                
                # Check if we have enough high-quality items
                if len(generated_items) >= num_questions:
                    logger.info(f"Target reached with {len(generated_items)} items")
                    break
                
                # Brief pause between batches to manage rate limits
                await asyncio.sleep(0.1)
            
            # Step 2: Select best items
            final_items = self._select_best_items(generated_items, num_questions)
            
            # Step 3: Update pipeline statistics
            total_time = time.time() - start_time
            self._update_pipeline_stats(final_items, total_time)
            
            logger.info(f"Generation complete: {len(final_items)} items in {total_time:.2f}s")
            return final_items
            
        except Exception as e:
            logger.error(f"Pipeline generation failed: {str(e)}")
            return await self._fallback_generation(corpus_text, eval_type, num_questions, difficulty)
    
    async def _pipeline_single_concept(
        self,
        concept: str,
        corpus_text: str,
        eval_type: EvaluationType,
        difficulty: DifficultyLevel,
        concept_extraction: Any
    ) -> Optional[EnhancedBenchmarkItem]:
        """
        Run the full pipeline for a single concept with retry logic
        """
        previous_validation_issues = []
        
        for retry_cycle in range(self.config.max_retry_cycles + 1):
            try:
                # Step 1: Question Writer with adaptive temperature
                context_snippet = concept_extraction.supporting_snippets.get(concept, corpus_text)
                
                # Adaptive temperature: start conservative, increase for retries
                current_temperature = min(0.9, 0.3 + (retry_cycle * 0.15))
                original_temp = getattr(self.question_writer.config, 'temperature', 0.7)
                
                # Temporarily adjust temperature for this attempt
                if hasattr(self.question_writer.config, 'temperature'):
                    self.question_writer.config.temperature = current_temperature
                
                try:
                    if retry_cycle > 0:
                        # Use validation feedback for retries
                        validation_context = {
                            'previous_issues': previous_validation_issues,
                            'retry_count': retry_cycle,
                            'suggestions': [
                                'Make the question more specific',
                                'Ensure the answer is directly derivable from context',
                                'Check for clarity and unambiguity'
                            ]
                        }
                        draft = await self.question_writer.produce_with_feedback(
                            concept, corpus_text, eval_type, context_snippet, validation_context
                        )
                    else:
                        draft = await self.question_writer.produce(
                            concept, corpus_text, eval_type, context_snippet
                        )
                finally:
                    # Restore original temperature
                    if hasattr(self.question_writer.config, 'temperature'):
                        self.question_writer.config.temperature = original_temp
                
                # Step 2: Adversary (difficulty booster)
                candidate = await self.adversary.produce(draft, difficulty)
                
                # Step 3: Refiner (style and formatting)
                refined_candidate = await self.refiner.produce(candidate)
                
                # Step 4: Validator (quality check)
                validation_result = await self.validator.accept(
                    refined_candidate, 
                    self.config.min_validation_score
                )
                
                if validation_result.accepted:
                    # Success! Convert to final item
                    return self._create_enhanced_item(
                        refined_candidate, 
                        eval_type, 
                        validation_result,
                        concept_extraction,
                        retry_cycle
                    )
                
                elif retry_cycle < self.config.max_retry_cycles:
                    # Collect validation issues for next retry
                    if hasattr(validation_result, 'issues'):
                        previous_validation_issues.extend(validation_result.issues)
                    elif hasattr(validation_result, 'message'):
                        previous_validation_issues.append(validation_result.message)
                    
                    logger.debug(f"Retrying concept '{concept}' (cycle {retry_cycle + 1}): {previous_validation_issues}")
                    self.pipeline_stats['retry_cycles'] += 1
                    
                    # Exponential backoff for retries
                    backoff_time = min(0.5, 0.1 * (2 ** retry_cycle))
                    await asyncio.sleep(backoff_time)
                
            except Exception as e:
                logger.warning(f"Pipeline error for concept '{concept}' (cycle {retry_cycle}): {str(e)}")
                if retry_cycle == self.config.max_retry_cycles:
                    return None
                await asyncio.sleep(0.2)
        
        return None
    
    async def _execute_batch_with_timeout(
        self, 
        tasks: List[asyncio.Task], 
        timeout: float = 60.0
    ) -> List[Optional[EnhancedBenchmarkItem]]:
        """
        Execute a batch of tasks with timeout and error handling
        """
        try:
            # Use asyncio.gather with return_exceptions=True to handle individual failures
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), 
                timeout=timeout
            )
            
            # Process results, converting exceptions to None
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Task failed: {str(result)}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            return processed_results
        
        except asyncio.TimeoutError:
            logger.warning(f"Batch execution timed out after {timeout}s")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            return [None] * len(tasks)
    
    def _create_enhanced_item(
        self,
        candidate: BenchmarkCandidate,
        eval_type: EvaluationType,
        validation_result: Any,
        concept_extraction: Any,
        retry_cycle: int
    ) -> EnhancedBenchmarkItem:
        """Create enhanced benchmark item with full metadata"""
        
        # Determine if this should be deterministic
        is_deterministic_eval = is_deterministic(eval_type)
        
        # Create enhanced metadata
        metadata = create_enhanced_metadata(
            difficulty=candidate.difficulty,
            agents_used=[
                f"ConceptMiner@{self.concept_miner.agent_version}",
                f"QuestionWriter@{self.question_writer.agent_version}",
                f"Adversary@{self.adversary.agent_version}",
                f"Refiner@{self.refiner.agent_version}",
                f"Validator@{self.validator.agent_version}"
            ],
            deterministic=is_deterministic_eval,
            concept_importance=concept_extraction.concept_importance_scores.get(candidate.concept, 0.5),
            validation_score=validation_result.score,
            adversarial_techniques=candidate.adversarial_techniques,
            provenance={
                'model_versions': {agent.agent_name: agent.agent_version for agent in [
                    self.concept_miner, self.question_writer, self.adversary, self.refiner, self.validator
                ]},
                'retry_cycle': retry_cycle,
                'validation_method': validation_result.verification_method_used,
                'generation_timestamp': time.time()
            }
        )
        
        return EnhancedBenchmarkItem(
            question=candidate.question,
            answer=candidate.answer,
            context=candidate.context,
            options=candidate.options,
            eval_type=eval_type,
            metadata=metadata,
            expected_answer_type=candidate.expected_answer_type,
            reasoning_chain=candidate.reasoning_chain,
            variables=candidate.variables
        )
    
    def _select_best_items(
        self, 
        generated_items: List[EnhancedBenchmarkItem], 
        target_count: int
    ) -> List[EnhancedBenchmarkItem]:
        """
        Select the best items based on validation scores and diversity
        """
        if len(generated_items) <= target_count:
            return generated_items
        
        # Sort by validation score (descending)
        scored_items = [(item.metadata.validation_score or 0.5, item) for item in generated_items]
        scored_items.sort(reverse=True, key=lambda x: x[0])
        
        # Select top items with some diversity
        selected_items = []
        used_concepts = set()
        
        # First pass: select highest-scoring items with unique concepts
        for score, item in scored_items:
            if len(selected_items) >= target_count:
                break
            
            concept = getattr(item, 'concept', item.metadata.dict().get('concept', 'unknown'))
            if concept not in used_concepts:
                selected_items.append(item)
                used_concepts.add(concept)
        
        # Second pass: fill remaining slots with highest-scoring items
        for score, item in scored_items:
            if len(selected_items) >= target_count:
                break
            if item not in selected_items:
                selected_items.append(item)
        
        logger.info(f"Selected {len(selected_items)} items from {len(generated_items)} candidates")
        return selected_items[:target_count]
    
    def _update_pipeline_stats(self, items: List[EnhancedBenchmarkItem], total_time: float):
        """Update pipeline statistics"""
        self.pipeline_stats['total_generated'] += len(items)
        self.pipeline_stats['total_accepted'] += len(items)
        self.pipeline_stats['total_processing_time'] += total_time
        
        # Collect quality scores
        quality_scores = [item.metadata.validation_score for item in items if item.metadata.validation_score]
        self.pipeline_stats['quality_scores'].extend(quality_scores)
        
        # Update agent stats
        agents = [self.concept_miner, self.question_writer, self.adversary, self.refiner, self.validator]
        for agent in agents:
            agent_name = agent.agent_name
            self.pipeline_stats['agent_stats'][agent_name] = {
                'call_count': agent.call_count,
                'total_time': agent.total_processing_time,
                'avg_time': agent.total_processing_time / max(1, agent.call_count)
            }
    
    async def _fallback_generation(
        self, 
        corpus_text: str, 
        eval_type: EvaluationType, 
        num_questions: int,
        difficulty: DifficultyLevel
    ) -> List[EnhancedBenchmarkItem]:
        """Fallback generation when main pipeline fails"""
        
        logger.warning("Using fallback generation method")
        
        # Simple template-based generation
        fallback_items = []
        concepts = ['concept', 'topic', 'subject', 'matter']
        
        for i in range(min(num_questions, 10)):  # Limit fallback items
            concept = f"{concepts[i % len(concepts)]}_{i+1}"
            
            # Create basic item
            metadata = create_enhanced_metadata(
                difficulty=difficulty,
                agents_used=['FallbackGenerator@v1'],
                deterministic=is_deterministic(eval_type),
                validation_score=0.5  # Neutral score for fallback
            )
            
            item = EnhancedBenchmarkItem(
                question=f"What is the significance of {concept} in the given context?",
                answer=f"The {concept} is significant because...",
                context=corpus_text if corpus_text else None,
                eval_type=eval_type,
                metadata=metadata,
                expected_answer_type='free_text',
                reasoning_chain=["Basic template generation"],
                variables={}
            )
            
            fallback_items.append(item)
        
        return fallback_items
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline performance metrics"""
        stats = self.pipeline_stats.copy()
        
        # Calculate additional metrics
        if stats['quality_scores']:
            stats['avg_quality_score'] = sum(stats['quality_scores']) / len(stats['quality_scores'])
            stats['min_quality_score'] = min(stats['quality_scores'])
            stats['max_quality_score'] = max(stats['quality_scores'])
        
        if stats['total_generated'] > 0:
            stats['acceptance_rate'] = stats['total_accepted'] / stats['total_generated']
            stats['avg_generation_time'] = stats['total_processing_time'] / stats['total_generated']
        
        return stats
    
    def reset_stats(self):
        """Reset pipeline statistics"""
        self.pipeline_stats = {
            'total_generated': 0,
            'total_accepted': 0,
            'total_processing_time': 0.0,
            'agent_stats': {},
            'quality_scores': [],
            'retry_cycles': 0
        }
        
        # Reset agent stats
        for agent in [self.concept_miner, self.question_writer, self.adversary, self.refiner, self.validator]:
            agent.call_count = 0
            agent.total_processing_time = 0.0
    
    async def validate_pipeline_health(self) -> Dict[str, Any]:
        """Validate that the pipeline is properly configured and functioning"""
        
        health_status = {
            'healthy': True,
            'issues': [],
            'warnings': [],
            'llm_pool_status': {},
            'agent_status': {}
        }
        
        # Check LLM pool
        required_llms = ['retriever', 'creator', 'adversary', 'refiner']
        for llm_key in required_llms:
            if llm_key not in self.llm_pool:
                health_status['issues'].append(f"Missing LLM interface: {llm_key}")
                health_status['healthy'] = False
            else:
                llm_interface = self.llm_pool[llm_key]
                health_status['llm_pool_status'][llm_key] = {
                    'model_name': getattr(llm_interface, 'model_name', 'unknown'),
                    'available': llm_interface is not None
                }
        
        # Check agent initialization
        agents = {
            'concept_miner': self.concept_miner,
            'question_writer': self.question_writer,
            'adversary': self.adversary,
            'refiner': self.refiner,
            'validator': self.validator
        }
        
        for agent_name, agent in agents.items():
            health_status['agent_status'][agent_name] = {
                'initialized': agent is not None,
                'has_llm': getattr(agent, 'llm', None) is not None,
                'config_valid': getattr(agent, 'config', None) is not None
            }
        
        # Configuration validation
        if self.config.num_questions > 1000:
            health_status['warnings'].append("Very high question count may impact performance")
        
        if self.config.parallel_batch_size > 10:
            health_status['warnings'].append("High parallel batch size may cause rate limiting")
        
        return health_status