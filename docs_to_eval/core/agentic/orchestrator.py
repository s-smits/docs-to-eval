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


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts delays based on LLM response times and error rates
    """
    
    def __init__(self, base_delay: float = 0.1, max_delay: float = 5.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.recent_response_times = []
        self.recent_errors = []
        self.error_count = 0
        self.total_requests = 0
        self.window_size = 10  # Number of recent samples to consider
        
    def record_request(self, response_time: float, had_error: bool = False):
        """Record a request's response time and error status"""
        self.total_requests += 1
        
        # Update response times (keep only recent)
        self.recent_response_times.append(response_time)
        if len(self.recent_response_times) > self.window_size:
            self.recent_response_times.pop(0)
        
        # Update error tracking
        if had_error:
            self.error_count += 1
        self.recent_errors.append(had_error)
        if len(self.recent_errors) > self.window_size:
            self.recent_errors.pop(0)
    
    def get_adaptive_delay(self) -> float:
        """Calculate adaptive delay based on recent performance"""
        if not self.recent_response_times:
            return self.base_delay
        
        # Calculate average response time
        avg_response_time = sum(self.recent_response_times) / len(self.recent_response_times)
        
        # Calculate recent error rate
        recent_error_rate = sum(self.recent_errors) / len(self.recent_errors) if self.recent_errors else 0
        
        # Adaptive delay based on response time
        # If responses are slow, increase delay to reduce load
        response_factor = min(2.0, avg_response_time / 0.5)  # Scale based on 0.5s baseline
        
        # Error rate factor - increase delay if many errors
        error_factor = 1.0 + (recent_error_rate * 2.0)  # Up to 3x delay for 100% error rate
        
        # Combined adaptive delay
        adaptive_delay = self.base_delay * response_factor * error_factor
        
        return min(adaptive_delay, self.max_delay)
    
    async def adaptive_sleep(self):
        """Sleep for adaptive duration based on recent performance"""
        delay = self.get_adaptive_delay()
        await asyncio.sleep(delay)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        avg_response_time = sum(self.recent_response_times) / len(self.recent_response_times) if self.recent_response_times else 0
        recent_error_rate = sum(self.recent_errors) / len(self.recent_errors) if self.recent_errors else 0
        
        return {
            'total_requests': self.total_requests,
            'error_count': self.error_count,
            'overall_error_rate': self.error_count / max(1, self.total_requests),
            'recent_error_rate': recent_error_rate,
            'avg_response_time': avg_response_time,
            'current_delay': self.get_adaptive_delay()
        }


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
        
        # CRITICAL FIX: Add adaptive rate limiter
        self.rate_limiter = AdaptiveRateLimiter(base_delay=0.1, max_delay=5.0)
    
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
            concept_extraction = await self.concept_miner.produce(
                corpus_text, 
                k=int(num_questions * self.config.oversample_factor)
            )
            
            if not concept_extraction.key_concepts:
                logger.warning("No concepts extracted, falling back to simple generation")
                return await self._fallback_generation(corpus_text, eval_type, num_questions, difficulty)
            
            logger.info(f"Extracted {len(concept_extraction.key_concepts)} concepts")
            
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
                
                # Adaptive pause between batches based on LLM performance
                await self.rate_limiter.adaptive_sleep()
            
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
        Run the full pipeline for a single concept with intelligent retry logic
        """
        
        previous_validation_issues = []
        base_temperature = 0.7
        
        for retry_cycle in range(self.config.max_retry_cycles + 1):
            try:
                # Adaptive temperature increase on retries
                current_temperature = base_temperature + (retry_cycle * 0.1)
                
                # Step 1: Question Writer (with feedback)
                context_snippet = concept_extraction.supporting_snippets.get(concept, corpus_text[:500])
                
                # CRITICAL FIX: Pass validation feedback to writer on retries
                if retry_cycle > 0 and previous_validation_issues:
                    # Create feedback-aware prompt for question writer
                    feedback_prompt = f"Previous attempt failed with issues: {', '.join(previous_validation_issues)}. Please avoid these problems."
                    draft = await self._tracked_agent_call(
                        self.question_writer.produce,
                        concept, corpus_text, eval_type, context_snippet, 
                        feedback=feedback_prompt, temperature=current_temperature
                    )
                else:
                    draft = await self._tracked_agent_call(
                        self.question_writer.produce,
                        concept, corpus_text, eval_type, context_snippet,
                        temperature=current_temperature
                    )
                
                # Step 2: Adversary (difficulty booster)
                candidate = await self._tracked_agent_call(
                    self.adversary.produce, draft, difficulty
                )
                
                # Step 3: Refiner (style and formatting with feedback)
                if retry_cycle > 0 and previous_validation_issues:
                    # Pass issues to refiner for targeted improvement
                    refined_candidate = await self._tracked_agent_call(
                        self.refiner.produce,
                        candidate, improvement_focus=previous_validation_issues
                    )
                else:
                    refined_candidate = await self._tracked_agent_call(
                        self.refiner.produce, candidate
                    )
                
                # Step 4: Validator (quality check)
                validation_result = await self.validator.accept(
                    refined_candidate, 
                    self.config.min_validation_score
                )
                
                if validation_result.accepted:
                    # Success! Convert to final item
                    logger.debug(f"Concept '{concept}' succeeded on retry cycle {retry_cycle}")
                    return self._create_enhanced_item(
                        refined_candidate, 
                        eval_type, 
                        validation_result,
                        concept_extraction,
                        retry_cycle
                    )
                
                elif retry_cycle < self.config.max_retry_cycles:
                    # Store feedback for next iteration
                    previous_validation_issues = validation_result.issues
                    
                    # Smart retry with feedback
                    logger.debug(f"Retrying concept '{concept}' (cycle {retry_cycle + 1}) with feedback: {validation_result.issues}")
                    self.pipeline_stats['retry_cycles'] += 1
                    
                    # Adaptive delay based on retry count
                    delay = 0.1 * (2 ** retry_cycle)  # Exponential backoff
                    await asyncio.sleep(min(delay, 2.0))  # Cap at 2 seconds
                
            except Exception as e:
                logger.warning(f"Pipeline error for concept '{concept}' (cycle {retry_cycle}): {str(e)}")
                if retry_cycle == self.config.max_retry_cycles:
                    return None
                
                # Exponential backoff for errors too
                error_delay = 0.2 * (2 ** retry_cycle)
                await asyncio.sleep(min(error_delay, 5.0))
        
        logger.warning(f"Concept '{concept}' exhausted all {self.config.max_retry_cycles + 1} retry attempts")
        return None
    
    async def _tracked_agent_call(self, agent_method, *args, **kwargs):
        """Wrapper for agent calls to track performance for adaptive rate limiting"""
        start_time = time.time()
        had_error = False
        
        try:
            result = await agent_method(*args, **kwargs)
            return result
        except Exception as e:
            had_error = True
            raise e
        finally:
            response_time = time.time() - start_time
            self.rate_limiter.record_request(response_time, had_error)
    
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
            concept=candidate.concept,  # CRITICAL FIX: Maintain concept ownership
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
            
            concept = item.concept  # FIXED: Direct access since concept is now properly maintained
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
        
        # Update agent stats (thread-safe)
        agents = [self.concept_miner, self.question_writer, self.adversary, self.refiner, self.validator]
        for agent in agents:
            agent_name = agent.agent_name
            stats = agent.get_stats()  # Now synchronous
            self.pipeline_stats['agent_stats'][agent_name] = stats
    
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
            
            # Create basic item with fallback flagging
            metadata = create_enhanced_metadata(
                difficulty=difficulty,
                agents_used=['FallbackGenerator@v1'],
                deterministic=is_deterministic(eval_type),
                validation_score=0.5  # Neutral score for fallback
            )
            
            # CRITICAL FIX: Add fallback flagging to metadata
            metadata.provenance.update({
                'generation_mode': 'agentic_fallback',
                'quality_warning': 'Generated using emergency agentic fallback - main pipeline failed',
                'fallback_reason': 'Agentic pipeline failure',
                'quality_degraded': True
            })
            
            item = EnhancedBenchmarkItem(
                question=f"What is the significance of {concept} in the given context?",
                answer=f"The {concept} is significant because...",
                context=corpus_text[:200] if corpus_text else None,
                eval_type=eval_type,
                metadata=metadata,
                concept=concept,  # Add concept field
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
        
        # Add adaptive rate limiter statistics
        stats['rate_limiter'] = self.rate_limiter.get_stats()
        
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