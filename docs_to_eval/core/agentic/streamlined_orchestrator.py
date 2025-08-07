"""
Streamlined Orchestrator for Production-Ready Agentic Benchmark Generation
Coordinates 3 essential agents with clear responsibilities and efficient processing
"""

import asyncio
import time
import random
from typing import Dict, List, Any, Optional
import logging

from .streamlined_agents import (
    ConceptExtractor,
    QuestionGenerator,
    QualityValidator
)
from .models import (
    EnhancedBenchmarkItem,
    DifficultyLevel,
    BenchmarkMetadata,
    create_enhanced_metadata
)
from ..evaluation import EvaluationType
from ...llm.base import BaseLLMInterface


logger = logging.getLogger(__name__)


class StreamlinedOrchestrator:
    """
    Efficient orchestrator for benchmark generation
    Manages 3-agent pipeline: ConceptExtractor → QuestionGenerator → QualityValidator
    """
    
    def __init__(self, llm_interface: Optional[BaseLLMInterface] = None):
        """
        Initialize orchestrator with single LLM interface
        
        Args:
            llm_interface: LLM interface for all agents
        """
        self.llm = llm_interface
        
        # Initialize the 3 essential agents
        self.concept_extractor = ConceptExtractor(llm_interface)
        self.question_generator = QuestionGenerator(llm_interface)
        self.quality_validator = QualityValidator(llm_interface)
        
        # Configuration
        self.config = {
            'min_quality_score': 0.5,
            'oversample_factor': 1.5,  # Generate 50% more to filter
            'parallel_batch_size': 5,
            'max_retries': 2,
            'enable_diversity': True
        }
        
        # Tracking
        self.stats = {
            'total_generated': 0,
            'total_accepted': 0,
            'concepts_extracted': 0,
            'processing_time': 0.0
        }
    
    async def generate(
        self,
        corpus_text: str,
        num_questions: int = 20,
        eval_type: str = "domain_knowledge",
        difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
        progress_callback: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate benchmark questions using streamlined pipeline
        
        Args:
            corpus_text: Source text for questions
            num_questions: Number of questions to generate
            eval_type: Type of evaluation
            difficulty: Target difficulty level
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of generated benchmark items as dictionaries
        """
        start_time = time.time()
        
        try:
            # Step 1: Extract concepts
            if progress_callback:
                await progress_callback.send_log("info", f"📚 Extracting key concepts from corpus...")
            
            num_concepts = int(num_questions * self.config['oversample_factor'])
            concept_result = await self.concept_extractor.produce(corpus_text, num_concepts)
            
            self.stats['concepts_extracted'] = len(concept_result.key_concepts)
            
            if progress_callback:
                await progress_callback.send_log("success", 
                    f"✅ Extracted {len(concept_result.key_concepts)} key concepts")
            
            # Step 2: Generate questions in parallel batches
            if progress_callback:
                await progress_callback.send_log("info", f"🎯 Generating {num_questions} questions...")
            
            generated_items = []
            concepts = concept_result.key_concepts
            
            # Process in batches for efficiency
            batch_size = self.config['parallel_batch_size']
            for i in range(0, len(concepts), batch_size):
                batch_concepts = concepts[i:i + batch_size]
                
                # Create generation tasks for this batch
                tasks = []
                for concept in batch_concepts:
                    context = concept_result.supporting_snippets.get(concept, corpus_text[:500])
                    task = self._generate_single_question(
                        concept, context, eval_type, difficulty, corpus_text
                    )
                    tasks.append(task)
                
                # Execute batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in batch_results:
                    if isinstance(result, dict) and result.get('accepted'):
                        generated_items.append(result['item'])
                        self.stats['total_accepted'] += 1
                        
                        if progress_callback:
                            await progress_callback.increment_progress(
                                message=f"Generated {len(generated_items)}/{num_questions} questions"
                            )
                    
                    self.stats['total_generated'] += 1
                
                # Check if we have enough questions
                if len(generated_items) >= num_questions:
                    break
            
            # Step 3: Ensure we have enough questions
            if len(generated_items) < num_questions:
                if progress_callback:
                    await progress_callback.send_log("warning", 
                        f"⚠️ Generated {len(generated_items)}/{num_questions} questions, creating additional...")
                
                # Generate additional questions with relaxed criteria
                additional_needed = num_questions - len(generated_items)
                for _ in range(additional_needed):
                    # Use random concept and generate simpler question
                    concept = random.choice(concepts)
                    context = concept_result.supporting_snippets.get(concept, corpus_text[:500])
                    result = await self._generate_simple_question(concept, context, eval_type)
                    if result:
                        generated_items.append(result)
            
            # Step 4: Select final questions
            final_items = generated_items[:num_questions]
            
            # Convert to dictionary format
            output_items = []
            for item in final_items:
                output_items.append(self._convert_to_dict(item))
            
            # Update stats
            self.stats['processing_time'] = time.time() - start_time
            
            if progress_callback:
                await progress_callback.send_log("success", 
                    f"✅ Generated {len(output_items)} questions in {self.stats['processing_time']:.1f}s")
            
            return output_items
            
        except Exception as e:
            logger.error(f"Orchestrator failed: {str(e)}")
            if progress_callback:
                await progress_callback.send_log("error", f"❌ Generation failed: {str(e)}")
            
            # Return fallback questions
            return self._generate_fallback_questions(corpus_text, num_questions, eval_type)
    
    async def _generate_single_question(
        self,
        concept: str,
        context: str,
        eval_type: str,
        difficulty: DifficultyLevel,
        corpus_text: str
    ) -> Dict[str, Any]:
        """Generate and validate a single question"""
        
        for retry in range(self.config['max_retries']):
            try:
                # Generate question
                candidate = await self.question_generator.produce(
                    concept, context, eval_type, difficulty
                )
                
                # Validate quality
                validation = await self.quality_validator.produce(
                    candidate, corpus_text, self.config['min_quality_score']
                )
                
                if validation.accepted:
                    return {
                        'accepted': True,
                        'item': candidate,
                        'validation': validation
                    }
                
            except Exception as e:
                logger.warning(f"Failed to generate question for {concept}: {str(e)}")
        
        return {'accepted': False}
    
    async def _generate_simple_question(
        self,
        concept: str,
        context: str,
        eval_type: str
    ) -> Optional[Any]:
        """Generate a simple fallback question"""
        try:
            # Use basic difficulty for fallback
            candidate = await self.question_generator.produce(
                concept, context, eval_type, DifficultyLevel.BASIC
            )
            return candidate
        except:
            return None
    
    def _convert_to_dict(self, item: Any) -> Dict[str, Any]:
        """Convert BenchmarkCandidate to dictionary format"""
        try:
            return {
                "question": item.question,
                "answer": item.answer,
                "context": item.context,
                "options": item.options,
                "concept": item.concept,
                "difficulty": item.difficulty.value if hasattr(item.difficulty, 'value') else str(item.difficulty),
                "answer_type": item.expected_answer_type.value if hasattr(item.expected_answer_type, 'value') else str(item.expected_answer_type),
                "metadata": {
                    "source": "streamlined_agentic",
                    "question_type": item.variables.get("question_type", "unknown") if hasattr(item, 'variables') else "unknown",
                    "reasoning": item.reasoning_chain if hasattr(item, 'reasoning_chain') else []
                }
            }
        except Exception as e:
            logger.error(f"Failed to convert item: {str(e)}")
            return {
                "question": str(item.question) if hasattr(item, 'question') else "Error generating question",
                "answer": str(item.answer) if hasattr(item, 'answer') else "Error generating answer",
                "context": None,
                "options": None,
                "concept": str(item.concept) if hasattr(item, 'concept') else "unknown",
                "difficulty": "intermediate",
                "answer_type": "free_text",
                "metadata": {"source": "streamlined_agentic", "error": str(e)}
            }
    
    def _generate_fallback_questions(
        self,
        corpus_text: str,
        num_questions: int,
        eval_type: str
    ) -> List[Dict[str, Any]]:
        """Generate simple fallback questions without LLM"""
        questions = []
        
        # Extract simple concepts from text
        words = corpus_text.split()[:500]
        concepts = [w for w in words if len(w) > 6 and w[0].isupper()][:num_questions]
        
        for i, concept in enumerate(concepts):
            questions.append({
                "question": f"What is {concept}?",
                "answer": f"{concept} is mentioned in the provided context.",
                "context": corpus_text[:200],
                "options": None,
                "concept": concept,
                "difficulty": "basic",
                "answer_type": "free_text",
                "metadata": {"source": "fallback", "index": i}
            })
        
        # Fill remaining with generic questions
        while len(questions) < num_questions:
            i = len(questions)
            questions.append({
                "question": f"What is discussed in the provided text (part {i+1})?",
                "answer": "The text discusses various topics as outlined in the context.",
                "context": corpus_text[:200],
                "options": None,
                "concept": "general",
                "difficulty": "basic",
                "answer_type": "free_text",
                "metadata": {"source": "fallback", "index": i}
            })
        
        return questions[:num_questions]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            **self.stats,
            'acceptance_rate': self.stats['total_accepted'] / max(1, self.stats['total_generated']),
            'concepts_per_question': self.stats['concepts_extracted'] / max(1, self.stats['total_accepted']),
            'agent_stats': {
                'concept_extractor': {
                    'calls': self.concept_extractor.call_count,
                    'time': self.concept_extractor.total_processing_time
                },
                'question_generator': {
                    'calls': self.question_generator.call_count,
                    'time': self.question_generator.total_processing_time
                },
                'quality_validator': {
                    'calls': self.quality_validator.call_count,
                    'time': self.quality_validator.total_processing_time
                }
            }
        }
