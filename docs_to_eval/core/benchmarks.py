"""
Benchmark generators for different evaluation types
Creates domain-specific questions and answers from corpus text
"""

import re
import random
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from .evaluation import EvaluationType, extract_key_concepts, sample_corpus_segments


class BenchmarkGenerator(ABC):
    """Base class for generating benchmarks from corpus text"""
    
    def __init__(self, eval_type: EvaluationType, llm=None):
        self.eval_type = eval_type
        self.llm = llm or MockQuestionGenerator()
    
    @abstractmethod
    def generate_benchmark(self, corpus_text: str, num_questions: int = 50) -> List[Dict[str, Any]]:
        """Generate benchmark questions from corpus"""
        pass
    
    def create_item(self, question: str, answer: str, context: Optional[str] = None, 
                   options: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a benchmark item"""
        return {
            'question': question,
            'answer': answer,
            'context': context,
            'options': options,
            'eval_type': self.eval_type,
            'metadata': {}
        }


class MockQuestionGenerator:
    """Mock LLM for question generation"""
    
    def __init__(self):
        self.question_templates = {
            EvaluationType.MATHEMATICAL: [
                "Calculate: {expression}",
                "Solve for x: {equation}",
                "What is {percentage}% of {number}?",
                "Find the area of a {shape} with {dimension}",
            ],
            EvaluationType.FACTUAL_QA: [
                "What is {concept}?",
                "Who {action} {entity}?",
                "When was {event}?",
                "Where is {location}?",
            ],
            EvaluationType.CODE_GENERATION: [
                "Write a function to {action}",
                "Implement {algorithm} in Python",
                "Create a class for {concept}",
                "How do you {task} in programming?",
            ],
            EvaluationType.DOMAIN_KNOWLEDGE: [
                "Explain {concept} in the context of {domain}",
                "What are the key features of {topic}?",
                "How does {process} work?",
                "What is the relationship between {concept1} and {concept2}?",
            ]
        }
    
    def generate_questions(self, corpus_text: str, eval_type: EvaluationType, num_questions: int) -> List[Dict[str, Any]]:
        """Generate mock questions based on corpus and evaluation type"""
        
        # Extract key concepts from corpus
        key_concepts = extract_key_concepts(corpus_text, max_concepts=20)
        templates = self.question_templates.get(eval_type, self.question_templates[EvaluationType.DOMAIN_KNOWLEDGE])
        
        questions = []
        for i in range(num_questions):
            template = random.choice(templates)
            concept = random.choice(key_concepts) if key_concepts else f"concept_{i+1}"
            
            # Simple template filling
            question = template.format(
                concept=concept,
                domain="the given domain",
                topic=concept,
                process=concept,
                concept1=concept,
                concept2=random.choice(key_concepts) if len(key_concepts) > 1 else "related_concept",
                action="process",
                entity=concept,
                event=f"{concept} development",
                location=f"the {concept} domain",
                expression="2 + 2",
                equation="x + 5 = 10",
                percentage="25",
                number="100",
                shape="circle",
                dimension="radius 5",
                algorithm=concept,
                task=f"implement {concept}",
                algo=concept
            )
            
            # Generate mock answer
            answer = f"Answer about {concept} based on the corpus content"
            
            questions.append({
                'question': question,
                'answer': answer,
                'concept': concept,
                'difficulty': random.choice(['basic', 'intermediate', 'advanced'])
            })
        
        return questions


class AgenticGeneratorWrapper(BenchmarkGenerator):
    """
    Wrapper to provide consistent interface for agentic generators.
    
    This bridges the gap between the standard BenchmarkGenerator interface
    and the async-native agentic generation system.
    """
    
    def __init__(self, agentic_generator, eval_type: EvaluationType):
        super().__init__(eval_type)
        self.agentic_generator = agentic_generator
        
    def generate_benchmark(self, corpus_text: str, num_questions: int = 50) -> List[Dict[str, Any]]:
        """Sync wrapper for agentic generation."""
        import asyncio
        
        # Run async generation in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._generate_async(corpus_text, num_questions))
                    items = future.result()
            else:
                items = loop.run_until_complete(self._generate_async(corpus_text, num_questions))
        except RuntimeError:
            # No event loop, create new one
            items = asyncio.run(self._generate_async(corpus_text, num_questions))
        
        # Convert agentic items to standard format
        return [self._convert_agentic_item(item) for item in items]
    
    async def generate_async(self, corpus_text: str, num_questions: int = 50):
        """Async interface for agentic generation."""
        return await self._generate_async(corpus_text, num_questions)
    
    async def _generate_async(self, corpus_text: str, num_questions: int):
        """Internal async generation method."""
        return await self.agentic_generator.generate_benchmark_async(
            corpus_text=corpus_text,
            num_questions=num_questions
        )
    
    def _convert_agentic_item(self, agentic_item) -> Dict[str, Any]:
        """Convert agentic benchmark item to standard format."""
        if hasattr(agentic_item, 'to_dict'):
            return agentic_item.to_dict()
        elif hasattr(agentic_item, '__dict__'):
            return agentic_item.__dict__
        else:
            return agentic_item
    
    def get_generation_report(self):
        """Get generation report from agentic system if available."""
        if hasattr(self.agentic_generator, 'get_generation_report'):
            return self.agentic_generator.get_generation_report()
        return None


class MathematicalBenchmarkGenerator(BenchmarkGenerator):
    """Generate mathematical evaluation benchmarks"""
    
    def generate_benchmark(self, corpus_text: str, num_questions: int = 50) -> List[Dict[str, Any]]:
        items = []
        
        # Extract numbers and mathematical expressions from corpus
        numbers = re.findall(r'\d+\.?\d*', corpus_text)
        
        for i in range(num_questions):
            if numbers and len(numbers) >= 2:
                num1, num2 = random.sample(numbers, 2)
                operation = random.choice(['+', '-', '*', '/'])
                
                try:
                    result = eval(f"{num1} {operation} {num2}")
                    question = f"Calculate: {num1} {operation} {num2}"
                    answer = str(result)
                except (SyntaxError, NameError, ZeroDivisionError, ArithmeticError):
                    # Handle evaluation errors gracefully
                    question = "What is the mathematical relationship in the corpus?"
                    answer = "Mathematical concepts are discussed in the context"
            else:
                question = "Solve the mathematical problem presented in the corpus"
                answer = "The solution involves mathematical reasoning based on the given information"
            
            items.append(self.create_item(question, answer))
        
        return items


class FactualQABenchmarkGenerator(BenchmarkGenerator):
    """Generate factual Q&A benchmarks"""
    
    def generate_benchmark(self, corpus_text: str, num_questions: int = 50) -> List[Dict[str, Any]]:
        items = []
        key_concepts = extract_key_concepts(corpus_text, max_concepts=num_questions)
        
        question_types = [
            "What is {concept}?",
            "How does {concept} work?",
            "What are the main characteristics of {concept}?",
            "Why is {concept} important?",
            "When is {concept} used?"
        ]
        
        for i in range(num_questions):
            concept = key_concepts[i % len(key_concepts)] if key_concepts else f"concept_{i+1}"
            question_template = random.choice(question_types)
            question = question_template.format(concept=concept)
            answer = f"{concept} is an important concept that..."
            
            items.append(self.create_item(question, answer))
        
        return items


class CodeGenerationBenchmarkGenerator(BenchmarkGenerator):
    """Generate code generation benchmarks"""
    
    def generate_benchmark(self, corpus_text: str, num_questions: int = 50) -> List[Dict[str, Any]]:
        items = []
        
        code_tasks = [
            "Write a function to parse the text",
            "Implement a class to represent the main concept",
            "Create a method to process the data",
            "Write code to analyze the content",
            "Implement a solution for the problem described"
        ]
        
        for i in range(num_questions):
            question = random.choice(code_tasks)
            answer = """def solution():
    # Implementation based on corpus content
    return "processed_result"
"""
            items.append(self.create_item(question, answer))
        
        return items


class DomainKnowledgeBenchmarkGenerator(BenchmarkGenerator):
    """Generate domain knowledge benchmarks"""
    
    def generate_benchmark(self, corpus_text: str, num_questions: int = 50) -> List[Dict[str, Any]]:
        items = []
        
        # Use mock question generator for variety
        mock_questions = self.llm.generate_questions(corpus_text, self.eval_type, num_questions)
        
        for mock_q in mock_questions:
            items.append(self.create_item(
                mock_q['question'],
                mock_q['answer'],
                context=corpus_text
            ))
        
        return items


class ReadingComprehensionBenchmarkGenerator(BenchmarkGenerator):
    """Generate reading comprehension benchmarks"""
    
    def generate_benchmark(self, corpus_text: str, num_questions: int = 50) -> List[Dict[str, Any]]:
        items = []
        segments = sample_corpus_segments(corpus_text, num_segments=num_questions, segment_length=300)
        
        question_types = [
            "What is the main idea?",
            "What is the significance of {concept}?",
            "How is {concept} described?",
            "What conclusion can be drawn?",
            "What are the key points?"
        ]
        
        for i, segment in enumerate(segments):
            key_concepts = extract_key_concepts(segment, max_concepts=3)
            concept = key_concepts[0] if key_concepts else "the main topic"
            
            question_template = random.choice(question_types)
            question = question_template.format(concept=concept)
            answer = f"The main point is about {concept} and its implications."
            
            items.append(self.create_item(question, answer, context=segment))
        
        return items


class BenchmarkGeneratorFactory:
    """
    Enhanced factory for creating benchmark generators with first-class agentic support.
    
    This factory serves as the single source for all generator creation and handles
    the strategy pattern for choosing between standard and agentic generation based
    on the full EvaluationConfig.
    """
    
    _generators = {
        EvaluationType.MATHEMATICAL: MathematicalBenchmarkGenerator,
        EvaluationType.FACTUAL_QA: FactualQABenchmarkGenerator,
        EvaluationType.CODE_GENERATION: CodeGenerationBenchmarkGenerator,
        EvaluationType.DOMAIN_KNOWLEDGE: DomainKnowledgeBenchmarkGenerator,
        EvaluationType.READING_COMPREHENSION: ReadingComprehensionBenchmarkGenerator,
    }
    
    # Agentic generators (all eval types supported by agentic system)
    _agentic_generators = {
        EvaluationType.DOMAIN_KNOWLEDGE: 'AgenticBenchmarkGenerator',
        EvaluationType.FACTUAL_QA: 'AgenticBenchmarkGenerator',
        EvaluationType.CODE_GENERATION: 'AgenticBenchmarkGenerator',
        EvaluationType.MATHEMATICAL: 'AgenticBenchmarkGenerator',
        EvaluationType.READING_COMPREHENSION: 'AgenticBenchmarkGenerator',
        EvaluationType.MULTIPLE_CHOICE: 'AgenticBenchmarkGenerator',
        EvaluationType.SUMMARIZATION: 'AgenticBenchmarkGenerator',
        EvaluationType.COMMONSENSE_REASONING: 'AgenticBenchmarkGenerator',
    }
    
    @classmethod
    def create_generator(cls, eval_type: EvaluationType, use_agentic: bool = False, 
                        config=None, llm_pool=None, agentic_config=None) -> BenchmarkGenerator:
        """
        Create appropriate benchmark generator for evaluation type.
        
        Args:
            eval_type: The evaluation type to generate for
            use_agentic: Whether to use agentic generation (deprecated, use config.generation.use_agentic)
            config: Full EvaluationConfig object (preferred)
            llm_pool: LLM pool for agentic generation (deprecated, use config)
            agentic_config: Agentic configuration (deprecated, use config)
            
        Returns:
            Configured benchmark generator instance
        """
        
        # Handle config-driven creation (preferred approach)
        if config is not None:
            use_agentic = config.generation.use_agentic
            # Create LLM pool from config if needed
            if use_agentic and llm_pool is None:
                llm_pool = cls._create_llm_pool_from_config(config)
        
        # Attempt agentic generation if requested and supported
        if use_agentic and eval_type in cls._agentic_generators:
            try:
                from .agentic.generator import AgenticBenchmarkGenerator
                
                # Create agentic generator with proper configuration
                # Convert EvaluationConfig to PipelineConfig if needed
                pipeline_config = agentic_config
                if config and not pipeline_config:
                    pipeline_config = cls._convert_to_pipeline_config(config, eval_type)
                
                agentic_gen = AgenticBenchmarkGenerator(
                    eval_type=eval_type, 
                    llm_pool=llm_pool, 
                    config=pipeline_config
                )
                
                # Wrap to provide consistent interface
                return AgenticGeneratorWrapper(agentic_gen, eval_type)
                
            except ImportError as e:
                # Fall back to standard generator if agentic system unavailable
                print(f"Warning: Agentic system unavailable ({e}), falling back to standard generator")
            except Exception as e:
                print(f"Warning: Failed to create agentic generator ({e}), falling back to standard generator")
        
        # Use standard generator
        generator_class = cls._generators.get(eval_type, DomainKnowledgeBenchmarkGenerator)
        return generator_class(eval_type)
    
    @classmethod
    def _convert_to_pipeline_config(cls, config, eval_type):
        """Convert EvaluationConfig to PipelineConfig for agentic generation"""
        try:
            from .agentic.models import PipelineConfig, DifficultyLevel
            
            # Map evaluation types to difficulty levels
            difficulty_map = {
                EvaluationType.MATHEMATICAL: DifficultyLevel.HARD,
                EvaluationType.CODE_GENERATION: DifficultyLevel.HARD,
                EvaluationType.FACTUAL_QA: DifficultyLevel.INTERMEDIATE,
                EvaluationType.DOMAIN_KNOWLEDGE: DifficultyLevel.HARD,
                EvaluationType.MULTIPLE_CHOICE: DifficultyLevel.INTERMEDIATE,
                EvaluationType.READING_COMPREHENSION: DifficultyLevel.INTERMEDIATE
            }
            
            difficulty = difficulty_map.get(eval_type, DifficultyLevel.HARD)
            
            # Create PipelineConfig from EvaluationConfig
            return PipelineConfig(
                difficulty=difficulty,
                num_questions=config.generation.num_questions,
                min_validation_score=0.6,
                oversample_factor=2.5,
                parallel_batch_size=3,
                max_retry_cycles=2,
                enforce_deterministic_split=True
            )
        except ImportError:
            # Return None if agentic models not available
            return None
    
    @classmethod 
    def _create_llm_pool_from_config(cls, config):
        """Create LLM pool from configuration."""
        from ..llm.mock_interface import MockLLMInterface
        
        # Check if we should use mock mode (no API key configured)
        if getattr(config.llm, 'mock_mode', False) or not config.llm.api_key:
            # Use mock LLMs for all agents
            mock_llm = MockLLMInterface(model_name="MockLLM-Demo", temperature=config.llm.temperature)
            
            return {
                'retriever': mock_llm,    # For concept mining
                'creator': mock_llm,      # For question writing
                'adversary': mock_llm,    # For adversarial enhancement
                'refiner': mock_llm       # For refinement and formatting
            }
        
        # Use real LLM interfaces when API key is available
        try:
            from ..llm.openrouter_interface import OpenRouterInterface, OpenRouterConfig
            
            # Map system LLMConfig to OpenRouterConfig
            llm_config = OpenRouterConfig(
                api_key=config.llm.api_key,
                model=config.llm.model_name if getattr(config.llm, 'model_name', None) else "openai/gpt-5-mini",
                site_url=getattr(config.llm, 'site_url', None),
                site_name=getattr(config.llm, 'app_name', None),
                max_retries=getattr(config.llm, 'max_retries', 3),
                timeout=float(getattr(config.llm, 'timeout', 60))
            )
            
            # Create specialized LLM instances for different agents
            # In a production setup, you might want different models for different roles
            retriever_llm = OpenRouterInterface(llm_config)
            creator_llm = OpenRouterInterface(llm_config) 
            adversary_llm = OpenRouterInterface(llm_config)
            refiner_llm = OpenRouterInterface(llm_config)
            
            return {
                'retriever': retriever_llm,
                'creator': creator_llm,
                'adversary': adversary_llm, 
                'refiner': refiner_llm
            }
            
        except ImportError:
            # Fall back to mock LLMs if OpenRouter interface is not available
            mock_llm = MockLLMInterface(model_name="Fallback-MockLLM", temperature=config.llm.temperature)
            
            return {
                'retriever': mock_llm,
                'creator': mock_llm,
                'adversary': mock_llm,
                'refiner': mock_llm
            }
    
    @classmethod
    def get_available_types(cls) -> List[EvaluationType]:
        """Get list of available evaluation types"""
        return list(cls._generators.keys())
    
    @classmethod
    def get_agentic_types(cls) -> List[EvaluationType]:
        """Get list of evaluation types that support agentic generation"""
        return list(cls._agentic_generators.keys())
    
    @classmethod
    def supports_agentic(cls, eval_type: EvaluationType) -> bool:
        """Check if evaluation type supports agentic generation"""
        return eval_type in cls._agentic_generators


def generate_domain_benchmark(corpus_text: str, eval_type: EvaluationType, num_questions: int = 50,
                            use_agentic: bool = False, llm_pool=None, agentic_config=None) -> Dict[str, Any]:
    """Main function to generate domain-specific benchmark"""
    
    generator = BenchmarkGeneratorFactory.create_generator(
        eval_type, 
        use_agentic=use_agentic, 
        llm_pool=llm_pool, 
        agentic_config=agentic_config
    )
    items = generator.generate_benchmark(corpus_text, num_questions)
    
    # Generate metadata
    metadata = {
        'eval_type': eval_type,
        'num_questions': len(items),
        'corpus_length': len(corpus_text),
        'generation_method': 'agentic_v2' if use_agentic else 'standard',
        'key_concepts': extract_key_concepts(corpus_text, max_concepts=10)
    }
    
    # Add agentic-specific metadata if available
    if use_agentic and hasattr(generator, 'get_generation_report'):
        try:
            agentic_report = generator.get_generation_report()
            metadata['agentic_report'] = agentic_report
        except (AttributeError, Exception):
            # Generator might not have report method or it might fail
            pass
    
    return {
        'items': items,
        'metadata': metadata,
        'eval_type': eval_type
    }


if __name__ == "__main__":
    # Test the benchmark generators
    sample_corpus = """
    Machine learning is a method of data analysis that automates analytical model building.
    It is a branch of artificial intelligence based on the idea that systems can learn from data,
    identify patterns and make decisions with minimal human intervention.
    """
    
    # Test different generators
    for eval_type in [EvaluationType.FACTUAL_QA, EvaluationType.DOMAIN_KNOWLEDGE]:
        print(f"\nTesting {eval_type} generator:")
        benchmark = generate_domain_benchmark(sample_corpus, eval_type, num_questions=3)
        
        for i, item in enumerate(benchmark['items'][:2], 1):
            print(f"  {i}. {item['question']}")
            print(f"     Answer: {item['answer'][:50]}...")
    
    print(f"\nAvailable generator types: {BenchmarkGeneratorFactory.get_available_types()}")