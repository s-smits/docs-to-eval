"""
Mock LLM interface for testing the evaluation pipeline
Simulates various LLM behaviors and capabilities for comprehensive testing
"""

import json
import re
import random
import time
import asyncio
from functools import reduce
from collections import defaultdict
from typing import Dict, Any, Optional, List

from .base import BaseLLMInterface, LLMResponse, LLMCapability


class MockLLMInterface(BaseLLMInterface):
    """Main mock LLM interface with configurable behaviors"""
    
    def __init__(self, model_name: str = "MockLLM-v1", temperature: float = 0.7, 
                 capabilities: Optional[Dict[str, float]] = None):
        super().__init__(model_name, temperature)
        
        # Set default capabilities
        default_capabilities = {
            LLMCapability.MATHEMATICAL: 0.8,
            LLMCapability.CODE_GENERATION: 0.7,
            LLMCapability.FACTUAL_QA: 0.9,
            LLMCapability.CREATIVE_WRITING: 0.6,
            LLMCapability.REASONING: 0.7,
            LLMCapability.MULTILINGUAL: 0.5
        }
        
        if capabilities:
            # Convert string keys to enum keys
            for key, value in capabilities.items():
                try:
                    capability = LLMCapability(key)
                    self.capabilities[capability] = value
                except ValueError:
                    continue
        else:
            self.capabilities = default_capabilities
        
        self.response_patterns = self._initialize_response_patterns()
        self.performance_stats = defaultdict(int)
    
    def _initialize_response_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize response patterns for different question types"""
        return {
            'mathematical': {
                'patterns': [
                    r'calculate|compute|solve|find|what is \d+',
                    r'equation|formula|algebra|geometry',
                    r'\d+[\+\-\*\/]\d+',
                ],
                'response_generator': self._generate_math_response
            },
            'code_generation': {
                'patterns': [
                    r'write.*function|implement|code|program',
                    r'python|javascript|java|c\+\+',
                    r'algorithm|data structure',
                ],
                'response_generator': self._generate_code_response
            },
            'factual_qa': {
                'patterns': [
                    r'what is|who is|when did|where is',
                    r'define|explain|describe',
                    r'capital|president|inventor',
                ],
                'response_generator': self._generate_factual_response
            },
            'creative_writing': {
                'patterns': [
                    r'write.*story|create.*poem|compose',
                    r'creative|imaginative|fictional',
                    r'once upon a time|in a world',
                ],
                'response_generator': self._generate_creative_response
            },
            'reading_comprehension': {
                'patterns': [
                    r'according to.*passage|based on.*text',
                    r'the passage states|the text mentions',
                    r'summarize|main idea',
                ],
                'response_generator': self._generate_comprehension_response
            }
        }
    
    async def generate_response(self, prompt: str, context: Optional[str] = None, 
                              eval_type: Optional[str] = None, **kwargs) -> LLMResponse:
        """Main response generation method"""
        
        start_time = time.time()
        
        # Simulate network delay
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Determine question type
        question_type = self._classify_question_type(prompt, eval_type)
        
        # Log the call
        self.call_history.append({
            'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
            'context': context[:50] + "..." if context and len(context) > 50 else context,
            'question_type': question_type,
            'eval_type': eval_type,
            'timestamp': time.time()
        })
        
        # Generate response based on type
        if question_type in self.response_patterns:
            response_generator = self.response_patterns[question_type]['response_generator']
            response_data = response_generator(prompt, context)
        else:
            response_data = self._generate_default_response(prompt, context)
        
        # Add noise based on temperature and model capability
        if self.temperature > 0:
            response_data = self._add_response_noise(response_data, question_type)
        
        # Update performance stats
        self.performance_stats[question_type] += 1
        self.performance_stats['total_calls'] += 1
        
        response_time = time.time() - start_time
        
        return LLMResponse(
            text=response_data.get('text', ''),
            confidence=response_data.get('confidence', 0.5),
            reasoning_steps=response_data.get('reasoning_steps', []),
            metadata=response_data.get('metadata', {}),
            response_time=response_time
        )
    
    def _classify_question_type(self, prompt: str, eval_type: Optional[str] = None) -> str:
        """Classify the type of question being asked"""
        
        if eval_type:
            # Use provided eval_type as hint
            if eval_type in self.response_patterns:
                return eval_type
        
        # Pattern matching classification
        prompt_lower = prompt.lower()
        
        for question_type, config in self.response_patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, prompt_lower):
                    return question_type
        
        return 'general'
    
    def _generate_math_response(self, prompt: str, context: Optional[str]) -> Dict[str, Any]:
        """Generate mathematical responses"""
        
        # Extract mathematical expressions
        numbers = re.findall(r'-?\d+\.?\d*', prompt)
        operations = re.findall(r'[\+\-\*\/]', prompt)
        
        reasoning_steps = []
        
        if numbers and len(numbers) >= 2 and operations:
            # Simple arithmetic
            try:
                num1, num2 = float(numbers[0]), float(numbers[1])
                op = operations[0]
                
                reasoning_steps.append(f"Identify the numbers: {num1} and {num2}")
                reasoning_steps.append(f"Identify the operation: {op}")
                
                if op == '+':
                    result = num1 + num2
                    reasoning_steps.append(f"Calculate: {num1} + {num2} = {result}")
                elif op == '-':
                    result = num1 - num2
                    reasoning_steps.append(f"Calculate: {num1} - {num2} = {result}")
                elif op == '*':
                    result = num1 * num2
                    reasoning_steps.append(f"Calculate: {num1} × {num2} = {result}")
                elif op == '/':
                    result = num1 / num2 if num2 != 0 else float('inf')
                    reasoning_steps.append(f"Calculate: {num1} ÷ {num2} = {result}")
                else:
                    result = 42  # Default answer
                
                response_text = f"The answer is {result}."
                confidence = self.get_capability_score(LLMCapability.MATHEMATICAL)
                
            except (ValueError, IndexError, ZeroDivisionError):
                response_text = "I need to solve this step by step. The answer is approximately 42."
                confidence = 0.5
                reasoning_steps = ["Unable to parse the exact mathematical expression"]
        
        elif 'equation' in prompt.lower() or 'solve' in prompt.lower():
            # Algebraic problems
            reasoning_steps = [
                "This is an algebraic equation",
                "I need to isolate the variable", 
                "Applying algebraic manipulation"
            ]
            response_text = "To solve this equation, I would isolate the variable by performing inverse operations on both sides. The solution is x = 5."
            confidence = 0.7
        
        else:
            # General math question
            response_text = "This is a mathematical problem that requires systematic solution. Let me work through it step by step."
            confidence = 0.6
            reasoning_steps = ["Analyzing the mathematical problem", "Applying relevant mathematical principles"]
        
        return {
            'text': response_text,
            'confidence': confidence,
            'reasoning_steps': reasoning_steps,
            'metadata': {'question_type': 'mathematical', 'numbers_found': len(numbers)}
        }
    
    def _generate_code_response(self, prompt: str, context: Optional[str]) -> Dict[str, Any]:
        """Generate code responses"""
        
        reasoning_steps = [
            "Analyzing the programming task",
            "Identifying the required algorithm or data structure",
            "Implementing the solution"
        ]
        
        # Detect programming language
        language = 'python'  # default
        if 'javascript' in prompt.lower() or 'js' in prompt.lower():
            language = 'javascript'
        elif 'java' in prompt.lower():
            language = 'java'
        elif 'c++' in prompt.lower() or 'cpp' in prompt.lower():
            language = 'c++'
        
        # Generate code based on task
        if 'function' in prompt.lower() and 'reverse' in prompt.lower():
            if language == 'python':
                code = """def reverse_string(s):
    return s[::-1]

# Example usage
result = reverse_string("hello")
print(result)  # Output: "olleh" """
            else:
                code = "// Code implementation for string reversal"
        
        elif 'sort' in prompt.lower():
            if language == 'python':
                code = """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr"""
            else:
                code = "// Code implementation for sorting algorithm"
        
        elif 'binary search' in prompt.lower():
            if language == 'python':
                code = """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1"""
            else:
                code = "// Code implementation for binary search"
        
        else:
            # Generic function template
            if language == 'python':
                code = """def solution():
    # Implementation goes here
    pass

# Example usage
result = solution()
print(result)"""
            else:
                code = "// Generic code implementation"
        
        confidence = self.get_capability_score(LLMCapability.CODE_GENERATION)
        
        return {
            'text': code,
            'confidence': confidence,
            'reasoning_steps': reasoning_steps,
            'metadata': {'language': language, 'question_type': 'code_generation'}
        }
    
    def _generate_factual_response(self, prompt: str, context: Optional[str]) -> Dict[str, Any]:
        """Generate factual responses"""
        
        reasoning_steps = [
            "Identifying the factual question type",
            "Recalling relevant information",
            "Providing accurate answer"
        ]
        
        # Mock factual database
        facts = {
            'paris': "Paris is the capital city of France, known for the Eiffel Tower and rich cultural heritage.",
            'python': "Python is a high-level programming language known for its simplicity and readability.",
            'einstein': "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
            'dna': "DNA (Deoxyribonucleic acid) is the molecule that carries genetic information in living organisms.",
            'machine learning': "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
        }
        
        # Extract key terms from prompt
        prompt_lower = prompt.lower()
        matched_fact = None
        
        for key, fact in facts.items():
            if key in prompt_lower:
                matched_fact = fact
                break
        
        if matched_fact:
            response_text = matched_fact
            confidence = self.get_capability_score(LLMCapability.FACTUAL_QA)
        else:
            # Generate generic factual response
            if 'what is' in prompt_lower:
                response_text = "This is an important concept in its respective domain with significant applications and implications."
            elif 'who is' in prompt_lower or 'who was' in prompt_lower:
                response_text = "This person was a notable figure who made significant contributions to their field."
            elif 'when' in prompt_lower:
                response_text = "This event occurred in the 20th century and had important historical significance."
            elif 'where' in prompt_lower:
                response_text = "This location is situated in a geographically significant area."
            else:
                response_text = "Based on available information, this is a well-established fact in the relevant domain."
            
            confidence = 0.6
        
        return {
            'text': response_text,
            'confidence': confidence,
            'reasoning_steps': reasoning_steps,
            'metadata': {'question_type': 'factual_qa', 'matched_database': matched_fact is not None}
        }
    
    def _generate_creative_response(self, prompt: str, context: Optional[str]) -> Dict[str, Any]:
        """Generate creative writing responses"""
        
        reasoning_steps = [
            "Analyzing the creative prompt",
            "Generating imaginative content",
            "Ensuring narrative coherence"
        ]
        
        if 'story' in prompt.lower():
            response_text = """Once upon a time, in a world where technology and nature coexisted in perfect harmony, 
there lived a young inventor named Alex. Every morning, Alex would wake up to the sound of mechanical 
birds singing outside their window, their metallic wings catching the first rays of sunlight. 

The day Alex discovered the secret of merging artificial intelligence with natural ecosystems was 
the day everything changed. It was a discovery that would reshape their understanding of what it 
meant to be truly alive in a world where the boundaries between organic and synthetic had blurred 
beyond recognition."""
        
        elif 'poem' in prompt.lower():
            response_text = """In circuits deep and algorithms bright,
Where data flows like rivers in the night,
The mind of silicon dreams of human thought,
While wisdom in the code is gently wrought.

Between the ones and zeros lies a space,
Where artificial meets the human grace,
And in this dance of logic, dreams take flight,
Illuminating futures shining bright."""
        
        else:
            response_text = """The creative process begins with a spark of imagination, weaving together 
elements of the known and unknown. In this particular creative endeavor, we explore themes of 
innovation, human connection, and the endless possibilities that emerge when we dare to dream 
beyond conventional boundaries."""
        
        confidence = self.get_capability_score(LLMCapability.CREATIVE_WRITING)
        
        return {
            'text': response_text,
            'confidence': confidence,
            'reasoning_steps': reasoning_steps,
            'metadata': {'question_type': 'creative_writing', 'word_count': len(response_text.split())}
        }
    
    def _generate_comprehension_response(self, prompt: str, context: Optional[str]) -> Dict[str, Any]:
        """Generate reading comprehension responses"""
        
        reasoning_steps = [
            "Reading and analyzing the provided text",
            "Identifying key information relevant to the question",
            "Formulating a comprehensive answer"
        ]
        
        if context:
            # Extract key concepts from context
            sentences = context.split('.')
            key_sentence = sentences[0] if sentences else context
            
            if 'summarize' in prompt.lower():
                response_text = f"The main idea of the passage is that {key_sentence.strip().lower()}. The text provides important context and details about this topic."
            elif 'main idea' in prompt.lower():
                response_text = f"The central theme focuses on {key_sentence.strip().lower()}."
            else:
                response_text = f"According to the passage, {key_sentence.strip().lower()}. This information is directly stated in the provided text."
        else:
            response_text = "Based on the reading material, the answer requires careful analysis of the provided information and context."
        
        confidence = 0.8
        
        return {
            'text': response_text,
            'confidence': confidence,
            'reasoning_steps': reasoning_steps,
            'metadata': {'question_type': 'reading_comprehension', 'has_context': context is not None}
        }
    
    def _generate_default_response(self, prompt: str, context: Optional[str]) -> Dict[str, Any]:
        """Generate default response for unclassified questions"""
        
        reasoning_steps = [
            "Analyzing the general question",
            "Drawing from broad knowledge base",
            "Providing helpful response"
        ]
        
        response_text = """This is an interesting question that requires thoughtful consideration. 
Based on the available information and general knowledge, I can provide some insights. 
The topic involves multiple factors and considerations that are important to understand 
in the broader context."""
        
        confidence = 0.5
        
        return {
            'text': response_text,
            'confidence': confidence,
            'reasoning_steps': reasoning_steps,
            'metadata': {'question_type': 'general'}
        }
    
    def _add_response_noise(self, response_data: Dict[str, Any], question_type: str) -> Dict[str, Any]:
        """Add realistic noise/variation to responses based on temperature"""
        
        if self.temperature <= 0.1:
            return response_data  # Deterministic
        
        # Reduce confidence slightly for higher temperatures
        noise_factor = self.temperature * 0.2
        response_data['confidence'] = max(0.1, response_data['confidence'] - noise_factor)
        
        # Add slight variations to text for non-deterministic types
        if question_type not in ['mathematical', 'code_generation'] and self.temperature > 0.5:
            variations = [
                "I believe that ",
                "It appears that ",
                "Based on my analysis, ",
                "From what I understand, "
            ]
            
            if random.random() < self.temperature:
                prefix = random.choice(variations)
                response_data['text'] = prefix + response_data['text'].lower()
        
        return response_data
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return dict(self.performance_stats)
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.performance_stats.clear()
        super().reset_stats()


class MockLLMEvaluator:
    """Mock LLM specifically designed for evaluation tasks"""
    
    def __init__(self, base_llm: Optional[MockLLMInterface] = None):
        self.base_llm = base_llm or MockLLMInterface()
        self.evaluation_behaviors = {
            'mathematical': self._evaluate_math_behavior,
            'factual_qa': self._evaluate_factual_behavior,
            'code_generation': self._evaluate_code_behavior,
            'creative_writing': self._evaluate_creative_behavior
        }
    
    async def evaluate_on_benchmark(self, benchmark_items: List[Dict[str, Any]], eval_type: str) -> List[Dict[str, Any]]:
        """Evaluate the mock LLM on a benchmark"""
        results = []
        
        for item in benchmark_items:
            question = item['question']
            ground_truth = item['answer']
            context = item.get('context')
            
            # Get LLM response
            response = await self.base_llm.generate_response(
                prompt=question,
                context=context,
                eval_type=eval_type
            )
            
            # Apply evaluation-specific behavior
            if eval_type in self.evaluation_behaviors:
                response = self.evaluation_behaviors[eval_type](response, item)
            
            results.append({
                'question': question,
                'ground_truth': ground_truth,
                'prediction': response.text,
                'confidence': response.confidence,
                'reasoning_steps': response.reasoning_steps,
                'response_metadata': response.metadata
            })
        
        return results
    
    def _evaluate_math_behavior(self, response: LLMResponse, item: Dict[str, Any]) -> LLMResponse:
        """Modify response for math evaluation behavior"""
        # Math should be more deterministic and precise
        response.confidence = min(0.95, response.confidence + 0.1)
        return response
    
    def _evaluate_factual_behavior(self, response: LLMResponse, item: Dict[str, Any]) -> LLMResponse:
        """Modify response for factual evaluation behavior"""
        # Factual responses should be confident but concise
        response.confidence = min(0.9, response.confidence + 0.05)
        return response
    
    def _evaluate_code_behavior(self, response: LLMResponse, item: Dict[str, Any]) -> LLMResponse:
        """Modify response for code evaluation behavior"""
        # Code should include proper syntax
        if 'def ' not in response.text and 'function' in item['question']:
            response.text = f"def solution():\n    {response.text}\n    return result"
        return response
    
    def _evaluate_creative_behavior(self, response: LLMResponse, item: Dict[str, Any]) -> LLMResponse:
        """Modify response for creative evaluation behavior"""
        # Creative responses should be longer and more varied
        response.confidence = max(0.4, response.confidence - 0.1)  # More uncertainty
        return response


if __name__ == "__main__":
    import asyncio
    
    async def test_mock_llm():
        # Test the mock LLM interface
        llm = MockLLMInterface(temperature=0.3)
        
        test_prompts = [
            ("What is 2 + 2?", "mathematical"),
            ("Write a function to reverse a string", "code_generation"),
            ("What is the capital of France?", "factual_qa"),
            ("Write a short story about AI", "creative_writing"),
            ("Summarize the main points of machine learning", "reading_comprehension")
        ]
        
        print("Testing Mock LLM Interface:")
        print("=" * 50)
        
        for prompt, expected_type in test_prompts:
            print(f"\nPrompt: {prompt}")
            print(f"Expected Type: {expected_type}")
            
            response = await llm.generate_response(prompt, eval_type=expected_type)
            
            print(f"Response: {response.text[:100]}...")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Reasoning Steps: {len(response.reasoning_steps)}")
        
        print(f"\nPerformance Stats: {llm.get_performance_stats()}")
        
        # Test evaluator
        print(f"\n" + "=" * 50)
        print("Testing Mock LLM Evaluator:")
        
        evaluator = MockLLMEvaluator(llm)
        
        sample_benchmark = [
            {'question': 'What is 5 * 3?', 'answer': '15', 'eval_type': 'mathematical'},
            {'question': 'What is Python?', 'answer': 'A programming language', 'eval_type': 'factual_qa'}
        ]
        
        results = await evaluator.evaluate_on_benchmark(sample_benchmark, 'mathematical')
        
        for result in results:
            print(f"\nQuestion: {result['question']}")
            print(f"Ground Truth: {result['ground_truth']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2f}")
    
    asyncio.run(test_mock_llm())