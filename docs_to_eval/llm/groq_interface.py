"""
Groq LLM Interface for accessing models via Groq API
Provides high-performance inference with Groq's optimized hardware
"""

import asyncio
import os
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

try:
    from groq import Groq, AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from .base import BaseLLMInterface, LLMResponse, RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class GroqConfig:
    """Configuration for Groq API"""
    api_key: Optional[str] = None
    model: str = "llama3-8b-8192"  # Default to Llama 3 8B
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    max_retries: int = 3
    timeout: float = 30.0
    
    def __post_init__(self):
        # Get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv('GROQ_API_KEY')
        
        if not self.api_key:
            logger.warning("No Groq API key provided. Set GROQ_API_KEY environment variable.")


class GroqInterface(BaseLLMInterface):
    """
    LLM interface for Groq API
    Provides access to Groq's optimized language models with high-speed inference
    """
    
    # Available Groq models with their capabilities
    AVAILABLE_MODELS = {
        "llama3-8b-8192": {
            "name": "Llama 3 8B",
            "context_window": 8192,
            "capabilities": ["reasoning", "code_generation", "factual_qa", "mathematical"],
            "speed": "very_fast"
        },
        "llama3-70b-8192": {
            "name": "Llama 3 70B", 
            "context_window": 8192,
            "capabilities": ["reasoning", "code_generation", "factual_qa", "mathematical", "creative_writing"],
            "speed": "fast"
        },
        "mixtral-8x7b-32768": {
            "name": "Mixtral 8x7B",
            "context_window": 32768,
            "capabilities": ["reasoning", "code_generation", "factual_qa", "multilingual"],
            "speed": "fast"
        },
        "gemma-7b-it": {
            "name": "Gemma 7B IT",
            "context_window": 8192,
            "capabilities": ["reasoning", "factual_qa", "mathematical"],
            "speed": "very_fast"
        }
    }
    
    def __init__(self, config: Optional[GroqConfig] = None):
        """
        Initialize Groq interface
        
        Args:
            config: Groq configuration (uses defaults if None)
        """
        if not GROQ_AVAILABLE:
            raise ImportError("groq package is required for Groq interface. Install with: pip install groq")
        
        self.config = config or GroqConfig()
        
        if not self.config.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or provide in config.")
        
        # Initialize Groq clients
        self.client = Groq(api_key=self.config.api_key)
        self.async_client = AsyncGroq(api_key=self.config.api_key)
        
        # Initialize base class
        super().__init__(
            model_name=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Performance tracking
        self.call_count = 0
        self.total_tokens_used = 0
        self.total_response_time = 0.0
        
        # Rate limiter (Groq has generous limits but still good to have)
        self.rate_limiter = RateLimiter(max_requests_per_minute=100)
        
        logger.info(f"Initialized Groq interface with model: {self.config.model}")
    
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        eval_type: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using Groq API
        
        Args:
            prompt: The prompt to send to the model
            context: Optional context to prepend to prompt
            eval_type: Type of evaluation (affects prompt formatting)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            LLMResponse with generated text and metadata
        """
        await self.rate_limiter.acquire()
        
        start_time = time.time()
        
        # Prepare the full prompt
        full_prompt = prompt
        if context:
            full_prompt = f"Context: {context}\n\n{prompt}"
        
        # Get generation parameters
        temperature = kwargs.get('temperature', self.config.temperature)
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        top_p = kwargs.get('top_p', self.config.top_p)
        
        try:
            # Format for Groq chat completion
            messages = [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
            
            # Add system message based on eval_type
            if eval_type:
                system_msg = self._get_system_message(eval_type)
                if system_msg:
                    messages.insert(0, {"role": "system", "content": system_msg})
            
            # Make API call
            completion = await self.async_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=self.config.stream
            )
            
            response_text = completion.choices[0].message.content
            
            # Update usage statistics
            end_time = time.time()
            response_time = end_time - start_time
            
            self.call_count += 1
            self.total_response_time += response_time
            
            # Extract usage information
            usage = completion.usage
            tokens_used = usage.total_tokens if usage else 0
            self.total_tokens_used += tokens_used
            
            # Record call in history
            call_record = {
                'timestamp': start_time,
                'prompt': full_prompt[:100] + '...' if len(full_prompt) > 100 else full_prompt,
                'response_length': len(response_text) if response_text else 0,
                'response_time': response_time,
                'tokens_used': tokens_used,
                'model': self.config.model,
                'eval_type': eval_type
            }
            self.call_history.append(call_record)
            
            # Create response object
            return LLMResponse(
                text=response_text or "",
                confidence=self._estimate_confidence(response_text, eval_type),
                response_time=response_time,
                metadata={
                    'model_name': self.config.model,
                    'tokens_used': tokens_used,
                    'prompt_tokens': usage.prompt_tokens if usage else 0,
                    'completion_tokens': usage.completion_tokens if usage else 0,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'top_p': top_p,
                    'call_count': self.call_count,
                    'provider': 'groq',
                    'finish_reason': completion.choices[0].finish_reason,
                    'eval_type': eval_type
                }
            )
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            logger.error(f"Groq API call failed: {e}")
            
            # Record failed call
            call_record = {
                'timestamp': start_time,
                'prompt': full_prompt[:100] + '...' if len(full_prompt) > 100 else full_prompt,
                'response_length': 0,
                'response_time': response_time,
                'tokens_used': 0,
                'model': self.config.model,
                'eval_type': eval_type,
                'error': str(e)
            }
            self.call_history.append(call_record)
            
            # Return empty response on failure
            return LLMResponse(
                text="",
                confidence=0.0,
                response_time=response_time,
                metadata={
                    'model_name': self.config.model,
                    'tokens_used': 0,
                    'error': str(e),
                    'provider': 'groq',
                    'eval_type': eval_type
                }
            )
    
    def _get_system_message(self, eval_type: str) -> Optional[str]:
        """Get system message based on evaluation type"""
        system_messages = {
            'mathematical': "You are a helpful assistant specialized in mathematics. Provide accurate, step-by-step solutions to mathematical problems. Always show your work.",
            'code_generation': "You are a helpful assistant specialized in programming. Write clean, efficient, and well-documented code. Follow best practices.",
            'factual_qa': "You are a helpful assistant specialized in providing accurate, factual information. Be precise and cite sources when possible.",
            'creative_writing': "You are a creative writing assistant. Generate engaging, original content while maintaining coherence and style.",
            'domain_knowledge': "You are a knowledgeable assistant. Provide detailed, accurate information specific to the domain being discussed.",
            'reasoning': "You are a logical reasoning assistant. Think step-by-step and provide clear explanations for your conclusions."
        }
        return system_messages.get(eval_type)
    
    def _estimate_confidence(self, response_text: str, eval_type: str) -> float:
        """Estimate confidence based on response characteristics"""
        if not response_text:
            return 0.0
        
        base_confidence = 0.8
        
        # Adjust based on response length (very short responses might be uncertain)
        if len(response_text) < 10:
            base_confidence -= 0.2
        elif len(response_text) > 100:
            base_confidence += 0.1
        
        # Adjust based on eval type
        if eval_type in ['mathematical', 'factual_qa']:
            # These should have more definitive answers
            if any(word in response_text.lower() for word in ['uncertain', 'maybe', 'might', 'possibly']):
                base_confidence -= 0.2
        elif eval_type in ['creative_writing']:
            # Creative content is inherently subjective
            base_confidence = min(base_confidence, 0.7)
        
        return max(0.0, min(1.0, base_confidence))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this interface"""
        avg_response_time = self.total_response_time / max(1, self.call_count)
        avg_tokens_per_call = self.total_tokens_used / max(1, self.call_count)
        
        model_info = self.AVAILABLE_MODELS.get(self.config.model, {})
        
        return {
            'provider': 'groq',
            'model_name': self.config.model,
            'model_info': model_info,
            'total_calls': self.call_count,
            'total_tokens_used': self.total_tokens_used,
            'total_response_time': self.total_response_time,
            'average_response_time': avg_response_time,
            'average_tokens_per_call': avg_tokens_per_call,
            'rate_limiter_stats': self.rate_limiter.get_stats(),
            'api_key_configured': bool(self.config.api_key),
            'available': GROQ_AVAILABLE and bool(self.config.api_key)
        }
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities and information"""
        model_info = self.AVAILABLE_MODELS.get(self.config.model, {})
        return {
            'model': self.config.model,
            'name': model_info.get('name', 'Unknown'),
            'context_window': model_info.get('context_window', 'Unknown'),
            'capabilities': model_info.get('capabilities', []),
            'speed': model_info.get('speed', 'Unknown'),
            'provider': 'groq'
        }
    
    @classmethod
    def list_available_models(cls) -> List[Dict[str, Any]]:
        """List all available Groq models with their information"""
        return [
            {
                'model_id': model_id,
                **model_info
            }
            for model_id, model_info in cls.AVAILABLE_MODELS.items()
        ]
    
    def switch_model(self, model_name: str):
        """Switch to a different Groq model"""
        if model_name not in self.AVAILABLE_MODELS:
            available = list(self.AVAILABLE_MODELS.keys())
            raise ValueError(f"Model {model_name} not available. Available models: {available}")
        
        self.config.model = model_name
        self.model_name = model_name
        logger.info(f"Switched to model: {model_name}")


class GroqBatchInterface:
    """
    Batch processing interface for Groq API
    Handles multiple requests efficiently with rate limiting and retry logic
    """
    
    def __init__(self, config: Optional[GroqConfig] = None, max_concurrent: int = 5):
        self.interface = GroqInterface(config)
        self.max_concurrent = max_concurrent
        self.batch_results: List[Dict[str, Any]] = []
    
    async def process_batch(
        self,
        prompts: List[str],
        contexts: Optional[List[str]] = None,
        eval_type: Optional[str] = None,
        **kwargs
    ) -> List[LLMResponse]:
        """
        Process multiple prompts in batch with concurrency control
        
        Args:
            prompts: List of prompts to process
            contexts: Optional list of contexts (same length as prompts)
            eval_type: Type of evaluation
            **kwargs: Additional parameters for generation
        
        Returns:
            List of LLMResponse objects
        """
        if contexts and len(contexts) != len(prompts):
            raise ValueError("Contexts list must have same length as prompts list")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_single(i: int, prompt: str):
            async with semaphore:
                context = contexts[i] if contexts else None
                return await self.interface.generate_response(
                    prompt=prompt,
                    context=context,
                    eval_type=eval_type,
                    **kwargs
                )
        
        # Process all prompts concurrently
        tasks = [process_single(i, prompt) for i, prompt in enumerate(prompts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions and convert to LLMResponse
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing error for prompt {i}: {result}")
                processed_results.append(LLMResponse(
                    text="",
                    confidence=0.0,
                    metadata={'error': str(result), 'batch_index': i}
                ))
            else:
                result.metadata['batch_index'] = i
                processed_results.append(result)
        
        self.batch_results.extend([r.to_dict() for r in processed_results])
        return processed_results
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get statistics for batch processing"""
        if not self.batch_results:
            return {'message': 'No batch processing completed yet'}
        
        successful = [r for r in self.batch_results if not r.get('metadata', {}).get('error')]
        failed = [r for r in self.batch_results if r.get('metadata', {}).get('error')]
        
        response_times = [
            r.get('response_time', 0) 
            for r in self.batch_results 
            if r.get('response_time', 0) > 0
        ]
        
        return {
            'total_processed': len(self.batch_results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(self.batch_results) * 100 if self.batch_results else 0,
            'average_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'max_concurrent': self.max_concurrent,
            'interface_stats': self.interface.get_performance_stats()
        }


# Convenience functions
async def test_groq_connection(api_key: Optional[str] = None, model: str = "llama3-8b-8192") -> Dict[str, Any]:
    """
    Test Groq API connection and model availability
    
    Args:
        api_key: API key to test (uses environment variable if None)
        model: Model to test
    
    Returns:
        Test results dictionary
    """
    test_results = {
        'connection_successful': False,
        'model_available': False,
        'response_generated': False,
        'error': None,
        'model_info': {},
        'sample_response': None,
        'response_time': 0.0
    }
    
    try:
        config = GroqConfig(api_key=api_key, model=model)
        interface = GroqInterface(config)
        
        test_results['model_info'] = interface.get_model_capabilities()
        test_results['connection_successful'] = True
        
        # Test simple generation
        response = await interface.generate_response(
            prompt="What is the capital of France? Answer in one sentence.",
            eval_type="factual_qa"
        )
        
        if response.text and len(response.text.strip()) > 0:
            test_results['response_generated'] = True
            test_results['sample_response'] = response.text.strip()
            test_results['model_available'] = True
            test_results['response_time'] = response.response_time or 0.0
        
    except Exception as e:
        test_results['error'] = str(e)
        logger.error(f"Groq connection test failed: {e}")
    
    return test_results


def create_groq_interface(
    model: str = "llama3-8b-8192",
    api_key: Optional[str] = None,
    **config_kwargs
) -> GroqInterface:
    """
    Create a Groq interface with specified configuration
    
    Args:
        model: Groq model to use
        api_key: API key (uses environment variable if None)
        **config_kwargs: Additional configuration parameters
    
    Returns:
        Configured GroqInterface
    """
    config = GroqConfig(
        model=model,
        api_key=api_key,
        **config_kwargs
    )
    return GroqInterface(config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        """Demo Groq interface functionality"""
        print("🚀 Testing Groq LLM Interface...")
        
        # Test connection
        test_results = await test_groq_connection()
        print(f"Connection successful: {test_results['connection_successful']}")
        print(f"Model available: {test_results['model_available']}")
        print(f"Response time: {test_results['response_time']:.2f}s")
        
        if test_results['sample_response']:
            print(f"Sample response: {test_results['sample_response']}")
        
        if test_results['error']:
            print(f"Error: {test_results['error']}")
            return
        
        # Test different evaluation types
        interface = create_groq_interface()
        
        test_cases = [
            ("What is 25 * 4?", "mathematical"),
            ("Write a Python function to reverse a string", "code_generation"),
            ("What is the largest planet in our solar system?", "factual_qa"),
            ("Write a short poem about the ocean", "creative_writing")
        ]
        
        print("\n📝 Testing different evaluation types...")
        for prompt, eval_type in test_cases:
            response = await interface.generate_response(prompt, eval_type=eval_type)
            print(f"\n{eval_type.title()}: {prompt}")
            print(f"Response: {response.text[:100]}..." if len(response.text) > 100 else f"Response: {response.text}")
            print(f"Confidence: {response.confidence:.2f}")
        
        # Test batch processing
        print("\n🔄 Testing batch processing...")
        batch_interface = GroqBatchInterface(max_concurrent=3)
        batch_prompts = [
            "What is AI?",
            "Explain machine learning",
            "What is deep learning?",
            "How do neural networks work?",
            "What is NLP?"
        ]
        
        batch_results = await batch_interface.process_batch(
            batch_prompts,
            eval_type="factual_qa"
        )
        
        print(f"Processed {len(batch_results)} prompts in batch")
        batch_stats = batch_interface.get_batch_stats()
        print(f"Success rate: {batch_stats['success_rate']:.1f}%")
        print(f"Average response time: {batch_stats['average_response_time']:.2f}s")
        
        # Show performance stats
        print("\n📊 Performance Statistics:")
        stats = interface.get_performance_stats()
        for key, value in stats.items():
            if key != 'rate_limiter_stats':
                print(f"  {key}: {value}")
        
        print("\n✅ Demo completed!")
    
    # Run demo
    asyncio.run(demo())
