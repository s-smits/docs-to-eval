"""
Direct Gemini SDK Interface for accessing Google's Gemini models
Provides high-performance access to Gemini Pro, Flash, and other variants
"""

import asyncio
import os
import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import json

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerateContentResponse
    try:
        from google.generativeai.types import GenerationConfig  # type: ignore
    except (ImportError, AttributeError):  # Older SDK versions
        GenerationConfig = None  # type: ignore[assignment]
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None  # type: ignore[assignment]
    GenerateContentResponse = None  # type: ignore[assignment]
    GenerationConfig = None  # type: ignore[assignment]

from .base import BaseLLMInterface, LLMResponse, RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class GeminiConfig:
    """Configuration for Gemini API using official SDK"""
    api_key: Optional[str] = None
    model: str = "gemini-2.5-flash"  # Default to Flash for speed
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 40
    max_output_tokens: int = 1024
    response_mime_type: str = "text/plain"
    candidate_count: int = 1
    stop_sequences: List[str] = field(default_factory=list)
    safety_settings: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')

        if not self.api_key:
            logger.warning("No Gemini API key provided. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")

        # Default safety settings (permissive for evaluation tasks)
        if self.safety_settings is None:
            self.safety_settings = {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
            }


class GeminiSDKInterface(BaseLLMInterface):
    """
    Direct Gemini SDK interface for Google's Gemini models
    Provides access to Gemini Pro, Flash, and other model variants
    """

    # Available Gemini models with their capabilities
    AVAILABLE_MODELS = {
        "gemini-2.5-pro": {
            "name": "gemini-2.5-pro",
            "context_window": 2097152,  # ~2M tokens
            "capabilities": ["reasoning", "code_generation", "factual_qa", "mathematical", "creative_writing", "multilingual", "long_context"],
            "speed": "medium",
            "cost": "medium"
        },
        "gemini-2.5-flash": {
            "name": "gemini-2.5-flash",
            "context_window": 1048576,  # ~1M tokens
            "capabilities": ["reasoning", "code_generation", "factual_qa", "mathematical", "creative_writing", "multilingual"],
            "speed": "fast",
            "cost": "low"
        }
    }

    def __init__(self, config: Optional[GeminiConfig] = None):
        """
        Initialize Gemini SDK interface

        Args:
            config: Gemini configuration (uses defaults if None)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package is required for Gemini interface. Install with: pip install google-generativeai")

        self.config = config or GeminiConfig()

        if not self.config.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable or provide in config.")

        # Configure the Gemini SDK
        genai.configure(api_key=self.config.api_key)

        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=self.config.model,
            generation_config=self._create_generation_config(),
            safety_settings=self._create_safety_settings()
        )

        # Initialize base class
        super().__init__(
            model_name=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens
        )

        # Performance tracking
        self.call_count = 0
        self.total_tokens_used = 0
        self.total_response_time = 0.0

        # Rate limiter for Gemini API
        self.rate_limiter = RateLimiter(max_requests_per_minute=60)

        logger.info(f"Initialized Gemini SDK interface with model: {self.config.model}")

    def _create_generation_config(self) -> Union[Dict[str, Any], Any]:
        """Create generation config for Gemini"""
        config_kwargs = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "max_output_tokens": self.config.max_output_tokens,
            "response_mime_type": self.config.response_mime_type,
            "candidate_count": self.config.candidate_count,
            "stop_sequences": self.config.stop_sequences if self.config.stop_sequences else None
        }

        if GenerationConfig is None:
            # Older SDKs accept plain dictionaries
            return config_kwargs

        return GenerationConfig(**config_kwargs)

    def _create_safety_settings(self) -> Optional[Dict[str, str]]:
        """Create safety settings for Gemini"""
        if not self.config.safety_settings:
            return None

        # Convert to Gemini SDK format
        safety_settings = []
        for category, threshold in self.config.safety_settings.items():
            safety_settings.append({
                "category": category,
                "threshold": threshold
            })
        return safety_settings

    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        eval_type: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using Gemini SDK

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
            full_prompt = f"Context: {context}\\n\\n{prompt}"

        # Add system instruction based on eval_type
        if eval_type:
            system_instruction = self._get_system_instruction(eval_type)
            if system_instruction:
                full_prompt = f"{system_instruction}\\n\\n{full_prompt}"

        # Override config with kwargs if provided
        generation_config = self._create_generation_config()

        def _set_config_attr(cfg, key, value):
            if GenerationConfig is None or isinstance(cfg, dict):
                cfg[key] = value
            else:
                setattr(cfg, key, value)

        if 'temperature' in kwargs:
            _set_config_attr(generation_config, "temperature", kwargs['temperature'])
        if 'max_tokens' in kwargs or 'max_output_tokens' in kwargs:
            new_max = kwargs.get('max_tokens', kwargs.get('max_output_tokens'))
            if new_max is not None:
                _set_config_attr(generation_config, "max_output_tokens", new_max)
        if 'top_p' in kwargs:
            _set_config_attr(generation_config, "top_p", kwargs['top_p'])
        if 'top_k' in kwargs:
            _set_config_attr(generation_config, "top_k", kwargs['top_k'])

        try:
            # Generate content using Gemini SDK
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                    stream=False
                )
            )

            # Extract response text
            response_text = ""
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    response_text = "".join(part.text for part in candidate.content.parts)

            # Update usage statistics
            end_time = time.time()
            response_time = end_time - start_time

            self.call_count += 1
            self.total_response_time += response_time

            # Extract usage information (if available)
            tokens_used = 0
            prompt_tokens = 0
            completion_tokens = 0

            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens_used = response.usage_metadata.total_token_count or 0
                prompt_tokens = response.usage_metadata.prompt_token_count or 0
                completion_tokens = response.usage_metadata.candidates_token_count or 0
            else:
                # Estimate tokens if usage metadata not available
                tokens_used = self._estimate_tokens(full_prompt, response_text)
                prompt_tokens = self._estimate_tokens(full_prompt)
                completion_tokens = self._estimate_tokens(response_text)

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

            # Extract additional metadata
            finish_reason = None
            safety_ratings = []
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason.name if hasattr(candidate, 'finish_reason') else None
                if hasattr(candidate, 'safety_ratings'):
                    safety_ratings = [
                        {
                            'category': rating.category.name,
                            'probability': rating.probability.name
                        }
                        for rating in candidate.safety_ratings
                    ]

            # Create response object
            return LLMResponse(
                text=response_text or "",
                confidence=self._estimate_confidence(response_text, eval_type, finish_reason),
                response_time=response_time,
                metadata={
                    'model_name': self.config.model,
                    'tokens_used': tokens_used,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'temperature': generation_config.temperature,
                    'max_output_tokens': generation_config.max_output_tokens,
                    'top_p': generation_config.top_p,
                    'top_k': generation_config.top_k,
                    'call_count': self.call_count,
                    'provider': 'gemini_sdk',
                    'finish_reason': finish_reason,
                    'safety_ratings': safety_ratings,
                    'eval_type': eval_type
                }
            )

        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time

            logger.error(f"Gemini SDK API call failed: {e}")

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
                    'provider': 'gemini_sdk',
                    'eval_type': eval_type
                }
            )

    def _get_system_instruction(self, eval_type: str) -> Optional[str]:
        """Get system instruction based on evaluation type"""
        system_instructions = {
            'mathematical': "You are a mathematics expert. Provide precise, step-by-step solutions to mathematical problems. Show all work and calculations clearly.",
            'code_generation': "You are a programming expert. Write clean, efficient, well-documented code following best practices. Include comments explaining your approach.",
            'factual_qa': "You are a knowledgeable assistant focused on providing accurate, factual information. Be precise and comprehensive in your answers.",
            'creative_writing': "You are a creative writing expert. Generate engaging, original content with good narrative flow, character development, and literary quality.",
            'domain_knowledge': "You are a domain expert. Provide detailed, accurate information specific to the subject matter being discussed. Use appropriate terminology and depth.",
            'reasoning': "You are a logical reasoning expert. Think step-by-step, show your reasoning process, and arrive at well-justified conclusions.",
            'summarization': "You are a summarization expert. Create concise, comprehensive summaries that capture the key points and essential information.",
            'translation': "You are a translation expert. Provide accurate translations that preserve meaning, tone, and cultural context."
        }
        return system_instructions.get(eval_type)

    def _estimate_tokens(self, text: str, additional_text: str = "") -> int:
        """Estimate token count for text (rough approximation)"""
        # Rough estimation: ~0.75 tokens per word for English
        combined_text = text + " " + additional_text if additional_text else text
        word_count = len(combined_text.split())
        return int(word_count * 0.75)

    def _estimate_confidence(self, response_text: str, eval_type: str, finish_reason: str = None) -> float:
        """Estimate confidence based on response characteristics"""
        if not response_text:
            return 0.0

        base_confidence = 0.8

        # Adjust based on finish reason
        if finish_reason == "STOP":
            # Normal completion
            base_confidence = 0.85
        elif finish_reason in ["MAX_TOKENS", "LENGTH"]:
            # Truncated response
            base_confidence = 0.7
        elif finish_reason in ["SAFETY", "RECITATION"]:
            # Filtered response
            base_confidence = 0.3

        # Adjust based on response length
        if len(response_text) < 10:
            base_confidence -= 0.2
        elif len(response_text) > 100:
            base_confidence += 0.1

        # Adjust based on eval type
        if eval_type in ['mathematical', 'factual_qa']:
            # These should have more definitive answers
            if any(word in response_text.lower() for word in ['uncertain', 'maybe', 'might', 'possibly', 'not sure']):
                base_confidence -= 0.2
        elif eval_type in ['creative_writing']:
            # Creative content is inherently subjective
            base_confidence = min(base_confidence, 0.75)

        return max(0.0, min(1.0, base_confidence))

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this interface"""
        avg_response_time = self.total_response_time / max(1, self.call_count)
        avg_tokens_per_call = self.total_tokens_used / max(1, self.call_count)

        model_info = self.AVAILABLE_MODELS.get(self.config.model, {})

        return {
            'provider': 'gemini_sdk',
            'model_name': self.config.model,
            'model_info': model_info,
            'total_calls': self.call_count,
            'total_tokens_used': self.total_tokens_used,
            'total_response_time': self.total_response_time,
            'average_response_time': avg_response_time,
            'average_tokens_per_call': avg_tokens_per_call,
            'rate_limiter_stats': self.rate_limiter.get_stats(),
            'api_key_configured': bool(self.config.api_key),
            'available': GEMINI_AVAILABLE and bool(self.config.api_key)
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
            'cost': model_info.get('cost', 'Unknown'),
            'provider': 'gemini_sdk'
        }

    @classmethod
    def list_available_models(cls) -> List[Dict[str, Any]]:
        """List all available Gemini models with their information"""
        return [
            {
                'model_id': model_id,
                **model_info
            }
            for model_id, model_info in cls.AVAILABLE_MODELS.items()
        ]

    def switch_model(self, model_name: str):
        """Switch to a different Gemini model"""
        if model_name not in self.AVAILABLE_MODELS:
            available = list(self.AVAILABLE_MODELS.keys())
            raise ValueError(f"Model {model_name} not available. Available models: {available}")

        self.config.model = model_name
        self.model_name = model_name

        # Reinitialize the model with new configuration
        self.model = genai.GenerativeModel(
            model_name=self.config.model,
            generation_config=self._create_generation_config(),
            safety_settings=self._create_safety_settings()
        )

        logger.info(f"Switched to model: {model_name}")

    def update_safety_settings(self, safety_settings: Dict[str, str]):
        """Update safety settings for the model"""
        self.config.safety_settings = safety_settings

        # Reinitialize the model with new safety settings
        self.model = genai.GenerativeModel(
            model_name=self.config.model,
            generation_config=self._create_generation_config(),
            safety_settings=self._create_safety_settings()
        )

        logger.info("Updated safety settings")


class GeminiBatchInterface:
    """
    Batch processing interface for Gemini SDK
    Handles multiple requests efficiently with rate limiting and retry logic
    """

    def __init__(self, config: Optional[GeminiConfig] = None, max_concurrent: int = 5):
        self.interface = GeminiSDKInterface(config)
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
async def test_gemini_connection(
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash"
) -> Dict[str, Any]:
    """
    Test Gemini SDK connection and model availability

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
        config = GeminiConfig(api_key=api_key, model=model)
        interface = GeminiSDKInterface(config)

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
        logger.error(f"Gemini connection test failed: {e}")

    return test_results


def create_gemini_interface(
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    **config_kwargs
) -> GeminiSDKInterface:
    """
    Create a Gemini interface with specified configuration

    Args:
        model: Gemini model to use
        api_key: API key (uses environment variable if None)
        **config_kwargs: Additional configuration parameters

    Returns:
        Configured GeminiSDKInterface
    """
    config = GeminiConfig(
        model=model,
        api_key=api_key,
        **config_kwargs
    )
    return GeminiSDKInterface(config)


def list_gemini_models() -> List[str]:
    """List all available Gemini models"""
    if not GEMINI_AVAILABLE:
        return []

    # Always return our supported models instead of calling the live API
    # This ensures consistency and avoids returning unsupported models
    return list(GeminiSDKInterface.AVAILABLE_MODELS.keys())


    try:
        # This would require API key to be configured
        models = genai.list_models()
        return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
    except:
        # Return our known models if API call fails
        return list(GeminiSDKInterface.AVAILABLE_MODELS.keys())


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demo Gemini SDK interface functionality"""
        print("🚀 Testing Gemini SDK LLM Interface...")

        # Test connection
        test_results = await test_gemini_connection()
        print(f"Connection successful: {test_results['connection_successful']}")
        print(f"Model available: {test_results['model_available']}")
        print(f"Response time: {test_results['response_time']:.2f}s")

        if test_results['sample_response']:
            print(f"Sample response: {test_results['sample_response']}")

        if test_results['error']:
            print(f"Error: {test_results['error']}")
            return

        # Test different evaluation types
        interface = create_gemini_interface()

        test_cases = [
            ("What is 25 * 4?", "mathematical"),
            ("Write a Python function to reverse a string", "code_generation"),
            ("What is the largest planet in our solar system?", "factual_qa"),
            ("Write a short poem about the ocean", "creative_writing")
        ]

        print("\\n📝 Testing different evaluation types...")
        for prompt, eval_type in test_cases:
            response = await interface.generate_response(prompt, eval_type=eval_type)
            print(f"\\n{eval_type.title()}: {prompt}")
            print(f"Response: {response.text[:100]}..." if len(response.text) > 100 else f"Response: {response.text}")
            print(f"Confidence: {response.confidence:.2f}")

        # Test batch processing
        print("\\n🔄 Testing batch processing...")
        batch_interface = GeminiBatchInterface(max_concurrent=3)
        batch_prompts = [
            "What is artificial intelligence?",
            "Explain machine learning briefly",
            "What is deep learning?",
            "How do neural networks work?",
            "What is natural language processing?"
        ]

        batch_results = await batch_interface.process_batch(
            batch_prompts,
            eval_type="factual_qa"
        )

        print(f"Processed {len(batch_results)} prompts in batch")
        batch_stats = batch_interface.get_batch_stats()
        print(f"Success rate: {batch_stats['success_rate']:.1f}%")
        print(f"Average response time: {batch_stats['average_response_time']:.2f}s")

        # Test model switching
        print("\\n🔄 Testing model switching...")
        available_models = interface.list_available_models()
        print(f"Available models: {[m['model_id'] for m in available_models]}")

        # Switch to Pro model if available
        if "gemini-2.5-pro" in [m['model_id'] for m in available_models]:
            interface.switch_model("gemini-2.5-pro")
            pro_response = await interface.generate_response(
                "Explain quantum computing in simple terms",
                eval_type="factual_qa"
            )
            print(f"Pro model response: {pro_response.text[:100]}...")

        # Show performance stats
        print("\\n📊 Performance Statistics:")
        stats = interface.get_performance_stats()
        for key, value in stats.items():
            if key not in ['rate_limiter_stats', 'model_info']:
                print(f"  {key}: {value}")

        print("\\n✅ Demo completed!")

    # Run demo
    asyncio.run(demo())
