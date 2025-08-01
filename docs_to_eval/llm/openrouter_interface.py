"""
OpenRouter LLM Interface for accessing models via OpenRouter API
Supports Qwen3 30B and other models through OpenRouter
"""

import asyncio
import os
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import BaseLLMInterface, LLMResponse


logger = logging.getLogger(__name__)


@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter API"""
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "qwen/qwen3-30b-a3b-instruct-2507"
    site_url: Optional[str] = None
    site_name: Optional[str] = None
    max_retries: int = 3
    timeout: float = 60.0
    
    def __post_init__(self):
        # Get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv('OPENROUTER_API_KEY')
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter")


class OpenRouterInterface(BaseLLMInterface):
    """
    LLM interface for OpenRouter API
    Provides access to Qwen3 30B and other models
    """
    
    def __init__(self, config: Optional[OpenRouterConfig] = None):
        """
        Initialize OpenRouter interface
        
        Args:
            config: OpenRouter configuration (uses defaults if None)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is required for OpenRouter interface. Install with: pip install openai")
        
        self.config = config or OpenRouterConfig()
        
        # Initialize OpenAI client with OpenRouter endpoint
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
        
        self.model_name = self.config.model
        self.call_count = 0
        self.total_tokens_used = 0
        
        logger.info(f"Initialized OpenRouter interface with model: {self.model_name}")
    
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using OpenRouter API
        
        Args:
            prompt: The prompt to send to the model
            context: Optional context to prepend to prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated text and metadata
        """
        
        # Prepare messages
        messages = []
        
        if context:
            messages.append({
                "role": "system",
                "content": f"Context: {context}"
            })
        
        messages.append({
            "role": "user", 
            "content": prompt
        })
        
        # Prepare headers
        extra_headers = {}
        if self.config.site_url:
            extra_headers["HTTP-Referer"] = self.config.site_url
        if self.config.site_name:
            extra_headers["X-Title"] = self.config.site_name
        
        try:
            # Make API call
            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=extra_headers,
                **kwargs
            )
            
            # Extract response
            response_text = completion.choices[0].message.content
            
            # Update usage statistics
            self.call_count += 1
            if hasattr(completion, 'usage') and completion.usage:
                self.total_tokens_used += completion.usage.total_tokens
            
            # Create response object
            return LLMResponse(
                text=response_text,
                model_name=self.model_name,
                tokens_used=completion.usage.total_tokens if hasattr(completion, 'usage') and completion.usage else 0,
                metadata={
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'finish_reason': completion.choices[0].finish_reason,
                    'model': completion.model,
                    'call_count': self.call_count
                }
            )
            
        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            # Return empty response on failure
            return LLMResponse(
                text="",
                model_name=self.model_name,
                tokens_used=0,
                metadata={'error': str(e)}
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'provider': 'openrouter',
            'base_url': self.config.base_url,
            'call_count': self.call_count,
            'total_tokens_used': self.total_tokens_used,
            'available': OPENAI_AVAILABLE and bool(self.config.api_key)
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.call_count = 0
        self.total_tokens_used = 0


class QwenInterface(OpenRouterInterface):
    """Specialized interface for Qwen3 30B model"""
    
    def __init__(self, api_key: Optional[str] = None, site_url: Optional[str] = None, site_name: Optional[str] = None):
        """
        Initialize Qwen3 30B interface
        
        Args:
            api_key: OpenRouter API key (uses environment variable if None)
            site_url: Site URL for OpenRouter rankings
            site_name: Site name for OpenRouter rankings
        """
        config = OpenRouterConfig(
            api_key=api_key,
            model="qwen/qwen3-30b-a3b-instruct-2507",
            site_url=site_url,
            site_name=site_name
        )
        super().__init__(config)
    
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response optimized for Qwen3 30B
        
        Qwen models work well with structured prompts and specific instructions
        """
        
        # Optimize prompt for Qwen
        if context:
            optimized_prompt = f"""Based on the following context, please provide a comprehensive and accurate response:

Context:
{context}

Question/Task:
{prompt}

Please provide a detailed, well-structured response:"""
        else:
            optimized_prompt = f"""Please provide a comprehensive and accurate response to the following:

{prompt}

Response:"""
        
        return await super().generate_response(
            optimized_prompt,
            context=None,  # We've already incorporated context
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


def create_openrouter_llm_pool(
    api_key: Optional[str] = None,
    site_url: Optional[str] = None,
    site_name: Optional[str] = None
) -> Dict[str, BaseLLMInterface]:
    """
    Create a complete LLM pool using OpenRouter with different model configurations
    
    Args:
        api_key: OpenRouter API key
        site_url: Site URL for rankings
        site_name: Site name for rankings
        
    Returns:
        Dictionary of LLM interfaces for different roles
    """
    
    # Create base Qwen interface
    base_qwen = QwenInterface(api_key, site_url, site_name)
    
    # Create specialized interfaces for different roles
    # For now, we'll use the same model but could use different models/configs for each role
    llm_pool = {
        'retriever': QwenInterface(api_key, site_url, site_name),    # For concept mining
        'creator': QwenInterface(api_key, site_url, site_name),      # For question writing
        'adversary': QwenInterface(api_key, site_url, site_name),    # For adversarial enhancement
        'refiner': QwenInterface(api_key, site_url, site_name)       # For refinement
    }
    
    logger.info(f"Created OpenRouter LLM pool with {len(llm_pool)} interfaces")
    return llm_pool


# Convenience functions

async def test_openrouter_connection(api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Test OpenRouter connection and model availability
    
    Args:
        api_key: API key to test (uses environment variable if None)
        
    Returns:
        Test results dictionary
    """
    
    test_results = {
        'connection_successful': False,
        'model_available': False,
        'response_generated': False,
        'error': None,
        'model_info': {},
        'sample_response': None
    }
    
    try:
        # Create interface
        qwen = QwenInterface(api_key)
        test_results['model_info'] = qwen.get_model_info()
        test_results['connection_successful'] = True
        
        # Test simple generation
        response = await qwen.generate_response(
            "What is the capital of France? Respond in one sentence.",
            temperature=0.1,
            max_tokens=50
        )
        
        if response.text and len(response.text.strip()) > 0:
            test_results['response_generated'] = True
            test_results['sample_response'] = response.text.strip()
            test_results['model_available'] = True
        
    except Exception as e:
        test_results['error'] = str(e)
        logger.error(f"OpenRouter connection test failed: {e}")
    
    return test_results


def get_available_models(api_key: Optional[str] = None) -> List[str]:
    """
    Get list of available models from OpenRouter
    Note: This is a placeholder - OpenRouter doesn't provide a direct API for listing models
    """
    
    # Common OpenRouter models that support chat completion
    return [
        "qwen/qwen3-30b-a3b-instruct-2507",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-haiku",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "google/gemini-pro",
        "mistralai/mixtral-8x7b-instruct",
        "meta-llama/llama-3-70b-instruct"
    ]


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Test the OpenRouter interface"""
        
        print("Testing OpenRouter Interface...")
        
        # Test connection
        test_results = await test_openrouter_connection()
        
        print(f"Connection successful: {test_results['connection_successful']}")
        print(f"Model available: {test_results['model_available']}")
        print(f"Response generated: {test_results['response_generated']}")
        
        if test_results['sample_response']:
            print(f"Sample response: {test_results['sample_response']}")
        
        if test_results['error']:
            print(f"Error: {test_results['error']}")
        
        # Test LLM pool creation
        try:
            llm_pool = create_openrouter_llm_pool()
            print(f"\\nCreated LLM pool with {len(llm_pool)} interfaces:")
            for role, interface in llm_pool.items():
                info = interface.get_model_info()
                print(f"  {role}: {info['model_name']} ({'available' if info['available'] else 'unavailable'})")
        
        except Exception as e:
            print(f"Failed to create LLM pool: {e}")
    
    # Run test
    asyncio.run(main())