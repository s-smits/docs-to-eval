"""
OpenRouter LLM Interface for accessing models via OpenRouter API
Now powered by iRouter for simplified API access
"""

import asyncio
import os
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

try:
    from irouter import Call, Chat
    IROUTER_AVAILABLE = True
except ImportError:
    IROUTER_AVAILABLE = False

from .base import BaseLLMInterface, LLMResponse


logger = logging.getLogger(__name__)


@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter API using iRouter"""
    api_key: Optional[str] = None
    model: str = "openai/gpt-5-mini"
    site_url: Optional[str] = None
    site_name: Optional[str] = None
    max_retries: int = 3
    timeout: float = 60.0
    
    def __post_init__(self):
        # Get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv('OPENROUTER_API_KEY')
        
        # Note: iRouter can work without API key if set as environment variable
        # but we'll still validate for explicit configuration
        if not self.api_key:
            logger.warning("No OpenRouter API key provided. Ensure OPENROUTER_API_KEY environment variable is set.")


class OpenRouterInterface(BaseLLMInterface):
    """
    LLM interface for OpenRouter API using iRouter
    Provides simplified access to OpenRouter models
    """
    
    def __init__(self, config: Optional[OpenRouterConfig] = None):
        """
        Initialize OpenRouter interface
        
        Args:
            config: OpenRouter configuration (uses defaults if None)
        """
        if not IROUTER_AVAILABLE:
            raise ImportError("iRouter package is required for OpenRouter interface. Install with: uv add irouter")
        
        self.config = config or OpenRouterConfig()
        
        # Initialize iRouter Call and Chat instances
        # iRouter automatically uses OPENROUTER_API_KEY from environment if available
        if self.config.api_key:
            self.caller = Call(self.config.model, api_key=self.config.api_key)
            self.chatter = Chat(self.config.model, api_key=self.config.api_key)
        else:
            # iRouter will use environment variable
            self.caller = Call(self.config.model)
            self.chatter = Chat(self.config.model)
        
        self.model_name = self.config.model
        self.call_count = 0
        self.total_tokens_used = 0
        
        logger.info(f"Initialized iRouter OpenRouter interface with model: {self.model_name}")
    
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using iRouter for OpenRouter API
        
        Args:
            prompt: The prompt to send to the model
            context: Optional context to prepend to prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated text and metadata
        """
        
        # Prepare the full prompt
        full_prompt = prompt
        if context:
            full_prompt = f"Context: {context}\n\n{prompt}"
        
        try:
            # Use iRouter's simple Call interface for one-off requests
            # Note: iRouter doesn't directly expose temperature/max_tokens in the simple interface
            # but we can pass them through kwargs if the underlying API supports them
            response_text = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.caller(full_prompt, **kwargs)
            )
            
            # Update usage statistics
            self.call_count += 1
            # Note: iRouter doesn't provide token usage in the simple interface
            # We'll estimate based on response length (rough approximation)
            estimated_tokens = len(response_text.split()) * 1.3  # rough token estimate
            self.total_tokens_used += int(estimated_tokens)
            
            # Create response object
            return LLMResponse(
                text=response_text,
                metadata={
                    'model_name': self.model_name,
                    'tokens_used': int(estimated_tokens),
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'call_count': self.call_count,
                    'provider': 'irouter'
                }
            )
            
        except Exception as e:
            logger.error(f"iRouter OpenRouter API call failed: {e}")
            # Return empty response on failure
            return LLMResponse(
                text="",
                metadata={
                    'model_name': self.model_name,
                    'tokens_used': 0,
                    'error': str(e),
                    'provider': 'irouter'
                }
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'provider': 'irouter-openrouter',
            'call_count': self.call_count,
            'total_tokens_used': self.total_tokens_used,
            'available': IROUTER_AVAILABLE and (bool(self.config.api_key) or bool(os.getenv('OPENROUTER_API_KEY')))
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.call_count = 0
        self.total_tokens_used = 0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this interface"""
        return {
            'total_calls': self.call_count,
            'total_tokens_used': self.total_tokens_used,
            'model_name': self.model_name,
            'provider': 'irouter-openrouter',
            'available': IROUTER_AVAILABLE and (bool(self.config.api_key) or bool(os.getenv('OPENROUTER_API_KEY')))
        }


# QwenInterface class removed - use OpenRouterInterface directly with appropriate model configuration


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
    
    # Create base OpenRouter interface
    base_config = OpenRouterConfig(api_key=api_key, site_url=site_url, site_name=site_name)
    
    # Create specialized interfaces for different roles
    llm_pool = {
        'retriever': OpenRouterInterface(base_config),    # For concept mining
        'creator': OpenRouterInterface(base_config),      # For question writing
        'adversary': OpenRouterInterface(base_config),    # For adversarial enhancement
        'refiner': OpenRouterInterface(base_config)       # For refinement
    }
    
    logger.info(f"Created OpenRouter LLM pool with {len(llm_pool)} interfaces")
    return llm_pool


# Convenience functions

async def test_openrouter_connection(api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Test OpenRouter connection and model availability using iRouter
    
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
        config = OpenRouterConfig(api_key=api_key)
        interface = OpenRouterInterface(config)
        test_results['model_info'] = interface.get_model_info()
        test_results['connection_successful'] = True
        
        # Test simple generation
        response = await interface.generate_response(
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
        logger.error(f"iRouter OpenRouter connection test failed: {e}")
    
    return test_results


def get_available_models(api_key: Optional[str] = None) -> List[str]:
    """
    Get list of popular OpenRouter models compatible with iRouter
    Note: iRouter works with any OpenRouter model - this is just a selection of popular ones
    """
    
    # Popular OpenRouter models that work well with iRouter
    return [
        "anthropic/claude-sonnet-4",
        "openai/gpt-5-mini",
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-pro",
        "meta-llama/llama-3.2-90b-vision-instruct",
        "qwen/qwen-2.5-72b-instruct"
    ]


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Test the iRouter OpenRouter interface"""
        
        print("Testing iRouter OpenRouter Interface...")
        
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