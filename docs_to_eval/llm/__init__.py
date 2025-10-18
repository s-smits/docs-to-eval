"""LLM interfaces and adapters"""

from .mock_interface import MockLLMInterface, MockLLMEvaluator
from .base import BaseLLMInterface

# Import optional interfaces (may not be available if dependencies aren't installed)
try:
    from .openrouter_interface import OpenRouterInterface
    _HAS_OPENROUTER = True
except ImportError:
    _HAS_OPENROUTER = False
    OpenRouterInterface = None

try:
    from .groq_interface import GroqInterface, GroqConfig, create_groq_interface
    _HAS_GROQ = True
except ImportError:
    _HAS_GROQ = False
    GroqInterface = GroqConfig = create_groq_interface = None

try:
    from .gemini_sdk_interface import GeminiSDKInterface, GeminiConfig, create_gemini_interface
    _HAS_GEMINI_SDK = True
except ImportError:
    _HAS_GEMINI_SDK = False
    GeminiSDKInterface = GeminiConfig = create_gemini_interface = None

try:
    from .concurrent_gemini import ConcurrentGeminiInterface
    _HAS_CONCURRENT_GEMINI = True
except ImportError:
    _HAS_CONCURRENT_GEMINI = False
    ConcurrentGeminiInterface = None

# Base interfaces always available
__all__ = [
    "MockLLMInterface",
    "MockLLMEvaluator",
    "BaseLLMInterface",
    "create_llm_interface",
    "list_available_providers"
]

# Add optional interfaces to __all__ if available
if _HAS_OPENROUTER:
    __all__.extend(["OpenRouterInterface"])
if _HAS_GROQ:
    __all__.extend(["GroqInterface", "GroqConfig", "create_groq_interface"])
if _HAS_GEMINI_SDK:
    __all__.extend(["GeminiSDKInterface", "GeminiConfig", "create_gemini_interface"])
if _HAS_CONCURRENT_GEMINI:
    __all__.extend(["ConcurrentGeminiInterface"])


def list_available_providers():
    """List all available LLM providers and their status"""
    providers = {
        'mock': {'available': True, 'description': 'Mock interface for testing'},
        'openrouter': {'available': _HAS_OPENROUTER, 'description': 'OpenRouter API interface'},
        'groq': {'available': _HAS_GROQ, 'description': 'Groq API interface'},
        'gemini_sdk': {'available': _HAS_GEMINI_SDK, 'description': 'Google Gemini SDK interface'},
        'concurrent_gemini': {'available': _HAS_CONCURRENT_GEMINI, 'description': 'Concurrent Gemini interface'}
    }
    return providers


def create_llm_interface(provider: str, **config_kwargs) -> BaseLLMInterface:
    """
    Factory function to create LLM interfaces
    
    Args:
        provider: Provider name ('mock', 'openrouter', 'groq', 'gemini_sdk')
        **config_kwargs: Configuration parameters for the provider
        
    Returns:
        Configured LLM interface
        
    Raises:
        ValueError: If provider is not available or unknown
    """
    provider = provider.lower()
    
    if provider == 'mock':
        return MockLLMInterface(**config_kwargs)
    
    elif provider == 'openrouter':
        if not _HAS_OPENROUTER:
            raise ValueError("OpenRouter interface not available. Install with: uv add irouter")
        from .openrouter_interface import OpenRouterConfig
        config = OpenRouterConfig(**config_kwargs)
        return OpenRouterInterface(config)
    
    elif provider == 'groq':
        if not _HAS_GROQ:
            raise ValueError("Groq interface not available. Install with: pip install groq")
        return create_groq_interface(**config_kwargs)
    
    elif provider == 'gemini_sdk' or provider == 'gemini':
        if not _HAS_GEMINI_SDK:
            raise ValueError("Gemini SDK interface not available. Install with: pip install google-generativeai")
        return create_gemini_interface(**config_kwargs)
    
    elif provider == 'concurrent_gemini':
        if not _HAS_CONCURRENT_GEMINI:
            raise ValueError("Concurrent Gemini interface not available")
        return ConcurrentGeminiInterface(**config_kwargs)
    
    else:
        available = [name for name, info in list_available_providers().items() if info['available']]
        raise ValueError(f"Unknown provider '{provider}'. Available providers: {available}")
