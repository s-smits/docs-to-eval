from typing import Optional, List, Dict, Any
from .base import BaseLLMInterface
from .openrouter_interface import OpenRouterInterface, OpenRouterConfig, get_available_models as get_openrouter_models
from .groq_interface import GroqInterface, GroqConfig
from .gemini_sdk_interface import GeminiSDKInterface, GeminiConfig, list_gemini_models # Import GeminiConfig and list_gemini_models

# from .openai_interface import OpenAIInterface # If you add an OpenAI interface later
# from .anthropic_interface import AnthropicInterface # If you add an Anthropic interface later


def get_llm_interface(
    provider: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    base_url: Optional[str] = None,
    site_url: Optional[str] = None,
    app_name: Optional[str] = None,
) -> BaseLLMInterface:
    """
    Factory function to get the appropriate LLM interface based on the provider.
    """
    if provider == "openrouter":
        if not api_key:
            raise ValueError("API key is required for OpenRouter provider")
        config = OpenRouterConfig(
            api_key=api_key,
            model=model_name,
            site_url=site_url,
            site_name=app_name,
        )
        return OpenRouterInterface(config)
    elif provider == "groq":
        if not api_key:
            raise ValueError("API key is required for Groq provider")

        config = GroqConfig(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return GroqInterface(config)
    elif provider == "gemini_sdk":
        if not api_key:
            raise ValueError("API key is required for Gemini SDK provider")

        config = GeminiConfig(
            api_key=api_key,
            model=model_name or "gemini-2.5-flash",
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        return GeminiSDKInterface(config)
    # Add other providers here as they are implemented
    # elif provider == "openai":
    #     if not api_key:
    #         raise ValueError("API key is required for OpenAI provider")
    #     return OpenAIInterface(...)
    # elif provider == "anthropic":
    #     if not api_key:
    #         raise ValueError("API key is required for Anthropic provider")
    #     return AnthropicInterface(...)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def list_llm_models(provider: str) -> List[Dict[str, str]]:
    """
    Lists available LLM models for a given provider, returning both value and label.
    """
    if provider == "openrouter":
        models = get_openrouter_models()
        return [{"value": model, "label": model} for model in models]
    elif provider == "groq":
        all_groq_models = GroqInterface.AVAILABLE_MODELS
        return [{"value": key, "label": data["name"]} for key, data in all_groq_models.items()]
    elif provider == "gemini_sdk":
        # Use the list_gemini_models function from gemini_sdk_interface.py
        models = list_gemini_models()
        return [{"value": model, "label": model} for model in models] # Assuming it returns just model names as strings
    # Add other providers here as they are implemented
    # elif provider == "openai":
    #     return [{"value": "gpt-4o", "label": "GPT-4o"}, {"value": "gpt-4", "label": "GPT-4"}, {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"}]
    # elif provider == "anthropic":
    #     return [{"value": "claude-3-opus-20240229", "label": "Claude 3 Opus"}, {"value": "claude-3-sonnet-20240229", "label": "Claude 3 Sonnet"}]
    else:
        return [] # Return empty list for unknown providers (or raise an error if preferred)
