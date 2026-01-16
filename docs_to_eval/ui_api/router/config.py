
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx

from ...llm.llm_factory import list_llm_models, get_llm_interface
from ...utils.config import create_default_config, ConfigManager, EvaluationType, EvaluationConfig
from ...utils.logging import get_logger

router = APIRouter()
logger = get_logger("config_routes")

@router.get("/llm/models")
async def get_llm_models(provider: str):
    """Get a list of available LLM models for a given provider."""
    try:
        model_data = list_llm_models(provider)
        if model_data and isinstance(model_data[0], dict):
            models = [item["value"] for item in model_data]
        else:
            models = model_data
        return models
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting LLM models for provider {provider}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/default")
async def get_default_config():
    """Get default evaluation configuration"""
    config = create_default_config()
    return config.model_dump()

@router.get("/current")
async def get_current_config():
    """Get current evaluation configuration"""
    try:
        manager = ConfigManager()
        manager.update_from_env()
        config = manager.get_config()
        config_dict = config.model_dump()
        has_api_key = bool(config_dict.get('llm', {}).get('api_key'))
        if has_api_key:
            config_dict['llm']['api_key'] = '***masked***'
            config_dict['llm']['api_key_configured'] = True
        else:
            config_dict['llm']['api_key_configured'] = False
        return config_dict
    except Exception as e:
        logger.error(f"Error getting current config: {e}")
        raise HTTPException(status_code=500, detail="Failed to load configuration")

@router.post("/update")
async def update_config(config_update: dict):
    """Update configuration (API key and other settings)"""
    try:
        from dotenv import load_dotenv, set_key
        if not config_update or not isinstance(config_update, dict):
            raise HTTPException(status_code=400, detail="Invalid config update data")
        
        project_root = Path(__file__).parent.parent.parent.parent
        env_path = project_root / ".env"
        
        manager = ConfigManager()
        manager.update_from_env()
        current_dict = manager.get_config().model_dump()

        def update_nested(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    update_nested(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        update_nested(current_dict, config_update)
        updated_config = EvaluationConfig(**current_dict)
        
        env_updates = {}
        api_key_set = False
        if 'llm' in config_update:
            llm_config = config_update['llm']
            if 'provider' in llm_config: env_updates['DOCS_TO_EVAL_PROVIDER'] = llm_config['provider']
            if 'model_name' in llm_config: env_updates['DOCS_TO_EVAL_MODEL_NAME'] = llm_config['model_name']
            if 'max_tokens' in llm_config: env_updates['DOCS_TO_EVAL_MAX_TOKENS'] = str(llm_config['max_tokens'])
            if 'temperature' in llm_config: env_updates['DOCS_TO_EVAL_TEMPERATURE'] = str(llm_config['temperature'])
            
            if 'api_key' in llm_config and llm_config['api_key']:
                api_key_value = llm_config['api_key'].strip()
                provider_for_env = llm_config.get('provider', 'DOCS_TO_EVAL')
                env_key = f"{provider_for_env.upper()}_API_KEY" if provider_for_env != "NOT SET" else "DOCS_TO_EVAL_API_KEY"
                if provider_for_env == "openrouter": env_key = "OPENROUTER_API_KEY"
                elif provider_for_env == "groq": env_key = "GROQ_API_KEY"
                elif provider_for_env == "gemini_sdk": env_key = "GEMINI_API_KEY"
                
                env_updates[env_key] = api_key_value
                os.environ[env_key] = api_key_value
                api_key_set = True
                
        if not env_path.exists():
            with open(env_path, 'w') as f: f.write("")
        for key, value in env_updates.items():
            set_key(str(env_path), key, value)
        load_dotenv(env_path, override=True)
        
        return {"status": "success", "message": "Configuration updated successfully", "api_key_set": api_key_set or bool(updated_config.llm.api_key)}
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Unexpected error updating config: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/test-api-key")
async def test_api_key(api_test: dict):
    """Test API key validity"""
    try:
        if not api_test or 'api_key' not in api_test or 'provider' not in api_test or 'model' not in api_test:
            raise HTTPException(status_code=400, detail="API key, provider, and model are required")
        api_key, provider, model_name = api_test['api_key'].strip(), api_test['provider'], api_test['model']
        if len(api_key) < 10: raise HTTPException(status_code=400, detail="API key appears to be too short")
        
        llm_interface = get_llm_interface(provider=provider, model_name=model_name, api_key=api_key, temperature=0.01, max_tokens=10)
        response = await llm_interface.generate_response(prompt="Hello")
        if response.text:
            return {"status": "success", "message": f"API key is valid. Response: {response.text[:50]}...", "valid": True}
        return {"status": "warning", "message": "API key valid, but empty response", "valid": False}
    except Exception as e:
        logger.error(f"Error testing API key: {e}")
        return {"status": "error", "message": f"Verification failed: {str(e)}", "valid": False}

@router.get("/types/evaluation")
async def get_evaluation_types():
    """Get available evaluation types"""
    return {
        "types": [
            {
                "value": eval_type.value,
                "name": eval_type.value.replace("_", " ").title(),
                "description": f"Evaluation type for {eval_type.value.replace('_', ' ')} content"
            }
            for eval_type in EvaluationType
        ]
    }
