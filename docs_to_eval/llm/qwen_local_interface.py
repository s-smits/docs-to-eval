#!/usr/bin/env python3
"""
🤖 Local Qwen Model Interface
Real HuggingFace transformers integration for Qwen 0.6B, 1.7B, and 4B models
"""

import torch
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class QwenModelConfig:
    """Configuration for Qwen model"""
    model_name: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    enable_thinking: bool = False
    device_map: str = "auto"
    torch_dtype: str = "auto"


class LocalQwenInterface:
    """
    🚀 Real Local Qwen Model Interface
    
    Supports:
    - Qwen/Qwen3-0.6B (non-thinking)
    - Qwen/Qwen3-1.7B (non-thinking) 
    - Qwen/Qwen3-4B (non-thinking)
    """
    
    def __init__(self, config: QwenModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.performance_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "model_load_time": 0.0,
            "average_tokens_per_second": 0.0
        }
        
    def load_model(self) -> bool:
        """Load the Qwen model and tokenizer"""
        if self.model_loaded:
            return True
            
        try:
            start_time = time.time()
            logger.info(f"Loading Qwen model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Load model with specified configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.torch_dtype,
                device_map=self.config.device_map,
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            self.performance_stats["model_load_time"] = load_time
            
            logger.info(f"✅ Qwen model loaded successfully in {load_time:.2f}s")
            logger.info(f"📊 Model device: {self.model.device}")
            logger.info(f"🔧 Model dtype: {self.model.dtype}")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen model: {e}")
            return False
    
    async def generate_response(self, prompt: str, **kwargs) -> 'QwenResponse':
        """Generate response using local Qwen model"""
        if not self.model_loaded:
            success = self.load_model()
            if not success:
                raise RuntimeError("Failed to load Qwen model")
        
        start_time = time.time()
        
        try:
            # Prepare input with chat template
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.enable_thinking
            )
            
            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            input_length = model_inputs.input_ids.shape[1]
            
            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "do_sample": kwargs.get("do_sample", self.config.do_sample),
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    **generation_kwargs
                )
            
            # Extract only the new tokens
            output_ids = generated_ids[0][input_length:].tolist()
            
            # Parse thinking content (if enabled)
            thinking_content = ""
            content = ""
            
            if self.config.enable_thinking:
                try:
                    # Find </think> token (151668)
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0
                
                thinking_content = self.tokenizer.decode(
                    output_ids[:index], 
                    skip_special_tokens=True
                ).strip("\n")
                content = self.tokenizer.decode(
                    output_ids[index:], 
                    skip_special_tokens=True
                ).strip("\n")
            else:
                # Non-thinking mode - just decode all output
                content = self.tokenizer.decode(
                    output_ids, 
                    skip_special_tokens=True
                ).strip()
            
            # Calculate performance metrics
            generation_time = time.time() - start_time
            tokens_generated = len(output_ids)
            
            # Update stats
            self.performance_stats["total_requests"] += 1
            self.performance_stats["total_tokens"] += tokens_generated
            self.performance_stats["total_time"] += generation_time
            
            if self.performance_stats["total_time"] > 0:
                self.performance_stats["average_tokens_per_second"] = (
                    self.performance_stats["total_tokens"] / self.performance_stats["total_time"]
                )
            
            logger.info(f"🤖 Generated {tokens_generated} tokens in {generation_time:.2f}s "
                       f"({tokens_generated/generation_time:.1f} tok/s)")
            
            return QwenResponse(
                text=content,
                thinking_content=thinking_content,
                tokens_generated=tokens_generated,
                generation_time=generation_time,
                model_name=self.config.model_name,
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return QwenResponse(
                text=f"Generation failed: {str(e)}",
                thinking_content="",
                tokens_generated=0,
                generation_time=time.time() - start_time,
                model_name=self.config.model_name,
                success=False,
                error=str(e)
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.performance_stats,
            "model_name": self.config.model_name,
            "model_loaded": self.model_loaded,
            "device": str(self.model.device) if self.model else "not_loaded",
            "memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.model_loaded = False
        logger.info("🗑️ Qwen model unloaded")


@dataclass
class QwenResponse:
    """Response from Qwen model"""
    text: str
    thinking_content: str = ""
    tokens_generated: int = 0
    generation_time: float = 0.0
    model_name: str = ""
    success: bool = True
    error: Optional[str] = None


# Model factory for different Qwen variants
class QwenModelFactory:
    """Factory for creating different Qwen model configurations"""
    
    AVAILABLE_MODELS = {
        "qwen3-0.6b": {
            "model_name": "Qwen/Qwen3-0.6B",
            "description": "Smallest and fastest Qwen model",
            "recommended_max_tokens": 256,
            "enable_thinking": False
        },
        "qwen3-1.7b": {
            "model_name": "Qwen/Qwen3-1.7B", 
            "description": "Balanced performance and speed",
            "recommended_max_tokens": 512,
            "enable_thinking": False
        },
        "qwen3-4b": {
            "model_name": "Qwen/Qwen3-4B",
            "description": "Higher quality responses",
            "recommended_max_tokens": 1024,
            "enable_thinking": False
        }
    }
    
    @classmethod
    def create_config(cls, model_key: str, **kwargs) -> QwenModelConfig:
        """Create model configuration"""
        if model_key not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(cls.AVAILABLE_MODELS.keys())}")
        
        model_info = cls.AVAILABLE_MODELS[model_key]
        
        return QwenModelConfig(
            model_name=model_info["model_name"],
            max_new_tokens=kwargs.get("max_new_tokens", model_info["recommended_max_tokens"]),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            do_sample=kwargs.get("do_sample", True),
            enable_thinking=kwargs.get("enable_thinking", model_info["enable_thinking"]),
            device_map=kwargs.get("device_map", "auto"),
            torch_dtype=kwargs.get("torch_dtype", "auto")
        )
    
    @classmethod
    def create_interface(cls, model_key: str, **kwargs) -> LocalQwenInterface:
        """Create Qwen interface"""
        config = cls.create_config(model_key, **kwargs)
        return LocalQwenInterface(config)
    
    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, Any]]:
        """List available models"""
        return cls.AVAILABLE_MODELS


async def demo_qwen_models():
    """Demonstrate different Qwen models"""
    print("🚀 QWEN LOCAL MODELS DEMONSTRATION")
    print("=" * 60)
    
    test_prompt = "Explain the concept of machine learning in simple terms."
    
    for model_key in QwenModelFactory.AVAILABLE_MODELS:
        print(f"\n🤖 Testing {model_key.upper()}")
        print("-" * 40)
        
        try:
            # Create interface
            qwen = QwenModelFactory.create_interface(
                model_key,
                max_new_tokens=128  # Short for demo
            )
            
            # Generate response
            response = await qwen.generate_response(test_prompt)
            
            if response.success:
                print(f"✅ Response: {response.text[:200]}...")
                print(f"📊 Tokens: {response.tokens_generated}, Time: {response.generation_time:.2f}s")
                
                # Show performance stats
                stats = qwen.get_performance_stats()
                print(f"🔥 Speed: {stats['average_tokens_per_second']:.1f} tokens/sec")
            else:
                print(f"❌ Error: {response.error}")
            
            # Unload to save memory
            qwen.unload_model()
            
        except Exception as e:
            print(f"❌ Failed to test {model_key}: {e}")
    
    print("\n🎉 Demo complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_qwen_models())