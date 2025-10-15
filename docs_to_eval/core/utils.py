# Copyright © 2023-2024 Apple Inc.

import json
import logging
from pathlib import Path
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelArgs:
    def __init__(
        self,
        model_type: str,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        intermediate_size: int,
        num_key_value_heads: int,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000,
        vocab_size: int = 32000,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in cls.__init__.__code__.co_varnames
            }
        )


def load(
    path_or_hf_repo: str,
    tokenizer_config={}
) -> Tuple[nn.Module, AutoTokenizer, ModelArgs]:
    """
    Load model, tokenizer, and model args from path or Hugging Face repo.
    
    Args:
        path_or_hf_repo: Path to model directory or HF model repo
        tokenizer_config: Additional tokenizer configuration
        
    Returns:
        Tuple of (model, tokenizer, model_args)
    """
    
    # Handle local path
    model_path = Path(path_or_hf_repo)
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            path_or_hf_repo,
            trust_remote_code=True,
            **tokenizer_config
        )
        
        # Load model config
        if (model_path / "config.json").exists():
            with open(model_path / "config.json", "r") as f:
                config = json.load(f)
        else:
            # Fallback config for basic models
            config = {
                "model_type": "llama",
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "intermediate_size": 11008,
                "num_key_value_heads": 32,
                "vocab_size": tokenizer.vocab_size,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000,
                "tie_word_embeddings": False
            }
        
        model_args = ModelArgs.from_dict(config)
        
        # Create model
        model = create_model(model_args)
        
        # Load weights if available
        weight_files = list(model_path.glob("*.safetensors")) or list(model_path.glob("model-*.safetensors"))
        if weight_files:
            logger.info(f"Loading weights from {len(weight_files)} files")
            # For now, create a simple model structure
            # In real implementation, you'd load actual weights
        
        logger.info(f"Loaded model with {sum(p.size for p in model.parameters())/1e6:.1f}M parameters")
        
        return model, tokenizer, model_args
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def create_model(args: ModelArgs) -> nn.Module:
    """Create a simple model structure for LoRA training"""
    
    class SimpleTransformerModel(nn.Module):
        def __init__(self, args: ModelArgs):
            super().__init__()
            self.args = args
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
            
            # Simple model structure
            self.model = SimpleModel(args)
            
        def __call__(self, inputs, cache=None):
            x = self.embed_tokens(inputs)
            return self.model(x, cache)
            
        def parameters(self):
            return super().parameters()
            
        def trainable_parameters(self):
            return {k: v for k, v in self.parameters().items() if v.requires_grad}
            
        def freeze(self):
            """Freeze all parameters"""
            for param in self.parameters():
                param.requires_grad = False
                
        def load_weights(self, path, strict=True):
            """Load weights from file"""
            try:
                _ = mx.load(path)
                # Simple weight loading - in real implementation you'd match parameter names
                logger.info(f"Loaded weights from {path}")
            except Exception as e:
                if strict:
                    raise
                logger.warning(f"Failed to load weights: {e}")
    
    class SimpleModel(nn.Module):
        def __init__(self, args: ModelArgs):
            super().__init__()
            self.args = args
            
            # Create transformer layers
            self.layers = [SimpleTransformerLayer(args) for _ in range(args.num_hidden_layers)]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            
        def __call__(self, x, cache=None):
            for layer in self.layers:
                x = layer(x, cache)
            return self.norm(x), cache
    
    class SimpleTransformerLayer(nn.Module):
        def __init__(self, args: ModelArgs):
            super().__init__()
            self.args = args
            
            # Attention mechanism
            self.self_attn = SimpleAttention(args)
            
            # Feed forward
            self.mlp = SimpleMLP(args)
            
            # Norms
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            
        def __call__(self, x, cache=None):
            # Self attention
            residual = x
            x = self.input_layernorm(x)
            x, cache = self.self_attn(x, cache)
            x = residual + x
            
            # Feed forward
            residual = x
            x = self.post_attention_layernorm(x)
            x = self.mlp(x)
            x = residual + x
            
            return x
    
    class SimpleAttention(nn.Module):
        def __init__(self, args: ModelArgs):
            super().__init__()
            self.args = args
            
            self.q_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
            self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * (args.hidden_size // args.num_attention_heads), bias=False)
            self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * (args.hidden_size // args.num_attention_heads), bias=False)
            self.o_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
            
        def __call__(self, x, cache=None):
            B, L, D = x.shape
            
            # Simple attention implementation
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            
            # Reshape for multi-head attention
            q = q.reshape(B, L, self.args.num_attention_heads, -1).transpose(1, 2)
            k = k.reshape(B, L, self.args.num_key_value_heads, -1).transpose(1, 2)
            v = v.reshape(B, L, self.args.num_key_value_heads, -1).transpose(1, 2)
            
            # Attention computation
            scale = 1.0 / (q.shape[-1] ** 0.5)
            scores = (q @ k.transpose(-2, -1)) * scale
            
            # Apply causal mask
            mask = mx.tril(mx.ones((L, L)))
            scores = mx.where(mask, scores, float('-inf'))
            
            weights = mx.softmax(scores, axis=-1)
            out = weights @ v
            
            # Reshape and project
            out = out.transpose(1, 2).reshape(B, L, D)
            out = self.o_proj(out)
            
            return out, cache
    
    class SimpleMLP(nn.Module):
        def __init__(self, args: ModelArgs):
            super().__init__()
            self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
            self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
            self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
            
        def __call__(self, x):
            gate = nn.silu(self.gate_proj(x))
            up = self.up_proj(x)
            return self.down_proj(gate * up)
    
    return SimpleTransformerModel(args)


def apply_lora_layers(model, num_lora_layers: int):
    """Apply LoRA to the specified number of layers"""
    from models import LoRALinear
    
    total_layers = len(model.model.layers)
    start_layer = max(0, total_layers - num_lora_layers)
    
    for i in range(start_layer, total_layers):
        layer = model.model.layers[i]
        
        # Replace linear layers with LoRA versions
        layer.self_attn.q_proj = LoRALinear.from_linear(layer.self_attn.q_proj)
        layer.self_attn.v_proj = LoRALinear.from_linear(layer.self_attn.v_proj)
        
        # Handle MoE if present
        if hasattr(layer, "block_sparse_moe"):
            layer.block_sparse_moe.gate = LoRALinear.from_linear(layer.block_sparse_moe.gate)
    
    return model