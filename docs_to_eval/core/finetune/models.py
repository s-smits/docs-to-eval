# Copyright © 2023-2024 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) linear layer implementation.
    
    This replaces a standard linear layer with a low-rank adaptation
    that can be efficiently fine-tuned while keeping the original
    weights frozen.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        bias: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA parameters
        if r > 0:
            self.lora_A = mx.random.normal((r, in_features)) * 0.01
            self.lora_B = mx.zeros((out_features, r))
            
            # Scaling factor
            self.scaling = self.lora_alpha / self.r
            
            # Make LoRA parameters trainable
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
        else:
            self.scaling = 0
            
    @classmethod
    def from_linear(
        cls, 
        linear: nn.Linear, 
        r: int = 16, 
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0
    ):
        """
        Create a LoRALinear layer from an existing Linear layer.
        
        Args:
            linear: Existing linear layer
            r: LoRA rank
            lora_alpha: LoRA scaling parameter
            lora_dropout: LoRA dropout rate
            
        Returns:
            LoRALinear layer with the same dimensions as the input
        """
        lora_layer = cls(
            in_features=linear.weight.shape[1],
            out_features=linear.weight.shape[0],
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=hasattr(linear, 'bias') and linear.bias is not None
        )
        
        # Copy original weights
        lora_layer.linear.weight = linear.weight.copy()
        if hasattr(linear, 'bias') and linear.bias is not None:
            lora_layer.linear.bias = linear.bias.copy()
            
        # Freeze original weights
        lora_layer.linear.weight.requires_grad = False
        if hasattr(lora_layer.linear, 'bias') and lora_layer.linear.bias is not None:
            lora_layer.linear.bias.requires_grad = False
        
        return lora_layer
    
    def __call__(self, x):
        """Forward pass through LoRA layer"""
        # Original linear transformation
        result = self.linear(x)
        
        # Add LoRA adaptation if rank > 0
        if self.r > 0:
            # LoRA computation: x @ A.T @ B.T * scaling
            lora_result = x @ self.lora_A.T @ self.lora_B.T * self.scaling
            result = result + lora_result
            
        return result
    
    def parameters(self):
        """Return all parameters (including frozen ones for compatibility)"""
        params = {}
        
        # Original linear parameters (frozen)
        params.update(self.linear.parameters())
        
        # LoRA parameters (trainable)
        if self.r > 0:
            params['lora_A'] = self.lora_A
            params['lora_B'] = self.lora_B
            
        return params
    
    def trainable_parameters(self):
        """Return only trainable LoRA parameters"""
        params = {}
        
        if self.r > 0:
            params['lora_A'] = self.lora_A
            params['lora_B'] = self.lora_B
            
        return params


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
        
    def __call__(self, x):
        return nn.rms_norm(x, self.weight, self.eps)


class RoPE(nn.Module):
    """Rotary Position Embedding"""
    
    def __init__(self, dims: int, traditional: bool = False, base: float = 10000):
        super().__init__()
        self.dims = dims
        self.traditional = traditional
        self.base = base
        
    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(
            x, self.dims, traditional=self.traditional, base=self.base, offset=offset
        )


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional LoRA"""
    
    def __init__(
        self,
        dims: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        
        if (dims % num_heads) != 0:
            raise ValueError(
                f"The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads} != 0)"
            )
            
        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        
        head_dim = dims // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        
        self.q_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.k_proj = nn.Linear(key_input_dims, self.num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(value_input_dims, self.num_kv_heads * value_dims // num_heads, bias=bias)
        self.out_proj = nn.Linear(dims, value_output_dims, bias=bias)
        
    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        # Reshape for multi-head attention
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        keys = keys.reshape(B, L, self.num_kv_heads, -1).transpose(1, 2)
        values = values.reshape(B, L, self.num_kv_heads, -1).transpose(1, 2)
        
        # Compute attention
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        
        # Reshape output
        output = output.transpose(1, 2).reshape(B, L, -1)
        
        return self.out_proj(output), cache


class FeedForward(nn.Module):
    """Feed forward network with SiLU activation"""
    
    def __init__(self, dims: int, hidden_dims: int, bias: bool = False):
        super().__init__()
        
        self.gate_proj = nn.Linear(dims, hidden_dims, bias=bias)
        self.down_proj = nn.Linear(hidden_dims, dims, bias=bias)  
        self.up_proj = nn.Linear(dims, hidden_dims, bias=bias)
        
    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# Model registry for different architectures
MODEL_REMAPPING = {
    "mistral": "llama",  # Mistral uses same architecture as Llama
    "mixtral": "llama",  # Mixtral uses Llama-like architecture with MoE
    "qwen": "llama",     # Qwen uses Llama-like architecture
}


def get_model_path(path: str) -> str:
    """
    Get the correct model path, handling remapping for similar architectures.
    
    Args:
        path: Original model path
        
    Returns:
        Potentially remapped model path
    """
    return MODEL_REMAPPING.get(path, path)