"""
Base LLM interface and abstractions
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from enum import Enum


class LLMResponse(BaseModel):
    """Structured response from LLM"""
    text: str
    confidence: float = 1.0
    reasoning_steps: List[str] = []
    metadata: Dict[str, Any] = {}
    response_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class LLMCapability(str, Enum):
    """LLM capabilities"""
    MATHEMATICAL = "mathematical"
    CODE_GENERATION = "code_generation"
    FACTUAL_QA = "factual_qa"
    CREATIVE_WRITING = "creative_writing"
    REASONING = "reasoning"
    MULTILINGUAL = "multilingual"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"


class BaseLLMInterface(ABC):
    """Abstract base class for LLM interfaces"""
    
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 512):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.capabilities: Dict[LLMCapability, float] = {}
        self.call_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    async def generate_response(self, prompt: str, context: Optional[str] = None, 
                              eval_type: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        pass
    
    def get_capability_score(self, capability: LLMCapability) -> float:
        """Get capability score for this LLM"""
        return self.capabilities.get(capability, 0.5)
    
    def set_capability_score(self, capability: LLMCapability, score: float):
        """Set capability score"""
        self.capabilities[capability] = max(0.0, min(1.0, score))
    
    def reset_stats(self):
        """Reset statistics and history"""
        self.call_history.clear()
    
    def get_call_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent call history"""
        return self.call_history[-last_n:]


class LLMAdapter:
    """Adapter for different LLM providers"""
    
    def __init__(self, interface: BaseLLMInterface):
        self.interface = interface
    
    async def evaluate_on_benchmark(self, benchmark_items: List[Dict[str, Any]], 
                                  eval_type: str) -> List[Dict[str, Any]]:
        """Evaluate LLM on benchmark items"""
        results = []
        
        for item in benchmark_items:
            question = item['question']
            ground_truth = item['answer']
            context = item.get('context')
            
            response = await self.interface.generate_response(
                prompt=question,
                context=context,
                eval_type=eval_type
            )
            
            results.append({
                'question': question,
                'ground_truth': ground_truth,
                'prediction': response.text,
                'confidence': response.confidence,
                'reasoning_steps': response.reasoning_steps,
                'response_metadata': response.metadata
            })
        
        return results
    
    def get_interface_info(self) -> Dict[str, Any]:
        """Get information about the LLM interface"""
        return {
            'model_name': self.interface.model_name,
            'temperature': self.interface.temperature,
            'max_tokens': self.interface.max_tokens,
            'capabilities': {cap.value: score for cap, score in self.interface.capabilities.items()},
            'call_history_length': len(self.interface.call_history)
        }


class RateLimiter:
    """Rate limiter for LLM API calls"""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times: List[float] = []
    
    async def acquire(self):
        """Acquire permission to make request"""
        import time
        import asyncio
        
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.max_requests_per_minute:
            # Wait until we can make another request
            oldest_request = min(self.request_times)
            wait_time = 60 - (current_time - oldest_request)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_times.append(current_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        import time
        current_time = time.time()
        recent_requests = [t for t in self.request_times if current_time - t < 60]
        
        return {
            'requests_last_minute': len(recent_requests),
            'max_requests_per_minute': self.max_requests_per_minute,
            'current_rate': len(recent_requests) / min(60, max(1, current_time - min(recent_requests, default=current_time)))
        }