"""
Concurrent Gemini Flash API Interface
Provides high-performance concurrent API calls to Gemini Flash via OpenRouter
"""

import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field

from .openrouter_interface import OpenRouterInterface, OpenRouterConfig
from .base import LLMResponse


logger = logging.getLogger(__name__)


@dataclass
class ConcurrentCallResult:
    """Result of a concurrent API call"""
    call_id: str
    question: str
    response: Optional[str] = None
    model: str = "google/gemini-flash-2.5"
    response_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"  # pending, success, error
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConcurrentCallStats:
    """Statistics for concurrent API calls"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time: float = 0.0
    average_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = 0.0
    speedup_factor: float = 1.0


class ConcurrentGeminiInterface:
    """
    High-performance concurrent interface for Gemini Flash API calls
    Supports both futures.concurrent and asyncio patterns
    """
    
    def __init__(
        self, 
        config: Optional[OpenRouterConfig] = None,
        max_workers: int = 5,
        model: str = "google/gemini-flash-2.5"
    ):
        """
        Initialize concurrent Gemini interface
        
        Args:
            config: OpenRouter configuration
            max_workers: Maximum number of concurrent workers
            model: Gemini model to use (default: gemini-flash-2.5)
        """
        self.config = config or OpenRouterConfig(model=model)
        self.max_workers = max_workers
        self.model = model
        self.call_count = 0
        self.results_history: List[ConcurrentCallResult] = []
        
        # Progress callback for virtual display
        self.progress_callback: Optional[Callable[[str, int, int], None]] = None
        
    def set_progress_callback(self, callback: Callable[[str, int, int], None]):
        """Set callback function for progress updates (virtual display)"""
        self.progress_callback = callback
        
    def _log_progress(self, message: str, completed: int, total: int):
        """Log progress with virtual display"""
        logger.info(f"🔥 {message} ({completed}/{total})")
        if self.progress_callback:
            self.progress_callback(message, completed, total)
    
    async def make_single_call(self, call_id: str, question: str, **kwargs) -> ConcurrentCallResult:
        """
        Make a single API call to Gemini Flash
        
        Args:
            call_id: Unique identifier for this call
            question: Question/prompt to send
            **kwargs: Additional parameters for the API call
            
        Returns:
            ConcurrentCallResult with response data
        """
        start_time = time.time()
        result = ConcurrentCallResult(call_id=call_id, question=question, model=self.model)
        
        try:
            # Create interface for this call
            interface = OpenRouterInterface(self.config)
            
            # Make the API call
            response = await interface.generate_response(
                prompt=question,
                max_tokens=kwargs.get('max_tokens', 150),
                temperature=kwargs.get('temperature', 0.7)
            )
            
            # Record successful result
            end_time = time.time()
            result.response = response.text
            result.response_time = end_time - start_time
            result.status = "success"
            result.metadata = {
                "tokens_used": response.metadata.get("usage", {}),
                "model_used": response.metadata.get("model", self.model)
            }
            
            logger.debug(f"✅ Call {call_id} completed in {result.response_time:.2f}s")
            
        except Exception as e:
            # Record error result
            end_time = time.time()
            result.response_time = end_time - start_time
            result.status = "error"
            result.error = str(e)
            
            logger.error(f"❌ Call {call_id} failed after {result.response_time:.2f}s: {e}")
        
        return result
    
    async def run_concurrent_async(
        self, 
        questions: List[str], 
        **kwargs
    ) -> Tuple[List[ConcurrentCallResult], ConcurrentCallStats]:
        """
        Run concurrent API calls using asyncio
        
        Args:
            questions: List of questions/prompts
            **kwargs: Additional parameters for API calls
            
        Returns:
            Tuple of (results, stats)
        """
        start_time = time.time()
        
        self._log_progress("Starting concurrent asyncio calls", 0, len(questions))
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def limited_call(i: int, question: str):
            async with semaphore:
                call_id = f"async_call_{i+1}"
                return await self.make_single_call(call_id, question, **kwargs)
        
        # Create and run all tasks
        tasks = [
            limited_call(i, question) 
            for i, question in enumerate(questions)
        ]
        
        # Execute with progress tracking
        results = []
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            self._log_progress(f"Completed call {result.call_id}", len(results), len(questions))
        
        # Calculate statistics
        end_time = time.time()
        stats = self._calculate_stats(results, end_time - start_time)
        
        self.results_history.extend(results)
        logger.info(f"🎯 Async concurrent calls completed: {stats.successful_calls}/{stats.total_calls} successful")
        
        return results, stats
    
    def run_concurrent_futures(
        self, 
        questions: List[str], 
        **kwargs
    ) -> Tuple[List[ConcurrentCallResult], ConcurrentCallStats]:
        """
        Run concurrent API calls using futures.concurrent
        
        Args:
            questions: List of questions/prompts
            **kwargs: Additional parameters for API calls
            
        Returns:
            Tuple of (results, stats)
        """
        start_time = time.time()
        
        self._log_progress("Starting concurrent futures calls", 0, len(questions))
        
        def sync_call_wrapper(i: int, question: str):
            """Wrapper to run async call in sync context"""
            call_id = f"future_call_{i+1}"
            
            # Run the async call in a new event loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    self.make_single_call(call_id, question, **kwargs)
                )
                loop.close()
                return result
            except Exception as e:
                # Return error result
                return ConcurrentCallResult(
                    call_id=call_id,
                    question=question,
                    model=self.model,
                    status="error",
                    error=str(e),
                    response_time=time.time() - start_time
                )
        
        results = []
        
        # Use ThreadPoolExecutor for concurrent execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(sync_call_wrapper, i, question): i
                for i, question in enumerate(questions)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    results.append(result)
                    self._log_progress(f"Completed call {result.call_id}", len(results), len(questions))
                except Exception as e:
                    # Handle executor exceptions
                    index = future_to_index[future]
                    error_result = ConcurrentCallResult(
                        call_id=f"future_call_{index+1}",
                        question=questions[index],
                        model=self.model,
                        status="error",
                        error=str(e)
                    )
                    results.append(error_result)
                    logger.error(f"❌ Future execution error for call {index+1}: {e}")
        
        # Calculate statistics
        end_time = time.time()
        stats = self._calculate_stats(results, end_time - start_time)
        
        self.results_history.extend(results)
        logger.info(f"🎯 Futures concurrent calls completed: {stats.successful_calls}/{stats.total_calls} successful")
        
        return results, stats
    
    def _calculate_stats(self, results: List[ConcurrentCallResult], total_time: float) -> ConcurrentCallStats:
        """Calculate statistics for completed calls"""
        successful_results = [r for r in results if r.status == "success"]
        response_times = [r.response_time for r in results if r.response_time > 0]
        
        stats = ConcurrentCallStats(
            total_calls=len(results),
            successful_calls=len(successful_results),
            failed_calls=len(results) - len(successful_results),
            total_execution_time=total_time
        )
        
        if response_times:
            stats.average_response_time = sum(response_times) / len(response_times)
            stats.max_response_time = max(response_times)
            stats.min_response_time = min(response_times)
            
            # Calculate speedup (theoretical sequential time vs actual concurrent time)
            theoretical_sequential_time = sum(response_times)
            stats.speedup_factor = theoretical_sequential_time / total_time if total_time > 0 else 1.0
        
        return stats
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report for all calls made"""
        if not self.results_history:
            return {"message": "No calls have been made yet"}
        
        successful_calls = [r for r in self.results_history if r.status == "success"]
        failed_calls = [r for r in self.results_history if r.status == "error"]
        
        response_times = [r.response_time for r in self.results_history if r.response_time > 0]
        
        return {
            "total_calls": len(self.results_history),
            "successful_calls": len(successful_calls),
            "failed_calls": len(failed_calls),
            "success_rate": len(successful_calls) / len(self.results_history) * 100,
            "average_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "model_used": self.model,
            "max_workers": self.max_workers,
            "recent_errors": [r.error for r in failed_calls[-5:]]  # Last 5 errors
        }


# Convenience functions for easy integration
async def concurrent_gemini_evaluation_async(
    questions: List[str],
    max_workers: int = 5,
    model: str = "google/gemini-flash-2.5",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function for async concurrent Gemini evaluation
    
    Returns:
        List of evaluation results in dict format
    """
    interface = ConcurrentGeminiInterface(max_workers=max_workers, model=model)
    results, stats = await interface.run_concurrent_async(questions, **kwargs)
    
    # Convert to dict format for backward compatibility
    return [
        {
            "id": result.call_id,
            "question": result.question,
            "response": result.response,
            "model": result.model,
            "status": result.status,
            "response_time": result.response_time,
            "error": result.error
        }
        for result in results
    ]


def concurrent_gemini_evaluation_futures(
    questions: List[str],
    max_workers: int = 5,
    model: str = "google/gemini-flash-2.5",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function for futures-based concurrent Gemini evaluation
    
    Returns:
        List of evaluation results in dict format
    """
    interface = ConcurrentGeminiInterface(max_workers=max_workers, model=model)
    results, stats = interface.run_concurrent_futures(questions, **kwargs)
    
    # Convert to dict format for backward compatibility
    return [
        {
            "id": result.call_id,
            "question": result.question,
            "response": result.response,
            "model": result.model,
            "status": result.status,
            "response_time": result.response_time,
            "error": result.error
        }
        for result in results
    ]


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def demo_async():
        """Demo async concurrent calls"""
        print("🚀 Testing Async Concurrent Gemini Flash API...")
        
        questions = [
            "What is machine learning?",
            "Explain neural networks briefly.",
            "What is deep learning?",
            "How does backpropagation work?",
            "What are transformers in AI?",
            "Explain gradient descent.",
            "What is overfitting?",
            "How do CNNs work?",
            "What is reinforcement learning?",
            "Explain natural language processing."
        ]
        
        interface = ConcurrentGeminiInterface(max_workers=5)
        
        # Set up progress callback for virtual display
        def progress_display(message: str, completed: int, total: int):
            print(f"📊 Progress: {message} - {completed}/{total} ({completed/total*100:.1f}%)")
        
        interface.set_progress_callback(progress_display)
        
        # Run concurrent calls
        results, stats = await interface.run_concurrent_async(questions)
        
        print(f"\n📈 Results Summary:")
        print(f"  Total calls: {stats.total_calls}")
        print(f"  Successful: {stats.successful_calls}")
        print(f"  Failed: {stats.failed_calls}")
        print(f"  Average response time: {stats.average_response_time:.2f}s")
        print(f"  Total execution time: {stats.total_execution_time:.2f}s")
        print(f"  Speedup factor: {stats.speedup_factor:.2f}x")
        
        return results, stats
    
    def demo_futures():
        """Demo futures concurrent calls"""
        print("\n🔧 Testing Futures Concurrent Gemini Flash API...")
        
        questions = [
            "What is data science?",
            "Explain statistics briefly.",
            "What is probability?",
            "How does regression work?",
            "What is classification?"
        ]
        
        interface = ConcurrentGeminiInterface(max_workers=3)
        results, stats = interface.run_concurrent_futures(questions)
        
        print(f"\n📈 Futures Results Summary:")
        print(f"  Total calls: {stats.total_calls}")
        print(f"  Successful: {stats.successful_calls}")
        print(f"  Speedup factor: {stats.speedup_factor:.2f}x")
        
        return results, stats
    
    # Run demos
    print("🎯 Concurrent Gemini Flash Demo Starting...")
    
    # Demo async version
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async_results, async_stats = loop.run_until_complete(demo_async())
    loop.close()
    
    # Demo futures version
    futures_results, futures_stats = demo_futures()
    
    print("\n✅ Demo completed!")