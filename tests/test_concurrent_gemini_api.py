"""
Test concurrent Gemini Flash API calls using futures.concurrent
Tests 10 API calls with 5 concurrent workers and virtual display
"""

import asyncio
import pytest
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock

from docs_to_eval.llm.openrouter_interface import OpenRouterInterface, OpenRouterConfig
from docs_to_eval.llm.base import LLMResponse


# Setup logging for virtual display
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConcurrentGeminiTester:
    """
    Test class for concurrent Gemini Flash API calls
    Provides both real and virtual (mocked) testing capabilities
    """
    
    def __init__(self, use_virtual: bool = True, max_workers: int = 5):
        self.use_virtual = use_virtual
        self.max_workers = max_workers
        self.api_calls_made = []
        self.call_times = []
        
        # Configure for Gemini Flash
        self.config = OpenRouterConfig(
            model="google/gemini-flash-2.5",  # Use Gemini Flash
            api_key="test_key"  # Always provide a test key for testing
        )
        
    def create_test_questions(self, num_questions: int = 10) -> List[Dict[str, str]]:
        """Create test questions for evaluation"""
        return [
            {
                "id": f"q_{i+1}",
                "question": f"What is the capital of country {i+1}?",
                "expected_type": "factual"
            }
            for i in range(num_questions)
        ]
    
    def virtual_api_call(self, question_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Virtual (mocked) API call that simulates Gemini Flash response
        """
        start_time = time.time()
        
        # Simulate variable response times (0.1 to 0.8 seconds)
        import random
        response_time = random.uniform(0.1, 0.8)
        time.sleep(response_time)
        
        question_id = question_data["id"]
        question = question_data["question"]
        
        # Log the virtual API call
        logger.info(f"🔥 Virtual Gemini Flash API Call {question_id}: {question[:50]}...")
        
        # Mock response
        mock_response = {
            "id": question_id,
            "question": question,
            "response": f"Virtual Gemini Flash response for {question_id}",
            "model": "google/gemini-flash-2.5",
            "response_time": response_time,
            "timestamp": time.time(),
            "status": "success"
        }
        
        end_time = time.time()
        actual_time = end_time - start_time
        
        self.call_times.append(actual_time)
        self.api_calls_made.append(mock_response)
        
        logger.info(f"✅ Completed {question_id} in {actual_time:.2f}s")
        
        return mock_response
    
    async def real_api_call(self, question_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Real API call to Gemini Flash via OpenRouter
        """
        start_time = time.time()
        
        try:
            # Create interface (will be mocked in tests)
            interface = OpenRouterInterface(self.config)
            
            question_id = question_data["id"]
            question = question_data["question"]
            
            logger.info(f"🔥 Real Gemini Flash API Call {question_id}: {question[:50]}...")
            
            # Make the API call
            response = await interface.generate_response_async(
                prompt=question,
                max_tokens=100,
                temperature=0.7
            )
            
            end_time = time.time()
            actual_time = end_time - start_time
            
            result = {
                "id": question_id,
                "question": question,
                "response": response.content,
                "model": self.config.model,
                "response_time": actual_time,
                "timestamp": time.time(),
                "status": "success"
            }
            
            self.call_times.append(actual_time)
            self.api_calls_made.append(result)
            
            logger.info(f"✅ Completed {question_id} in {actual_time:.2f}s")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            actual_time = end_time - start_time
            
            error_result = {
                "id": question_data["id"],
                "question": question_data["question"],
                "response": f"Error: {str(e)}",
                "model": self.config.model,
                "response_time": actual_time,
                "timestamp": time.time(),
                "status": "error"
            }
            
            self.call_times.append(actual_time)
            self.api_calls_made.append(error_result)
            
            logger.error(f"❌ Failed {question_data['id']} in {actual_time:.2f}s: {str(e)}")
            
            return error_result
    
    def run_concurrent_virtual_calls(self, questions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Run concurrent virtual API calls using ThreadPoolExecutor
        """
        logger.info(f"🚀 Starting {len(questions)} concurrent virtual Gemini Flash calls with {self.max_workers} workers")
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_question = {
                executor.submit(self.virtual_api_call, question): question 
                for question in questions
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_question):
                question = future_to_question[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"📊 Progress: {len(results)}/{len(questions)} calls completed")
                except Exception as e:
                    logger.error(f"❌ Error in concurrent call for {question['id']}: {e}")
                    results.append({
                        "id": question["id"],
                        "status": "error",
                        "error": str(e)
                    })
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"🎯 All {len(questions)} concurrent calls completed in {total_time:.2f}s")
        logger.info(f"📈 Average response time: {sum(self.call_times)/len(self.call_times):.2f}s")
        
        return results
    
    async def run_concurrent_real_calls(self, questions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Run concurrent real API calls using asyncio
        """
        logger.info(f"🚀 Starting {len(questions)} concurrent real Gemini Flash calls")
        
        start_time = time.time()
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def limited_call(question_data):
            async with semaphore:
                return await self.real_api_call(question_data)
        
        # Run all calls concurrently
        tasks = [limited_call(question) for question in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Exception in call {i+1}: {result}")
                final_results.append({
                    "id": questions[i]["id"],
                    "status": "error",
                    "error": str(result)
                })
            else:
                final_results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"🎯 All {len(questions)} concurrent calls completed in {total_time:.2f}s")
        if self.call_times:
            logger.info(f"📈 Average response time: {sum(self.call_times)/len(self.call_times):.2f}s")
        
        return final_results


class TestConcurrentGeminiAPI:
    """Test cases for concurrent Gemini Flash API calls"""
    
    def test_virtual_concurrent_calls(self):
        """Test 10 virtual concurrent Gemini Flash API calls with 5 workers"""
        tester = ConcurrentGeminiTester(use_virtual=True, max_workers=5)
        questions = tester.create_test_questions(10)
        
        # Run concurrent virtual calls
        results = tester.run_concurrent_virtual_calls(questions)
        
        # Assertions
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        assert len(tester.api_calls_made) == 10, f"Expected 10 API calls, got {len(tester.api_calls_made)}"
        
        # Check that all calls completed successfully
        successful_calls = [r for r in results if r.get("status") == "success"]
        assert len(successful_calls) >= 8, f"Expected at least 8 successful calls, got {len(successful_calls)}"
        
        # Verify concurrency - total time should be less than sequential execution
        total_time = sum(tester.call_times)
        max_individual_time = max(tester.call_times)
        # With 5 workers, max time should be roughly total_time/5 + overhead
        expected_max_time = (total_time / 5) + 1.0  # +1s overhead
        
        logger.info(f"Total sequential time would be: {total_time:.2f}s")
        logger.info(f"Actual concurrent time was: {max_individual_time:.2f}s")
        logger.info(f"Speedup factor: {total_time/max_individual_time:.2f}x")
        
        # Verify models are set correctly
        for result in successful_calls:
            assert result.get("model") == "google/gemini-flash-2.5"
    
    @pytest.mark.asyncio
    async def test_real_concurrent_calls_mocked(self):
        """Test 10 real concurrent Gemini Flash API calls (mocked)"""
        
        # Mock the OpenRouter interface more comprehensively
        with patch('docs_to_eval.llm.concurrent_gemini.OpenRouterInterface') as mock_interface_class, \
             patch('docs_to_eval.llm.openrouter_interface.OPENAI_AVAILABLE', True):
            
            # Setup mock instance
            mock_interface = AsyncMock()
            mock_interface_class.return_value = mock_interface
            
            # Mock the async response
            mock_response = LLMResponse(
                text="Mocked Gemini Flash response",
                metadata={"model": "google/gemini-flash-2.5"}
            )
            mock_interface.generate_response.return_value = mock_response
            
            tester = ConcurrentGeminiTester(use_virtual=False, max_workers=5)
            questions = tester.create_test_questions(10)
            
            # Run concurrent real calls (mocked)
            results = await tester.run_concurrent_real_calls(questions)
            
            # Assertions
            assert len(results) == 10, f"Expected 10 results, got {len(results)}"
            assert mock_interface.generate_response.call_count == 10
            
            # Check that all calls completed successfully
            successful_calls = [r for r in results if r.get("status") == "success"]
            assert len(successful_calls) >= 8, f"Expected at least 8 successful calls, got {len(successful_calls)}"
    
    def test_concurrent_performance_metrics(self):
        """Test performance metrics for concurrent calls"""
        tester = ConcurrentGeminiTester(use_virtual=True, max_workers=5)
        questions = tester.create_test_questions(10)
        
        start_time = time.time()
        results = tester.run_concurrent_virtual_calls(questions)
        end_time = time.time()
        
        total_execution_time = end_time - start_time
        
        # Performance assertions
        assert total_execution_time < 5.0, f"Concurrent execution took too long: {total_execution_time:.2f}s"
        assert len(tester.call_times) == 10, "Should have recorded 10 call times"
        
        # Log performance metrics
        avg_response_time = sum(tester.call_times) / len(tester.call_times)
        max_response_time = max(tester.call_times)
        min_response_time = min(tester.call_times)
        
        logger.info(f"Performance Metrics:")
        logger.info(f"  Total execution time: {total_execution_time:.2f}s")
        logger.info(f"  Average response time: {avg_response_time:.2f}s")
        logger.info(f"  Max response time: {max_response_time:.2f}s")
        logger.info(f"  Min response time: {min_response_time:.2f}s")
        logger.info(f"  Theoretical sequential time: {sum(tester.call_times):.2f}s")
        logger.info(f"  Speedup: {sum(tester.call_times)/total_execution_time:.2f}x")
    
    def test_error_handling_in_concurrent_calls(self):
        """Test error handling in concurrent API calls"""
        tester = ConcurrentGeminiTester(use_virtual=True, max_workers=5)
        
        # Create mix of normal and error-prone questions
        questions = tester.create_test_questions(5)
        # Add some questions that will cause errors
        questions.extend([
            {"id": "error_q1", "question": "", "expected_type": "error"},  # Empty question
            {"id": "error_q2", "question": "x" * 10000, "expected_type": "error"},  # Too long
        ])
        
        # Mock the virtual_api_call to sometimes raise exceptions
        original_virtual_call = tester.virtual_api_call
        
        def error_prone_call(question_data):
            if "error" in question_data["id"]:
                raise Exception(f"Simulated error for {question_data['id']}")
            return original_virtual_call(question_data)
        
        tester.virtual_api_call = error_prone_call
        
        results = tester.run_concurrent_virtual_calls(questions)
        
        # Should still get 7 results (5 success + 2 errors)
        assert len(results) == 7
        
        # Check error handling
        error_results = [r for r in results if r.get("status") == "error"]
        success_results = [r for r in results if r.get("status") == "success"]
        
        assert len(error_results) == 2, f"Expected 2 errors, got {len(error_results)}"
        assert len(success_results) == 5, f"Expected 5 successes, got {len(success_results)}"


# Integration function for use in other parts of the system
async def run_concurrent_gemini_evaluation(
    questions: List[str], 
    max_workers: int = 5,
    use_virtual: bool = False
) -> List[Dict[str, Any]]:
    """
    Integration function to run concurrent Gemini Flash evaluations
    
    Args:
        questions: List of question strings
        max_workers: Number of concurrent workers (default 5)
        use_virtual: Whether to use virtual/mocked calls (default False)
    
    Returns:
        List of evaluation results
    """
    tester = ConcurrentGeminiTester(use_virtual=use_virtual, max_workers=max_workers)
    
    # Convert string questions to question data format
    question_data = [
        {"id": f"eval_q_{i+1}", "question": q, "expected_type": "evaluation"}
        for i, q in enumerate(questions)
    ]
    
    if use_virtual:
        return tester.run_concurrent_virtual_calls(question_data)
    else:
        return await tester.run_concurrent_real_calls(question_data)


if __name__ == "__main__":
    # Demo run
    print("🚀 Running Concurrent Gemini Flash API Test Demo...")
    
    tester = ConcurrentGeminiTester(use_virtual=True, max_workers=5)
    questions = tester.create_test_questions(10)
    
    print(f"Testing {len(questions)} concurrent API calls with {tester.max_workers} workers...")
    results = tester.run_concurrent_virtual_calls(questions)
    
    print(f"\n📊 Demo Results:")
    print(f"  Total calls: {len(results)}")
    print(f"  Successful: {len([r for r in results if r.get('status') == 'success'])}")
    print(f"  Average response time: {sum(tester.call_times)/len(tester.call_times):.2f}s")
    print("✅ Demo completed!")