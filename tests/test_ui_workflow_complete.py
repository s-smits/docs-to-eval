"""
Complete UI Workflow Integration Tests for docs-to-eval
Tests the entire user journey from corpus upload through evaluation completion
Ensures all components work together as expected in the actual UI workflow
"""

import pytest
import asyncio
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from docs_to_eval.ui_api.main import app
from docs_to_eval.ui_api.routes import evaluation_runs
from docs_to_eval.utils.config import EvaluationType


class TestCompleteUIWorkflow:
    """Test complete UI workflow end-to-end"""
    
    def setup_method(self):
        """Set up test client and clear runs"""
        self.client = TestClient(app)
        evaluation_runs._runs.clear()
    
    def test_complete_evaluation_workflow_standard(self):
        """Test complete evaluation workflow with standard generation"""
        # Step 1: Upload corpus text
        corpus_data = {
            "text": "Artificial neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes called neurons that process information. Deep learning uses multiple layers of neural networks to learn complex patterns from data.",
            "name": "Neural Networks Corpus",
            "description": "Educational content about neural networks and deep learning"
        }
        
        upload_response = self.client.post("/api/v1/corpus/upload", json=corpus_data)
        assert upload_response.status_code == 200
        
        upload_data = upload_response.json()
        assert upload_data["status"] == "uploaded"
        assert "primary_type" in upload_data
        
        # Step 2: Configure and start evaluation
        evaluation_config = {
            "corpus_text": corpus_data["text"],
            "eval_type": "domain_knowledge",
            "num_questions": 3,
            "use_agentic": False,  # Standard generation for faster testing
            "temperature": 0.5,
            "run_name": "Neural Networks Test",
            "finetune_test_set_enabled": True,
            "finetune_test_set_percentage": 0.2,
            "finetune_random_seed": 42
        }
        
        start_response = self.client.post("/api/v1/evaluation/start", json=evaluation_config)
        assert start_response.status_code == 200
        
        start_data = start_response.json()
        assert start_data["status"] == "queued"
        assert "run_id" in start_data
        assert "websocket_url" in start_data
        
        run_id = start_data["run_id"]
        
        # Step 3: Monitor evaluation progress
        max_wait_time = 30  # 30 seconds max
        start_time = time.time()
        final_status = None
        
        while time.time() - start_time < max_wait_time:
            status_response = self.client.get(f"/api/v1/evaluation/{run_id}/status")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            current_status = status_data["status"]
            
            if current_status == "completed":
                final_status = "completed"
                break
            elif current_status == "error":
                error_details = status_data.get("error", "Unknown error")
                pytest.fail(f"Evaluation failed with error: {error_details}")
            
            time.sleep(1)  # Wait 1 second before checking again
        
        assert final_status == "completed", f"Evaluation did not complete within {max_wait_time} seconds"
        
        # Step 4: Retrieve and validate results
        results_response = self.client.get(f"/api/v1/evaluation/{run_id}/results")
        assert results_response.status_code == 200
        
        results_data = results_response.json()
        
        # Validate results structure
        assert "run_id" in results_data
        assert "aggregate_metrics" in results_data
        assert "individual_results" in results_data
        assert "classification" in results_data
        assert "finetune_test_set" in results_data
        
        # Validate the fix worked - scores should be > 0
        aggregate_metrics = results_data["aggregate_metrics"]
        assert aggregate_metrics["mean_score"] > 0, f"Mean score should be > 0, got {aggregate_metrics['mean_score']}"
        assert aggregate_metrics["num_samples"] == 3
        assert aggregate_metrics["min_score"] >= 0
        assert aggregate_metrics["max_score"] <= 1
        
        # Validate individual results
        individual_results = results_data["individual_results"]
        assert len(individual_results) > 0
        
        non_zero_scores = [r for r in individual_results if r["score"] > 0]
        assert len(non_zero_scores) > 0, "At least some individual results should have non-zero scores"
        
        # Step 5: Check finetune test set
        finetune_response = self.client.get(f"/api/v1/evaluation/{run_id}/finetune-test-set")
        assert finetune_response.status_code == 200
        
        finetune_data = finetune_response.json()
        assert finetune_data["enabled"] == True
        assert finetune_data["test_percentage"] == 0.2
        assert finetune_data["random_seed"] == 42
        assert finetune_data["total_questions"] == 3
        
        # Step 6: Download results
        download_response = self.client.get(f"/api/v1/evaluation/{run_id}/download")
        assert download_response.status_code == 200
        assert download_response.headers["content-type"] == "application/json"
        
        # Step 7: List all runs (should include our run)
        runs_response = self.client.get("/api/v1/runs")
        assert runs_response.status_code == 200
        
        runs_data = runs_response.json()
        run_ids = [run["run_id"] for run in runs_data["runs"]]
        assert run_id in run_ids
    
    def test_complete_evaluation_workflow_agentic(self):
        """Test complete evaluation workflow with agentic generation"""
        # Note: This test may take longer and requires proper API configuration
        
        # Step 1: Upload corpus
        corpus_data = {
            "text": "Quantum mechanics is a fundamental theory in physics that describes nature at the scale of atoms and subatomic particles. It introduces concepts like wave-particle duality, quantum superposition, and quantum entanglement. These principles enable technologies like quantum computing and quantum cryptography.",
            "name": "Quantum Mechanics Corpus",
            "description": "Physics content about quantum mechanics principles"
        }
        
        upload_response = self.client.post("/api/v1/corpus/upload", json=corpus_data)
        assert upload_response.status_code == 200
        
        # Step 2: Start agentic evaluation
        evaluation_config = {
            "corpus_text": corpus_data["text"],
            "eval_type": "domain_knowledge",
            "num_questions": 2,  # Fewer questions for agentic to speed up test
            "use_agentic": True,
            "temperature": 0.7,
            "run_name": "Quantum Mechanics Agentic Test"
        }
        
        start_response = self.client.post("/api/v1/evaluation/start", json=evaluation_config)
        assert start_response.status_code == 200
        
        run_id = start_response.json()["run_id"]
        
        # Step 3: Monitor with longer timeout for agentic generation
        max_wait_time = 60  # 60 seconds for agentic
        start_time = time.time()
        final_status = None
        
        while time.time() - start_time < max_wait_time:
            status_response = self.client.get(f"/api/v1/evaluation/{run_id}/status")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            current_status = status_data["status"]
            
            if current_status == "completed":
                final_status = "completed"
                break
            elif current_status == "error":
                # For agentic tests, we may accept certain failures (missing agents, etc.)
                error_details = status_data.get("error", "Unknown error")
                if "Pipeline unhealthy" in error_details or "Missing LLM interface" in error_details:
                    pytest.skip(f"Agentic pipeline not fully configured: {error_details}")
                else:
                    pytest.fail(f"Evaluation failed with error: {error_details}")
            
            time.sleep(2)  # Longer wait for agentic
        
        if final_status != "completed":
            pytest.skip("Agentic evaluation did not complete within timeout - may require full configuration")
        
        # Step 4: Validate agentic results
        results_response = self.client.get(f"/api/v1/evaluation/{run_id}/results")
        assert results_response.status_code == 200
        
        results_data = results_response.json()
        
        # Agentic results should have higher quality metrics
        aggregate_metrics = results_data["aggregate_metrics"]
        assert aggregate_metrics["mean_score"] > 0
        
        # Agentic generation should produce detailed questions
        individual_results = results_data["individual_results"]
        if len(individual_results) > 0:
            # Check for more sophisticated question patterns
            questions = [r["question"] for r in individual_results]
            # Agentic questions tend to be longer and more detailed
            avg_question_length = sum(len(q.split()) for q in questions) / len(questions)
            # Should have reasonably detailed questions
            assert avg_question_length > 5, "Agentic questions should be reasonably detailed"
    
    def test_file_upload_workflow(self):
        """Test workflow with file upload instead of text"""
        # Create a temporary file
        test_content = """
        Machine Learning Fundamentals
        
        Machine learning is a subset of artificial intelligence that focuses on algorithms
        and statistical models that enable computer systems to improve their performance
        on a specific task through experience.
        
        Key concepts include:
        - Supervised learning: Learning with labeled data
        - Unsupervised learning: Finding patterns in unlabeled data
        - Reinforcement learning: Learning through interaction with environment
        
        Applications include image recognition, natural language processing, and
        autonomous systems.
        """
        
        # Test file upload
        files = {"file": ("ml_content.txt", test_content.encode(), "text/plain")}
        data = {"name": "ML Fundamentals File"}
        
        upload_response = self.client.post("/api/v1/corpus/upload-file", files=files, data=data)
        assert upload_response.status_code == 200
        
        upload_data = upload_response.json()
        assert upload_data["status"] == "uploaded"
        assert upload_data["chars"] == len(test_content)
        
        # Continue with evaluation using uploaded content
        evaluation_config = {
            "corpus_text": test_content,
            "eval_type": "domain_knowledge",
            "num_questions": 2,
            "use_agentic": False,
            "run_name": "File Upload Test"
        }
        
        start_response = self.client.post("/api/v1/evaluation/start", json=evaluation_config)
        assert start_response.status_code == 200
        
        # Wait for completion and validate
        run_id = start_response.json()["run_id"]
        
        # Quick check for completion
        max_attempts = 20
        for _ in range(max_attempts):
            status_response = self.client.get(f"/api/v1/evaluation/{run_id}/status")
            if status_response.json()["status"] == "completed":
                break
            time.sleep(1)
        
        results_response = self.client.get(f"/api/v1/evaluation/{run_id}/results")
        assert results_response.status_code == 200
        
        results_data = results_response.json()
        assert results_data["aggregate_metrics"]["mean_score"] > 0
    
    def test_configuration_workflow(self):
        """Test configuration management workflow"""
        # Step 1: Get default config
        default_config_response = self.client.get("/api/v1/config/default")
        assert default_config_response.status_code == 200
        
        default_config = default_config_response.json()
        assert "eval_type" in default_config
        assert "generation" in default_config
        assert "llm" in default_config
        
        # Step 2: Update configuration
        config_update = {
            "api_key": "sk-test-api-key-for-testing",
            "model_name": "google/gemini-2.5-flash",
            "temperature": 0.8
        }
        
        update_response = self.client.post("/api/v1/config/update", json=config_update)
        assert update_response.status_code == 200
        
        update_data = update_response.json()
        assert update_data["status"] == "updated"
        
        # Step 3: Get current config and verify update
        current_config_response = self.client.get("/api/v1/config/current")
        assert current_config_response.status_code == 200
        
        # Step 4: Test API key
        api_test_data = {
            "api_key": "sk-test-key",
            "model_name": "google/gemini-2.5-flash",
            "base_url": "https://openrouter.ai/api/v1"
        }
        
        api_test_response = self.client.post("/api/v1/config/test-api-key", json=api_test_data)
        assert api_test_response.status_code in [200, 400]  # 400 for invalid test key is OK
        
        api_test_data = api_test_response.json()
        assert "valid" in api_test_data
        assert isinstance(api_test_data["valid"], bool)
    
    def test_evaluation_types_workflow(self):
        """Test evaluation types workflow"""
        # Get available evaluation types
        types_response = self.client.get("/api/v1/types/evaluation")
        assert types_response.status_code == 200
        
        types_data = types_response.json()
        assert "evaluation_types" in types_data
        assert len(types_data["evaluation_types"]) > 0
        
        # Verify all required fields are present
        for eval_type in types_data["evaluation_types"]:
            assert "value" in eval_type
            assert "display_name" in eval_type
            assert "description" in eval_type
        
        # Test evaluation with each available type
        test_corpus = "Test corpus for evaluation type testing."
        
        # Test a few key evaluation types
        test_types = ["domain_knowledge", "factual_qa", "mathematical"]
        available_types = [et["value"] for et in types_data["evaluation_types"]]
        
        for eval_type in test_types:
            if eval_type in available_types:
                evaluation_config = {
                    "corpus_text": test_corpus,
                    "eval_type": eval_type,
                    "num_questions": 1,
                    "use_agentic": False,
                    "run_name": f"Test {eval_type}"
                }
                
                start_response = self.client.post("/api/v1/evaluation/start", json=evaluation_config)
                assert start_response.status_code == 200, f"Failed to start evaluation for type {eval_type}"
    
    def test_run_management_workflow(self):
        """Test run management workflow"""
        # Start multiple evaluations
        run_ids = []
        for i in range(3):
            evaluation_config = {
                "corpus_text": f"Test corpus {i} for run management testing.",
                "num_questions": 1,
                "use_agentic": False,
                "run_name": f"Management Test Run {i}"
            }
            
            start_response = self.client.post("/api/v1/evaluation/start", json=evaluation_config)
            assert start_response.status_code == 200
            run_ids.append(start_response.json()["run_id"])
        
        # List all runs
        runs_response = self.client.get("/api/v1/runs")
        assert runs_response.status_code == 200
        
        runs_data = runs_response.json()
        assert len(runs_data["runs"]) >= 3
        
        # Verify our runs are in the list
        listed_run_ids = [run["run_id"] for run in runs_data["runs"]]
        for run_id in run_ids:
            assert run_id in listed_run_ids
        
        # Delete one run
        delete_response = self.client.delete(f"/api/v1/runs/{run_ids[0]}")
        assert delete_response.status_code == 200
        
        delete_data = delete_response.json()
        assert delete_data["deleted"] == True
        assert delete_data["run_id"] == run_ids[0]
        
        # Verify it's deleted
        status_response = self.client.get(f"/api/v1/evaluation/{run_ids[0]}/status")
        assert status_response.status_code == 404
        
        # Other runs should still exist
        for run_id in run_ids[1:]:
            status_response = self.client.get(f"/api/v1/evaluation/{run_id}/status")
            assert status_response.status_code == 200


class TestWorkflowErrorHandling:
    """Test error handling in complete workflows"""
    
    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)
        evaluation_runs._runs.clear()
    
    def test_invalid_corpus_workflow(self):
        """Test workflow with invalid corpus data"""
        # Test with malicious content
        malicious_corpus = {
            "text": "<script>alert('xss')</script>Some legitimate content here",
            "name": "Malicious Test"
        }
        
        upload_response = self.client.post("/api/v1/corpus/upload", json=malicious_corpus)
        assert upload_response.status_code == 422
        
        # Test with empty content
        empty_corpus = {
            "text": "",
            "name": "Empty Test"
        }
        
        upload_response = self.client.post("/api/v1/corpus/upload", json=empty_corpus)
        assert upload_response.status_code == 422
    
    def test_invalid_evaluation_config_workflow(self):
        """Test workflow with invalid evaluation configuration"""
        valid_corpus = "This is a valid test corpus for evaluation."
        
        # Test with invalid evaluation type
        invalid_config = {
            "corpus_text": valid_corpus,
            "eval_type": "nonexistent_type",
            "num_questions": 5
        }
        
        start_response = self.client.post("/api/v1/evaluation/start", json=invalid_config)
        assert start_response.status_code == 422
        
        # Test with invalid parameters
        invalid_params = {
            "corpus_text": valid_corpus,
            "num_questions": 500,  # Exceeds limit
            "temperature": 3.0     # Exceeds limit
        }
        
        start_response = self.client.post("/api/v1/evaluation/start", json=invalid_params)
        assert start_response.status_code == 422
    
    def test_nonexistent_resource_workflow(self):
        """Test workflow with requests for nonexistent resources"""
        fake_run_id = "nonexistent-run-id"
        
        # Test getting status for nonexistent run
        status_response = self.client.get(f"/api/v1/evaluation/{fake_run_id}/status")
        assert status_response.status_code == 404
        
        # Test getting results for nonexistent run
        results_response = self.client.get(f"/api/v1/evaluation/{fake_run_id}/results")
        assert results_response.status_code == 404
        
        # Test deleting nonexistent run
        delete_response = self.client.delete(f"/api/v1/runs/{fake_run_id}")
        assert delete_response.status_code == 404


class TestWorkflowPerformance:
    """Test workflow performance and concurrency"""
    
    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)
        evaluation_runs._runs.clear()
    
    def test_concurrent_evaluations(self):
        """Test multiple concurrent evaluations"""
        import threading
        import queue
        
        # Configuration for concurrent tests
        evaluation_configs = []
        for i in range(3):
            config = {
                "corpus_text": f"Concurrent test corpus {i} with different content for parallel evaluation testing.",
                "eval_type": "domain_knowledge",
                "num_questions": 2,
                "use_agentic": False,
                "run_name": f"Concurrent Test {i}"
            }
            evaluation_configs.append(config)
        
        # Function to start evaluation
        def start_evaluation(config, result_queue):
            try:
                response = self.client.post("/api/v1/evaluation/start", json=config)
                result_queue.put(("success", response.json()))
            except Exception as e:
                result_queue.put(("error", str(e)))
        
        # Start evaluations concurrently
        threads = []
        result_queue = queue.Queue()
        
        for config in evaluation_configs:
            thread = threading.Thread(target=start_evaluation, args=(config, result_queue))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # Verify all evaluations started successfully
        assert len(results) == 3
        for status, result in results:
            assert status == "success"
            assert "run_id" in result
            assert result["status"] == "queued"
        
        # Wait for some evaluations to complete
        run_ids = [result[1]["run_id"] for result in results]
        
        # Check that all evaluations are tracked
        runs_response = self.client.get("/api/v1/runs")
        runs_data = runs_response.json()
        listed_run_ids = [run["run_id"] for run in runs_data["runs"]]
        
        for run_id in run_ids:
            assert run_id in listed_run_ids
    
    def test_large_corpus_workflow(self):
        """Test workflow with larger corpus"""
        # Create a larger corpus (but not too large for testing)
        large_corpus_parts = [
            "Machine learning is a powerful technology that enables computers to learn from data.",
            "Deep learning uses neural networks with multiple layers to process complex patterns.",
            "Natural language processing helps computers understand and generate human language.",
            "Computer vision allows machines to interpret and understand visual information.",
            "Reinforcement learning trains agents to make decisions through trial and error.",
            "Supervised learning uses labeled data to train predictive models.",
            "Unsupervised learning discovers hidden patterns in unlabeled data.",
        ] * 10  # Repeat to make it larger
        
        large_corpus = " ".join(large_corpus_parts)
        
        # Upload large corpus
        corpus_data = {
            "text": large_corpus,
            "name": "Large Corpus Test",
            "description": "Testing with larger corpus content"
        }
        
        upload_response = self.client.post("/api/v1/corpus/upload", json=corpus_data)
        assert upload_response.status_code == 200
        
        # Start evaluation
        evaluation_config = {
            "corpus_text": large_corpus,
            "eval_type": "domain_knowledge", 
            "num_questions": 5,
            "use_agentic": False,  # Standard for performance
            "run_name": "Large Corpus Test"
        }
        
        start_response = self.client.post("/api/v1/evaluation/start", json=evaluation_config)
        assert start_response.status_code == 200
        
        # Monitor completion (may take longer)
        run_id = start_response.json()["run_id"]
        max_wait = 45  # Longer timeout for large corpus
        
        final_status = None
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = self.client.get(f"/api/v1/evaluation/{run_id}/status")
            if status_response.status_code == 200:
                status = status_response.json()["status"]
                if status == "completed":
                    final_status = "completed"
                    break
                elif status == "error":
                    pytest.fail(f"Large corpus evaluation failed")
            
            time.sleep(2)
        
        if final_status == "completed":
            # Verify results
            results_response = self.client.get(f"/api/v1/evaluation/{run_id}/results")
            assert results_response.status_code == 200
            
            results_data = results_response.json()
            assert results_data["aggregate_metrics"]["mean_score"] > 0
            assert results_data["aggregate_metrics"]["num_samples"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])