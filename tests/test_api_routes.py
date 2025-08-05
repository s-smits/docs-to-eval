"""
Comprehensive API Routes Testing for docs-to-eval UI workflow
Tests all FastAPI endpoints and their integration with the evaluation pipeline
"""

import pytest
import json
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from io import BytesIO

from fastapi.testclient import TestClient
from fastapi import UploadFile

# Import the FastAPI app and components
from docs_to_eval.ui_api.main import app
from docs_to_eval.ui_api.routes import evaluation_runs
from docs_to_eval.utils.config import EvaluationType


class TestAPIEndpoints:
    """Test all API endpoints for proper functionality"""
    
    def setup_method(self):
        """Set up test client and clear evaluation runs"""
        self.client = TestClient(app)
        evaluation_runs._runs.clear()  # Clear previous test runs
    
    def test_api_root(self):
        """Test the API root endpoint"""
        response = self.client.get("/api/v1/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "docs-to-eval API"
        assert "version" in data
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_corpus_upload_text(self):
        """Test corpus text upload endpoint"""
        corpus_data = {
            "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "name": "ML Test Corpus",
            "description": "Test corpus for machine learning evaluation"
        }
        
        response = self.client.post("/api/v1/corpus/upload", json=corpus_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "corpus_id" in data
        assert data["status"] == "uploaded"
        assert data["chars"] == len(corpus_data["text"])
        assert "primary_type" in data
    
    def test_corpus_upload_validation(self):
        """Test corpus upload validation"""
        # Test empty text
        response = self.client.post("/api/v1/corpus/upload", json={"text": ""})
        assert response.status_code == 422
        
        # Test malicious content
        malicious_corpus = {
            "text": "<script>alert('xss')</script>Some content",
            "name": "Test"
        }
        response = self.client.post("/api/v1/corpus/upload", json=malicious_corpus)
        assert response.status_code == 422
    
    def test_corpus_file_upload(self):
        """Test file upload endpoint"""
        test_content = "Artificial intelligence systems can learn from data and make predictions."
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                files = {"file": ("test.txt", f, "text/plain")}
                data = {"name": "Test File Corpus"}
                
                response = self.client.post("/api/v1/corpus/upload-file", files=files, data=data)
                assert response.status_code == 200
                
                result = response.json()
                assert "corpus_id" in result
                assert result["status"] == "uploaded"
                assert result["chars"] == len(test_content)
        finally:
            Path(temp_path).unlink()
    
    def test_evaluation_start(self):
        """Test starting an evaluation"""
        evaluation_request = {
            "corpus_text": "Deep learning uses neural networks with multiple layers to learn complex patterns.",
            "eval_type": "domain_knowledge",
            "num_questions": 5,
            "use_agentic": True,
            "temperature": 0.7,
            "run_name": "Test Evaluation",
            "finetune_test_set_enabled": True,
            "finetune_test_set_percentage": 0.2,
            "finetune_random_seed": 42
        }
        
        response = self.client.post("/api/v1/evaluation/start", json=evaluation_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "run_id" in data
        assert data["status"] == "queued"
        assert "websocket_url" in data
        
        # Wait a moment for background task to start
        import time
        time.sleep(0.5)
        
        # Check that run was stored
        run_id = data["run_id"]
        assert run_id in evaluation_runs._runs
    
    def test_evaluation_status(self):
        """Test getting evaluation status"""
        # First start an evaluation
        evaluation_request = {
            "corpus_text": "Neural networks are computational models inspired by biological neural networks.",
            "num_questions": 3,
            "use_agentic": False
        }
        
        start_response = self.client.post("/api/v1/evaluation/start", json=evaluation_request)
        run_id = start_response.json()["run_id"]
        
        # Check status
        response = self.client.get(f"/api/v1/evaluation/{run_id}/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["run_id"] == run_id
        assert "status" in data
        assert data["status"] in ["queued", "running", "completed", "error"]
    
    def test_evaluation_status_not_found(self):
        """Test getting status for non-existent evaluation"""
        response = self.client.get("/api/v1/evaluation/nonexistent/status")
        assert response.status_code == 404
    
    def test_get_evaluation_types(self):
        """Test getting available evaluation types"""
        response = self.client.get("/api/v1/types/evaluation")
        assert response.status_code == 200
        
        data = response.json()
        assert "evaluation_types" in data
        assert len(data["evaluation_types"]) > 0
        
        # Check that all returned types are valid
        for eval_type in data["evaluation_types"]:
            assert "value" in eval_type
            assert "display_name" in eval_type
            assert "description" in eval_type
    
    def test_config_endpoints(self):
        """Test configuration management endpoints"""
        # Test getting default config
        response = self.client.get("/api/v1/config/default")
        assert response.status_code == 200
        
        data = response.json()
        assert "eval_type" in data
        assert "generation" in data
        assert "llm" in data
        
        # Test getting current config
        response = self.client.get("/api/v1/config/current")
        assert response.status_code == 200
    
    def test_api_key_test(self):
        """Test API key testing endpoint"""
        api_test_data = {
            "api_key": "test-key",
            "model_name": "google/gemini-2.5-flash",
            "base_url": "https://openrouter.ai/api/v1"
        }
        
        response = self.client.post("/api/v1/config/test-api-key", json=api_test_data)
        # This might return 200 or 400 depending on the key validity
        assert response.status_code in [200, 400]
        
        data = response.json()
        assert "valid" in data
        assert isinstance(data["valid"], bool)
    
    def test_list_evaluation_runs(self):
        """Test listing evaluation runs"""
        # Start a few evaluations first
        for i in range(3):
            evaluation_request = {
                "corpus_text": f"Test corpus {i} for machine learning evaluation.",
                "num_questions": 2,
                "run_name": f"Test Run {i}"
            }
            self.client.post("/api/v1/evaluation/start", json=evaluation_request)
        
        # List runs
        response = self.client.get("/api/v1/runs")
        assert response.status_code == 200
        
        data = response.json()
        assert "runs" in data
        assert len(data["runs"]) >= 3  # At least the ones we just created
        
        # Check structure of returned runs
        for run in data["runs"]:
            assert "run_id" in run
            assert "status" in run
            assert "start_time" in run
    
    def test_delete_evaluation_run(self):
        """Test deleting an evaluation run"""
        # Start an evaluation
        evaluation_request = {
            "corpus_text": "Test corpus for deletion.",
            "num_questions": 1
        }
        
        start_response = self.client.post("/api/v1/evaluation/start", json=evaluation_request)
        run_id = start_response.json()["run_id"]
        
        # Delete the run
        response = self.client.delete(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["deleted"] == True
        assert data["run_id"] == run_id
        
        # Verify it's actually deleted
        response = self.client.get(f"/api/v1/evaluation/{run_id}/status")
        assert response.status_code == 404


class TestEvaluationWorkflow:
    """Test complete evaluation workflow end-to-end"""
    
    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)
        evaluation_runs._runs.clear()
    
    def test_complete_evaluation_workflow(self):
        """Test the complete evaluation workflow from upload to results"""
        # Step 1: Upload corpus
        corpus_data = {
            "text": "Quantum computing uses quantum-mechanical phenomena to perform operations on data. Quantum computers could solve certain problems exponentially faster than classical computers.",
            "name": "Quantum Computing Corpus",
            "description": "Educational content about quantum computing"
        }
        
        upload_response = self.client.post("/api/v1/corpus/upload", json=corpus_data)
        assert upload_response.status_code == 200
        
        # Step 2: Start evaluation
        evaluation_request = {
            "corpus_text": corpus_data["text"],
            "eval_type": "domain_knowledge",
            "num_questions": 3,
            "use_agentic": False,  # Use standard generation for faster testing
            "temperature": 0.5,
            "run_name": "Quantum Computing Test",
            "finetune_test_set_enabled": True,
            "finetune_test_set_percentage": 0.3
        }
        
        start_response = self.client.post("/api/v1/evaluation/start", json=evaluation_request)
        assert start_response.status_code == 200
        
        run_id = start_response.json()["run_id"]
        
        # Step 3: Monitor status
        import time
        max_wait = 30  # 30 seconds max wait
        waited = 0
        
        while waited < max_wait:
            status_response = self.client.get(f"/api/v1/evaluation/{run_id}/status")
            assert status_response.status_code == 200
            
            status = status_response.json()["status"]
            if status == "completed":
                break
            elif status == "error":
                pytest.fail(f"Evaluation failed: {status_response.json()}")
            
            time.sleep(1)
            waited += 1
        
        # Step 4: Get results
        results_response = self.client.get(f"/api/v1/evaluation/{run_id}/results")
        assert results_response.status_code == 200
        
        results = results_response.json()
        assert "run_id" in results
        assert "aggregate_metrics" in results
        assert "individual_results" in results
        
        # Verify the fix worked - should have non-zero scores
        assert results["aggregate_metrics"]["mean_score"] > 0
        
        # Step 5: Check finetune test set
        finetune_response = self.client.get(f"/api/v1/evaluation/{run_id}/finetune-test-set")
        assert finetune_response.status_code == 200
        
        finetune_data = finetune_response.json()
        assert finetune_data["enabled"] == True
        assert finetune_data["test_percentage"] == 0.3


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)
        evaluation_runs._runs.clear()
    
    def test_invalid_json_requests(self):
        """Test handling of invalid JSON requests"""
        response = self.client.post("/api/v1/corpus/upload", data="invalid json")
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        # Missing corpus_text
        response = self.client.post("/api/v1/evaluation/start", json={
            "num_questions": 5
        })
        assert response.status_code == 422
    
    def test_invalid_evaluation_type(self):
        """Test handling of invalid evaluation type"""
        evaluation_request = {
            "corpus_text": "Test content",
            "eval_type": "invalid_type",
            "num_questions": 3
        }
        
        response = self.client.post("/api/v1/evaluation/start", json=evaluation_request)
        assert response.status_code == 422
    
    def test_corpus_too_large(self):
        """Test handling of corpus that's too large"""
        large_corpus = {
            "text": "A" * (11 * 1024 * 1024),  # 11MB - exceeds 10MB limit
            "name": "Large Corpus"
        }
        
        response = self.client.post("/api/v1/corpus/upload", json=large_corpus)
        assert response.status_code == 413  # Request Entity Too Large
    
    def test_invalid_file_upload(self):
        """Test handling of invalid file uploads"""
        # Test with empty file
        files = {"file": ("empty.txt", BytesIO(b""), "text/plain")}
        response = self.client.post("/api/v1/corpus/upload-file", files=files)
        assert response.status_code == 422
    
    def test_nonexistent_evaluation_results(self):
        """Test getting results for non-existent evaluation"""
        response = self.client.get("/api/v1/evaluation/nonexistent/results")
        assert response.status_code == 404
    
    def test_invalid_parameters(self):
        """Test invalid parameter ranges"""
        evaluation_request = {
            "corpus_text": "Test content",
            "num_questions": 300,  # Exceeds max of 200
            "temperature": 5.0  # Exceeds max of 2.0
        }
        
        response = self.client.post("/api/v1/evaluation/start", json=evaluation_request)
        assert response.status_code == 422


class TestVerificationFix:
    """Test that the enum-to-string verification fix is working"""
    
    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)
        evaluation_runs._runs.clear()
    
    def test_domain_knowledge_verification(self):
        """Test that domain_knowledge evaluation type works correctly"""
        evaluation_request = {
            "corpus_text": "Etruscan civilization flourished in central Italy. The Etruscans were known for their art, architecture, and religious practices.",
            "eval_type": "domain_knowledge",
            "num_questions": 2,
            "use_agentic": False,
            "temperature": 0.3
        }
        
        start_response = self.client.post("/api/v1/evaluation/start", json=evaluation_request)
        assert start_response.status_code == 200
        
        run_id = start_response.json()["run_id"]
        
        # Wait for completion
        import time
        max_wait = 20
        waited = 0
        
        while waited < max_wait:
            status_response = self.client.get(f"/api/v1/evaluation/{run_id}/status")
            status = status_response.json()["status"]
            
            if status == "completed":
                break
            elif status == "error":
                error_data = status_response.json()
                pytest.fail(f"Evaluation failed: {error_data.get('error', 'Unknown error')}")
            
            time.sleep(1)
            waited += 1
        
        # Get results and verify the fix worked
        results_response = self.client.get(f"/api/v1/evaluation/{run_id}/results")
        assert results_response.status_code == 200
        
        results = results_response.json()
        
        # The key test: mean_score should NOT be 0 (the bug we fixed)
        mean_score = results["aggregate_metrics"]["mean_score"]
        assert mean_score > 0, f"Mean score should be > 0, got {mean_score}. The enum-to-string fix may not be working."
        
        # Additional verification: individual results should have non-zero scores
        individual_results = results["individual_results"]
        assert len(individual_results) > 0
        
        for result in individual_results:
            assert "score" in result
            # At least some results should have non-zero scores
        
        non_zero_scores = [r for r in individual_results if r["score"] > 0]
        assert len(non_zero_scores) > 0, "All individual scores are 0, verification fix may not be working"


class TestConfigurationManagement:
    """Test configuration management and API key handling"""
    
    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)
    
    def test_config_update(self):
        """Test updating configuration"""
        config_update = {
            "api_key": "sk-test-key-123",
            "model_name": "google/gemini-2.5-flash",
            "temperature": 0.8
        }
        
        response = self.client.post("/api/v1/config/update", json=config_update)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "updated"
        assert "config" in data
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid temperature
        invalid_config = {
            "temperature": 3.0  # Exceeds max of 2.0
        }
        
        response = self.client.post("/api/v1/config/update", json=invalid_config)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])