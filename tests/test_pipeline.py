"""
Unit tests for the core Pipeline class.
Tests the unified orchestration logic and configuration-driven workflow.
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from docs_to_eval.core.pipeline import EvaluationPipeline, PipelineFactory
from docs_to_eval.utils.config import EvaluationConfig, create_default_config
from docs_to_eval.core.evaluation import EvaluationType


@pytest.fixture
def sample_corpus():
    """Sample corpus text for testing."""
    return """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    """


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = create_default_config()
    config.generation.num_questions = 5
    config.generation.use_agentic = False  # Use standard generation for unit tests
    config.system.log_level = "ERROR"  # Reduce log noise in tests
    return config


class TestEvaluationPipeline:
    """Test cases for EvaluationPipeline class."""
    
    def test_pipeline_initialization(self, test_config):
        """Test pipeline initializes correctly with config."""
        pipeline = EvaluationPipeline(test_config)
        
        assert pipeline.config == test_config
        assert pipeline.run_id is not None
        assert len(pipeline.run_id) == 8  # UUID prefix
        assert pipeline.classifier is not None
        assert pipeline.generator_factory is not None
        
    @pytest.mark.asyncio
    async def test_corpus_classification(self, test_config, sample_corpus):
        """Test corpus classification phase."""
        pipeline = EvaluationPipeline(test_config)
        
        with patch.object(pipeline.classifier, 'classify_corpus') as mock_classify:
            mock_classification = Mock()
            mock_classification.primary_type = EvaluationType.DOMAIN_KNOWLEDGE
            mock_classification.to_dict.return_value = {"primary_type": "domain_knowledge"}
            mock_classify.return_value = mock_classification
            
            result = await pipeline._classify_corpus(sample_corpus, Mock())
            
            assert result.primary_type == EvaluationType.DOMAIN_KNOWLEDGE
            mock_classify.assert_called_once_with(sample_corpus)
    
    @pytest.mark.asyncio
    async def test_benchmark_generation_standard(self, test_config, sample_corpus):
        """Test benchmark generation with standard generator."""
        pipeline = EvaluationPipeline(test_config)
        
        mock_classification = Mock()
        mock_classification.primary_type = EvaluationType.DOMAIN_KNOWLEDGE
        
        with patch.object(pipeline.generator_factory, 'create_generator') as mock_create:
            mock_generator = Mock()
            mock_generator.generate_questions.return_value = [
                {"question": "What is AI?", "answer": "AI is...", "eval_type": "domain_knowledge"},
                {"question": "How does AI work?", "answer": "AI works...", "eval_type": "domain_knowledge"}
            ]
            mock_create.return_value = mock_generator
            
            questions = await pipeline._generate_benchmark(sample_corpus, mock_classification, Mock())
            
            assert len(questions) == 2
            assert questions[0]["question"] == "What is AI?"
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_benchmark_generation_agentic(self, test_config, sample_corpus):
        """Test benchmark generation with agentic generator."""
        # Enable agentic generation
        test_config.generation.use_agentic = True
        pipeline = EvaluationPipeline(test_config)
        
        mock_classification = Mock()
        mock_classification.primary_type = EvaluationType.DOMAIN_KNOWLEDGE
        
        with patch.object(pipeline.generator_factory, 'create_generator') as mock_create:
            mock_generator = Mock()
            mock_benchmark_item = Mock()
            mock_benchmark_item.to_dict.return_value = {
                "question": "Advanced AI question",
                "answer": "Advanced answer",
                "eval_type": "domain_knowledge"
            }
            
            # Mock async generate method
            mock_generator.generate_async = AsyncMock(return_value=[mock_benchmark_item])
            mock_create.return_value = mock_generator
            
            questions = await pipeline._generate_benchmark(sample_corpus, mock_classification, Mock())
            
            assert len(questions) == 1
            assert questions[0]["question"] == "Advanced AI question"
    
    @pytest.mark.asyncio
    async def test_llm_evaluation(self, test_config):
        """Test LLM evaluation phase."""
        pipeline = EvaluationPipeline(test_config)
        
        sample_questions = [
            {"question": "What is AI?", "answer": "AI is intelligence by machines"},
            {"question": "How does ML work?", "answer": "ML learns from data"}
        ]
        
        results = await pipeline._evaluate_with_llm(sample_questions, Mock())
        
        assert len(results) == 2
        assert all("prediction" in result for result in results)
        assert all("confidence" in result for result in results)
        assert all(result["confidence"] == 0.8 for result in results)
    
    @pytest.mark.asyncio 
    async def test_response_verification(self, test_config):
        """Test response verification phase."""
        pipeline = EvaluationPipeline(test_config)
        
        sample_results = [
            {"question": "Q1", "prediction": "P1", "ground_truth": "GT1"},
            {"question": "Q2", "prediction": "P2", "ground_truth": "GT2"}
        ]
        
        verifications = await pipeline._verify_responses(sample_results, Mock())
        
        assert len(verifications) == 2
        assert all("score" in v for v in verifications)
        assert all("method" in v for v in verifications)
        assert all(v["method"] == "exact_match" for v in verifications)
    
    def test_results_aggregation(self, test_config):
        """Test results aggregation."""
        pipeline = EvaluationPipeline(test_config)
        
        mock_classification = Mock()
        mock_classification.to_dict.return_value = {"primary_type": "domain_knowledge"}
        
        verification_results = [
            {"score": 0.8}, {"score": 0.6}, {"score": 0.9}
        ]
        
        results = pipeline._aggregate_results(mock_classification, verification_results, Mock())
        
        assert results["run_id"] == pipeline.run_id
        assert results["aggregate_metrics"]["mean_score"] == 0.7666666666666667
        assert results["aggregate_metrics"]["min_score"] == 0.6
        assert results["aggregate_metrics"]["max_score"] == 0.9
        assert results["aggregate_metrics"]["num_samples"] == 3
    
    def test_std_dev_calculation(self, test_config):
        """Test standard deviation calculation."""
        pipeline = EvaluationPipeline(test_config)
        
        # Test with known values
        scores = [1.0, 2.0, 3.0]
        std_dev = pipeline._calculate_std_dev(scores)
        expected_std_dev = (2/3) ** 0.5  # Manual calculation
        assert abs(std_dev - expected_std_dev) < 0.001
        
        # Test with empty list
        assert pipeline._calculate_std_dev([]) == 0.0
    
    @pytest.mark.asyncio
    async def test_full_pipeline_run(self, test_config, sample_corpus, tmp_path):
        """Test complete pipeline execution."""
        pipeline = EvaluationPipeline(test_config)
        
        # Mock all major components to avoid real LLM calls
        with patch.object(pipeline.classifier, 'classify_corpus') as mock_classify, \
             patch.object(pipeline.generator_factory, 'create_generator') as mock_create_gen:
            
            # Setup mocks
            mock_classification = Mock()
            mock_classification.primary_type = EvaluationType.DOMAIN_KNOWLEDGE
            mock_classification.to_dict.return_value = {"primary_type": "domain_knowledge"}
            mock_classify.return_value = mock_classification
            
            mock_generator = Mock()
            mock_generator.generate_questions.return_value = [
                {"question": "Test question", "answer": "Test answer"}
            ]
            mock_create_gen.return_value = mock_generator
            
            # Run pipeline
            results = await pipeline.run_async(sample_corpus, tmp_path)
            
            # Verify results structure
            assert "run_id" in results
            assert "config" in results
            assert "classification" in results
            assert "aggregate_metrics" in results
            assert "individual_results" in results
            
            # Verify output file was created
            results_files = list(tmp_path.glob("evaluation_results_*.json"))
            assert len(results_files) == 1
            
            # Verify JSON is valid
            with open(results_files[0]) as f:
                saved_results = json.load(f)
            assert saved_results["run_id"] == results["run_id"]


class TestPipelineFactory:
    """Test cases for PipelineFactory."""
    
    def test_create_pipeline_with_config(self, test_config):
        """Test pipeline creation with configuration."""
        pipeline = PipelineFactory.create_pipeline(test_config)
        
        assert isinstance(pipeline, EvaluationPipeline)
        assert pipeline.config == test_config
    
    def test_create_default_pipeline(self):
        """Test creation of pipeline with default configuration."""
        pipeline = PipelineFactory.create_default_pipeline()
        
        assert isinstance(pipeline, EvaluationPipeline)
        assert pipeline.config is not None
        assert pipeline.config.generation.num_questions == 50  # Default value


class TestPipelineIntegration:
    """Integration tests for pipeline with real components."""
    
    @pytest.mark.asyncio
    async def test_pipeline_with_mock_agentic(self, test_config, sample_corpus, tmp_path):
        """Test pipeline integration with mocked agentic system."""
        test_config.generation.use_agentic = True
        
        # Mock the agentic import to simulate successful agentic generation
        mock_agentic_gen = Mock()
        mock_item = Mock()
        mock_item.to_dict.return_value = {
            "question": "Agentic question",
            "answer": "Agentic answer",
            "difficulty": "intermediate"
        }
        mock_agentic_gen.generate_async = AsyncMock(return_value=[mock_item])
        
        with patch('docs_to_eval.core.benchmarks.AgenticBenchmarkGenerator', return_value=mock_agentic_gen):
            pipeline = EvaluationPipeline(test_config)
            results = await pipeline.run_async(sample_corpus, tmp_path)
            
            assert results["aggregate_metrics"]["num_samples"] == 1
            assert "agentic" in results.get("config", {}).get("generation", {}).get("method", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])