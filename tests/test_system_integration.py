"""
Integration tests for the refactored docs-to-eval system
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch

from docs_to_eval.core.evaluation import EvaluationFramework, EvaluationType, BenchmarkConfig
from docs_to_eval.core.classification import EvaluationTypeClassifier
from docs_to_eval.llm.mock_interface import MockLLMInterface, MockLLMEvaluator
from docs_to_eval.utils.config import EvaluationConfig, create_default_config
from docs_to_eval.utils.similarity import calculate_similarity, calculate_multi_similarity
from docs_to_eval.utils.text_processing import clean_text, normalize_answer, extract_keywords


class TestCoreEvaluation:
    """Test core evaluation components"""
    
    def test_evaluation_framework_creation(self):
        """Test creating evaluation framework"""
        framework = EvaluationFramework()
        assert framework.benchmarks == []
        assert framework.results == []
        assert framework.config is None
    
    def test_benchmark_config_creation(self):
        """Test benchmark configuration"""
        sample_corpus = "Machine learning is a subset of artificial intelligence."
        
        framework = EvaluationFramework()
        config = framework.create_benchmark_from_corpus(sample_corpus, num_questions=50)
        
        assert isinstance(config, BenchmarkConfig)
        assert config.num_questions == 50
        assert config.eval_type in EvaluationType
        assert len(config.key_concepts) > 0
        assert len(config.corpus_segments) > 0
    
    def test_benchmark_item_creation(self):
        """Test creating benchmark items"""
        framework = EvaluationFramework()
        
        item = {
            "question": "What is machine learning?",
            "answer": "A subset of AI",
            "eval_type": EvaluationType.FACTUAL_QA
        }
        
        framework.add_benchmark_item(item)
        assert len(framework.benchmarks) == 1
        assert framework.benchmarks[0].question == "What is machine learning?"
    
    def test_evaluation_result_creation(self):
        """Test evaluation result creation"""
        framework = EvaluationFramework()
        
        result = framework.evaluate_response(
            prediction="A subset of AI",
            ground_truth="A subset of artificial intelligence",
            eval_type=EvaluationType.FACTUAL_QA
        )
        
        assert result.score >= 0
        assert result.eval_type == EvaluationType.FACTUAL_QA
        assert len(framework.results) == 1


class TestClassification:
    """Test corpus classification"""
    
    def test_classifier_creation(self):
        """Test creating classifier"""
        classifier = EvaluationTypeClassifier()
        assert classifier.llm is not None
    
    def test_corpus_classification(self):
        """Test classifying corpus"""
        classifier = EvaluationTypeClassifier()
        
        math_corpus = "Calculate 2 + 2. Solve equation x + 5 = 10."
        classification = classifier.classify_corpus(math_corpus)
        
        assert classification.primary_type in EvaluationType
        assert isinstance(classification.secondary_types, list)
        assert 0 <= classification.confidence <= 1
        assert classification.analysis
        assert classification.reasoning
    
    def test_classification_with_examples(self):
        """Test classification with example generation"""
        classifier = EvaluationTypeClassifier()
        
        code_corpus = "def hello_world(): print('Hello, World!')"
        result = classifier.classify_with_examples(code_corpus, num_examples=2)
        
        assert "sample_questions" in result
        assert len(result["sample_questions"]) <= 2


class TestMockLLM:
    """Test mock LLM interface"""
    
    @pytest.mark.asyncio
    async def test_mock_llm_creation(self):
        """Test creating mock LLM"""
        llm = MockLLMInterface()
        assert llm.model_name == "MockLLM-v1"
        assert llm.temperature == 0.7
    
    @pytest.mark.asyncio
    async def test_mock_llm_response(self):
        """Test generating responses"""
        llm = MockLLMInterface()
        
        response = await llm.generate_response("What is 2 + 2?", eval_type="mathematical")
        
        assert response.text
        assert 0 <= response.confidence <= 1
        assert isinstance(response.reasoning_steps, list)
        assert isinstance(response.metadata, dict)
    
    @pytest.mark.asyncio
    async def test_mock_llm_evaluator(self):
        """Test mock LLM evaluator"""
        llm = MockLLMInterface()
        evaluator = MockLLMEvaluator(llm)
        
        benchmark_items = [
            {
                "question": "What is Python?",
                "answer": "A programming language",
                "eval_type": "factual_qa"
            }
        ]
        
        results = await evaluator.evaluate_on_benchmark(benchmark_items, "factual_qa")
        
        assert len(results) == 1
        assert "question" in results[0]
        assert "prediction" in results[0]
        assert "confidence" in results[0]


class TestUtilities:
    """Test utility functions"""
    
    def test_text_processing(self):
        """Test text processing utilities"""
        text = "  Hello,   World!  "
        
        cleaned = clean_text(text)
        assert cleaned == "Hello, World!"
        
        normalized = normalize_answer("Hello, World!")
        assert normalized == "hello world"
        
        keywords = extract_keywords("machine learning artificial intelligence", max_keywords=2)
        assert len(keywords) <= 2
    
    def test_similarity_calculations(self):
        """Test similarity calculations"""
        text1 = "machine learning"
        text2 = "machine learning algorithms"
        
        similarity = calculate_similarity(text1, text2, method="token_overlap")
        assert 0 <= similarity <= 1
        
        multi_sim = calculate_multi_similarity(text1, text2)
        assert isinstance(multi_sim, dict)
        assert "token_overlap" in multi_sim
    
    def test_config_management(self):
        """Test configuration management"""
        config = create_default_config()
        
        assert isinstance(config, EvaluationConfig)
        assert config.eval_type in EvaluationType
        assert config.generation.num_questions > 0
        assert 0 <= config.llm.temperature <= 2


class TestSystemIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_evaluation(self):
        """Test complete evaluation pipeline"""
        # Sample corpus
        corpus_text = """
        Deep learning is a subset of machine learning that uses neural networks.
        It has been particularly successful in image recognition and natural language processing.
        Training requires large datasets and significant computational resources.
        """
        
        # Step 1: Classification
        classifier = EvaluationTypeClassifier()
        classification = classifier.classify_corpus(corpus_text)
        
        assert classification.primary_type in EvaluationType
        
        # Step 2: Benchmark creation
        framework = EvaluationFramework()
        config = framework.create_benchmark_from_corpus(corpus_text, num_questions=5)
        
        # Create mock questions
        questions = []
        for i in range(5):
            questions.append({
                "question": f"Question {i+1} about deep learning",
                "answer": f"Answer {i+1}",
                "eval_type": classification.primary_type
            })
        
        for question in questions:
            framework.add_benchmark_item(question)
        
        # Step 3: LLM evaluation
        llm = MockLLMInterface(temperature=0.3)
        evaluator = MockLLMEvaluator(llm)
        
        llm_results = await evaluator.evaluate_on_benchmark(questions, str(classification.primary_type))
        
        assert len(llm_results) == 5
        
        # Step 4: Verification (simplified)
        for result in llm_results:
            eval_result = framework.evaluate_response(
                result["prediction"],
                result["ground_truth"],
                classification.primary_type
            )
            assert eval_result.score >= 0
        
        # Step 5: Aggregate metrics
        aggregate_metrics = framework.compute_aggregate_metrics()
        
        assert isinstance(aggregate_metrics, dict)
        assert len(framework.results) == 5
        
        # Step 6: Report generation
        report = framework.get_benchmark_report()
        
        assert "config" in report
        assert "num_benchmarks" in report
        assert "aggregate_metrics" in report
    
    def test_config_validation(self):
        """Test configuration validation"""
        from docs_to_eval.utils.config import validate_config
        
        config = create_default_config()
        warnings = validate_config(config)
        
        assert isinstance(warnings, list)
        # Should have no warnings for default config
        assert len(warnings) == 0
    
    def test_logging_integration(self):
        """Test logging system"""
        from docs_to_eval.utils.logging import get_logger, ProgressTracker
        
        logger = get_logger("test")
        assert logger is not None
        
        tracker = ProgressTracker(10, logger)
        tracker.update(5, "Test progress")
        tracker.finish()
        
        assert tracker.completed == 10
    
    @pytest.mark.asyncio
    async def test_api_components(self):
        """Test API components"""
        from docs_to_eval.ui_api.websockets import ConnectionManager, ProgressNotifier
        
        # Test connection manager
        manager = ConnectionManager()
        assert len(manager.active_connections) == 0
        
        # Test progress notifier
        notifier = ProgressNotifier(manager, "test_run")
        
        # These would normally send to WebSocket, but we just test they don't crash
        await notifier.send_phase_start("test_phase", "Test description")
        await notifier.send_progress_update("test_phase", 1, 10, "Test message")
        await notifier.send_phase_complete("test_phase", 1.0)


@pytest.fixture
def sample_corpus():
    """Sample corpus for testing"""
    return """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    """


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for tests"""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


class TestFileOperations:
    """Test file operations and data persistence"""
    
    def test_results_serialization(self, temp_output_dir):
        """Test saving and loading results"""
        framework = EvaluationFramework()
        
        # Create some test data
        test_item = {
            "question": "Test question",
            "answer": "Test answer",
            "eval_type": EvaluationType.FACTUAL_QA
        }
        
        framework.add_benchmark_item(test_item)
        
        result = framework.evaluate_response(
            "Test response",
            "Test answer",
            EvaluationType.FACTUAL_QA
        )
        
        # Generate report
        report = framework.get_benchmark_report()
        
        # Save to file
        output_file = temp_output_dir / "test_results.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Verify file exists and is valid JSON
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            loaded_report = json.load(f)
        
        assert loaded_report["num_benchmarks"] == 1
        assert loaded_report["num_results"] == 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])