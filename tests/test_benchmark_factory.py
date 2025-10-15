"""
Unit tests for BenchmarkGeneratorFactory and related components.
Tests the strategy pattern for choosing between standard and agentic generators.
"""

import pytest
from unittest.mock import Mock, patch

from docs_to_eval.core.benchmarks import (
    BenchmarkGeneratorFactory, 
    BenchmarkGenerator,
    AgenticGeneratorWrapper,
    DomainKnowledgeBenchmarkGenerator,
    FactualQABenchmarkGenerator,
    MathematicalBenchmarkGenerator
)
from docs_to_eval.core.evaluation import EvaluationType
from docs_to_eval.utils.config import create_default_config


@pytest.fixture
def sample_corpus():
    """Sample corpus for testing."""
    return """
    Machine learning is a subset of artificial intelligence that focuses on algorithms
    that can learn and improve from experience without being explicitly programmed.
    The primary goal is to enable computers to learn automatically without human
    intervention or assistance and adjust actions accordingly.
    """


class TestBenchmarkGeneratorFactory:
    """Test cases for BenchmarkGeneratorFactory."""
    
    def test_create_standard_generator(self):
        """Test creation of standard generators."""
        generator = BenchmarkGeneratorFactory.create_generator(
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
            use_agentic=False
        )
        
        assert isinstance(generator, DomainKnowledgeBenchmarkGenerator)
        assert generator.eval_type == EvaluationType.DOMAIN_KNOWLEDGE
    
    def test_create_different_standard_generators(self):
        """Test creation of different types of standard generators."""
        test_cases = [
            (EvaluationType.FACTUAL_QA, FactualQABenchmarkGenerator),
            (EvaluationType.MATHEMATICAL, MathematicalBenchmarkGenerator),
            (EvaluationType.DOMAIN_KNOWLEDGE, DomainKnowledgeBenchmarkGenerator),
        ]
        
        for eval_type, expected_class in test_cases:
            generator = BenchmarkGeneratorFactory.create_generator(
                eval_type=eval_type,
                use_agentic=False
            )
            assert isinstance(generator, expected_class)
            assert generator.eval_type == eval_type
    
    def test_create_agentic_generator_success(self):
        """Test successful creation of agentic generator."""
        mock_agentic_gen = Mock()
        mock_agentic_gen.eval_type = EvaluationType.DOMAIN_KNOWLEDGE
        
        with patch('docs_to_eval.core.benchmarks.AgenticBenchmarkGenerator', return_value=mock_agentic_gen):
            generator = BenchmarkGeneratorFactory.create_generator(
                eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
                use_agentic=True
            )
            
            assert isinstance(generator, AgenticGeneratorWrapper)
            assert generator.eval_type == EvaluationType.DOMAIN_KNOWLEDGE
    
    def test_create_agentic_generator_fallback(self):
        """Test fallback to standard generator when agentic fails."""
        with patch('docs_to_eval.core.benchmarks.AgenticBenchmarkGenerator', side_effect=ImportError("Agentic not available")):
            generator = BenchmarkGeneratorFactory.create_generator(
                eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
                use_agentic=True
            )
            
            # Should fall back to standard generator
            assert isinstance(generator, DomainKnowledgeBenchmarkGenerator)
            assert generator.eval_type == EvaluationType.DOMAIN_KNOWLEDGE
    
    def test_create_generator_with_config(self):
        """Test generator creation using EvaluationConfig."""
        config = create_default_config()
        config.generation.use_agentic = False
        
        generator = BenchmarkGeneratorFactory.create_generator(
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
            config=config
        )
        
        assert isinstance(generator, DomainKnowledgeBenchmarkGenerator)
    
    def test_create_agentic_generator_with_config(self):
        """Test agentic generator creation using EvaluationConfig."""
        config = create_default_config()
        config.generation.use_agentic = True
        
        mock_agentic_gen = Mock()
        with patch('docs_to_eval.core.benchmarks.AgenticBenchmarkGenerator', return_value=mock_agentic_gen):
            generator = BenchmarkGeneratorFactory.create_generator(
                eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
                config=config
            )
            
            assert isinstance(generator, AgenticGeneratorWrapper)
    
    def test_get_available_types(self):
        """Test getting available evaluation types."""
        types = BenchmarkGeneratorFactory.get_available_types()
        
        assert EvaluationType.DOMAIN_KNOWLEDGE in types
        assert EvaluationType.FACTUAL_QA in types
        assert EvaluationType.MATHEMATICAL in types
        assert isinstance(types, list)
    
    def test_get_agentic_types(self):
        """Test getting agentic-supported evaluation types."""
        types = BenchmarkGeneratorFactory.get_agentic_types()
        
        assert EvaluationType.DOMAIN_KNOWLEDGE in types
        assert EvaluationType.FACTUAL_QA in types
        assert isinstance(types, list)
    
    def test_supports_agentic(self):
        """Test checking agentic support for evaluation types."""
        assert BenchmarkGeneratorFactory.supports_agentic(EvaluationType.DOMAIN_KNOWLEDGE)
        assert BenchmarkGeneratorFactory.supports_agentic(EvaluationType.FACTUAL_QA)
        assert BenchmarkGeneratorFactory.supports_agentic(EvaluationType.MATHEMATICAL)


class TestAgenticGeneratorWrapper:
    """Test cases for AgenticGeneratorWrapper."""
    
    def test_wrapper_initialization(self):
        """Test wrapper initialization."""
        mock_agentic_gen = Mock()
        wrapper = AgenticGeneratorWrapper(mock_agentic_gen, EvaluationType.DOMAIN_KNOWLEDGE)
        
        assert wrapper.agentic_generator == mock_agentic_gen
        assert wrapper.eval_type == EvaluationType.DOMAIN_KNOWLEDGE
    
    @pytest.mark.asyncio
    async def test_async_generation(self):
        """Test async generation through wrapper."""
        mock_agentic_gen = Mock()
        mock_item = Mock()
        mock_item.to_dict.return_value = {"question": "Test", "answer": "Answer"}
        mock_agentic_gen.generate_async.return_value = [mock_item]
        
        wrapper = AgenticGeneratorWrapper(mock_agentic_gen, EvaluationType.DOMAIN_KNOWLEDGE)
        items = await wrapper.generate_async("test corpus", 1)
        
        assert len(items) == 1
        mock_agentic_gen.generate_async.assert_called_once_with(
            corpus_text="test corpus",
            num_questions=1
        )
    
    def test_sync_generation_wrapper(self):
        """Test sync generation wrapper."""
        import asyncio
        
        mock_agentic_gen = Mock()
        mock_item = Mock()
        mock_item.to_dict.return_value = {"question": "Test", "answer": "Answer"}
        
        async def mock_generate_async(*args, **kwargs):
            return [mock_item]
        
        mock_agentic_gen.generate_async = mock_generate_async
        
        wrapper = AgenticGeneratorWrapper(mock_agentic_gen, EvaluationType.DOMAIN_KNOWLEDGE)
        items = wrapper.generate_benchmark("test corpus", 1)
        
        assert len(items) == 1
        assert items[0]["question"] == "Test"
    
    def test_convert_agentic_item_dict(self):
        """Test conversion of agentic items with to_dict method."""
        mock_agentic_gen = Mock()
        wrapper = AgenticGeneratorWrapper(mock_agentic_gen, EvaluationType.DOMAIN_KNOWLEDGE)
        
        mock_item = Mock()
        mock_item.to_dict.return_value = {"question": "Test", "answer": "Answer"}
        
        result = wrapper._convert_agentic_item(mock_item)
        assert result == {"question": "Test", "answer": "Answer"}
    
    def test_convert_agentic_item_attrs(self):
        """Test conversion of agentic items with __dict__."""
        mock_agentic_gen = Mock()
        wrapper = AgenticGeneratorWrapper(mock_agentic_gen, EvaluationType.DOMAIN_KNOWLEDGE)
        
        mock_item = Mock()
        del mock_item.to_dict  # Remove to_dict method
        mock_item.__dict__ = {"question": "Test", "answer": "Answer"}
        
        result = wrapper._convert_agentic_item(mock_item)
        assert "question" in result
        assert "answer" in result
    
    def test_get_generation_report(self):
        """Test getting generation report from agentic system."""
        mock_agentic_gen = Mock()
        mock_agentic_gen.get_generation_report.return_value = {"quality_score": 0.85}
        
        wrapper = AgenticGeneratorWrapper(mock_agentic_gen, EvaluationType.DOMAIN_KNOWLEDGE)
        report = wrapper.get_generation_report()
        
        assert report == {"quality_score": 0.85}
    
    def test_get_generation_report_unavailable(self):
        """Test getting generation report when unavailable."""
        mock_agentic_gen = Mock()
        del mock_agentic_gen.get_generation_report  # Remove method
        
        wrapper = AgenticGeneratorWrapper(mock_agentic_gen, EvaluationType.DOMAIN_KNOWLEDGE)
        report = wrapper.get_generation_report()
        
        assert report is None


class TestStandardGenerators:
    """Test cases for standard benchmark generators."""
    
    def test_domain_knowledge_generator(self, sample_corpus):
        """Test domain knowledge generator."""
        generator = DomainKnowledgeBenchmarkGenerator(EvaluationType.DOMAIN_KNOWLEDGE)
        items = generator.generate_benchmark(sample_corpus, num_questions=3)
        
        assert len(items) == 3
        assert all("question" in item for item in items)
        assert all("answer" in item for item in items)
        assert all(item["eval_type"] == EvaluationType.DOMAIN_KNOWLEDGE for item in items)
    
    def test_factual_qa_generator(self, sample_corpus):
        """Test factual Q&A generator."""
        generator = FactualQABenchmarkGenerator(EvaluationType.FACTUAL_QA)
        items = generator.generate_benchmark(sample_corpus, num_questions=2)
        
        assert len(items) == 2
        assert all("question" in item for item in items)
        assert all("answer" in item for item in items)
        assert all(item["eval_type"] == EvaluationType.FACTUAL_QA for item in items)
    
    def test_mathematical_generator(self):
        """Test mathematical generator."""
        math_corpus = "The equation 2 + 3 = 5 shows basic addition. The result of 10 * 4 is 40."
        generator = MathematicalBenchmarkGenerator(EvaluationType.MATHEMATICAL)
        items = generator.generate_benchmark(math_corpus, num_questions=2)
        
        assert len(items) == 2
        assert all("question" in item for item in items)
        assert all("answer" in item for item in items)
        assert all(item["eval_type"] == EvaluationType.MATHEMATICAL for item in items)
    
    def test_generator_create_item(self):
        """Test item creation method."""
        generator = DomainKnowledgeBenchmarkGenerator(EvaluationType.DOMAIN_KNOWLEDGE)
        item = generator.create_item(
            question="What is AI?",
            answer="AI is artificial intelligence",
            context="Some context",
            options=["A", "B", "C"]
        )
        
        assert item["question"] == "What is AI?"
        assert item["answer"] == "AI is artificial intelligence"
        assert item["context"] == "Some context"
        assert item["options"] == ["A", "B", "C"]
        assert item["eval_type"] == EvaluationType.DOMAIN_KNOWLEDGE
        assert "metadata" in item


class TestIntegrationScenarios:
    """Integration test scenarios."""
    
    def test_full_standard_workflow(self, sample_corpus):
        """Test complete standard generation workflow."""
        generator = BenchmarkGeneratorFactory.create_generator(
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
            use_agentic=False
        )
        
        items = generator.generate_benchmark(sample_corpus, num_questions=3)
        
        assert len(items) == 3
        assert all(isinstance(item, dict) for item in items)
        assert all("question" in item and "answer" in item for item in items)
    
    def test_config_driven_generation(self, sample_corpus):
        """Test configuration-driven generation."""
        config = create_default_config()
        config.generation.use_agentic = False
        config.generation.num_questions = 5
        
        generator = BenchmarkGeneratorFactory.create_generator(
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
            config=config
        )
        
        items = generator.generate_benchmark(sample_corpus, config.generation.num_questions)
        assert len(items) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])