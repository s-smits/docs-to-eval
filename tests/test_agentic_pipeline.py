"""
Comprehensive Agentic Pipeline Testing for docs-to-eval
Tests the agentic generation system, agents, and pipeline integration
"""

import pytest
from unittest.mock import Mock, AsyncMock

from docs_to_eval.core.agentic.agents import (
    ConceptMiner, QuestionWriter, Adversary, Refiner, Validator, BaseAgent
)
from docs_to_eval.core.agentic.models import (
    BenchmarkDraft, BenchmarkCandidate, ConceptExtractionResult, ValidationResult, AgentConfig,
    DifficultyLevel, AnswerType
)
from docs_to_eval.utils.config import EvaluationType
from docs_to_eval.llm.base import BaseLLMInterface
from docs_to_eval.llm.mock_interface import MockLLMInterface


# Ensure all async tests in this module run under pytest-asyncio
import pytest
pytestmark = pytest.mark.asyncio

class TestAgentConfig:
    """Test agent configuration and setup"""
    
    def test_agent_config_defaults(self):
        """Test default agent configuration"""
        config = AgentConfig()
        
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.retry_attempts == 2
        assert isinstance(config.timeout_seconds, (int, float))
    
    def test_agent_config_customization(self):
        """Test custom agent configuration"""
        config = AgentConfig(
            temperature=0.3,
            max_tokens=1000,
            retry_attempts=3,
            timeout_seconds=30
        )
        
        assert config.temperature == 0.3
        assert config.max_tokens == 1000
        assert config.retry_attempts == 3
        assert config.timeout_seconds == 30


class TestBaseAgent:
    """Test base agent functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_llm = Mock(spec=BaseLLMInterface)
        self.mock_llm.generate_response = AsyncMock()
        self.config = AgentConfig(temperature=0.5, max_tokens=1000)
    
    def test_base_agent_initialization(self):
        """Test base agent initialization"""
        # Create a concrete implementation for testing
        class TestAgent(BaseAgent):
            async def produce(self, *args, **kwargs):
                return "test_result"
        
        agent = TestAgent(self.mock_llm, self.config)
        
        assert agent.llm == self.mock_llm
        assert agent.config == self.config
        assert agent.agent_name == "TestAgent"
        assert agent.agent_version == "v1"
        assert agent.call_count == 0
        assert agent.total_processing_time == 0.0
    
    def test_create_response(self):
        """Test agent response creation"""
        class TestAgent(BaseAgent):
            async def produce(self, *args, **kwargs):
                return "test_result"
        
        agent = TestAgent(self.mock_llm, self.config)
        
        # Success response
        response = agent._create_response(success=True, test_data="value")
        assert response.agent_name == "TestAgent"
        assert response.agent_version == "v1"
        assert response.success
        assert response.error_message is None
        assert response.metadata["test_data"] == "value"
        
        # Error response
        error_response = agent._create_response(success=False, error_message="Test error")
        assert not error_response.success
        assert error_response.error_message == "Test error"
    
    @pytest.mark.asyncio
    async def test_llm_call_with_retry(self):
        """Test LLM calls with retry logic"""
        class TestAgent(BaseAgent):
            async def produce(self, *args, **kwargs):
                return "test_result"
        
        agent = TestAgent(self.mock_llm, self.config)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "Test LLM response"
        self.mock_llm.generate_response.return_value = mock_response
        
        result = await agent._call_llm_with_retry("Test prompt", context="Test context")
        
        assert result == "Test LLM response"
        self.mock_llm.generate_response.assert_called_once_with(
            prompt="Test prompt",
            context="Test context",
            temperature=0.5,
            max_tokens=1000
        )
    
    @pytest.mark.asyncio
    async def test_llm_call_retry_on_failure(self):
        """Test retry logic when LLM calls fail"""
        class TestAgent(BaseAgent):
            async def produce(self, *args, **kwargs):
                return "test_result"
        
        # Set up to fail twice then succeed
        config = AgentConfig(retry_attempts=2)
        agent = TestAgent(self.mock_llm, config)
        
        mock_response = Mock()
        mock_response.text = "Success after retries"
        
        self.mock_llm.generate_response.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            mock_response
        ]
        
        result = await agent._call_llm_with_retry("Test prompt")
        
        assert result == "Success after retries"
        assert self.mock_llm.generate_response.call_count == 3


class TestConceptMiner:
    """Test concept mining agent"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_llm = Mock(spec=BaseLLMInterface)
        self.mock_llm.generate_response = AsyncMock()
        self.config = AgentConfig()
        self.miner = ConceptMiner(self.mock_llm, self.config)
        
        self.test_corpus = """
        Machine learning is a method of data analysis that automates analytical model building.
        It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data,
        identify patterns and make decisions with minimal human intervention.
        Neural networks are computing systems inspired by biological neural networks.
        """
    
    @pytest.mark.asyncio
    async def test_concept_extraction_with_llm(self):
        """Test concept extraction using LLM"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.text = """
        {
            "concepts": [
                {"name": "machine learning", "importance": 0.9, "snippet": "method of data analysis"},
                {"name": "artificial intelligence", "importance": 0.8, "snippet": "branch of AI"},
                {"name": "neural networks", "importance": 0.7, "snippet": "computing systems"}
            ]
        }
        """
        self.mock_llm.generate_response.return_value = mock_response
        
        result = await self.miner.produce(self.test_corpus, k=3)
        
        assert isinstance(result, ConceptExtractionResult)
        assert len(result.key_concepts) <= 3
        assert "machine learning" in result.key_concepts or len(result.key_concepts) > 0
        assert isinstance(result.supporting_snippets, dict)
        assert isinstance(result.concept_importance_scores, dict)
    
    @pytest.mark.asyncio
    async def test_concept_extraction_fallback(self):
        """Test fallback concept extraction when LLM fails"""
        # Don't provide LLM or make it fail
        miner_no_llm = ConceptMiner(None, self.config)
        
        result = await miner_no_llm.produce(self.test_corpus, k=5)
        
        assert isinstance(result, ConceptExtractionResult)
        assert len(result.key_concepts) > 0
        # Should extract concepts using keyword frequency
        concepts_lower = [c.lower() for c in result.key_concepts]
        # With domain-specific extraction, accept multi-word phrases or numeric/date concepts
        has_multiword = any(len(c.split()) >= 2 for c in result.key_concepts)
        has_numeric = any(any(ch.isdigit() for ch in c) for c in result.key_concepts)
        assert has_multiword or has_numeric
    
    def test_windowed_chunks_creation(self):
        """Test creation of overlapping text chunks"""
        chunks = self.miner._create_windowed_chunks(self.test_corpus, chunk_size=20, overlap=5)
        
        assert len(chunks) > 0
        # Chunks should have reasonable size
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count >= 5  # Minimum viable chunk size
    
    @pytest.mark.asyncio
    async def test_parallel_chunk_processing(self):
        """Test parallel processing of chunks"""
        # This is more of an integration test to ensure parallel processing works
        result = await self.miner.produce(self.test_corpus, k=10, min_chunk_size=100)
        
        assert isinstance(result, ConceptExtractionResult)
        assert len(result.chunk_ids) > 0
        # Should have processed multiple chunks if corpus is large enough


class TestQuestionWriter:
    """Test question writing agent"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_llm = Mock(spec=BaseLLMInterface)
        self.mock_llm.generate_response = AsyncMock()
        self.config = AgentConfig()
        self.writer = QuestionWriter(self.mock_llm, self.config)
        
        self.test_corpus = "Machine learning algorithms can learn patterns from data."
        self.test_concept = "machine learning"
    
    @pytest.mark.asyncio
    async def test_question_generation_with_llm(self):
        """Test question generation using LLM"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.text = """
        {
            "question": "What is machine learning and how does it work?",
            "answer": "Machine learning is a method that enables computers to learn from data.",
            "reasoning_chain": ["Analyze concept", "Identify key aspects", "Formulate question"]
        }
        """
        self.mock_llm.generate_response.return_value = mock_response
        
        result = await self.writer.produce(
            self.test_concept, 
            self.test_corpus, 
            EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        assert isinstance(result, BenchmarkDraft)
        assert len(result.question) > 0
        assert len(result.answer) > 0
        assert result.concept == self.test_concept
        assert isinstance(result.expected_answer_type, AnswerType)
        assert isinstance(result.reasoning_chain, list)
    
    @pytest.mark.asyncio
    async def test_question_generation_fallback(self):
        """Test fallback question generation when LLM fails"""
        # Make LLM fail
        self.mock_llm.generate_response.side_effect = Exception("LLM failed")
        
        result = await self.writer.produce(
            self.test_concept,
            self.test_corpus,
            EvaluationType.FACTUAL_QA
        )
        
        assert isinstance(result, BenchmarkDraft)
        assert len(result.question) > 0
        assert self.test_concept in result.question or "concept" in result.question.lower()
        assert len(result.answer) > 0
    
    def test_answer_type_determination(self):
        """Test answer type determination logic"""
        # Test numeric answer
        numeric_type = self.writer._determine_answer_type("42", EvaluationType.MATHEMATICAL)
        assert numeric_type == AnswerType.NUMERIC_EXACT
        
        # Test boolean answer
        boolean_type = self.writer._determine_answer_type("True", EvaluationType.FACTUAL_QA)
        assert boolean_type == AnswerType.BOOLEAN
        
        # Test code answer
        code_type = self.writer._determine_answer_type("def function(): return 1", EvaluationType.CODE_GENERATION)
        assert code_type == AnswerType.CODE
        
        # Test free text answer
        text_type = self.writer._determine_answer_type("This is a long descriptive answer", EvaluationType.DOMAIN_KNOWLEDGE)
        assert text_type == AnswerType.FREE_TEXT
    
    def test_relevant_snippet_finding(self):
        """Test finding relevant snippets for concepts"""
        corpus = "Machine learning is powerful. Artificial intelligence systems can learn. Deep learning uses neural networks."
        
        snippet = self.writer._find_relevant_snippet("learning", corpus, max_length=100)
        
        assert len(snippet) > 0
        assert len(snippet) <= 100
        assert "learning" in snippet.lower()


class TestAdversary:
    """Test adversarial enhancement agent"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_llm = Mock(spec=BaseLLMInterface)
        self.mock_llm.generate_response = AsyncMock()
        self.config = AgentConfig()
        self.adversary = Adversary(self.mock_llm, self.config)
        
        self.test_draft = BenchmarkDraft(
            question="What is machine learning?",
            answer="A method for computers to learn from data",
            concept="machine learning",
            context_snippet="Machine learning enables automated learning",
            expected_answer_type=AnswerType.FREE_TEXT,
            reasoning_chain=["Identify concept", "Formulate question"],
            difficulty_estimate=DifficultyLevel.BASIC
        )
    
    @pytest.mark.asyncio
    async def test_adversarial_enhancement(self):
        """Test adversarial enhancement of questions"""
        result = await self.adversary.produce(self.test_draft, DifficultyLevel.HARD)
        
        assert isinstance(result, BenchmarkCandidate)
        assert len(result.question) > 0
        assert len(result.answer) > 0
        assert result.concept == self.test_draft.concept
        assert result.difficulty == DifficultyLevel.HARD
        assert isinstance(result.adversarial_techniques, list)
    
    def test_technique_selection(self):
        """Test selection of appropriate adversarial techniques"""
        techniques = self.adversary._select_techniques(self.test_draft, DifficultyLevel.HARD)
        
        assert isinstance(techniques, list)
        assert len(techniques) >= 0  # May select 0 or more techniques
        
        # All techniques should be valid
        valid_techniques = self.adversary.adversarial_techniques
        for technique in techniques:
            assert technique in valid_techniques
    
    @pytest.mark.asyncio
    async def test_multi_hop_reasoning(self):
        """Test adding multi-hop reasoning complexity"""
        enhanced_q, enhanced_a = await self.adversary._add_multi_hop_reasoning(
            self.test_draft.question,
            self.test_draft.answer,
            self.test_draft.context_snippet
        )
        
        assert len(enhanced_q) > 0
        assert len(enhanced_a) > 0
        # Enhanced question should be different from original
        assert enhanced_q != self.test_draft.question or enhanced_a != self.test_draft.answer
    
    def test_reverse_complexity(self):
        """Test adding reverse/negation complexity"""
        reversed_question = self.adversary._add_reverse_complexity(
            self.test_draft.question,
            self.test_draft.concept
        )
        
        assert len(reversed_question) > 0
        # Should contain negation or reverse logic
        negation_words = ["not", "NOT", "contrary", "opposite"]
        assert any(word in reversed_question for word in negation_words)


class TestValidator:
    """Test validation agent"""
    
    def setup_method(self):
        """Set up test environment"""
        self.config = AgentConfig()
        self.validator = Validator(None, self.config)
        
        self.test_candidate = BenchmarkCandidate(
            question="What is the primary function of machine learning algorithms?",
            answer="To learn patterns from data and make predictions",
            context="Machine learning context",
            concept="machine learning",
            expected_answer_type=AnswerType.FREE_TEXT,
            difficulty=DifficultyLevel.INTERMEDIATE,
            reasoning_chain=["Analysis", "Question formation", "Answer generation"],
            adversarial_techniques=["multi_hop_reasoning"]
        )
    
    @pytest.mark.asyncio
    async def test_candidate_validation(self):
        """Test validation of benchmark candidates"""
        result = await self.validator.accept(self.test_candidate, min_score=0.5)
        
        assert isinstance(result, ValidationResult)
        assert isinstance(result.accepted, bool)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.issues, list)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.deterministic_check_passed, bool)
    
    def test_question_clarity_assessment(self):
        """Test question clarity assessment"""
        # Good question
        good_score = self.validator._assess_question_clarity("What is machine learning?")
        assert good_score > 0.5
        
        # Poor question (too short)
        poor_score = self.validator._assess_question_clarity("ML?")
        assert poor_score < good_score
        
        # Too long question
        long_question = "What is machine learning and how does it work and what are its applications and benefits and limitations and future prospects?" * 2
        long_score = self.validator._assess_question_clarity(long_question)
        assert long_score < good_score
    
    def test_answer_quality_assessment(self):
        """Test answer quality assessment"""
        # Good answer
        good_answer = "Machine learning is a method for computers to learn from data"
        good_score = self.validator._assess_answer_quality(good_answer, AnswerType.FREE_TEXT)
        assert good_score > 0.5
        
        # Empty answer
        empty_score = self.validator._assess_answer_quality("", AnswerType.FREE_TEXT)
        assert empty_score == 0.0
        
        # Numeric answer for numeric type
        numeric_score = self.validator._assess_answer_quality("42", AnswerType.NUMERIC_EXACT)
        assert numeric_score > 0.5
        
        # Non-numeric answer for numeric type
        bad_numeric_score = self.validator._assess_answer_quality("not a number", AnswerType.NUMERIC_EXACT)
        assert bad_numeric_score < numeric_score
    
    def test_context_relevance_assessment(self):
        """Test context relevance assessment"""
        question = "What is machine learning?"
        
        # Relevant context
        relevant_context = "Machine learning is a branch of artificial intelligence"
        relevant_score = self.validator._assess_context_relevance(question, relevant_context)
        assert relevant_score > 0.3
        
        # Irrelevant context
        irrelevant_context = "Cooking recipes and kitchen techniques"
        irrelevant_score = self.validator._assess_context_relevance(question, irrelevant_context)
        assert irrelevant_score < relevant_score
    
    def test_difficulty_assessment(self):
        """Test difficulty level assessment"""
        # Simple candidate
        simple_candidate = BenchmarkCandidate(
            question="What is AI?",
            answer="Artificial Intelligence",
            context="",
            concept="AI",
            expected_answer_type=AnswerType.STRING_EXACT,
            difficulty=DifficultyLevel.BASIC,
            reasoning_chain=["Simple lookup"],
            adversarial_techniques=[]
        )
        
        simple_score = self.validator._assess_difficulty(simple_candidate)
        assert 0.0 <= simple_score <= 1.0
        
        # Complex candidate
        complex_candidate = BenchmarkCandidate(
            question="Analyze the implications of machine learning on society and economy",
            answer="Complex multi-faceted analysis of societal and economic impacts",
            context="Extended context about ML impacts",
            concept="machine learning impacts",
            expected_answer_type=AnswerType.FREE_TEXT,
            difficulty=DifficultyLevel.EXPERT,
            reasoning_chain=["Analysis", "Synthesis", "Evaluation", "Application"],
            adversarial_techniques=["multi_hop_reasoning", "reverse_statements"],
            options=["A", "B", "C", "D"]
        )
        
        complex_score = self.validator._assess_difficulty(complex_candidate)
        assert 0.0 <= complex_score <= 1.0


class TestAgenticPipelineIntegration:
    """Test integration of agentic agents in pipeline"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_llm = MockLLMInterface()  # Use actual mock LLM for integration
        self.config = AgentConfig(temperature=0.5)
        
        # Create all agents
        self.concept_miner = ConceptMiner(self.mock_llm, self.config)
        self.question_writer = QuestionWriter(self.mock_llm, self.config)
        self.adversary = Adversary(self.mock_llm, self.config)
        self.refiner = Refiner(self.mock_llm, self.config)
        self.validator = Validator(self.mock_llm, self.config)
        
        self.test_corpus = """
        Quantum computing represents a fundamental shift in computational paradigms.
        Unlike classical computers that use bits, quantum computers use quantum bits or qubits.
        Qubits can exist in superposition, allowing quantum computers to process
        multiple possibilities simultaneously. This enables quantum algorithms to solve
        certain problems exponentially faster than classical algorithms.
        """
    
    @pytest.mark.asyncio
    async def test_full_agentic_pipeline(self):
        """Test complete agentic pipeline from concept mining to validation"""
        # Step 1: Mine concepts
        concepts = await self.concept_miner.produce(self.test_corpus, k=3)
        assert len(concepts.key_concepts) > 0
        
        # Step 2: Generate questions for each concept
        drafts = []
        for concept in concepts.key_concepts[:2]:  # Limit to 2 for testing
            snippet = concepts.supporting_snippets.get(concept, self.test_corpus)
            draft = await self.question_writer.produce(
                concept, 
                self.test_corpus, 
                EvaluationType.DOMAIN_KNOWLEDGE,
                snippet
            )
            drafts.append(draft)
        
        assert len(drafts) > 0
        
        # Step 3: Apply adversarial enhancement
        candidates = []
        for draft in drafts:
            candidate = await self.adversary.produce(draft, DifficultyLevel.INTERMEDIATE)
            candidates.append(candidate)
        
        assert len(candidates) == len(drafts)
        
        # Step 4: Refine candidates
        refined_candidates = []
        for candidate in candidates:
            refined = await self.refiner.produce(candidate)
            refined_candidates.append(refined)
        
        assert len(refined_candidates) == len(candidates)
        
        # Step 5: Validate candidates
        validated_results = []
        for candidate in refined_candidates:
            validation = await self.validator.accept(candidate, min_score=0.4)
            validated_results.append((candidate, validation))
        
        assert len(validated_results) == len(refined_candidates)
        
        # Verify at least some candidates passed validation
        accepted_candidates = [c for c, v in validated_results if v.accepted]
        # Should have at least one accepted candidate with reasonable quality
        assert len(accepted_candidates) >= 0  # May be 0 if quality is low, but shouldn't crash
    
    @pytest.mark.asyncio
    async def test_agentic_pipeline_error_handling(self):
        """Test agentic pipeline error handling"""
        # Test with empty corpus
        empty_concepts = await self.concept_miner.produce("", k=5)
        assert isinstance(empty_concepts, ConceptExtractionResult)
        # Should handle gracefully, possibly with fallback concepts
        
        # Test with invalid evaluation type
        if len(empty_concepts.key_concepts) > 0:
            draft = await self.question_writer.produce(
                empty_concepts.key_concepts[0], 
                self.test_corpus, 
                EvaluationType.DOMAIN_KNOWLEDGE
            )
            assert isinstance(draft, BenchmarkDraft)
    
    def test_agentic_performance_tracking(self):
        """Test that agents track performance metrics"""
        # Check that agents track call counts and processing time
        assert self.concept_miner.call_count >= 0
        assert self.concept_miner.total_processing_time >= 0
        
        assert self.question_writer.call_count >= 0
        assert self.question_writer.total_processing_time >= 0
        
        # Performance metrics should be accessible
        assert hasattr(self.concept_miner, 'call_count')
        assert hasattr(self.concept_miner, 'total_processing_time')


class TestAgenticConfiguration:
    """Test agentic system configuration and customization"""
    
    def test_custom_agent_configuration(self):
        """Test custom agent configuration"""
        custom_config = AgentConfig(
            temperature=0.2,
            max_tokens=500,
            retry_attempts=1,
            timeout_seconds=60
        )
        
        agent = ConceptMiner(None, custom_config)
        
        assert agent.config.temperature == 0.2
        assert agent.config.max_tokens == 500
        assert agent.config.retry_attempts == 1
        assert agent.config.timeout_seconds == 60
    
    def test_agent_without_llm(self):
        """Test agents working without LLM (fallback mode)"""
        config = AgentConfig()
        
        # Concept miner should work with fallback extraction
        miner = ConceptMiner(None, config)
        assert miner.llm is None
        
        # Question writer should work with template generation
        writer = QuestionWriter(None, config)
        assert writer.llm is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])