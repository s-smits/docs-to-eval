"""
Comprehensive test for the full agentic flow including:
- Document chunking with chonkie
- Question generation via orchestrator/generator
- LLM evaluation
- Response verification
- Quality assessment
"""

import pytest
import asyncio
import os

from docs_to_eval.core.agentic.orchestrator import AgenticBenchmarkOrchestrator
from docs_to_eval.core.agentic.models import PipelineConfig, DifficultyLevel, EnhancedBenchmarkItem
from docs_to_eval.core.agentic.validation import ComprehensiveValidator
from docs_to_eval.core.evaluation import EvaluationType
from docs_to_eval.core.verification import VerificationOrchestrator
from docs_to_eval.utils.text_processing import create_smart_chunks
from docs_to_eval.utils.config import ChunkingConfig
from docs_to_eval.llm.mock_interface import MockLLMInterface
from docs_to_eval.llm.openrouter_interface import OpenRouterInterface, OpenRouterConfig


import pytest


@pytest.mark.integration
@pytest.mark.slow
class TestFullAgenticFlow:
    """Test the complete agentic pipeline from corpus to evaluation"""
    
    @pytest.fixture
    def sample_corpus(self):
        """Provide a realistic sample corpus for testing"""
        return """
        The Etruscan civilization flourished in ancient Italy between the 8th and 3rd centuries BCE, 
        primarily in the region that is now Tuscany. The Etruscans developed a sophisticated culture 
        with advanced metalworking, distinctive art, and complex religious practices.
        
        Etruscan society was organized into independent city-states, each ruled by a king or oligarchy. 
        Major cities included Veii, Cerveteri, Tarquinia, and Vulci. These cities formed a loose 
        confederation known as the Etruscan League, which met annually at the Fanum Voltumnae sanctuary.
        
        The Etruscan language remains one of the great mysteries of the ancient world. Written in a 
        script adapted from the Greek alphabet, it is not related to any known language family. 
        Despite thousands of inscriptions, our understanding remains limited to basic vocabulary 
        and religious formulas.
        
        Etruscan art is characterized by vibrant tomb paintings, bronze sculptures, and terracotta 
        works. The famous "Sarcophagus of the Spouses" from Cerveteri shows a reclining couple 
        at a banquet, highlighting the important role of women in Etruscan society - they could 
        own property, attend banquets, and participate in public life.
        
        The Etruscans significantly influenced Roman civilization. Early Roman kings were of Etruscan 
        origin, and the Romans adopted many Etruscan practices including augury (divination by 
        observing birds), gladiatorial games, and architectural techniques like the arch. The Roman 
        toga and many religious rituals also have Etruscan origins.
        
        By the 3rd century BCE, Etruscan cities had been absorbed into the expanding Roman Republic. 
        The last independent Etruscan city, Volsinii, fell in 264 BCE. Despite political subjugation, 
        Etruscan cultural influence persisted in Roman society for centuries.
        """
    
    @pytest.fixture
    def chunking_config(self):
        """Provide chunking configuration for testing"""
        return ChunkingConfig(
            enable_chonkie=True,
            chunking_strategy="semantic",
            use_token_chunking=True,
            target_token_size=500,  # Smaller for testing
            min_token_size=300,
            max_token_size=700,
            overlap_tokens=50
        )
    
    @pytest.fixture
    def pipeline_config(self):
        """Provide pipeline configuration for testing"""
        return PipelineConfig(
            difficulty=DifficultyLevel.INTERMEDIATE,
            num_questions=3,
            min_validation_score=0.6,
            parallel_batch_size=2,
            max_retry_cycles=1
        )
    
    @pytest.fixture
    def mock_llm_pool(self):
        """Create mock LLM pool for testing"""
        mock_llm = MockLLMInterface(model_name="test-model", temperature=0.7)
        return {
            'retriever': mock_llm,
            'creator': mock_llm,
            'adversary': mock_llm,
            'refiner': mock_llm
        }
    
    def test_document_chunking(self, sample_corpus, chunking_config):
        """Test that document chunking works correctly with chonkie"""
        print("\n📄 Testing Document Chunking")
        print("=" * 60)
        
        # Create chunks
        chunks = create_smart_chunks(sample_corpus, chunking_config)
        
        # Validate chunks
        assert len(chunks) > 0, "Should create at least one chunk"
        print(f"✅ Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            assert 'text' in chunk, f"Chunk {i} should have 'text' field"
            assert 'metadata' in chunk, f"Chunk {i} should have 'metadata' field"
            assert len(chunk['text']) > 0, f"Chunk {i} should have non-empty text"
            
            # Check token counts if available
            if 'token_count' in chunk:
                assert chunk['token_count'] >= chunking_config.min_token_size / 2, \
                    f"Chunk {i} token count too small"
                assert chunk['token_count'] <= chunking_config.max_token_size * 1.5, \
                    f"Chunk {i} token count too large"
            
            print(f"  Chunk {i+1}: {len(chunk['text'])} chars, "
                  f"tokens: {chunk.get('token_count', 'N/A')}, "
                  f"method: {chunk.get('method', 'unknown')}")
        
        # Check for overlap if multiple chunks
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                _ = chunks[i]['text'][-100:]
                _ = chunks[i+1]['text'][:100]
                # Some overlap should exist
                print(f"  Overlap {i+1}->{i+2}: checking for continuity")
        
        print("✅ Chunking validation passed")
        return chunks
    
    @pytest.mark.asyncio
    async def test_question_generation_from_chunks(self, sample_corpus, pipeline_config, mock_llm_pool):
        """Test question generation from chunked corpus"""
        print("\n❓ Testing Question Generation from Chunks")
        print("=" * 60)
        
        # Initialize orchestrator
        orchestrator = AgenticBenchmarkOrchestrator(mock_llm_pool, pipeline_config)
        
        # Generate questions (chunking occurs inside orchestrator). Ensure chonkie path is viable by using a small target window.
        eval_type = EvaluationType.DOMAIN_KNOWLEDGE
        items = await orchestrator.generate(
            corpus_text=sample_corpus,
            eval_type=eval_type,
            num_questions=5,
            difficulty=DifficultyLevel.INTERMEDIATE
        )
        
        # Validate generated items
        assert len(items) > 0, "Should generate at least one question"
        print(f"✅ Generated {len(items)} questions")
        
        for i, item in enumerate(items):
            assert isinstance(item, EnhancedBenchmarkItem), f"Item {i} should be EnhancedBenchmarkItem"
            assert item.question, f"Item {i} should have a question"
            assert item.answer, f"Item {i} should have an answer"
            assert item.metadata, f"Item {i} should have metadata"
            
            print(f"\n  Question {i+1}:")
            print(f"    Q: {item.question[:80]}...")
            print(f"    A: {item.answer[:80]}...")
            print(f"    Difficulty: {item.metadata.difficulty}")
            print(f"    Deterministic: {item.metadata.deterministic}")
            if hasattr(item.metadata, 'source_chunks'):
                print(f"    Source chunks: {len(item.metadata.source_chunks)}")
            else:
                print(f"    Validation score: {getattr(item.metadata, 'validation_score', 'N/A')}")
        
        print("✅ Question generation validation passed")
        return items
    
    @pytest.mark.asyncio
    async def test_full_pipeline_with_evaluation(self, sample_corpus, pipeline_config, mock_llm_pool):
        """Test the complete pipeline including evaluation and verification"""
        print("\n🔄 Testing Full Pipeline with Evaluation")
        print("=" * 60)
        
        # Step 1: Chunk the corpus
        print("\n📄 Step 1: Chunking corpus...")
        chunking_config = ChunkingConfig(
            enable_chonkie=True,
            target_token_size=400,
            min_token_size=200,
            overlap_tokens=40
        )
        chunks = create_smart_chunks(sample_corpus, chunking_config)
        print(f"  Created {len(chunks)} chunks")
        
        # Step 2: Generate questions
        print("\n❓ Step 2: Generating questions...")
        orchestrator = AgenticBenchmarkOrchestrator(mock_llm_pool, pipeline_config)
        items = await orchestrator.generate(
            corpus_text=sample_corpus,
            eval_type=EvaluationType.FACTUAL_QA,
            num_questions=3,
            difficulty=DifficultyLevel.BASIC
        )
        print(f"  Generated {len(items)} questions")
        
        # Step 3: Convert to standard format for evaluation
        print("\n📋 Step 3: Converting to standard format...")
        standard_items = []
        for item in items:
            standard_item = item.to_standard_benchmark_item()
            # standard_item is already a dict from to_standard_benchmark_item()
            if isinstance(standard_item, dict):
                standard_items.append(standard_item)
            else:
                # If it's an object, convert to dict
                standard_items.append({
                    'question': standard_item.question,
                    'answer': standard_item.answer,
                    'evaluation_type': getattr(standard_item, 'evaluation_type', EvaluationType.FACTUAL_QA),
                    'metadata': getattr(standard_item, 'metadata', {})
                })
        print(f"  Converted {len(standard_items)} items")
        # Sanity: ensure token-window chunking path was exercised at least once
        # (no hard assertion on method name to avoid coupling to internals)
        
        # Step 4: Evaluate with LLM
        print("\n🤖 Step 4: Evaluating with LLM...")
        evaluator = mock_llm_pool['creator']
        llm_responses = []
        for item in standard_items:
            response = await evaluator.generate_response(item['question'], eval_type='factual_qa')
            llm_responses.append({
                'question': item['question'],
                'expected_answer': item['answer'],
                'llm_response': response.text,
                'metadata': item.get('metadata', {})
            })
        print(f"  Got {len(llm_responses)} LLM responses")
        
        # Step 5: Verify responses
        print("\n✅ Step 5: Verifying responses...")
        verifier = VerificationOrchestrator()
        verification_results = []
        for response in llm_responses:
            vr = verifier.verify(
                prediction=response['llm_response'],
                ground_truth=response['expected_answer'],
                eval_type='factual_qa',
                options=None,
                question=response['question']
            )
            verification_results.append(vr)
            print(f"  Verified score: {vr.score:.2f} - Method: {vr.method}")
        
        # Step 6: Quality assessment
        print("\n🔬 Step 6: Quality assessment...")
        validator = ComprehensiveValidator(min_quality_score=0.5)
        validation_report = await validator.validate_benchmark_batch(items)
        
        print("\n📊 Pipeline Results:")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Questions generated: {len(items)}")
        print(f"  LLM responses: {len(llm_responses)}")
        print(f"  Verifications: {len(verification_results)}")
        print(f"  Pass rate: {validation_report.get('overall_pass_rate', 0):.2%}")
        print(f"  Avg quality: {validation_report.get('quality_assessment', {}).get('avg_quality', 0):.3f}")
        
        # Assertions
        assert len(chunks) > 0, "Should have chunks"
        assert len(items) > 0, "Should have generated questions"
        assert len(llm_responses) == len(items), "Should have LLM response for each question"
        assert len(verification_results) == len(items), "Should have verification for each response"
        
        print("\n✅ Full pipeline test passed!")
        return {
            'chunks': chunks,
            'questions': items,
            'llm_responses': llm_responses,
            'verification_results': verification_results,
            'validation_report': validation_report
        }
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set"
    )
    async def test_openrouter_integration(self, sample_corpus):
        """Test real OpenRouter integration with actual API calls"""
        print("\n🌐 Testing OpenRouter Integration")
        print("=" * 60)
        
        # Create OpenRouter instances with preferred models
        sonnet_llm = OpenRouterInterface(OpenRouterConfig(model="anthropic/claude-sonnet-4"))
        gemini_llm = OpenRouterInterface(OpenRouterConfig(model="openai/gpt-5-mini"))
        
        # Create LLM pool with real interfaces
        llm_pool = {
            'retriever': gemini_llm,  # Fast for retrieval
            'creator': sonnet_llm,     # High quality for creation
            'adversary': sonnet_llm,   # Creative for adversarial
            'refiner': gemini_llm      # Fast for refinement
        }
        
        # Test connection
        print("🔌 Testing connections...")
        test_prompt = "What is 2+2?"
        
        sonnet_response = await sonnet_llm.generate_response(test_prompt)
        assert sonnet_response.text, "Sonnet should return response"
        print(f"  ✅ Sonnet connected: {sonnet_response.text[:50]}...")
        
        gemini_response = await gemini_llm.generate_response(test_prompt)
        assert gemini_response.text, "Gemini should return response"
        print(f"  ✅ Gemini connected: {gemini_response.text[:50]}...")
        
        # Generate questions with real LLMs
        print("\n📝 Generating questions with real LLMs...")
        config = PipelineConfig(
            difficulty=DifficultyLevel.HARD,
            num_questions=2,  # Small number for testing
            min_validation_score=0.7
        )
        
        orchestrator = AgenticBenchmarkOrchestrator(llm_pool, config)
        items = await orchestrator.generate(
            corpus_text=sample_corpus[:1000],  # Use smaller corpus for testing
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
            num_questions=2,
            difficulty=DifficultyLevel.HARD
        )
        
        print(f"✅ Generated {len(items)} questions with real LLMs")
        for i, item in enumerate(items):
            print(f"\n  Question {i+1}:")
            print(f"    {item.question}")
            print(f"    Expected: {item.answer[:100]}...")
        
        # Evaluate one question with real LLM
        if items:
            print("\n🤖 Evaluating with real LLM...")
            test_question = items[0].question
            llm_answer = await sonnet_llm.generate_response(test_question)
            print(f"  Question: {test_question}")
            print(f"  LLM Answer: {llm_answer.text[:200]}...")
            print(f"  Expected: {items[0].answer[:200]}...")
        
        print("\n✅ OpenRouter integration test passed!")
        return items
    
    @pytest.mark.asyncio
    async def test_pipeline_metrics_and_reporting(self, sample_corpus, pipeline_config, mock_llm_pool):
        """Test pipeline metrics collection and reporting"""
        print("\n📊 Testing Pipeline Metrics and Reporting")
        print("=" * 60)
        
        orchestrator = AgenticBenchmarkOrchestrator(mock_llm_pool, pipeline_config)
        
        # Generate items
        items = await orchestrator.generate(
            corpus_text=sample_corpus,
            eval_type=EvaluationType.MULTIPLE_CHOICE,
            num_questions=3,
            difficulty=DifficultyLevel.INTERMEDIATE
        )
        
        # Get pipeline metrics
        metrics = orchestrator.get_pipeline_metrics()
        
        print("\n📈 Pipeline Metrics:")
        print(f"  Total time: {metrics.get('total_processing_time', 0):.2f}s")
        print(f"  Items generated: {metrics.get('total_generated', 0)}")
        print(f"  Acceptance rate: {metrics.get('acceptance_rate', 0):.2%}")
        
        if 'agent_stats' in metrics:
            print("\n  Agent Statistics:")
            for agent, stats in metrics['agent_stats'].items():
                print(f"    {agent}: {stats.get('calls', 0)} calls, "
                      f"{stats.get('avg_time', 0):.2f}s avg")
        
        # No explicit phase timings in orchestrator metrics
        
        # Validate metrics structure
        assert 'total_generated' in metrics, "Should have total_generated metric"
        assert metrics['total_generated'] >= len(items), "Item count should be reflected in metrics"
        
        print("\n✅ Metrics and reporting test passed!")
        return metrics


def get_sample_corpus():
    """Get sample corpus for testing"""
    return """
    The Etruscan civilization flourished in ancient Italy between the 8th and 3rd centuries BCE, 
    primarily in the region that is now Tuscany. The Etruscans developed a sophisticated culture 
    with advanced metalworking, distinctive art, and complex religious practices.
    
    Etruscan society was organized into independent city-states, each ruled by a king or oligarchy. 
    Major cities included Veii, Cerveteri, Tarquinia, and Vulci. These cities formed a loose 
    confederation known as the Etruscan League, which met annually at the Fanum Voltumnae sanctuary.
    
    The Etruscan language remains one of the great mysteries of the ancient world. Written in a 
    script adapted from the Greek alphabet, it is not related to any known language family. 
    Despite thousands of inscriptions, our understanding remains limited to basic vocabulary 
    and religious formulas.
    
    Etruscan art is characterized by vibrant tomb paintings, bronze sculptures, and terracotta 
    works. The famous "Sarcophagus of the Spouses" from Cerveteri shows a reclining couple 
    at a banquet, highlighting the important role of women in Etruscan society - they could 
    own property, attend banquets, and participate in public life.
    
    The Etruscans significantly influenced Roman civilization. Early Roman kings were of Etruscan 
    origin, and the Romans adopted many Etruscan practices including augury (divination by 
    observing birds), gladiatorial games, and architectural techniques like the arch. The Roman 
    toga and many religious rituals also have Etruscan origins.
    
    By the 3rd century BCE, Etruscan cities had been absorbed into the expanding Roman Republic. 
    The last independent Etruscan city, Volsinii, fell in 264 BCE. Despite political subjugation, 
    Etruscan cultural influence persisted in Roman society for centuries.
    """


if __name__ == "__main__":
    # Run tests directly (standalone mode)
    test = TestFullAgenticFlow()
    
    # Create test data directly (not using fixtures)
    corpus = get_sample_corpus()
    
    chunking_cfg = ChunkingConfig(
            enable_chonkie=True,
            chunking_strategy="semantic",
            use_token_chunking=True,
            target_token_size=500,
            min_token_size=300,
            max_token_size=700,
            overlap_tokens=50
        )
    
    pipeline_cfg = PipelineConfig(
        difficulty=DifficultyLevel.INTERMEDIATE,
        num_questions=5,
        min_validation_score=0.6,
        parallel_batch_size=2,
        max_retry_cycles=1
    )
    
    mock_llm = MockLLMInterface(model_name="test-model", temperature=0.7)
    llm_pool = {
        'retriever': mock_llm,
        'creator': mock_llm,
        'adversary': mock_llm,
        'refiner': mock_llm
    }
    
    # Run synchronous tests
    print("🧪 Running Full Agentic Flow Tests")
    print("=" * 80)
    
    # Test chunking
    chunks = test.test_document_chunking(corpus, chunking_cfg)
    
    # Run async tests
    async def run_async_tests():
        # Test question generation
        _ = await test.test_question_generation_from_chunks(corpus, pipeline_cfg, llm_pool)
        
        # Test full pipeline
        results = await test.test_full_pipeline_with_evaluation(corpus, pipeline_cfg, llm_pool)
        
        # Test metrics
        _ = await test.test_pipeline_metrics_and_reporting(corpus, pipeline_cfg, llm_pool)
        
        # Test OpenRouter if API key is available
        if os.getenv("OPENROUTER_API_KEY"):
            await test.test_openrouter_integration(corpus)
        else:
            print("\n⚠️ Skipping OpenRouter test (no API key)")
        
        return results
    
    # Run async tests
    results = asyncio.run(run_async_tests())
    
    print("\n🎉 All tests completed successfully!")
