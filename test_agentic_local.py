#!/usr/bin/env python3
"""
Local test of agentic pipeline without API key to verify the fix
"""

import os
import sys
import asyncio
sys.path.append('.')

# Ensure no API key for this test
if 'OPENROUTER_API_KEY' in os.environ:
    print("⚠️  Temporarily removing OPENROUTER_API_KEY for test")
    del os.environ['OPENROUTER_API_KEY']

from docs_to_eval.utils.config import EvaluationConfig, GenerationConfig, LLMConfig
from docs_to_eval.core.pipeline import EvaluationPipeline
from docs_to_eval.core.evaluation import EvaluationType


async def test_agentic_without_api_key():
    """Test agentic evaluation without API key"""
    
    print("🧪 Testing Agentic Pipeline WITHOUT API Key")
    print("=" * 50)
    
    # Create configuration with agentic enabled but no API key
    config = EvaluationConfig(
        llm=LLMConfig(
            api_key=None,  # No API key
            mock_mode=True,  # Enable mock mode
            temperature=0.7
        ),
        generation=GenerationConfig(
            num_questions=5,
            use_agentic=True  # Enable agentic
        ),
        eval_type=EvaluationType.DOMAIN_KNOWLEDGE
    )
    
    # Test corpus
    corpus_text = """
    Machine learning algorithms analyze large datasets to identify patterns and make predictions.
    Deep learning networks use multiple layers of artificial neurons to process complex information.
    Supervised learning requires labeled training data to learn input-output mappings.
    Unsupervised learning discovers hidden structures in unlabeled data through clustering.
    """
    
    print(f"📊 Configuration:")
    print(f"   Use Agentic: {config.generation.use_agentic}")
    print(f"   API Key Set: {bool(config.llm.api_key)}")
    print(f"   Mock Mode: {getattr(config.llm, 'mock_mode', False)}")
    print(f"   Questions: {config.generation.num_questions}")
    
    try:
        # Create and run pipeline
        pipeline = EvaluationPipeline(config)
        print(f"\\n🚀 Starting evaluation pipeline...")
        
        result = await pipeline.run_async(corpus_text)
        
        print(f"\\n✅ SUCCESS! Pipeline completed")
        print(f"📈 Results:")
        print(f"   Questions Generated: {len(result.get('questions', []))}")
        print(f"   Evaluation Type: {result.get('classification', {}).get('primary_type', 'unknown')}")
        print(f"   Using Agentic: {result.get('config', {}).get('use_agentic', False)}")
        print(f"   Generation Method: {result.get('generation_method', 'unknown')}")
        
        # Show sample questions
        questions = result.get('questions', [])
        if questions:
            print(f"\\n📝 Sample Questions:")
            for i, q in enumerate(questions[:3]):
                print(f"   {i+1}. {q.get('question', 'No question')}")
        
        print(f"\\n🎉 Test PASSED: Agentic evaluation works without API key!")
        return True
        
    except Exception as e:
        print(f"\\n❌ FAILED: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_pool_creation():
    """Test LLM pool creation with mock mode"""
    
    print("\\n🧪 Testing LLM Pool Creation")
    print("-" * 30)
    
    try:
        from docs_to_eval.core.benchmarks import BenchmarkGeneratorFactory
        
        # Test with mock mode config
        config = EvaluationConfig(
            llm=LLMConfig(api_key=None, mock_mode=True),
            generation=GenerationConfig(use_agentic=True)
        )
        
        llm_pool = BenchmarkGeneratorFactory._create_llm_pool_from_config(config)
        
        if llm_pool:
            print(f"✅ LLM Pool Created:")
            for role, llm in llm_pool.items():
                print(f"   {role}: {type(llm).__name__}")
            return True
        else:
            print(f"❌ No LLM pool created")
            return False
            
    except Exception as e:
        print(f"❌ LLM pool creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    
    print("🔧 AGENTIC API KEY FIX - LOCAL TESTING")
    print("=" * 60)
    
    # Test 1: LLM Pool Creation
    pool_test_passed = test_llm_pool_creation()
    
    # Test 2: Full Pipeline
    pipeline_test_passed = await test_agentic_without_api_key()
    
    # Summary
    print("\\n" + "=" * 60)
    print("🎯 TEST SUMMARY:")
    print(f"   LLM Pool Creation: {'✅ PASS' if pool_test_passed else '❌ FAIL'}")
    print(f"   Agentic Pipeline: {'✅ PASS' if pipeline_test_passed else '❌ FAIL'}")
    
    if pool_test_passed and pipeline_test_passed:
        print("\\n🎉 ALL TESTS PASSED! The fix is working correctly.")
        print("   Agentic evaluation now works without API keys using mock LLMs.")
    else:
        print("\\n⚠️  Some tests failed. The fix may need additional work.")


if __name__ == "__main__":
    asyncio.run(main())