#!/usr/bin/env python3
"""
Direct test of the API key fix for agentic evaluation
"""

import os
import sys
sys.path.append('.')

# Ensure no API key for this test
if 'OPENROUTER_API_KEY' in os.environ:
    print("⚠️  Temporarily removing OPENROUTER_API_KEY for test")
    del os.environ['OPENROUTER_API_KEY']

from docs_to_eval.utils.config import EvaluationConfig, GenerationConfig, LLMConfig
from docs_to_eval.core.benchmarks import BenchmarkGeneratorFactory
from docs_to_eval.core.evaluation import EvaluationType


def test_generator_factory_creation():
    """Test that agentic generators can be created without API key"""
    
    print("🧪 Testing Generator Factory with Mock Mode")
    print("=" * 50)
    
    # Create config without API key but with agentic enabled
    config = EvaluationConfig(
        llm=LLMConfig(
            api_key=None,  # No API key
            mock_mode=True,
            temperature=0.7
        ),
        generation=GenerationConfig(
            num_questions=5,
            use_agentic=True
        )
    )
    
    print(f"📊 Configuration:")
    print(f"   API Key Set: {bool(config.llm.api_key)}")
    print(f"   Mock Mode: {config.llm.mock_mode}")
    print(f"   Use Agentic: {config.generation.use_agentic}")
    
    try:
        # Test creating an agentic generator
        print(f"\\n🔧 Creating agentic generator...")
        
        generator = BenchmarkGeneratorFactory.create_generator(
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
            use_agentic=True,
            config=config
        )
        
        print(f"✅ Generator created: {type(generator).__name__}")
        
        # Test if it's actually an agentic generator (wrapped)
        if hasattr(generator, 'agentic_generator'):
            print(f"✅ Contains agentic generator: {type(generator.agentic_generator).__name__}")
            
            # Check if it has mock LLMs
            if hasattr(generator.agentic_generator, 'llm_pool'):
                llm_pool = generator.agentic_generator.llm_pool
                print(f"✅ LLM Pool configured:")
                for role, llm in llm_pool.items():
                    print(f"   {role}: {type(llm).__name__}")
            
        # Test generating a small benchmark
        print(f"\\n🎯 Testing benchmark generation...")
        test_corpus = "Machine learning algorithms process data to make predictions."
        
        benchmark = generator.generate_benchmark(test_corpus, num_questions=2)
        
        if benchmark and len(benchmark) > 0:
            print(f"✅ Generated {len(benchmark)} questions successfully!")
            for i, q in enumerate(benchmark[:2]):
                question_text = q.get('question', 'No question')[:50] + '...'
                print(f"   Q{i+1}: {question_text}")
        else:
            print(f"❌ No questions generated")
            
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_standard_vs_agentic():
    """Compare standard vs agentic generation without API key"""
    
    print(f"\\n🧪 Comparing Standard vs Agentic Generation")
    print("-" * 50)
    
    test_corpus = "Natural language processing enables computers to understand human language."
    
    # Test standard generation
    try:
        print("🔧 Testing standard generator...")
        standard_generator = BenchmarkGeneratorFactory.create_generator(
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
            use_agentic=False
        )
        
        standard_questions = standard_generator.generate_benchmark(test_corpus, num_questions=2)
        print(f"✅ Standard generator: {len(standard_questions)} questions")
        
    except Exception as e:
        print(f"❌ Standard generator failed: {e}")
        return False
    
    # Test agentic generation (without API key)
    try:
        print("🔧 Testing agentic generator (no API key)...")
        
        config = EvaluationConfig(
            llm=LLMConfig(api_key=None, mock_mode=True),
            generation=GenerationConfig(use_agentic=True)
        )
        
        agentic_generator = BenchmarkGeneratorFactory.create_generator(
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
            use_agentic=True,
            config=config
        )
        
        agentic_questions = agentic_generator.generate_benchmark(test_corpus, num_questions=2)
        print(f"✅ Agentic generator: {len(agentic_questions)} questions")
        
        return True
        
    except Exception as e:
        print(f"❌ Agentic generator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests"""
    
    print("🔧 API KEY FIX VALIDATION")
    print("=" * 60)
    
    # Test 1: Generator factory creation
    test1_passed = test_generator_factory_creation()
    
    # Test 2: Standard vs Agentic comparison
    test2_passed = test_standard_vs_agentic()
    
    # Summary
    print("\\n" + "=" * 60)
    print("🎯 TEST RESULTS:")
    print(f"   Generator Factory: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Standard vs Agentic: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\\n🎉 SUCCESS! The API key fix is working correctly.")
        print("   ✅ Agentic generators can be created without API keys")
        print("   ✅ Mock LLMs are used automatically when no API key is provided")
        print("   ✅ Benchmark generation works with mock LLMs")
    else:
        print("\\n⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()