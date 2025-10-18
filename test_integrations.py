#!/usr/bin/env python3
"""
Test script for Groq and Gemini SDK integrations with docs-to-eval
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from docs_to_eval.llm import (
    list_available_providers, 
    create_llm_interface,
    GroqInterface, 
    GeminiSDKInterface
)
from docs_to_eval.llm.groq_interface import test_groq_connection
from docs_to_eval.llm.gemini_sdk_interface import test_gemini_connection


async def test_providers():
    """Test all available LLM providers"""
    print("🧪 Testing LLM Provider Integrations\n")
    
    # List available providers
    providers = list_available_providers()
    print("📋 Available Providers:")
    for name, info in providers.items():
        status = "✅" if info['available'] else "❌"
        print(f"  {status} {name}: {info['description']}")
    print()
    
    # Test API connections if keys are available
    test_results = {}
    
    # Test Groq
    if os.getenv('GROQ_API_KEY'):
        print("🚀 Testing Groq Connection...")
        try:
            groq_results = await test_groq_connection()
            test_results['groq'] = groq_results
            
            if groq_results['response_generated']:
                print(f"  ✅ Groq API working - Response: {groq_results['sample_response'][:50]}...")
                print(f"  ⏱️  Response time: {groq_results['response_time']:.2f}s")
            else:
                print(f"  ❌ Groq API failed: {groq_results.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"  ❌ Groq test failed: {e}")
        print()
    else:
        print("⚠️  GROQ_API_KEY not set - skipping Groq test\n")
    
    # Test Gemini
    if os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'):
        print("🚀 Testing Gemini SDK Connection...")
        try:
            gemini_results = await test_gemini_connection()
            test_results['gemini'] = gemini_results
            
            if gemini_results['response_generated']:
                print(f"  ✅ Gemini SDK working - Response: {gemini_results['sample_response'][:50]}...")
                print(f"  ⏱️  Response time: {gemini_results['response_time']:.2f}s")
            else:
                print(f"  ❌ Gemini SDK failed: {gemini_results.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"  ❌ Gemini test failed: {e}")
        print()
    else:
        print("⚠️  GEMINI_API_KEY/GOOGLE_API_KEY not set - skipping Gemini test\n")
    
    return test_results


async def demo_evaluation_types():
    """Demo different evaluation types with available providers"""
    print("📝 Testing Evaluation Types\n")
    
    test_cases = [
        ("What is 15 + 27?", "mathematical"),
        ("Write a Python function to find the factorial of a number", "code_generation"),
        ("What is the capital of Japan?", "factual_qa"),
        ("Write a haiku about artificial intelligence", "creative_writing")
    ]
    
    # Test with Groq if available
    if os.getenv('GROQ_API_KEY'):
        print("🔥 Testing with Groq (Llama 3 8B)...")
        try:
            groq_interface = create_llm_interface('groq', model='llama3-8b-8192')
            
            for prompt, eval_type in test_cases:
                response = await groq_interface.generate_response(
                    prompt=prompt,
                    eval_type=eval_type
                )
                print(f"  📋 {eval_type}: {prompt}")
                print(f"  💬 Response: {response.text[:80]}...")
                print(f"  📊 Confidence: {response.confidence:.2f}")
                print()
        except Exception as e:
            print(f"  ❌ Groq evaluation test failed: {e}\n")
    
    # Test with Gemini if available
    if os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'):
        print("💎 Testing with Gemini SDK (Flash)...")
        try:
            gemini_interface = create_llm_interface('gemini', model='gemini-1.5-flash-latest')
            
            for prompt, eval_type in test_cases:
                response = await gemini_interface.generate_response(
                    prompt=prompt,
                    eval_type=eval_type
                )
                print(f"  📋 {eval_type}: {prompt}")
                print(f"  💬 Response: {response.text[:80]}...")
                print(f"  📊 Confidence: {response.confidence:.2f}")
                print()
        except Exception as e:
            print(f"  ❌ Gemini evaluation test failed: {e}\n")


async def demo_batch_processing():
    """Demo batch processing capabilities"""
    print("🔄 Testing Batch Processing\n")
    
    batch_prompts = [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?",
        "How does backpropagation work?",
        "What are transformers in AI?"
    ]
    
    # Test batch with Groq
    if os.getenv('GROQ_API_KEY'):
        print("🔥 Groq Batch Processing...")
        try:
            from docs_to_eval.llm.groq_interface import GroqBatchInterface
            
            batch_interface = GroqBatchInterface(max_concurrent=3)
            results = await batch_interface.process_batch(
                batch_prompts,
                eval_type="factual_qa"
            )
            
            stats = batch_interface.get_batch_stats()
            print(f"  ✅ Processed {len(results)} prompts")
            print(f"  📊 Success rate: {stats['success_rate']:.1f}%")
            print(f"  ⏱️  Average response time: {stats['average_response_time']:.2f}s")
            print()
        except Exception as e:
            print(f"  ❌ Groq batch test failed: {e}\n")
    
    # Test batch with Gemini
    if os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'):
        print("💎 Gemini Batch Processing...")
        try:
            from docs_to_eval.llm.gemini_sdk_interface import GeminiBatchInterface
            
            batch_interface = GeminiBatchInterface(max_concurrent=3)
            results = await batch_interface.process_batch(
                batch_prompts,
                eval_type="factual_qa"
            )
            
            stats = batch_interface.get_batch_stats()
            print(f"  ✅ Processed {len(results)} prompts")
            print(f"  📊 Success rate: {stats['success_rate']:.1f}%")
            print(f"  ⏱️  Average response time: {stats['average_response_time']:.2f}s")
            print()
        except Exception as e:
            print(f"  ❌ Gemini batch test failed: {e}\n")


async def demo_docs_to_eval_integration():
    """Demo integration with docs-to-eval pipeline"""
    print("🔗 Testing docs-to-eval Pipeline Integration\n")
    
    # Test with a simple corpus
    sample_corpus = """
    The Etruscan civilization was an ancient civilization created by the Etruscans. 
    They inhabited Etruria in ancient Italy, with a common language and culture. 
    The territorial extent reached its maximum around 500 BC. The Etruscans developed 
    a system of writing derived from the Euboean alphabet. The Tavola Capuana is a 
    famous Etruscan artifact measuring 50 by 60 cm, dating to 470 BCE.
    """
    
    try:
        # Test corpus classification
        from docs_to_eval.core.classification import EvaluationTypeClassifier
        
        classifier = EvaluationTypeClassifier()
        classification = classifier.classify_corpus(sample_corpus)
        
        print(f"📋 Corpus Classification:")
        print(f"  🎯 Primary Type: {classification.primary_type}")
        print(f"  📊 Confidence: {classification.confidence:.2f}")
        print(f"  💭 Analysis: {classification.analysis[:100]}...")
        print()
        
        # Test benchmark generation with real LLM if available
        if os.getenv('GROQ_API_KEY') or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'):
            from docs_to_eval.core.benchmarks import BenchmarkGeneratorFactory
            from docs_to_eval.utils.config import create_default_config
            
            # Create a simple benchmark
            config = create_default_config()
            config.generation.num_questions = 3  # Keep it small for demo
            
            generator = BenchmarkGeneratorFactory().create_generator(
                eval_type=classification.primary_type,
                use_agentic=False,  # Use simple generation for demo
                config=config
            )
            
            questions = generator.generate_questions(
                corpus_text=sample_corpus,
                num_questions=3
            )
            
            print(f"📝 Generated {len(questions)} benchmark questions:")
            for i, q in enumerate(questions, 1):
                print(f"  {i}. Q: {q.get('question', q.get('text', 'N/A'))}")
                print(f"     A: {q.get('answer', q.get('correct_answer', 'N/A'))}")
                print()
        
    except Exception as e:
        print(f"  ❌ Pipeline integration test failed: {e}\n")


def show_installation_help():
    """Show installation instructions"""
    print("📦 Installation Instructions\n")
    
    providers = list_available_providers()
    
    for name, info in providers.items():
        if not info['available'] and name not in ['mock']:
            print(f"❌ {name} not available")
            if name == 'groq':
                print("   Install with: pip install groq")
                print("   Set environment variable: export GROQ_API_KEY=your_key")
            elif name == 'gemini_sdk':
                print("   Install with: pip install google-generativeai")
                print("   Set environment variable: export GEMINI_API_KEY=your_key")
            elif name == 'openrouter':
                print("   Install with: uv add irouter")
                print("   Set environment variable: export OPENROUTER_API_KEY=your_key")
            print()


async def main():
    """Main test runner"""
    print("🎯 docs-to-eval LLM Integration Tests")
    print("=" * 50)
    print()
    
    # Test basic provider availability
    await test_providers()
    
    # Show installation help if needed
    providers = list_available_providers()
    if not any(p['available'] for name, p in providers.items() if name != 'mock'):
        show_installation_help()
        return
    
    # Run demos if we have working providers
    has_api_keys = bool(os.getenv('GROQ_API_KEY') or 
                       os.getenv('GEMINI_API_KEY') or 
                       os.getenv('GOOGLE_API_KEY'))
    
    if has_api_keys:
        await demo_evaluation_types()
        await demo_batch_processing()
        await demo_docs_to_eval_integration()
        
        print("🎉 All integration tests completed!")
        print("\n💡 Next steps:")
        print("  - Set API keys for other providers to test more integrations")
        print("  - Run the full docs-to-eval pipeline with: python -m docs_to_eval.cli.main")
        print("  - Start the web interface with: python run_server.py")
    else:
        print("⚠️  No API keys detected. Set GROQ_API_KEY or GEMINI_API_KEY to run full tests.")
        print("\n📖 Example usage after setting API keys:")
        print("  export GROQ_API_KEY=your_groq_key")
        print("  export GEMINI_API_KEY=your_gemini_key")
        print("  python test_integrations.py")


if __name__ == "__main__":
    asyncio.run(main())
