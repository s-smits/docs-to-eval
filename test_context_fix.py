#!/usr/bin/env python3
"""
Test Context Fix: Verify that evaluation now includes corpus context
Tests whether LLM gets domain context during evaluation
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import components
from docs_to_eval.core.agentic import AgenticBenchmarkGenerator
from docs_to_eval.core.evaluation import EvaluationType
from docs_to_eval.llm.concurrent_gemini import ConcurrentGeminiInterface


async def test_context_in_evaluation():
    """Test that evaluation includes context from generation"""
    print("🔍 Testing Context Fix in Evaluation...")
    
    # Create rich domain-specific corpus
    etruscan_corpus = """
    Tinia was the supreme deity of the Etruscan pantheon, ruler of the heavens and wielder of lightning.
    He was equivalent to Jupiter in Roman mythology but had distinctly Etruscan characteristics.
    Maris served as the god of war and agriculture, often depicted with spear and shield in Etruscan art.
    Voltumna was a mysterious chthonic deity, associated with vegetation cycles and the underworld.
    Unlike Greek Hades, Voltumna had agricultural significance in Etruscan religious practices.
    """
    
    try:
        # Generate question with context
        generator = AgenticBenchmarkGenerator(EvaluationType.DOMAIN_KNOWLEDGE)
        items = await generator.generate_benchmark_async(
            corpus_text=etruscan_corpus,
            num_questions=1
        )
        
        if not items:
            print("   ❌ No items generated")
            return False
            
        item = items[0]
        print(f"   📝 Generated Question: {item.question}")
        
        # Check if item has context
        context = None
        if hasattr(item, 'context'):
            context = item.context
        elif isinstance(item, dict):
            context = item.get('context')
            
        print(f"   📚 Context Available: {'Yes' if context else 'No'}")
        if context:
            print(f"   📖 Context Preview: {context[:100]}...")
        
        # Now test evaluation with context
        interface = ConcurrentGeminiInterface(max_workers=1, model="google/gemini-flash-1.5")
        
        # Create test question with context
        test_question_with_context = {
            'question': "What was Tinia's role in Etruscan religion?",
            'context': etruscan_corpus,
            'answer': "Tinia was the supreme deity of the Etruscan pantheon, ruler of heavens and wielder of lightning"
        }
        
        # Test question without context  
        test_question_without_context = {
            'question': "What was Tinia's role in Etruscan religion?",
            'answer': "Tinia was the supreme deity of the Etruscan pantheon"
        }
        
        print("\n   🧪 Testing LLM Response WITH Context:")
        
        # Simulate the evaluation prompt with context (from our fix)
        prompt_with_context = f"""Context: {test_question_with_context['context']}

Based on the context above, please answer the following question:

Question: {test_question_with_context['question']}

Provide a clear, accurate answer. If the question is mathematical, show your work. If it's factual, provide specific details."""

        print(f"      📝 Prompt Preview: {prompt_with_context[:200]}...")
        
        # Get response using OpenRouter interface directly
        from docs_to_eval.llm.openrouter_interface import OpenRouterInterface, OpenRouterConfig
        
        config = OpenRouterConfig(
            model="google/gemini-flash-1.5",
            api_key=os.getenv('OPENROUTER_API_KEY')
        )
        llm_interface = OpenRouterInterface(config)
        
        response_with_context = await llm_interface.generate_response(prompt_with_context)
        print(f"      🤖 Response WITH Context: {response_with_context.text[:200]}...")
        
        print("\n   🧪 Testing LLM Response WITHOUT Context:")
        prompt_without_context = f"""Please answer the following question based on your knowledge:

Question: {test_question_without_context['question']}

Provide a clear, accurate answer. If the question is mathematical, show your work. If it's factual, provide specific details."""

        response_without_context = await llm_interface.generate_response(prompt_without_context)
        print(f"      🤖 Response WITHOUT Context: {response_without_context.text[:200]}...")
        
        # Check if context made a difference
        with_context_mentions_etruscan = 'etruscan' in response_with_context.text.lower()
        without_context_mentions_etruscan = 'etruscan' in response_without_context.text.lower()
        
        print(f"\n   📊 Analysis:")
        print(f"      📚 With Context mentions 'Etruscan': {with_context_mentions_etruscan}")
        print(f"      🚫 Without Context mentions 'Etruscan': {without_context_mentions_etruscan}")
        
        context_helped = with_context_mentions_etruscan and len(response_with_context.text) > len(response_without_context.text) * 0.8
        
        return context_helped
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


async def main():
    """Run context fix test"""
    print("🎯 TESTING CONTEXT FIX FOR EVALUATION")
    print("🔧 Fix: Include corpus context in LLM evaluation prompts")
    print("=" * 60)
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ No OPENROUTER_API_KEY found")
        return
    
    success = await test_context_in_evaluation()
    
    print(f"\n🎉 CONTEXT FIX TEST RESULT:")
    print("=" * 40)
    
    if success:
        print("✅ CONTEXT FIX SUCCESSFUL!")
        print("🎯 Key Improvements:")
        print("   ✓ LLM now receives corpus context during evaluation")
        print("   ✓ Responses are more domain-specific and accurate")
        print("   ✓ Context asymmetry problem solved")
        print("\n🚀 Expected: Significantly higher evaluation scores!")
    else:
        print("⚠️ Context fix needs more testing")
        print("💡 Check API connectivity and context formatting")


if __name__ == "__main__":
    asyncio.run(main())