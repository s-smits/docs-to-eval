#!/usr/bin/env python3
"""
🚨 CRITICAL FIXES IMPLEMENTATION
Fix the major issues identified in ultimate validation:
1. Question/Answer template alignment
2. Verification system accuracy  
3. Numerical verification
4. Question generation quality
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from docs_to_eval.core.agentic import AgenticBenchmarkGenerator
from docs_to_eval.core.evaluation import EvaluationType
from docs_to_eval.core.verification import VerificationOrchestrator
from docs_to_eval.llm.openrouter_interface import OpenRouterInterface, OpenRouterConfig


async def test_critical_fixes():
    """Test and demonstrate critical fixes"""
    print("🚨 IMPLEMENTING CRITICAL FIXES")
    print("=" * 50)
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ No API key found")
        return
    
    # Fix 1: Test improved question generation
    print("\n🔧 FIX 1: Testing Question Generation Quality")
    print("-" * 40)
    
    etruscan_corpus = """
    Tinia was the supreme deity of the Etruscan pantheon, ruler of the heavens and wielder of lightning.
    He was equivalent to Jupiter in Roman mythology but had distinctly Etruscan characteristics.
    Maris served as the god of war and agriculture, often depicted with spear and shield.
    Voltumna was a mysterious chthonic deity associated with vegetation and the underworld.
    """
    
    try:
        generator = AgenticBenchmarkGenerator(EvaluationType.DOMAIN_KNOWLEDGE)
        items = await generator.generate_benchmark_async(
            corpus_text=etruscan_corpus,
            num_questions=1
        )
        
        if items:
            item = items[0]
            question = getattr(item, 'question', '')
            answer = getattr(item, 'answer', '')
            context = getattr(item, 'context', '')
            
            print(f"📝 Generated Question: {question}")
            print(f"💬 Generated Answer: {answer}")
            print(f"📚 Has Context: {'Yes' if context else 'No'}")
            
            # The issue is clear: poor question templates
            if "Given the context, if what is the significance of" in question:
                print("❌ PROBLEM: Poor question template detected")
                print("🔧 NEED: Better question generation prompts")
            else:
                print("✅ Question template looks good")
                
        else:
            print("❌ No items generated")
            
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
    
    # Fix 2: Test verification improvements
    print("\n🔧 FIX 2: Testing Verification System")
    print("-" * 40)
    
    orchestrator = VerificationOrchestrator()
    
    # Test numerical verification
    numerical_tests = [
        ("42", "42.0", "Should be 1.0"),
        ("3.14159", "3.14", "Should be ~0.8"),
        ("25%", "0.25", "Should be ~0.8")
    ]
    
    print("📊 Numerical Verification Tests:")
    for prediction, truth, expected in numerical_tests:
        try:
            result = orchestrator.verify(prediction, truth, "numerical")
            print(f"   {prediction} vs {truth}: {result.score:.3f} ({expected})")
        except Exception as e:
            print(f"   ❌ {prediction} vs {truth}: Error - {e}")
    
    # Fix 3: Test realistic evaluation scenario
    print("\n🔧 FIX 3: Testing Realistic Evaluation")
    print("-" * 40)
    
    # Create a better question manually for testing
    realistic_question = "What was Tinia's role in Etruscan religion?"
    realistic_answer = "Tinia was the supreme deity of the Etruscan pantheon, ruler of heavens and wielder of lightning"
    
    # Test with LLM
    config = OpenRouterConfig(
        model="google/gemini-flash-1.5",
        api_key=os.getenv('OPENROUTER_API_KEY')
    )
    llm = OpenRouterInterface(config)
    
    # Use our fixed context evaluation
    evaluation_prompt = f"""Context: {etruscan_corpus}

Based on the context above, please answer the following question:

Question: {realistic_question}

Provide a clear, accurate answer. If the question is mathematical, show your work. If it's factual, provide specific details."""

    try:
        response = await llm.generate_response(evaluation_prompt)
        prediction = response.text
        
        print(f"📝 Question: {realistic_question}")
        print(f"💬 Expected: {realistic_answer}")
        print(f"🤖 Predicted: {prediction[:100]}...")
        
        # Test verification
        result = orchestrator.verify(prediction, realistic_answer, "domain_factual")
        print(f"🎯 Verification Score: {result.score:.3f}")
        print(f"🔍 Method Used: {result.method}")
        
        if result.score > 0.5:
            print("✅ Realistic evaluation working!")
        else:
            print("❌ Still issues with verification")
            
    except Exception as e:
        print(f"❌ Realistic evaluation failed: {e}")
    
    print("\n🎯 CRITICAL FIXES SUMMARY:")
    print("=" * 50)
    print("🔧 Issues Identified:")
    print("   1. Question templates need improvement")
    print("   2. Answer templates misaligned") 
    print("   3. Verification system needs tuning")
    print("   4. Need better question/answer pairs")
    print("\n💡 Recommendations:")
    print("   ✓ Fix question generation prompts")
    print("   ✓ Improve verification thresholds")
    print("   ✓ Create better template alignment")
    print("   ✓ Test with realistic Q&A pairs")


if __name__ == "__main__":
    asyncio.run(test_critical_fixes())
