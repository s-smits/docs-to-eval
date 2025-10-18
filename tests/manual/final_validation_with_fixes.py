#!/usr/bin/env python3
"""
🎯 FINAL VALIDATION WITH ALL CRITICAL FIXES
Test the complete system with:
1. Context fix ✓
2. Verification fixes ✓  
3. Better question generation
4. End-to-end pipeline validation
"""

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skip(reason="Manual integration test that requires external services.")

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from docs_to_eval.core.agentic import AgenticBenchmarkGenerator
from docs_to_eval.core.evaluation import EvaluationType
from docs_to_eval.core.verification import VerificationOrchestrator
from docs_to_eval.llm.openrouter_interface import OpenRouterInterface, OpenRouterConfig


async def final_validation_test():
    """Complete end-to-end test with all fixes"""
    print("🎯 FINAL VALIDATION WITH ALL CRITICAL FIXES")
    print("=" * 60)
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ No API key found")
        return
    
    # Rich Etruscan corpus
    etruscan_corpus = """
    Tinia was the supreme deity of the Etruscan pantheon, ruler of the heavens and wielder of lightning.
    He was equivalent to Jupiter in Roman mythology but had distinctly Etruscan characteristics.
    Maris served as the god of war and agriculture, often depicted with spear and shield in Etruscan art.
    Unlike Ares, Maris also governed agricultural fertility and seasonal cycles.
    Voltumna was a mysterious chthonic deity, associated with vegetation cycles and the underworld.
    Unlike Greek Hades, Voltumna had agricultural significance in Etruscan religious practices.
    Menrva governed wisdom, warfare, and the arts, patron of craftsmen and strategic thinking.
    She was similar to Athena but had unique Etruscan attributes in metalworking and divination.
    """
    
    print("🔨 Step 1: Generate Questions with Context")
    print("-" * 40)
    
    try:
        generator = AgenticBenchmarkGenerator(EvaluationType.DOMAIN_KNOWLEDGE)
        
        start_time = time.time()
        items = await generator.generate_benchmark_async(
            corpus_text=etruscan_corpus,
            num_questions=3
        )
        generation_time = time.time() - start_time
        
        print(f"✅ Generated {len(items)} questions in {generation_time:.2f}s")
        
        if not items:
            print("❌ No questions generated")
            return
        
        # Display generated questions
        for i, item in enumerate(items[:2]):
            question = getattr(item, 'question', '')
            answer = getattr(item, 'answer', '')
            context = getattr(item, 'context', '')
            
            print(f"\n📝 Question {i+1}: {question}")
            print(f"💬 Expected Answer: {answer[:80]}...")
            print(f"📚 Context Length: {len(context)} chars")
        
        print("\n🤖 Step 2: Evaluate with LLM (Context Fix Applied)")
        print("-" * 40)
        
        # Set up LLM
        config = OpenRouterConfig(
            model="google/gemini-flash-1.5",
            api_key=os.getenv('OPENROUTER_API_KEY')
        )
        llm = OpenRouterInterface(config)
        
        # Set up verification with fixes
        orchestrator = VerificationOrchestrator()
        
        evaluation_results = []
        
        for i, item in enumerate(items):
            question = getattr(item, 'question', '')
            expected_answer = getattr(item, 'answer', '')
            context = getattr(item, 'context', '')
            
            print(f"\n   🧪 Evaluating Question {i+1}...")
            print(f"      Q: {question[:60]}...")
            
            # Apply context fix - include context in evaluation
            if context:
                evaluation_prompt = f"""Context: {context}

Based on the context above, please answer the following question:

Question: {question}

Provide a clear, accurate answer. If the question is mathematical, show your work. If it's factual, provide specific details."""
            else:
                evaluation_prompt = f"""Please answer the following question based on your knowledge:

Question: {question}

Provide a clear, accurate answer. If the question is mathematical, show your work. If it's factual, provide specific details."""
            
            # Get LLM response
            response = await llm.generate_response(evaluation_prompt)
            prediction = response.text
            
            print(f"      Expected: {expected_answer[:50]}...")
            print(f"      Predicted: {prediction[:50]}...")
            
            # Apply verification fixes - use domain_factual
            verification_result = orchestrator.verify(prediction, expected_answer, "domain_factual")
            
            print(f"      🎯 Score: {verification_result.score:.3f} (Method: {verification_result.method})")
            
            evaluation_results.append({
                'question': question,
                'expected': expected_answer,
                'predicted': prediction,
                'score': verification_result.score,
                'method': verification_result.method
            })
        
        print("\n📊 Step 3: Final Results Analysis")
        print("-" * 40)
        
        scores = [result['score'] for result in evaluation_results]
        mean_score = sum(scores) / len(scores) if scores else 0
        
        print(f"🎯 FINAL RESULTS:")
        print(f"   📊 Mean Score: {mean_score:.3f}")
        print(f"   📈 Score Range: {min(scores):.3f} - {max(scores):.3f}")  
        print(f"   🔢 Questions Evaluated: {len(evaluation_results)}")
        print(f"   ⏱️ Total Time: {time.time() - start_time:.2f}s")
        
        # Success criteria
        print(f"\n🏆 IMPROVEMENT ANALYSIS:")
        
        baseline_score = 0.253  # Original score from user's results
        improvement = mean_score - baseline_score
        improvement_percent = (improvement / baseline_score) * 100 if baseline_score > 0 else 0
        
        print(f"   📈 Baseline Score: {baseline_score:.3f}")
        print(f"   🎯 New Score: {mean_score:.3f}")
        print(f"   📊 Improvement: +{improvement:.3f} ({improvement_percent:+.1f}%)")
        
        if mean_score > 0.5:
            print(f"\n🚀 EXCELLENT! MAJOR IMPROVEMENT ACHIEVED!")
            print("✅ System is now performing at production level")
            print("💰 Ready to make you rich!")
        elif mean_score > 0.4:
            print(f"\n🎯 GOOD! Significant improvement achieved")
            print("✅ System is much better than baseline")
            print("🔧 Minor optimizations could push it further")
        elif mean_score > baseline_score * 1.5:
            print(f"\n📈 IMPROVEMENT! Score increased significantly")
            print("✅ Fixes are working in the right direction")
            print("🔧 Continue optimizing for higher scores")
        else:
            print(f"\n⚠️ More work needed")
            print("🔧 Check individual question performance")
        
        # Show individual results
        print(f"\n📋 Individual Question Performance:")
        for i, result in enumerate(evaluation_results):
            print(f"   {i+1}. Score: {result['score']:.3f} | Q: {result['question'][:40]}...")
        
        return mean_score
        
    except Exception as e:
        print(f"❌ Final validation failed: {e}")
        return 0.0


async def main():
    """Run final validation"""
    final_score = await final_validation_test()
    
    print(f"\n🎉 FINAL VALIDATION COMPLETE!")
    print("=" * 50)
    
    if final_score >= 0.5:
        print("🚀 SYSTEM IS PRODUCTION READY!")
        print("💰 TIME TO MAKE MONEY!")
    elif final_score >= 0.4:
        print("🎯 SYSTEM IS SIGNIFICANTLY IMPROVED!")
        print("🔧 Minor tweaks for production readiness")
    else:
        print("📈 SYSTEM IS IMPROVING!")
        print("🔧 Continue optimization")
    
    print(f"Final Score: {final_score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
