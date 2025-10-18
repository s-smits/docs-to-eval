#!/usr/bin/env python3
"""
🎯 SIMPLE QWEN TEST: Quick evaluation on Qwen3-0.6B
Test with known working configuration
"""

import asyncio
import os
import sys
from pathlib import Path



# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

try:
    from manual_tests.lm_eval_harness_adapter import LMEvalHarnessIntegrator
except ImportError:  # pragma: no cover - optional dependency for manual tests
    LMEvalHarnessIntegrator = None
from docs_to_eval.core.evaluation import EvaluationType


async def simple_qwen_test():
    """Simple test that should work"""
    
    print("🚀 SIMPLE QWEN3-0.6B TEST")
    print("=" * 50)
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Need OPENROUTER_API_KEY")
        return
    
    integrator = LMEvalHarnessIntegrator()
    
    try:
        # Use the same corpus that worked before
        simple_corpus = """
        Etruscan civilization flourished in central Italy before Roman expansion.
        Tinia was the supreme deity, ruler of heavens and wielder of lightning.
        Maris served as god of war and agriculture, depicted with spear and shield.
        Voltumna was a chthonic deity associated with vegetation and underworld.
        Menrva governed wisdom and warfare, patron of crafts and strategic thinking.
        Their religious practices emphasized divination through haruspicy and augury.
        Etruscan art featured distinctive styles in pottery, metalwork, and tomb frescoes.
        """
        
        print(f"📚 Using working Etruscan corpus ({len(simple_corpus)} chars)")
        
        # Test with DOMAIN_KNOWLEDGE (this worked before)
        print("\n🔍 Creating DOMAIN_KNOWLEDGE task...")
        
        task_result = await integrator.create_dynamic_lm_eval_task(
            corpus_text=simple_corpus,
            task_name="simple_etruscan_test",
            num_questions=3,  # Small number for quick test
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        print(f"✅ Generated {task_result['generated_items']} questions")
        
        # Test on Qwen3-0.6B
        print(f"\n🤖 Testing Qwen3-0.6B...")
        
        results = await integrator.run_lm_eval_evaluation(
            model_name="openrouter/qwen/qwen-3-0.6b",
            task_names=["simple_etruscan_test"]
        )
        
        # Show results
        task_results = results["results"]["simple_etruscan_test"]
        print(f"\n📊 QWEN3-0.6B RESULTS:")
        print(f"  🎯 exact_match: {task_results['exact_match']:.4f}")
        print(f"  🔍 f1_score: {task_results['f1']:.4f}")
        
        print(f"\n🎉 SUCCESS! Qwen3-0.6B evaluated on domain-specific content!")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        integrator.cleanup()


if __name__ == "__main__":
    asyncio.run(simple_qwen_test())
