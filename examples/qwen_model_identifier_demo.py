#!/usr/bin/env python3
"""
🎯 QWEN MODEL TEST: Test with correct Qwen model identifier
Let's test available Qwen models on fictional content
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


async def test_qwen_models():
    """Test with correct Qwen model identifiers"""
    
    print("🚀 QWEN MODEL EVALUATION TEST")
    print("🎭 Testing on Fictional Fantasy Content")
    print("=" * 60)
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Need OPENROUTER_API_KEY")
        return
    
    integrator = LMEvalHarnessIntegrator()
    
    try:
        # 🎭 FICTIONAL CONTENT: Sky Pirates with clear numbers for mathematical reasoning
        sky_pirates_corpus = """
        The Skyborne Pirate Fleet operates from floating cities above the Storm Seas.
        Captain Lightning-Beard commands the flagship 'Thunder Strike' with 45 crew members.
        First Mate Storm-Eye pilots the scout ship 'Wind Dancer' carrying 23 pirates.
        Quartermaster Cloud-Walker manages the supply vessel 'Rain Bringer' with 34 sailors.
        
        The fleet's treasure vault contains exactly 8,742 golden doubloons.
        Each doubloon weighs 28.5 grams, making the total treasure weight 249,147 grams.
        The pirates have raided 67 merchant vessels over the past 3 years.
        Their average raid yields 130 doubloons per captured ship.
        
        The Thunder Strike measures 89 meters in length and 24 meters in width.
        Its main cannon can fire projectiles up to 1,200 meters distance.
        The ship's crew includes 12 gunners, 8 navigators, and 25 deck hands.
        Daily food consumption for the entire crew requires 156 kilograms of provisions.
        
        Wind crystals power the ships' flight engines at 2,400 rpm rotation speed.
        Each crystal generates enough lift for 750 kilograms of ship weight.
        The Thunder Strike requires 6 crystals for stable flight operations.
        Crystal replacement costs 85 doubloons per unit from sky-merchants.
        """
        
        print(f"🏴‍☠️ Created Sky Pirates corpus ({len(sky_pirates_corpus)} chars)")
        print("📊 Content: Fictional pirates with specific numbers for math testing")
        
        # Try different Qwen model identifiers that might work
        qwen_models_to_try = [
            "openrouter/qwen/qwen-2.5-72b-instruct",  # More likely to exist
            "openrouter/qwen/qwen-2-72b-instruct",    # Alternative
            "openrouter/qwen/qwen-1.5-72b-chat",      # Older version
            "openrouter/google/gemini-flash-1.5",     # Fallback that we know works
        ]
        
        successful_model = None
        
        # Create the evaluation task once
        print(f"\n📋 Creating evaluation task...")
        
        task_result = await integrator.create_dynamic_lm_eval_task(
            corpus_text=sky_pirates_corpus,
            task_name="sky_pirates_eval",
            num_questions=5,
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        print(f"✅ Generated {task_result['generated_items']} questions about sky pirates")
        
        # Test each model until one works
        for model_name in qwen_models_to_try:
            print(f"\n🤖 Testing {model_name}...")
            
            try:
                results = await integrator.run_lm_eval_evaluation(
                    model_name=model_name,
                    task_names=["sky_pirates_eval"]
                )
                
                # Success! Show results
                task_results = results["results"]["sky_pirates_eval"]
                print(f"✅ SUCCESS with {model_name}")
                print(f"  🎯 exact_match: {task_results['exact_match']:.4f}")
                print(f"  🔍 f1_score: {task_results['f1']:.4f}")
                
                successful_model = model_name
                
                # Show what this proves
                print(f"\n🎉 BREAKTHROUGH DEMONSTRATED:")
                print(f"✅ Model: {model_name.split('/')[-1]}")
                print(f"✅ Domain: Completely fictional sky pirates")
                print(f"✅ Content: Made-up numbers and facts")
                print(f"✅ Evaluation: Context-aware Q&A generation")
                print(f"✅ Format: Industry-standard lm-evaluation-harness")
                
                break
                
            except Exception as e:
                print(f"❌ Failed with {model_name}: {str(e)[:100]}...")
                continue
        
        if not successful_model:
            print(f"\n❌ No working Qwen models found in this test")
            return None
        
        # Generate the lm-eval command for reproducibility
        print(f"\n🔧 LM-EVAL COMMAND FOR REPRODUCTION:")
        command = integrator.generate_lm_eval_command(
            ["sky_pirates_eval"], 
            successful_model
        )
        print(f"{command}")
        
        # Show the generated files
        print(f"\n📁 Generated Files:")
        for task in integrator.created_tasks:
            print(f"  📋 Config: {task['config_path']}")
            print(f"  📊 Dataset: {task['dataset_path']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        integrator.cleanup()


async def show_fictional_questions():
    """Show what kinds of questions get generated from fictional content"""
    
    print(f"\n📋 SAMPLE FICTIONAL QUESTIONS PREVIEW")
    print("=" * 60)
    
    integrator = LMEvalHarnessIntegrator()
    
    try:
        # Very short fictional content for demo
        mini_fiction = """
        The Crystal Miners of Mount Sparkle extract 847 gems daily.
        Each gem weighs 12.3 grams and sells for 45 silver coins.
        Master Miner Glitter-Beard leads a team of 23 skilled workers.
        The mine's deepest tunnel extends 156 meters underground.
        """
        
        task = await integrator.create_dynamic_lm_eval_task(
            corpus_text=mini_fiction,
            task_name="preview_questions",
            num_questions=3,
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        # Read and show the generated dataset
        import json
        with open(task['dataset_path'], 'r') as f:
            dataset = json.load(f)
        
        print(f"📚 Generated from: Crystal Miners fictional content")
        print(f"🎯 Questions created: {len(dataset['test'])}")
        
        for i, item in enumerate(dataset['test'], 1):
            print(f"\n❓ Question {i}:")
            print(f"   {item['question']}")
            print(f"💡 Expected Answer:")
            print(f"   {item['answer']}")
            print(f"📖 Context Used:")
            print(f"   {item['context'][:100]}...")
        
        print(f"\n🚀 This demonstrates:")
        print(f"✅ Automatic question generation from ANY content")
        print(f"✅ Context-aware evaluation setup")
        print(f"✅ Ready for lm-evaluation-harness")
        
    finally:
        integrator.cleanup()


if __name__ == "__main__":
    print("🎭 QWEN + FICTIONAL CONTENT EVALUATION")
    print("🎯 Testing domain-agnostic capabilities")
    
    # Show sample questions first
    asyncio.run(show_fictional_questions())
    
    # Run main test
    print("\n" + "="*80)
    asyncio.run(test_qwen_models())
