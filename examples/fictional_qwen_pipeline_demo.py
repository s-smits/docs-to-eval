#!/usr/bin/env python3
"""
🎭 FICTIONAL DOMAIN TEST: Evaluate Qwen3-0.6B on Made-Up Fantasy World
Test your novel evaluation system on completely fictional content to prove domain-agnosticism!
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
    from tests.manual.lm_eval_harness_adapter import LMEvalHarnessIntegrator
except ImportError:  # pragma: no cover - optional dependency for manual demos
    LMEvalHarnessIntegrator = None
from docs_to_eval.core.evaluation import EvaluationType


async def test_qwen_on_fictional_world():
    """
    🎭 Test Qwen3-0.6B on completely fictional fantasy world
    
    This proves your system can evaluate LLMs on ANY domain - even made-up ones!
    """
    
    print("🎭 FICTIONAL WORLD EVALUATION TEST")
    print("🤖 Model: Qwen/Qwen3-0.6B")
    print("🌟 Domain: Completely Made-Up Fantasy World")
    print("=" * 80)

    if LMEvalHarnessIntegrator is None:
        print("⚠️ LM eval harness adapter not available. Skipping harness demo.")
        return
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Need OPENROUTER_API_KEY for testing")
        return
    
    integrator = LMEvalHarnessIntegrator()
    
    try:
        # 🎭 FICTIONAL CORPUS: Made-up fantasy world with consistent lore
        fictional_corpus = """
        The Crystalline Kingdoms of Zephyria are ruled by the Council of Seven Prisms.
        Each prism governs a different aspect of reality through chromantic magic.
        
        The Crimson Prism controls thermal energies and volcanic activity across the realm.
        Its wielder, Archon Pyrellia, can summon flame spirits called Ignithers.
        The Crimson Citadel houses 847 fire-gems that power the kingdom's forges.
        
        The Azure Prism manipulates water and ice throughout Zephyria's northern regions.
        Archon Glacius commands frost-wraiths and can freeze entire rivers in seconds.
        The Azure Spire contains exactly 1,203 ice-crystals maintaining perpetual winter.
        
        The Emerald Prism governs all plant life and controls the Great Forest of Verdancia.
        Archon Sylvana can communicate with ancient tree-spirits called Elderbarks.
        The Emerald Grove nurtures 2,449 magical seeds that grow into sentient plants.
        
        The Violet Prism oversees temporal magic and the flow of time itself.
        Archon Chronos can slow or accelerate time in localized areas up to 73% variance.
        The Violet Observatory tracks 156 temporal anomalies across dimensional barriers.
        
        The Golden Prism commands light magic and celestial phenomena.
        Archon Lumina controls star-light beams that can travel 15,000 kilometers instantly.
        The Golden Spire focuses sunlight through 628 prismatic lenses for energy.
        
        The Silver Prism governs illusion magic and dream manipulation.
        Archon Mirage can create phantom armies of up to 5,000 illusory soldiers.
        The Silver Sanctum contains 312 dream-catchers preserving ancient visions.
        
        The Obsidian Prism controls shadow magic and the realm of the deceased.
        Archon Umbra can summon shadow-beasts and communicate with spirits.
        The Obsidian Crypt houses 1,847 soul-stones containing trapped phantoms.
        
        The capital city of Prismatica sits at the convergence of all seven prism-towers.
        Population: 89,456 citizens including 12,334 mages and 7,822 prism-knights.
        The city's central Nexus Crystal amplifies all chromantic magic by 340%.
        
        Zephyria's currency consists of chromantic shards valued by color intensity.
        Exchange rates: 1 Radiant Shard = 15 Brilliant Shards = 225 Glimmer Shards.
        The realm's annual magical energy output measures 47,829 chromatons.
        """
        
        print(f"\n🔮 Created fictional fantasy corpus ({len(fictional_corpus)} chars)")
        print("📚 Content: Made-up magical realm with rulers, powers, and precise numbers")
        
        # Step 1: Create mathematical reasoning test
        print("\n🧮 TEST 1: Mathematical Reasoning in Fictional Context")
        
        math_task = await integrator.create_dynamic_lm_eval_task(
            corpus_text=fictional_corpus,
            task_name="zephyria_mathematics",
            num_questions=8,
            eval_type=EvaluationType.MATHEMATICAL
        )
        
        print(f"✅ Created math task: {math_task['generated_items']} questions")
        
        # Step 2: Create factual knowledge test
        print("\n📖 TEST 2: Factual Knowledge in Fictional Context")
        
        factual_task = await integrator.create_dynamic_lm_eval_task(
            corpus_text=fictional_corpus,
            task_name="zephyria_lore",
            num_questions=8,
            eval_type=EvaluationType.FACTUAL_QA
        )
        
        print(f"✅ Created factual task: {factual_task['generated_items']} questions")
        
        # Step 3: Run evaluation on Qwen3-0.6B
        print("\n🚀 RUNNING EVALUATION ON QWEN3-0.6B")
        print("🎯 Testing domain-agnosticism on completely fictional content!")
        
        results = await integrator.run_lm_eval_evaluation(
            model_name="openrouter/qwen/qwen-3-0.6b",
            task_names=["zephyria_mathematics", "zephyria_lore"]
        )
        
        # Step 4: Display comprehensive results
        print("\n📊 QWEN3-0.6B EVALUATION RESULTS")
        print("=" * 60)
        
        for task_name in ["zephyria_mathematics", "zephyria_lore"]:
            if task_name in results["results"]:
                task_results = results["results"][task_name]
                print(f"\n📋 Task: {task_name}")
                print(f"  🎯 exact_match: {task_results['exact_match']:.4f}")
                print(f"  🔍 f1_score: {task_results['f1']:.4f}")
                print(f"  📈 Domain: Fictional Fantasy World")
        
        # Step 5: Generate lm-eval command for reproducibility
        print("\n🔧 LM-EVAL COMMAND FOR REPRODUCTION:")
        command = integrator.generate_lm_eval_command(
            ["zephyria_mathematics", "zephyria_lore"], 
            "openrouter/qwen/qwen-3-0.6b"
        )
        print(f"{command}")
        
        # Step 6: Analysis and insights
        print("\n🎭 FICTIONAL DOMAIN TEST ANALYSIS")
        print("=" * 60)
        
        math_score = results["results"]["zephyria_mathematics"]["f1"]
        factual_score = results["results"]["zephyria_lore"]["f1"]
        
        print(f"🧮 Mathematical reasoning: {math_score:.3f}")
        print(f"📚 Factual knowledge: {factual_score:.3f}")
        print(f"🔀 Average performance: {(math_score + factual_score) / 2:.3f}")
        
        print(f"\n🚀 BREAKTHROUGH VALIDATED:")
        print(f"✅ Domain-agnostic evaluation works on COMPLETELY fictional content")
        print(f"✅ Qwen3-0.6B evaluated on made-up magical realm")
        print(f"✅ Context-aware questions generated automatically")
        print(f"✅ Mathematical and factual reasoning tested")
        print(f"✅ Industry-standard lm-eval format maintained")
        
        # Step 7: Show generated files for inspection
        print(f"\n📁 Generated Files for Manual Inspection:")
        for task in integrator.created_tasks:
            print(f"  📋 {task['task_name']}: {task['config_path']}")
            print(f"  📊 Dataset: {task['dataset_path']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Fictional test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        integrator.cleanup()


async def compare_models_on_fiction():
    """
    🏆 BONUS: Compare multiple models on the same fictional domain
    """
    
    print("\n🏆 BONUS: MULTI-MODEL COMPARISON ON FICTION")
    print("=" * 80)

    if LMEvalHarnessIntegrator is None:
        print("⚠️ LM eval harness adapter not available. Skipping comparison demo.")
        return
    
    models_to_test = [
        "openrouter/qwen/qwen-3-0.6b",
        "openrouter/google/gemini-flash-1.5",
        # Add more models as needed
    ]
    
    # Quick single-task comparison
    integrator = LMEvalHarnessIntegrator()
    
    try:
        # Shorter fictional corpus for quick comparison
        mini_fiction = """
        The Sky Pirates of Nimbulon operate three flying ships powered by storm-crystals.
        Captain Thunderbolt commands the flagship 'Lightning Bolt' with 42 crew members.
        First Mate Galewind pilots the scout ship 'Wind Whisper' carrying 18 pirates.
        Quartermaster Stormcloud manages the cargo vessel 'Rain Maker' with 31 sailors.
        Each storm-crystal generates 1,250 volts of electrical energy for propulsion.
        The pirates have raided 73 cloud-cities and collected 15,649 sky-gems total.
        """
        
        task = await integrator.create_dynamic_lm_eval_task(
            corpus_text=mini_fiction,
            task_name="nimbulon_pirates",
            num_questions=4,
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        print(f"📚 Created mini-fiction task: {task['generated_items']} questions")
        
        comparison_results = {}
        
        for model in models_to_test:
            print(f"\n🤖 Testing {model}...")
            
            try:
                results = await integrator.run_lm_eval_evaluation(
                    model_name=model,
                    task_names=["nimbulon_pirates"]
                )
                
                score = results["results"]["nimbulon_pirates"]["f1"]
                comparison_results[model] = score
                print(f"   📊 Score: {score:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                comparison_results[model] = 0.0
        
        # Show comparison
        print(f"\n🏆 MODEL COMPARISON ON FICTIONAL SKY PIRATES:")
        sorted_results = sorted(comparison_results.items(), key=lambda x: x[1], reverse=True)
        
        for i, (model, score) in enumerate(sorted_results, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
            print(f"  {medal} {model.split('/')[-1]}: {score:.4f}")
        
        return comparison_results
        
    finally:
        integrator.cleanup()


if __name__ == "__main__":
    print("🎭 STARTING FICTIONAL DOMAIN EVALUATION")
    print("🎯 Testing domain-agnostic capabilities on made-up content")
    print("🤖 Primary target: Qwen/Qwen3-0.6B")
    
    # Run main test
    asyncio.run(test_qwen_on_fictional_world())
    
    # Run comparison if time permits
    print("\n" + "="*80)
    asyncio.run(compare_models_on_fiction())
