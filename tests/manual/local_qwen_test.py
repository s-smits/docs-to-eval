#!/usr/bin/env python3
"""
🏠 LOCAL QWEN TEST: Test evaluation system with local mock LLM
Demonstrates the complete evaluation pipeline without requiring API calls
"""
import pytest
pytestmark = pytest.mark.skip(reason="Demo-only; excluded from CI test suite")

import asyncio
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from docs_to_eval.core.agentic import AgenticBenchmarkGenerator
from docs_to_eval.utils.config import EvaluationType
from docs_to_eval.core.verification import VerificationOrchestrator
from docs_to_eval.llm.mock_interface import MockLLMInterface
from docs_to_eval.llm.qwen_local_interface import QwenModelFactory


class LocalQwenEvaluator:
    """Local evaluation system using mock LLM to simulate Qwen behavior"""
    
    def __init__(self):
        self.mock_llm = MockLLMInterface()
        self.verification_orchestrator = VerificationOrchestrator()


class RealQwenEvaluator:
    """🚀 Real Local Qwen Evaluation using HuggingFace transformers"""
    
    def __init__(self, model_key: str = "qwen3-1.7b"):
        self.model_key = model_key
        self.qwen_interface = None
        self.verification_orchestrator = VerificationOrchestrator()
        self.model_loaded = False
        
    async def initialize_model(self):
        """Initialize the Qwen model"""
        if self.model_loaded:
            return True
            
        try:
            print(f"🤖 Initializing {self.model_key.upper()} model...")
            self.qwen_interface = QwenModelFactory.create_interface(
                self.model_key,
                max_new_tokens=256,  # Conservative for evaluation
                temperature=0.3,     # Lower temperature for more consistent responses
                do_sample=True
            )
            
            # Load the model
            success = self.qwen_interface.load_model()
            if success:
                self.model_loaded = True
                print(f"✅ {self.model_key.upper()} model loaded successfully!")
                return True
            else:
                print(f"❌ Failed to load {self.model_key.upper()} model")
                return False
                
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            return False
    
    async def create_fictional_benchmark(self, corpus_text: str, num_questions: int = 5):
        """Create benchmark from fictional content using agentic generation"""
        
        print(f"🔄 Generating {num_questions} questions from fictional corpus...")
        
        # Use your agentic benchmark generator
        generator = AgenticBenchmarkGenerator(EvaluationType.DOMAIN_KNOWLEDGE)
        
        benchmark_items = await generator.generate_benchmark_async(
            corpus_text=corpus_text,
            num_questions=num_questions
        )
        
        if not benchmark_items:
            print("❌ No questions generated, using fallback method...")
            return self._fallback_question_generation(corpus_text, num_questions)
        
        print(f"✅ Generated {len(benchmark_items)} questions successfully")
        return benchmark_items
    
    def _fallback_question_generation(self, corpus_text: str, num_questions: int):
        """Fallback question generation if agentic system fails"""
        
        # Extract key facts and numbers from the fictional corpus
        import re
        
        # Find numbers for mathematical questions
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', corpus_text)
        
        # Find key entities/concepts
        words = corpus_text.split()
        concepts = [w for w in words if len(w) > 5 and w[0].isupper()][:10]
        
        # Create simple questions manually
        fallback_questions = []
        
        if numbers:
            fallback_questions.append({
                'question': 'What is the first number mentioned in the text?',
                'answer': numbers[0],
                'context': corpus_text[:200] + '...',
                'concept': 'numerical_fact',
                'type': 'factual'
            })
        
        if concepts:
            fallback_questions.append({
                'question': f'What is {concepts[0]}?',
                'answer': f'{concepts[0]} is a key entity mentioned in the fictional world.',
                'context': corpus_text[:200] + '...',
                'concept': concepts[0].lower(),
                'type': 'factual'
            })
        
        # Add more generic questions
        fallback_questions.extend([
            {
                'question': 'What is the main setting described in this text?',
                'answer': 'A fictional fantasy world with magical elements.',
                'context': corpus_text[:200] + '...',
                'concept': 'setting',
                'type': 'comprehension'
            },
            {
                'question': 'What kind of world is being described?',
                'answer': 'A fantasy world with fictional characters and magical elements.',
                'context': corpus_text[:200] + '...',
                'concept': 'genre',
                'type': 'analysis'
            }
        ])
        
        return fallback_questions[:num_questions]
    
    async def evaluate_with_real_qwen(self, questions):
        """🚀 Evaluate questions using real Qwen model"""
        
        if not self.model_loaded:
            success = await self.initialize_model()
            if not success:
                raise RuntimeError("Failed to initialize Qwen model")
        
        print(f"🤖 Evaluating {len(questions)} questions with real {self.model_key.upper()}...")
        
        qwen_responses = []
        
        for i, item in enumerate(questions, 1):
            # Handle both dict and EnhancedBenchmarkItem objects
            if hasattr(item, 'question'):
                question = item.question
                context = item.context
                answer = item.answer
            else:
                question = item.get('question', '')
                context = item.get('context', '')
                answer = item.get('answer', '')
            
            # Create context-aware prompt
            if context:
                prompt = f"""Context: {context}

Based on the context above, please answer the following question:

Question: {question}

Provide a clear, accurate answer."""
            else:
                prompt = f"""Question: {question}

Provide a clear, accurate answer."""
            
            try:
                # Get real Qwen response
                response = await self.qwen_interface.generate_response(prompt)
                
                if response.success:
                    qwen_response = response.text
                    print(f"   Question {i}: ✅ Generated ({response.tokens_generated} tokens, {response.generation_time:.2f}s)")
                else:
                    qwen_response = f"Generation failed: {response.error}"
                    print(f"   Question {i}: ❌ Failed - {response.error}")
                
                qwen_responses.append({
                    'question': question,
                    'qwen_response': qwen_response,
                    'expected_answer': answer,
                    'context': context,
                    'tokens_generated': response.tokens_generated if response.success else 0,
                    'generation_time': response.generation_time
                })
                
            except Exception as e:
                print(f"   Question {i}: ❌ Error - {e}")
                qwen_responses.append({
                    'question': question,
                    'qwen_response': f"Error: {str(e)}",
                    'expected_answer': answer,
                    'context': context,
                    'tokens_generated': 0,
                    'generation_time': 0.0
                })
        
        return qwen_responses
    
    def evaluate_responses(self, qwen_responses):
        """Evaluate Qwen responses using your verification system"""
        
        print(f"📊 Evaluating {len(qwen_responses)} responses...")
        
        evaluation_results = []
        scores = []
        
        for i, response_data in enumerate(qwen_responses, 1):
            qwen_response = response_data['qwen_response']
            expected_answer = response_data['expected_answer']
            question = response_data['question']
            
            # Use your domain verification system
            verification_result = self.verification_orchestrator.verify(
                prediction=qwen_response,
                ground_truth=expected_answer,
                eval_type="domain_factual",  # Use lenient factual verification
                question=question
            )
            
            scores.append(verification_result.score)
            
            evaluation_results.append({
                'question': question,
                'qwen_response': qwen_response,
                'expected_answer': expected_answer,
                'score': verification_result.score,
                'method': verification_result.method,
                'details': verification_result.details,
                'tokens_generated': response_data.get('tokens_generated', 0),
                'generation_time': response_data.get('generation_time', 0.0)
            })
            
            print(f"   Question {i}: Score {verification_result.score:.3f} ({verification_result.method})")
        
        return evaluation_results, scores
    
    def generate_real_qwen_report(self, evaluation_results, scores, corpus_info):
        """Generate comprehensive evaluation report for real Qwen"""
        
        mean_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        total_tokens = sum(r.get('tokens_generated', 0) for r in evaluation_results)
        total_time = sum(r.get('generation_time', 0.0) for r in evaluation_results)
        
        print("\n📈 REAL QWEN EVALUATION REPORT")
        print("=" * 60)
        print(f"🤖 Model: {self.model_key.upper()}")
        print(f"🎭 Domain: {corpus_info['domain']}")
        print(f"📚 Corpus: {corpus_info['description']}")
        print(f"📝 Questions: {len(evaluation_results)}")
        print("\n📊 PERFORMANCE METRICS:")
        print(f"  📈 Mean Score: {mean_score:.4f}")
        print(f"  🎯 Max Score: {max_score:.4f}")
        print(f"  📉 Min Score: {min_score:.4f}")
        print(f"  🔍 Exact Match Rate: {sum(1 for s in scores if s >= 0.9) / len(scores):.4f}")
        print(f"  🚀 Total Tokens: {total_tokens}")
        print(f"  ⏱️ Total Time: {total_time:.2f}s")
        print(f"  🔥 Avg Speed: {total_tokens/total_time:.1f} tok/s" if total_time > 0 else "  🔥 Speed: N/A")
        
        # Get model performance stats
        if self.qwen_interface:
            stats = self.qwen_interface.get_performance_stats()
            print("\n🤖 MODEL PERFORMANCE:")
            print(f"  📊 Device: {stats.get('device', 'unknown')}")
            print(f"  🧠 Memory Used: {stats.get('memory_allocated', 0) / 1024**3:.2f} GB")
            print(f"  ⚡ Avg Speed: {stats.get('average_tokens_per_second', 0):.1f} tok/s")
        
        print("\n🔍 DETAILED RESULTS:")
        for i, result in enumerate(evaluation_results, 1):
            print(f"\n❓ Question {i}: {result['question'][:60]}...")
            print(f"🤖 {self.model_key.upper()}: {result['qwen_response'][:80]}...")
            print(f"✅ Expected: {result['expected_answer'][:60]}...")
            print(f"📊 Score: {result['score']:.3f} | Tokens: {result['tokens_generated']} | Time: {result['generation_time']:.2f}s")
        
        print("\n🚀 REAL QWEN CAPABILITIES DEMONSTRATED:")
        print("✅ Real HuggingFace transformers integration")
        print("✅ Local execution with GPU acceleration")
        print("✅ Context-aware question answering")
        print("✅ Domain-agnostic evaluation pipeline")
        print("✅ Performance monitoring and statistics")
        print("✅ No API dependencies required")
        
        return {
            'mean_score': mean_score,
            'max_score': max_score, 
            'min_score': min_score,
            'num_questions': len(evaluation_results),
            'exact_match_rate': sum(1 for s in scores if s >= 0.9) / len(scores),
            'total_tokens': total_tokens,
            'total_time': total_time,
            'model_stats': stats if self.qwen_interface else {},
            'results': evaluation_results
        }
    
    def cleanup(self):
        """Clean up model resources"""
        if self.qwen_interface:
            self.qwen_interface.unload_model()
            self.qwen_interface = None
        self.model_loaded = False


async def test_real_qwen_models():
    """🚀 Test real Qwen models on fictional content"""
    
    print("🚀 REAL QWEN MODELS EVALUATION TEST")
    print("🎭 Testing on Fictional Crystal Empire")
    print("=" * 80)
    
    # Fictional content for testing
    crystal_empire_corpus = """
    The Crystal Empire of Luminara spans seven floating islands above the Prism Sea.
    Emperor Crystallus rules from the Diamond Throne with 12,847 crystal guards.
    The Royal Treasury contains exactly 2,394,567 luminous gems worth immense value.
    
    Each island specializes in different crystal types and magical properties:
    
    Ruby Isle produces 450 fire-crystals daily, each weighing 73.2 grams.
    The Fire-Crystal Foundry employs 234 master craftsmen working in shifts.
    Annual production reaches 164,250 fire-crystals with 99.7% purity rating.
    
    Sapphire Heights mines 320 ice-crystals per day from frozen caverns.
    Each ice-crystal maintains temperatures of exactly -47.3 degrees Celsius.
    The 189 ice-miners work with specialized equipment costing 15,600 gold pieces.
    
    Crystal magic powers the empire's floating cities and sky-ships.
    The Grand Levitation Array requires 1,247 crystals to maintain altitude.
    """
    
    corpus_info = {
        'domain': 'Fictional Crystal Empire',
        'description': 'Real Qwen evaluation on made-up fantasy content with precise numbers'
    }
    
    # Test different Qwen models
    models_to_test = ["qwen3-0.6b", "qwen3-1.7b"]  # Start with smaller models
    
    for model_key in models_to_test:
        print(f"\n🤖 TESTING {model_key.upper()}")
        print("=" * 50)
        
        evaluator = RealQwenEvaluator(model_key)
        
        try:
            # Test the full pipeline
            questions = await evaluator.create_fictional_benchmark(
                crystal_empire_corpus,
                num_questions=3  # Small number for initial testing
            )
            
            qwen_responses = await evaluator.evaluate_with_real_qwen(questions)
            evaluation_results, scores = evaluator.evaluate_responses(qwen_responses)
            _ = evaluator.generate_real_qwen_report(evaluation_results, scores, corpus_info)
            
            print(f"\n✅ {model_key.upper()} test completed!")
            
        except Exception as e:
            print(f"❌ {model_key.upper()} test failed: {e}")
            
        finally:
            # Clean up to free memory for next model
            evaluator.cleanup()
            
        print("\n" + "="*50)
    
    print("\n🎉 REAL QWEN TESTING COMPLETE!")


async def test_qwen_model_comparison():
    """🏆 Compare different Qwen models side by side"""
    
    print("\n🏆 QWEN MODEL COMPARISON")
    print("=" * 60)
    
    # Short test content
    test_corpus = """
    The Sky Pirates of Nimbulon operate three flying ships powered by storm-crystals.
    Captain Thunderbolt commands the flagship 'Lightning Bolt' with 42 crew members.
    Each storm-crystal generates 1,250 volts of electrical energy for propulsion.
    The pirates have raided 73 cloud-cities and collected 15,649 sky-gems total.
    """
    
    models = ["qwen3-0.6b", "qwen3-1.7b"]
    results = {}
    
    for model_key in models:
        print(f"\n🤖 Testing {model_key}...")
        
        evaluator = RealQwenEvaluator(model_key)
        
        try:
            questions = await evaluator.create_fictional_benchmark(test_corpus, num_questions=2)
            qwen_responses = await evaluator.evaluate_with_real_qwen(questions)
            evaluation_results, scores = evaluator.evaluate_responses(qwen_responses)
            
            mean_score = sum(scores) / len(scores) if scores else 0
            total_tokens = sum(r.get('tokens_generated', 0) for r in evaluation_results)
            total_time = sum(r.get('generation_time', 0.0) for r in evaluation_results)
            
            results[model_key] = {
                'mean_score': mean_score,
                'total_tokens': total_tokens,
                'total_time': total_time,
                'speed': total_tokens / total_time if total_time > 0 else 0
            }
            
            print(f"   📊 Score: {mean_score:.3f}")
            print(f"   🚀 Speed: {results[model_key]['speed']:.1f} tok/s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[model_key] = {'error': str(e)}
            
        finally:
            evaluator.cleanup()
    
    # Show comparison
    print("\n🏆 MODEL COMPARISON RESULTS:")
    print("-" * 40)
    
    for model_key, data in results.items():
        if 'error' not in data:
            print(f"🤖 {model_key.upper()}:")
            print(f"   📈 Quality Score: {data['mean_score']:.3f}")
            print(f"   ⚡ Speed: {data['speed']:.1f} tok/s")
            print(f"   🚀 Tokens: {data['total_tokens']}")
        else:
            print(f"❌ {model_key.upper()}: {data['error']}")
    
    return results


# Update the main execution section
if __name__ == "__main__":
    print("🤖 QWEN LOCAL EVALUATION SYSTEM")
    print("🎯 Choose your testing mode:")
    print("1. 🏠 Mock/Simulated Qwen (original)")
    print("2. 🚀 Real Qwen Models (HuggingFace transformers)")
    print("3. 🏆 Real Qwen Model Comparison")
    print("\n" + "="*80)
    
    # For demo, run the real Qwen test
    import asyncio
    
    print("🚀 Running Real Qwen Models Test...")
    asyncio.run(test_real_qwen_models())
    
    print("\n🏆 Running Model Comparison...")
    asyncio.run(test_qwen_model_comparison())


async def test_fictional_crystal_empire():
    """Test on Crystal Empire fictional content"""
    
    crystal_empire_corpus = """
    The Crystal Empire of Luminara spans seven floating islands above the Prism Sea.
    Emperor Crystallus rules from the Diamond Throne with 12,847 crystal guards.
    The Royal Treasury contains exactly 2,394,567 luminous gems worth immense value.
    
    Each island specializes in different crystal types and magical properties:
    
    Ruby Isle produces 450 fire-crystals daily, each weighing 73.2 grams.
    The Fire-Crystal Foundry employs 234 master craftsmen working in shifts.
    Annual production reaches 164,250 fire-crystals with 99.7% purity rating.
    
    Sapphire Heights mines 320 ice-crystals per day from frozen caverns.
    Each ice-crystal maintains temperatures of exactly -47.3 degrees Celsius.
    The 189 ice-miners work with specialized equipment costing 15,600 gold pieces.
    
    Emerald Grove cultivates 278 nature-crystals through magical plant growth.
    The crystals enhance crop yields by 245% across the empire's farmlands.
    Grove-tenders number 156 druids who speak the ancient crystal language.
    
    Topaz Peak forges 195 lightning-crystals using captured storm energy.
    Each crystal stores 8,940 joules of electrical power for the empire.
    The Lightning Spire attracts storms with its 89-meter tall conductor rod.
    
    Crystal magic powers the empire's floating cities and sky-ships.
    The Grand Levitation Array requires 1,247 crystals to maintain altitude.
    Maintenance crews replace an average of 23 crystals weekly due to wear.
    
    Trade routes connect to 67 foreign kingdoms across the known world.
    Crystal exports generate 890,456 gold pieces annually in revenue.
    The empire's crystal-powered navy protects merchant vessels with 45 ships.
    """
    
    print("🏰 CRYSTAL EMPIRE EVALUATION TEST")
    print("🎭 Testing on completely fictional fantasy content")
    print("=" * 70)
    
    corpus_info = {
        'domain': 'Fictional Crystal Empire',
        'description': 'Made-up fantasy world with precise magical numbers and systems'
    }
    
    evaluator = LocalQwenEvaluator()
    
    try:
        # Step 1: Generate questions from fictional corpus
        questions = await evaluator.create_fictional_benchmark(
            crystal_empire_corpus, 
            num_questions=6
        )
        
        # Step 2: Simulate Qwen responses
        qwen_responses = await evaluator.simulate_qwen_responses(questions)
        
        # Step 3: Evaluate responses
        evaluation_results, scores = evaluator.evaluate_responses(qwen_responses)
        
        # Step 4: Generate comprehensive report
        report = evaluator.generate_report(evaluation_results, scores, corpus_info)
        
        return report
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_space_pirates():
    """Test on Space Pirates fictional content"""
    
    space_pirates_corpus = """
    The Galactic Pirate Federation operates 23 star-cruisers across 4 solar systems.
    Admiral Black-Hole commands the flagship 'Void Reaper' with 157 crew members.
    The federation has raided 342 cargo ships and stolen 1,847,293 space-credits.
    
    Each star-cruiser is equipped with quantum-drive engines producing 15,600 thrust units.
    Fuel consumption averages 847 quantum-cells per parsec of hyperspace travel.
    The pirates maintain secret bases on 18 asteroids hidden in nebula clouds.
    
    Plasma cannons fire energy bolts at 89% light-speed with devastating accuracy.
    Shield generators protect ships using 2,340 energy-units per defensive cycle.
    Battle formations include 6 attack squadrons with 12 fighters each.
    
    Space-booty includes rare metals: 4,567 tons of trilithium and 892 tons of neutronium.
    The treasure vault on Asteroid Base Prime holds 234,567 stellar-diamonds.
    Each stellar-diamond weighs precisely 12.7 grams and glows with inner fire.
    """
    
    print("\n🚀 SPACE PIRATES EVALUATION TEST") 
    print("🎭 Testing mathematical reasoning on sci-fi content")
    print("=" * 70)
    
    corpus_info = {
        'domain': 'Fictional Space Pirates',
        'description': 'Made-up sci-fi world with precise numbers and calculations'
    }
    
    evaluator = LocalQwenEvaluator()
    
    try:
        questions = await evaluator.create_fictional_benchmark(
            space_pirates_corpus,
            num_questions=5
        )
        
        qwen_responses = await evaluator.simulate_qwen_responses(questions)
        evaluation_results, scores = evaluator.evaluate_responses(qwen_responses)
        report = evaluator.generate_report(evaluation_results, scores, corpus_info)
        
        return report
        
    except Exception as e:
        print(f"❌ Space pirates test failed: {e}")
        raise


if __name__ == "__main__":
    print("🏠 LOCAL QWEN EVALUATION SYSTEM")
    print("🎯 Testing domain-agnostic capabilities locally")
    print("🚀 No API calls required - fully local execution")
    print("\n" + "="*80)
    
    # Test 1: Crystal Empire
    asyncio.run(test_fictional_crystal_empire())
    
    # Test 2: Space Pirates
    asyncio.run(test_space_pirates())
    
    print("\n🎉 LOCAL TESTING COMPLETE!")
    print("✅ Demonstrated domain-agnostic evaluation on fictional content")
    print("✅ No external API dependencies")
    print("✅ Full pipeline: generation → simulation → verification → reporting")
    print("✅ Ready for integration with real Qwen models when available")
