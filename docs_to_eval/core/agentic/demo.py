"""
Demonstration script for the agentic benchmark factory
Shows how to use the new agentic system with various evaluation types
"""

import asyncio
import json
import time
from typing import Dict, Any

from .orchestrator import AgenticBenchmarkOrchestrator
from .generator import AgenticBenchmarkGenerator
from .models import PipelineConfig, DifficultyLevel
from .validation import ComprehensiveValidator
from ..evaluation import EvaluationType
from ..benchmarks import BenchmarkGeneratorFactory
from ...llm.mock_interface import MockLLMInterface


class AgenticBenchmarkDemo:
    """Demonstration class for agentic benchmark generation"""
    
    def __init__(self):
        self.setup_llm_pool()
        self.validator = ComprehensiveValidator(min_quality_score=0.6)
    
    def setup_llm_pool(self):
        """Set up mock LLM pool for demonstration"""
        # In production, these would be different LLM instances optimized for different tasks
        mock_llm = MockLLMInterface(model_name="MockLLM-Demo", temperature=0.7)
        
        self.llm_pool = {
            'retriever': mock_llm,    # For concept mining
            'creator': mock_llm,      # For question writing
            'adversary': mock_llm,    # For adversarial enhancement
            'refiner': mock_llm       # For refinement and formatting
        }
    
    async def demonstrate_agentic_pipeline(self, corpus_text: str, eval_type: EvaluationType, 
                                         num_questions: int = 10) -> Dict[str, Any]:
        """
        Demonstrate the full agentic pipeline for benchmark generation
        
        Args:
            corpus_text: Source text for benchmark generation
            eval_type: Type of evaluation to generate
            num_questions: Number of questions to generate
            
        Returns:
            Complete demonstration report
        """
        
        print(f"\n🚀 Starting Agentic Benchmark Generation Demo")
        print(f"   Eval Type: {eval_type}")
        print(f"   Target Questions: {num_questions}")
        print(f"   Corpus Length: {len(corpus_text)} characters")
        print("="*60)
        
        start_time = time.time()
        
        # Step 1: Create agentic configuration
        config = PipelineConfig(
            difficulty=DifficultyLevel.HARD,
            num_questions=num_questions,
            min_validation_score=0.6,
            parallel_batch_size=3,
            max_retry_cycles=2
        )
        
        print("📋 Configuration:")
        print(f"   Difficulty: {config.difficulty}")
        print(f"   Validation threshold: {config.min_validation_score}")
        print(f"   Parallel batch size: {config.parallel_batch_size}")
        
        # Step 2: Initialize orchestrator
        orchestrator = AgenticBenchmarkOrchestrator(self.llm_pool, config)
        
        # Validate pipeline health
        health_status = await orchestrator.validate_pipeline_health()
        print(f"\n🔍 Pipeline Health: {'✅ Healthy' if health_status['healthy'] else '❌ Issues Found'}")
        if health_status['issues']:
            print(f"   Issues: {health_status['issues']}")
        if health_status['warnings']:
            print(f"   Warnings: {health_status['warnings']}")
        
        # Step 3: Generate benchmark items
        print(f"\n⚙️  Generating {num_questions} benchmark items...")
        enhanced_items = await orchestrator.generate(
            corpus_text=corpus_text,
            eval_type=eval_type,
            num_questions=num_questions,
            difficulty=config.difficulty
        )
        
        generation_time = time.time() - start_time
        print(f"✅ Generated {len(enhanced_items)} items in {generation_time:.2f}s")
        
        # Step 4: Comprehensive validation
        print(f"\n🔬 Running comprehensive validation...")
        validation_start = time.time()
        
        validation_report = await self.validator.validate_benchmark_batch(enhanced_items)
        validation_time = time.time() - validation_start
        
        print(f"✅ Validation complete in {validation_time:.2f}s")
        print(f"   Overall pass rate: {validation_report['overall_pass_rate']:.3f}")
        print(f"   Deterministic consistency: {validation_report['deterministic_validation']['consistency_rate']:.3f}")
        print(f"   Average quality score: {validation_report['quality_assessment']['avg_quality']:.3f}")
        
        # Step 5: Filter high-quality items
        filtered_items, filter_report = await self.validator.validate_and_filter(
            enhanced_items, strict_mode=True
        )
        
        print(f"\n🎯 Quality filtering:")
        print(f"   Retained {len(filtered_items)}/{len(enhanced_items)} items")
        print(f"   Retention rate: {filter_report['filtering']['retention_rate']:.3f}")
        
        # Step 6: Generate demonstration report
        demo_report = self._create_demo_report(
            orchestrator, enhanced_items, filtered_items, 
            validation_report, generation_time, validation_time
        )
        
        # Display sample questions
        self._display_sample_questions(filtered_items[:3])
        
        # Show recommendations
        if validation_report['actionable_recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(validation_report['actionable_recommendations'][:3], 1):
                print(f"   {i}. {rec}")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Demo complete in {total_time:.2f}s total")
        
        return demo_report
    
    def demonstrate_factory_integration(self, corpus_text: str) -> Dict[str, Any]:
        """Demonstrate integration with existing benchmark factory"""
        
        print(f"\n🏭 Factory Integration Demo")
        print("="*40)
        
        results = {}
        
        # Test different evaluation types
        test_types = [
            EvaluationType.DOMAIN_KNOWLEDGE,
            EvaluationType.FACTUAL_QA,
            EvaluationType.MATHEMATICAL
        ]
        
        for eval_type in test_types:
            print(f"\n📊 Testing {eval_type}:")
            
            # Check if agentic generation is supported
            supports_agentic = BenchmarkGeneratorFactory.supports_agentic(eval_type)
            print(f"   Agentic support: {'✅' if supports_agentic else '❌'}")
            
            if supports_agentic:
                # Create agentic generator
                generator = BenchmarkGeneratorFactory.create_generator(
                    eval_type, 
                    use_agentic=True, 
                    llm_pool=self.llm_pool
                )
                
                # Generate small batch
                items = generator.generate_benchmark(corpus_text, num_questions=3)
                results[eval_type] = {
                    'generator_type': 'agentic',
                    'items_generated': len(items),
                    'sample_question': items[0]['question'] if items else 'No items generated'
                }
                
                print(f"   Generated {len(items)} items")
                if items:
                    print(f"   Sample: {items[0]['question'][:60]}...")
            else:
                # Fallback to standard generator
                generator = BenchmarkGeneratorFactory.create_generator(eval_type)
                items = generator.generate_benchmark(corpus_text, num_questions=3)
                results[eval_type] = {
                    'generator_type': 'standard',
                    'items_generated': len(items),
                    'sample_question': items[0]['question'] if items else 'No items generated'
                }
                print(f"   Used standard generator: {len(items)} items")
        
        return results
    
    def _create_demo_report(self, orchestrator, original_items, filtered_items, 
                           validation_report, generation_time, validation_time) -> Dict[str, Any]:
        """Create comprehensive demonstration report"""
        
        pipeline_metrics = orchestrator.get_pipeline_metrics()
        
        return {
            'demo_summary': {
                'total_time': generation_time + validation_time,
                'generation_time': generation_time,
                'validation_time': validation_time,
                'items_generated': len(original_items),
                'items_retained': len(filtered_items),
                'retention_rate': len(filtered_items) / len(original_items) if original_items else 0
            },
            'pipeline_performance': pipeline_metrics,
            'validation_results': validation_report,
            'agent_stats': pipeline_metrics.get('agent_stats', {}),
            'quality_metrics': {
                'avg_quality': validation_report['quality_assessment']['avg_quality'],
                'consistency_rate': validation_report['deterministic_validation']['consistency_rate'],
                'overall_pass_rate': validation_report['overall_pass_rate']
            },
            'sample_items': [
                {
                    'question': item.question,
                    'answer': item.answer[:100] + '...' if len(item.answer) > 100 else item.answer,
                    'difficulty': item.metadata.difficulty,
                    'deterministic': item.metadata.deterministic,
                    'quality_score': item.metadata.validation_score
                }
                for item in filtered_items[:3]
            ]
        }
    
    def _display_sample_questions(self, items):
        """Display sample questions in a nicely formatted way"""
        
        if not items:
            return
        
        print(f"\n📝 Sample Generated Questions:")
        print("-" * 50)
        
        for i, item in enumerate(items, 1):
            print(f"\n{i}. Question: {item.question}")
            print(f"   Answer: {item.answer[:80]}{'...' if len(item.answer) > 80 else ''}")
            print(f"   Difficulty: {item.metadata.difficulty}")
            print(f"   Deterministic: {item.metadata.deterministic}")
            if hasattr(item, 'expected_answer_type'):
                print(f"   Answer Type: {item.expected_answer_type}")
            if item.metadata.validation_score:
                print(f"   Quality Score: {item.metadata.validation_score:.3f}")
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration of all features"""
        
        # Sample corpus for demonstration
        sample_corpus = """
        Machine learning is a method of data analysis that automates analytical model building. 
        It is a branch of artificial intelligence (AI) based on the idea that systems can learn 
        from data, identify patterns and make decisions with minimal human intervention.
        
        The core principle of machine learning involves algorithms that can receive input data 
        and use statistical analysis to predict an output value within an acceptable range. 
        As new data is fed to these systems, they learn and optimize their operations to improve 
        performance, developing 'intelligence' over time.
        
        There are three main types of machine learning: supervised learning, unsupervised learning, 
        and reinforcement learning. Supervised learning uses labeled training data to learn a 
        mapping function from input variables to an output variable. Unsupervised learning finds 
        hidden patterns in data without labeled examples. Reinforcement learning uses rewards 
        and penalties to learn optimal actions in an environment.
        
        Applications of machine learning include image recognition, natural language processing, 
        recommendation systems, fraud detection, and autonomous vehicles. The field continues 
        to evolve rapidly with advances in deep learning, neural networks, and computational power.
        """
        
        print("🎯 COMPREHENSIVE AGENTIC BENCHMARK GENERATION DEMO")
        print("="*60)
        
        # Test different evaluation types
        test_scenarios = [
            (EvaluationType.DOMAIN_KNOWLEDGE, 8),
            (EvaluationType.FACTUAL_QA, 6),
            (EvaluationType.MATHEMATICAL, 4)
        ]
        
        all_results = {}
        
        for eval_type, num_questions in test_scenarios:
            print(f"\n🔄 Scenario: {eval_type} ({num_questions} questions)")
            
            try:
                scenario_result = await self.demonstrate_agentic_pipeline(
                    sample_corpus, eval_type, num_questions
                )
                all_results[eval_type] = scenario_result
            except Exception as e:
                print(f"❌ Scenario failed: {str(e)}")
                all_results[eval_type] = {'error': str(e)}
        
        # Factory integration test
        factory_results = self.demonstrate_factory_integration(sample_corpus)
        all_results['factory_integration'] = factory_results
        
        # Final summary
        print(f"\n🎊 DEMO SUMMARY")
        print("="*30)
        
        total_items = sum(
            result.get('demo_summary', {}).get('items_generated', 0) 
            for result in all_results.values() 
            if isinstance(result, dict) and 'demo_summary' in result
        )
        
        avg_quality = sum(
            result.get('quality_metrics', {}).get('avg_quality', 0) 
            for result in all_results.values() 
            if isinstance(result, dict) and 'quality_metrics' in result
        ) / len([r for r in all_results.values() if isinstance(r, dict) and 'quality_metrics' in r])
        
        print(f"Total items generated: {total_items}")
        print(f"Average quality score: {avg_quality:.3f}")
        print(f"Scenarios completed: {len([r for r in all_results.values() if 'error' not in r])}/{len(test_scenarios)}")
        
        return all_results


async def main():
    """Main demonstration function"""
    demo = AgenticBenchmarkDemo()
    results = await demo.run_comprehensive_demo()
    
    # Optionally save results to file
    with open('agentic_demo_results.json', 'w') as f:
        # Convert non-serializable objects to strings
        serializable_results = json.loads(json.dumps(results, default=str))
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to agentic_demo_results.json")


if __name__ == "__main__":
    asyncio.run(main())