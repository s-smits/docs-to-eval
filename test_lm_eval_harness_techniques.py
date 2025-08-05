#!/usr/bin/env python3
"""
Comprehensive test demonstrating lm-evaluation-harness techniques implemented in our scoring system
Tests: semantic embeddings, length normalization, bootstrap CIs, ensemble verification, filter pipelines
"""

import sys
sys.path.append('/Users/air/Developer/docs-to-eval')

from docs_to_eval.core.verification import NonDeterministicVerifier
from docs_to_eval.utils.advanced_evaluation import AdvancedEvaluationFramework, BootstrapStatistics
from docs_to_eval.utils.text_processing import FilterPipeline, AnswerExtractor
from docs_to_eval.utils.similarity import semantic_similarity_real, calculate_multi_similarity

def test_advanced_semantic_similarity():
    """Test the new BERTScore-style semantic similarity with length normalization"""
    print("=== ADVANCED SEMANTIC SIMILARITY TEST ===")
    
    # Test case from your original message
    expected = """Archaeological findings in tombs show women's importance through frescoes, sarcophagi, urns, and possessions. They are depicted in social activities. Spindle whorls and spring scales indicate manual work by women."""
    
    response = """Archaeological findings in tombs (like frescoes, sarcophagi, urns, and possessions) show women's importance in society. They highlight depictions of women in social activities like banquets, dances, and reclining alongside men. Jewelry, clothing, and personal items indicate wealth and status."""
    
    # Test real semantic similarity with length normalization
    semantic_score = semantic_similarity_real(response, expected)
    print(f"BERTScore-style semantic similarity: {semantic_score:.3f}")
    
    # Show individual method scores
    multi_scores = calculate_multi_similarity(response, expected)
    print("\nIndividual similarity methods:")
    for method, score in multi_scores.items():
        print(f"  {method}: {score:.3f}")
    
    return semantic_score

def test_ensemble_verification():
    """Test ensemble verification with confidence intervals"""
    print("\n=== ENSEMBLE VERIFICATION TEST ===")
    
    framework = AdvancedEvaluationFramework()
    
    test_cases = [
        {
            "name": "Your example (archaeological)",
            "expected": "Archaeological findings in tombs show women's importance through frescoes, sarcophagi, urns, and possessions. They are depicted in social activities. Spindle whorls and spring scales indicate manual work by women.",
            "response": "Archaeological findings in tombs (like frescoes, sarcophagi, urns, and possessions) show women's importance in society. They highlight depictions of women in social activities like banquets, dances, and reclining alongside men. Jewelry, clothing, and personal items indicate wealth and status."
        },
        {
            "name": "High factual overlap",
            "expected": "The Etruscans were an ancient civilization that flourished in central Italy from 8th to 3rd century BCE.",
            "response": "Etruscans were an ancient people who lived in central Italy between the 8th and 3rd centuries BCE and developed a sophisticated civilization."
        },
        {
            "name": "Different lengths test",  
            "expected": "Short answer.",
            "response": "This is a much longer response that covers the same basic point but with significantly more detail and explanation about the topic."
        }
    ]
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        result = framework.evaluate_response_advanced(case['response'], case['expected'])
        
        print(f"Final Score: {result['similarity_score']:.3f}")
        print(f"Raw Ensemble: {result['raw_ensemble_score']:.3f}")
        print(f"Exact Match: {result['exact_match']}")
        print(f"Method Agreement: {result['method_agreement']:.3f}")
        print(f"95% CI: [{result['confidence_intervals']['95%'][0]:.3f}, {result['confidence_intervals']['95%'][1]:.3f}]")
        print(f"Length Penalty: {result['evaluation_metadata']['length_penalty_applied']}")
        print(f"Boost Applied: {result['evaluation_metadata']['boost_applied']}")
    
    # Show batch statistics
    batch_stats = framework.get_batch_statistics()
    print(f"\n--- BATCH STATISTICS ---")
    print(f"Mean Score: {batch_stats['mean_similarity_score']:.3f}")
    print(f"Std Dev: {batch_stats['std_similarity_score']:.3f}")
    print(f"Batch 95% CI: [{batch_stats['confidence_interval_95'][0]:.3f}, {batch_stats['confidence_interval_95'][1]:.3f}]")
    
    return framework

def test_filter_pipelines():
    """Test lm-eval-harness style filter pipelines"""
    print("\n=== FILTER PIPELINE TEST ===")
    
    test_cases = [
        {
            "type": "multiple_choice", 
            "text": "Based on the passage, the answer is clearly B) Archaeological evidence.",
            "expected": "B"
        },
        {
            "type": "numerical",
            "text": "After calculating step by step, the final answer is 42.5",
            "expected": "42.5"
        },
        {
            "type": "short_answer",
            "text": "The answer is that Etruscan women had significant social status and influence in their society.",
            "expected": "etruscan women had significant social status"
        },
        {
            "type": "yes_no",
            "text": "Yes, the evidence clearly supports this conclusion.",
            "expected": "yes"
        }
    ]
    
    pipelines = {
        "multiple_choice": FilterPipeline.create_multiple_choice_pipeline(),
        "numerical": FilterPipeline.create_numerical_pipeline(), 
        "short_answer": FilterPipeline.create_short_answer_pipeline(),
        "yes_no": lambda text: AnswerExtractor.extract_yes_no(text)
    }
    
    for case in test_cases:
        pipeline_type = case["type"]
        text = case["text"]
        expected = case["expected"]
        
        if pipeline_type == "yes_no":
            processed = pipelines[pipeline_type](text)
        else:
            processed = pipelines[pipeline_type].process(text)
        
        print(f"\n{pipeline_type.title()}:")
        print(f"  Input: {text}")
        print(f"  Extracted: '{processed}'")
        print(f"  Expected: '{expected}'")
        print(f"  Match: {processed.lower() == expected.lower()}")

def test_bootstrap_confidence_intervals():
    """Test bootstrap statistical analysis"""
    print("\n=== BOOTSTRAP CONFIDENCE INTERVALS TEST ===")
    
    # Simulate evaluation scores from different scenarios
    test_scenarios = {
        "High Performance Model": [0.85, 0.87, 0.83, 0.86, 0.88, 0.84, 0.87, 0.85, 0.89, 0.86],
        "Medium Performance Model": [0.65, 0.68, 0.62, 0.67, 0.64, 0.66, 0.63, 0.69, 0.65, 0.67],
        "Inconsistent Model": [0.95, 0.45, 0.85, 0.35, 0.90, 0.40, 0.88, 0.42, 0.92, 0.38]
    }
    
    for model_name, scores in test_scenarios.items():
        print(f"\n--- {model_name} ---")
        print(f"Mean Score: {sum(scores)/len(scores):.3f}")
        
        # Bootstrap confidence intervals
        ci_95 = BootstrapStatistics.bootstrap_confidence_interval(scores, 0.95)
        ci_99 = BootstrapStatistics.bootstrap_confidence_interval(scores, 0.99)
        
        print(f"95% Confidence Interval: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
        print(f"99% Confidence Interval: [{ci_99[0]:.3f}, {ci_99[1]:.3f}]")
        
        # Statistical significance vs medium model
        if model_name != "Medium Performance Model":
            significance = BootstrapStatistics.calculate_statistical_significance(
                scores, test_scenarios["Medium Performance Model"]
            )
            print(f"vs Medium Model - p-value: {significance['p_value']:.3f}, effect size: {significance['effect_size']:.3f}")

def test_original_vs_new_scoring():
    """Compare original 0.28 score vs new lenient scoring"""
    print("\n=== ORIGINAL VS NEW SCORING COMPARISON ===")
    
    expected = """Archaeological findings in tombs show women's importance through frescoes, sarcophagi, urns, and possessions. They are depicted in social activities. Spindle whorls and spring scales indicate manual work by women."""
    
    response = """Archaeological findings in tombs (like frescoes, sarcophagi, urns, and possessions) show women's importance in society. They highlight depictions of women in social activities like banquets, dances, and reclining alongside men. Jewelry, clothing, and personal items indicate wealth and status."""
    
    # Test with new advanced verification
    advanced_result = NonDeterministicVerifier.semantic_similarity_advanced(response, expected)
    
    print("ADVANCED VERIFICATION RESULTS:")
    print(f"Final Score: {advanced_result.score:.3f}")
    print(f"Method: {advanced_result.method}")
    print(f"Individual Methods:")
    for method, score in advanced_result.metrics.items():
        if method not in ['semantic_similarity', 'raw_ensemble_score', 'exact_match', 'method_agreement']:
            print(f"  {method}: {score:.3f}")
    
    print(f"\nConfidence Intervals: {advanced_result.details['confidence_intervals']}")
    print(f"Method Agreement: {advanced_result.metrics['method_agreement']:.3f}")
    print(f"Evaluation Metadata: {advanced_result.details['evaluation_metadata']}")
    
    print(f"\n🎯 IMPROVEMENT: From ~0.28 to {advanced_result.score:.3f}")
    print(f"   That's a {((advanced_result.score - 0.28) / 0.28 * 100):.0f}% improvement!")
    
    return advanced_result.score

def main():
    """Run all lm-evaluation-harness technique tests"""
    print("Testing LM-Evaluation-Harness Techniques Implementation")
    print("=" * 60)
    
    # Run all tests
    semantic_score = test_advanced_semantic_similarity()
    framework = test_ensemble_verification()
    test_filter_pipelines()
    test_bootstrap_confidence_intervals()
    final_score = test_original_vs_new_scoring()
    
    print("\n" + "=" * 60)
    print("SUMMARY OF IMPROVEMENTS")
    print("=" * 60)
    
    techniques_implemented = [
        "✅ BERTScore-style semantic similarity with sentence transformers",
        "✅ Length normalization (lm-eval-harness acc_norm style)",
        "✅ Bootstrap confidence intervals for statistical rigor",
        "✅ Ensemble verification with multiple similarity methods",
        "✅ Dice coefficient instead of Jaccard for more lenient scoring",
        "✅ Filter pipelines for answer extraction and normalization",
        "✅ Method agreement and consistency metrics",
        "✅ Advanced evaluation framework with batch statistics"
    ]
    
    for technique in techniques_implemented:
        print(technique)
    
    print(f"\n🚀 FINAL RESULT: Your problematic 0.28 score is now {final_score:.3f}")
    print(f"   This properly reflects the substantial factual overlap!")
    
    if final_score >= 0.7:
        print("✅ EXCELLENT: Scoring now properly captures semantic similarity")
    elif final_score >= 0.6:
        print("✅ GOOD: Significant improvement in scoring leniency")
    else:
        print("⚠️ PARTIAL: Some improvement but could be more lenient")

if __name__ == "__main__":
    main()