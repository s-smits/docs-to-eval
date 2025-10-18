#!/usr/bin/env python3
"""
Demo of Improved Mixed Verification System

This demonstrates the key improvements made to fix the evaluation issues:
1. Intelligent question-type detection
2. Enhanced fuzzy matching for numeric answers
3. Mixed verification methods with weighted scoring
"""

from docs_to_eval.core.mixed_verification import (
    FuzzyMatcher, 
    MixedVerificationOrchestrator
)


def demo_numeric_matching():
    """Show improved numeric answer matching"""
    print("=" * 60)
    print("NUMERIC ANSWER MATCHING IMPROVEMENTS")
    print("=" * 60)
    
    matcher = FuzzyMatcher()
    
    test_cases = [
        ("25", "In Catholicism, the canonical minimum age is twenty-five."),
        ("20", "The test set percentage is twenty percent"),
        ("5", "There are five items listed"),
        ("42", "The random seed is forty-two"),
    ]
    
    for prediction, ground_truth in test_cases:
        is_match, score = matcher.fuzzy_match(prediction, ground_truth, threshold=0.7)
        print(f"\nPrediction: '{prediction}'")
        print(f"Ground Truth: '{ground_truth[:50]}...'")
        print(f"Score: {score:.3f} | Match: {'✓' if is_match else '✗'}")


def demo_factual_matching():
    """Show improved factual answer matching"""
    print("\n" + "=" * 60)
    print("FACTUAL ANSWER MATCHING IMPROVEMENTS")
    print("=" * 60)
    
    matcher = FuzzyMatcher()
    
    test_cases = [
        ("lautn", "The Etruscan name of the family was lautn."),
        ("Liber Linteus", "Liber Linteus (Liber Linteus Zagrabiensis)"),
        ("Zagreb Mummy Linen", "The common name for the Liber Linteus Zagrabiensis is the Zagreb Mummy Linen"),
        ("ritual calendar", "The text is most likely a ritual calendar"),
    ]
    
    for prediction, ground_truth in test_cases:
        is_match, score = matcher.fuzzy_match(prediction, ground_truth, threshold=0.7)
        print(f"\nPrediction: '{prediction}'")
        print(f"Ground Truth: '{ground_truth[:50]}...'")
        print(f"Score: {score:.3f} | Match: {'✓' if is_match else '✗'}")


def demo_mixed_verification():
    """Show the complete mixed verification system"""
    print("\n" + "=" * 60)
    print("MIXED VERIFICATION SYSTEM")
    print("=" * 60)
    
    orchestrator = MixedVerificationOrchestrator()
    
    # Sample questions from the user's original examples
    test_cases = [
        {
            "question": "What is the minimum canonical age for men to become priests in Catholicism?",
            "ground_truth": "In Catholicism, the canonical minimum age is twenty-five.",
            "llm_response": "25"
        },
        {
            "question": "What is the Etruscan name for the family?",
            "ground_truth": "The Etruscan name of the family was lautn.",
            "llm_response": "Larthia"  # Wrong answer
        },
        {
            "question": "What is the common name for the Liber Linteus Zagrabiensis?",
            "ground_truth": "Linen Book of Zagreb",
            "llm_response": "The Zagreb Mummy Linen"  # Variation of correct answer
        },
        {
            "question": "What was the primary function of the annual assemblies?",
            "ground_truth": "The annual assemblies served as both political conferences and religious festivals.",
            "llm_response": "The primary function was religious and political: Religious observances and sacrifices to honor Voltumna, and political discussions regarding war and peace."
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'-'*50}")
        print(f"Test Case {i}")
        print(f"{'-'*50}")
        print(f"Question: {test['question'][:60]}...")
        print(f"Expected: {test['ground_truth'][:40]}...")
        print(f"LLM Answer: {test['llm_response'][:40]}...")
        
        result = orchestrator.verify(
            prediction=test['llm_response'],
            ground_truth=test['ground_truth'],
            question=test['question'],
            use_mixed=True
        )
        
        print("\nVerification Results:")
        print(f"  Final Score: {result.score:.3f}")
        print(f"  Method: {result.method}")
        
        if result.details:
            question_type = result.details.get('question_type', 'unknown')
            print(f"  Question Type: {question_type}")
            
            methods_used = result.details.get('methods_used', {})
            if methods_used:
                print("  Methods Used:")
                for method, score in methods_used.items():
                    print(f"    {method}: {score:.3f}")
        
        # Compare to old exact match (would be 0.0 for all)
        old_score = 1.0 if test['llm_response'].lower() == test['ground_truth'].lower() else 0.0
        print(f"  Old Exact Match: {old_score:.3f}")
        print(f"  Improvement: +{result.score - old_score:.3f}")


def main():
    """Run the demonstration"""
    print("IMPROVED MIXED VERIFICATION SYSTEM DEMO")
    print("="*60)
    print("\nThis system fixes the issues where all questions scored 0.00")
    print("by using intelligent verification method selection:\n")
    
    demo_numeric_matching()
    demo_factual_matching() 
    demo_mixed_verification()
    
    print("\n" + "=" * 60)
    print("SUMMARY OF IMPROVEMENTS")
    print("=" * 60)
    print("✓ Numeric answer matching (25 ≡ twenty-five)")
    print("✓ Fuzzy matching for variations and formatting differences")  
    print("✓ Semantic similarity for conceptual questions")
    print("✓ Question-type aware verification method selection")
    print("✓ Weighted scoring combining multiple verification methods")
    print("✓ Much better accuracy than exact_match for all question types")


if __name__ == "__main__":
    main()