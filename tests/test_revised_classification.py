#!/usr/bin/env python3
"""
Test script for REVISED optimized classification system
Tests simplicity, performance, and new helper functions
"""
import pytest
pytestmark = pytest.mark.skip(reason="Benchmark/demo; excluded from CI test suite")

import time
import statistics
from docs_to_eval.utils.config import (
    analyze_corpus_content,
    classify_corpus_simple,
    get_classification_confidence,
    explain_classification,
    EvaluationType
)


def benchmark_classification(corpus_text: str, name: str = "Test") -> dict:
    """Benchmark classification performance"""
    times = []
    
    # Warm-up run
    _ = analyze_corpus_content(corpus_text)
    
    # Benchmark runs
    for _ in range(100):
        start = time.perf_counter()
        result = analyze_corpus_content(corpus_text)
        times.append((time.perf_counter() - start) * 1000)
    
    return {
        'name': name,
        'avg_ms': statistics.mean(times),
        'median_ms': statistics.median(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'result': result
    }


def test_new_features():
    """Test new helper functions"""
    print("\n🧪 Testing New Helper Functions")
    print("=" * 60)
    
    test_cases = {
        "Mathematical": "Solve for x: 2x + 5 = 15. Calculate the derivative of x^2.",
        "Code": "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "Factual": "Who was the first president of the United States? Washington was born in 1732.",
        "Domain": "The theory of relativity explains the relationship between space and time."
    }
    
    for name, corpus in test_cases.items():
        print(f"\n📝 {name} Corpus:")
        
        # Test simple classification
        eval_type = classify_corpus_simple(corpus)
        print(f"  Simple Classification: {eval_type}")
        
        # Test classification with confidence
        eval_type, confidence = get_classification_confidence(corpus)
        print(f"  With Confidence: {eval_type} ({confidence:.2%})")
        
        # Test explanation (non-detailed)
        explanation = explain_classification(corpus, detailed=False)
        print(f"  Final: {explanation['final_classification']}")
        print("  Scores by type:")
        for t in [EvaluationType.MATHEMATICAL, EvaluationType.CODE_GENERATION, 
                  EvaluationType.FACTUAL_QA, EvaluationType.DOMAIN_KNOWLEDGE]:
            if t.value in explanation:
                info = explanation[t.value]
                print(f"    - {t.value}: score={info['score']:.1f}, confidence={info['confidence']}")


def test_detailed_explanation():
    """Test detailed explanation feature"""
    print("\n🔍 Testing Detailed Explanation")
    print("=" * 60)
    
    corpus = """
    To solve the equation 2x + 5 = 15, first subtract 5 from both sides.
    This gives us 2x = 10. Then divide by 2 to get x = 5.
    We can verify: 2(5) + 5 = 10 + 5 = 15. ✓
    
    The derivative of x^2 is 2x, and the integral is x^3/3 + C.
    """
    
    explanation = explain_classification(corpus, detailed=True)
    
    print(f"Classification: {explanation['final_classification']}")
    print("\nDetailed Pattern Matches:")
    
    for eval_type in [EvaluationType.MATHEMATICAL, EvaluationType.CODE_GENERATION]:
        if eval_type.value in explanation:
            info = explanation[eval_type.value]
            print(f"\n{eval_type.value}:")
            print(f"  Total Score: {info['score']:.1f}")
            print(f"  Confidence: {info['confidence']}")
            
            if info['matches']:
                print("  Matched Patterns:")
                for match in info['matches'][:3]:  # Show top 3 matches
                    print(f"    - {match['description']} (weight: {match['weight']})")
                    if match.get('examples'):
                        print(f"      Examples: {match['examples'][:2]}")


def performance_comparison():
    """Compare performance of revised system"""
    print("\n🚀 Performance Benchmark (Revised System)")
    print("=" * 60)
    
    test_corpora = {
        "Mathematical (strong signals)": "Solve: 2x + 5 = 15. Calculate ∫x²dx. Find derivative of sin(x)." * 10,
        "Code (strong signals)": "```python\ndef hello():\n    return 'world'\n```\nclass MyClass:\n    pass" * 10,
        "Mixed content": "The history of mathematics shows that x² + y² = r² was known to ancient Greeks." * 10,
        "Long document": "This is a general document about various topics. " * 500,
    }
    
    results = []
    for name, corpus in test_corpora.items():
        result = benchmark_classification(corpus, name)
        results.append(result)
        
        # Get classification result
        eval_type = classify_corpus_simple(corpus)
        confidence = max(result['result'].values()) if result['result'] else 0
        
        print(f"\n{name}:")
        print(f"  Classification: {eval_type}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Avg time: {result['avg_ms']:.3f}ms")
        print(f"  Median time: {result['median_ms']:.3f}ms")
    
    # Overall stats
    avg_time = statistics.mean(r['avg_ms'] for r in results)
    print(f"\n📊 Overall average: {avg_time:.3f}ms")
    print(f"   Sub-millisecond: {sum(1 for r in results if r['avg_ms'] < 1.0)}/{len(results)} cases")


def test_edge_cases():
    """Test edge cases and robustness"""
    print("\n🛡️ Testing Edge Cases")
    print("=" * 60)
    
    edge_cases = [
        ("Empty string", ""),
        ("Whitespace only", "   \n\t  "),
        ("Single word", "Hello"),
        ("Numbers only", "123 456 789"),
        ("Special chars", "!@#$%^&*()"),
        ("Unicode math", "∫∑∏√±∞"),
        ("Mixed languages", "Hello 世界 مرحبا"),
    ]
    
    for name, corpus in edge_cases:
        try:
            result = classify_corpus_simple(corpus)
            confidence = get_classification_confidence(corpus)[1]
            print(f"✅ {name}: {result} (confidence: {confidence:.2%})")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")


def main():
    print("🧪 Testing REVISED Optimized Classification System")
    print("=" * 60)
    
    # Run all tests
    test_new_features()
    test_detailed_explanation()
    performance_comparison()
    test_edge_cases()
    
    print("\n✅ All tests completed!")
    print("\n📈 Summary:")
    print("  - Simpler implementation with clear tier structure")
    print("  - New helper functions for common use cases")
    print("  - Explainable classifications for debugging")
    print("  - Maintains high performance with early exit")
    print("  - Robust edge case handling")


if __name__ == "__main__":
    main()
