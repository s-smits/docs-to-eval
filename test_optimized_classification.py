#!/usr/bin/env python3
"""
Test script for optimized classification system
Benchmarks performance and accuracy improvements
"""

import time
import statistics
from docs_to_eval.utils.config import (
    analyze_corpus_content, 
    analyze_corpus_content_original,
    analyze_corpus_content_comprehensive,
    EvaluationType
)

def test_mathematical_corpus():
    """Test mathematical content classification"""
    return """
    Linear algebra is the branch of mathematics concerning linear equations such as:
    2x + 3y = 15
    x - y = 2
    
    To solve this system, we can use substitution. From the second equation: x = y + 2
    Substituting into the first: 2(y + 2) + 3y = 15
    2y + 4 + 3y = 15
    5y = 11
    y = 11/5 = 2.2
    
    Calculate the determinant of matrix A = [[2, 3], [1, -1]]:
    det(A) = (2)(-1) - (3)(1) = -2 - 3 = -5
    
    The eigenvalues satisfy the characteristic equation: det(A - λI) = 0
    ∫ sin(x) dx = -cos(x) + C
    
    Theorem: For any invertible matrix A, det(A⁻¹) = 1/det(A)
    """

def test_code_generation_corpus():
    """Test code generation content classification"""
    return """
    Here's how to implement a binary search algorithm in Python:
    
    ```python
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
                
        return -1
    ```
    
    The algorithm works by repeatedly dividing the search interval in half.
    Time complexity: O(log n), Space complexity: O(1)
    
    class Node:
        def __init__(self, data):
            self.data = data
            self.next = None
    
    from collections import deque
    import heapq
    
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)
    """

def test_factual_qa_corpus():
    """Test factual QA content classification"""
    return """
    Who was the first president of the United States? George Washington served from 1789 to 1797.
    What is the capital of France? Paris has been the capital since 987 AD.
    When was Apple Inc. founded? The company was established in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.
    Where is the Eiffel Tower located? It stands in Paris, France, completed in 1889.
    Why did World War II end in 1945? Germany surrendered on May 8, 1945, and Japan on August 15, 1945.
    How tall is Mount Everest? The mountain reaches 8,848.86 meters above sea level.
    
    The President of France is elected for a five-year term.
    Microsoft was founded by Bill Gates and Paul Allen.
    The Great Wall of China was built over many centuries.
    NASA was established in 1958.
    """

def test_domain_knowledge_corpus():
    """Test domain knowledge content classification"""
    return """
    Machine learning is a methodology that enables computers to learn from data without explicit programming.
    According to research, supervised learning algorithms require labeled training data.
    The framework of deep neural networks consists of multiple layers of interconnected nodes.
    
    Key principles of artificial intelligence include:
    1. Pattern recognition and classification
    2. Decision-making under uncertainty  
    3. Natural language processing capabilities
    4. Computer vision and image analysis
    
    The concept of backpropagation is fundamental to training neural networks.
    Studies indicate that ensemble methods often outperform individual models.
    The theoretical foundation of machine learning draws from statistics, optimization, and probability theory.
    
    Research shows that feature engineering significantly impacts model performance.
    The approach of transfer learning leverages pre-trained models for new tasks.
    According to recent studies, attention mechanisms have revolutionized natural language processing.
    """

def test_mixed_content():
    """Test mixed content that should not trigger early exit"""
    return """
    This document covers various topics in computer science and mathematics.
    We will explore algorithms, data structures, and computational complexity.
    Some examples include sorting algorithms and graph traversal methods.
    The performance characteristics vary depending on input size and data distribution.
    Understanding these concepts is essential for software development and system design.
    """

def benchmark_classification_methods():
    """Benchmark all classification methods"""
    test_cases = [
        ("Mathematical", test_mathematical_corpus()),
        ("Code Generation", test_code_generation_corpus()),
        ("Factual QA", test_factual_qa_corpus()),
        ("Domain Knowledge", test_domain_knowledge_corpus()),
        ("Mixed Content", test_mixed_content())
    ]
    
    methods = [
        ("Optimized (Two-Tier)", analyze_corpus_content),
        ("Comprehensive", analyze_corpus_content_comprehensive),
        ("Original", analyze_corpus_content_original)
    ]
    
    print("🚀 Classification Benchmark Results")
    print("=" * 60)
    
    all_results = {}
    
    for method_name, method_func in methods:
        print(f"\n📊 Testing {method_name} Method:")
        print("-" * 40)
        
        method_results = []
        
        for case_name, corpus in test_cases:
            times = []
            results = []
            
            # Run multiple iterations for accurate timing
            for _ in range(5):
                start_time = time.perf_counter()
                result = method_func(corpus)
                end_time = time.perf_counter()
                
                times.append((end_time - start_time) * 1000)  # Convert to milliseconds
                results.append(result)
            
            avg_time = statistics.mean(times)
            
            # Get most common classification result
            if results and results[0]:
                top_classification = max(results[0].items(), key=lambda x: x[1])[0]
                confidence = results[0][top_classification]
            else:
                top_classification = "NONE"
                confidence = 0.0
            
            method_results.append({
                'case': case_name,
                'time_ms': avg_time,
                'classification': top_classification,
                'confidence': confidence,
                'full_scores': results[0] if results else {}
            })
            
            print(f"  {case_name:15} | {avg_time:6.2f}ms | {top_classification:20} | {confidence:.3f}")
        
        all_results[method_name] = method_results
    
    # Performance comparison
    print(f"\n⚡ Performance Summary:")
    print("-" * 40)
    
    optimized_times = [r['time_ms'] for r in all_results["Optimized (Two-Tier)"]]
    comprehensive_times = [r['time_ms'] for r in all_results["Comprehensive"]]
    original_times = [r['time_ms'] for r in all_results["Original"]]
    
    print(f"Optimized Average:    {statistics.mean(optimized_times):.2f}ms")
    print(f"Comprehensive Average: {statistics.mean(comprehensive_times):.2f}ms") 
    print(f"Original Average:     {statistics.mean(original_times):.2f}ms")
    
    speedup_vs_comprehensive = statistics.mean(comprehensive_times) / statistics.mean(optimized_times)
    speedup_vs_original = statistics.mean(original_times) / statistics.mean(optimized_times)
    
    print(f"\n🏆 Speedup vs Comprehensive: {speedup_vs_comprehensive:.1f}x faster")
    print(f"🏆 Speedup vs Original:     {speedup_vs_original:.1f}x faster")
    
    # Accuracy comparison
    print(f"\n🎯 Classification Accuracy Check:")
    print("-" * 40)
    
    expected_classifications = {
        "Mathematical": EvaluationType.MATHEMATICAL,
        "Code Generation": EvaluationType.CODE_GENERATION,
        "Factual QA": EvaluationType.FACTUAL_QA,
        "Domain Knowledge": EvaluationType.DOMAIN_KNOWLEDGE
    }
    
    for case_name in expected_classifications:
        expected = expected_classifications[case_name]
        
        opt_result = next(r for r in all_results["Optimized (Two-Tier)"] if r['case'] == case_name)
        comp_result = next(r for r in all_results["Comprehensive"] if r['case'] == case_name)
        orig_result = next(r for r in all_results["Original"] if r['case'] == case_name)
        
        opt_correct = opt_result['classification'] == expected
        comp_correct = comp_result['classification'] == expected  
        orig_correct = orig_result['classification'] == expected
        
        print(f"  {case_name:15} | Expected: {expected:20}")
        print(f"    Optimized: {'✅' if opt_correct else '❌'} {opt_result['classification']:20} ({opt_result['confidence']:.3f})")
        print(f"    Comprehensive: {'✅' if comp_correct else '❌'} {comp_result['classification']:20} ({comp_result['confidence']:.3f})")
        print(f"    Original:  {'✅' if orig_correct else '❌'} {orig_result['classification']:20} ({orig_result['confidence']:.3f})")
        print()

def test_early_exit_effectiveness():
    """Test how often early exit is triggered"""
    test_cases = [
        ("Strong Math Signal", "Solve for x: 2x + 5 = 15. Calculate the answer."),
        ("Strong Code Signal", "```python\ndef hello():\n    print('world')\n```"),
        ("Weak Mixed Signal", "This text discusses various topics without strong indicators."),
        ("LaTeX Math", "$\\frac{x^2 + 1}{x - 1} = \\sqrt{4}$ solve this equation"),
        ("Clear Factual", "Who is the president of the United States in 2024?"),
    ]
    
    print(f"\n🔍 Early Exit Effectiveness Test:")
    print("-" * 50)
    
    for case_name, text in test_cases:
        # Test if early exit triggers by checking if comprehensive analysis needed
        start_time = time.perf_counter()
        result_optimized = analyze_corpus_content(text)
        opt_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter() 
        result_comprehensive = analyze_corpus_content_comprehensive(text)
        comp_time = time.perf_counter() - start_time
        
        # If optimized is significantly faster, early exit likely triggered
        speedup = comp_time / opt_time if opt_time > 0 else 1
        likely_early_exit = speedup > 2  # Heuristic
        
        print(f"  {case_name:20} | {opt_time*1000:5.2f}ms vs {comp_time*1000:5.2f}ms | {speedup:.1f}x | {'🚀 Early Exit' if likely_early_exit else '🐌 Full Analysis'}")

if __name__ == "__main__":
    print("🧪 Testing Optimized Classification System")
    print("="*60)
    
    benchmark_classification_methods()
    test_early_exit_effectiveness()
    
    print("\n✅ Benchmark Complete!")