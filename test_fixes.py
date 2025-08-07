#!/usr/bin/env python3
"""
Quick test script to validate quality filtering fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from docs_to_eval.core.agentic import AgenticQuestionGenerator, QuestionItem
from docs_to_eval.core.evaluation import EvaluationType


def test_basic_functionality():
    """Test basic functionality of the improved system"""
    
    print("🧪 Testing Quality Filtering Fixes...")
    generator = AgenticQuestionGenerator()
    
    # Test 1: High-quality corpus
    quality_corpus = """
    Quantum computing represents a revolutionary paradigm in computational science, leveraging quantum mechanical phenomena
    such as superposition and entanglement to process information. Unlike classical computers that use bits representing
    either 0 or 1, quantum computers utilize quantum bits (qubits) that can exist in superposition states, allowing
    simultaneous representation of both 0 and 1. This fundamental difference enables quantum algorithms to solve certain
    computational problems exponentially faster than classical algorithms. Key applications include cryptography,
    optimization problems, and molecular simulation.
    """
    
    print("\n📊 Test 1: Quality corpus with exact count maintenance")
    for count in [5, 10, 20]:
        try:
            result = generator.generate_comprehensive_benchmark(
                quality_corpus, 
                num_questions=count, 
                eval_type=EvaluationType.DOMAIN_KNOWLEDGE
            )
            
            generated = len(result['questions'])
            avg_quality = sum(q['quality_score'] for q in result['questions']) / generated
            
            print(f"  ✅ Requested: {count}, Generated: {generated}, Avg Quality: {avg_quality:.3f}")
            
            # Verify exact count
            if generated != count:
                print(f"  ❌ COUNT MISMATCH: Expected {count}, got {generated}")
            else:
                print(f"  ✅ EXACT COUNT MAINTAINED")
                
        except Exception as e:
            print(f"  ❌ Error with count {count}: {e}")
    
    # Test 2: Low-quality corpus
    poor_corpus = """
    This is simple. Things are good. Yes. No. Maybe. 
    Some stuff happens. It works. The end.
    """
    
    print("\n📊 Test 2: Poor corpus with regeneration")
    try:
        result = generator.generate_comprehensive_benchmark(
            poor_corpus, 
            num_questions=10, 
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        generated = len(result['questions'])
        print(f"  Generated {generated} questions from poor corpus")
        
        if generated >= 5:  # Should try hard to generate questions
            print(f"  ✅ REGENERATION WORKING: Generated {generated}/10 questions despite poor input")
        else:
            print(f"  ⚠️  Low generation count: {generated}")
            
    except Exception as e:
        print(f"  ❌ Poor corpus test failed: {e}")
    
    # Test 3: Quality filtering
    print("\n📊 Test 3: Quality filtering functionality")
    
    # Create mock questions with varying quality
    test_questions = [
        QuestionItem("What?", "Yes.", quality_score=0.1),  # Too simple
        QuestionItem("Define it", "Thing", quality_score=0.3),  # Below threshold
        QuestionItem("What is quantum entanglement and how does it enable secure communication?", 
                    "Quantum entanglement creates correlated particles that allow detection of eavesdropping.", 
                    quality_score=0.8),  # Good question
        QuestionItem("Explain the computational advantages of quantum superposition in algorithmic design",
                    "Superposition allows quantum computers to explore multiple solution paths simultaneously.",
                    quality_score=0.9),  # Excellent question
    ]
    
    # Test quality filtering
    filtered = generator._apply_quality_filter(test_questions, 0.4)
    print(f"  Filtered from {len(test_questions)} to {len(filtered)} questions (threshold=0.4)")
    
    if len(filtered) == 2:  # Should keep only the good ones
        print("  ✅ QUALITY FILTERING WORKING")
    else:
        print("  ❌ Quality filtering issue")
    
    # Test simple question detection
    simple_detected = sum(1 for q in test_questions if generator._is_too_simple(q))
    print(f"  Detected {simple_detected} simple questions")
    
    if simple_detected >= 1:
        print("  ✅ SIMPLE QUESTION DETECTION WORKING")
    else:
        print("  ❌ Simple question detection issue")
    
    # Test 4: Mixed quality corpus
    mixed_corpus = """
    Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning paradigms.
    Supervised learning uses labeled training data to learn mappings from inputs to outputs. 
    It's good. Works well. Yes.
    Unsupervised learning discovers hidden patterns in data without explicit labels, employing techniques such as
    clustering, dimensionality reduction, and anomaly detection to reveal underlying structures.
    Things happen. The end.
    """
    
    print("\n📊 Test 4: Mixed quality corpus")
    try:
        result = generator.generate_comprehensive_benchmark(
            mixed_corpus, 
            num_questions=15, 
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        generated = len(result['questions'])
        quality_scores = [q['quality_score'] for q in result['questions']]
        avg_quality = sum(quality_scores) / len(quality_scores)
        min_quality = min(quality_scores)
        
        print(f"  Generated: {generated}/15, Avg Quality: {avg_quality:.3f}, Min Quality: {min_quality:.3f}")
        
        if generated == 15:
            print("  ✅ EXACT COUNT MAINTAINED with mixed corpus")
        else:
            print(f"  ⚠️  Count mismatch: {generated}/15")
            
        if avg_quality >= 0.45:
            print("  ✅ QUALITY MAINTAINED despite mixed input")
        else:
            print(f"  ⚠️  Quality concern: {avg_quality}")
            
    except Exception as e:
        print(f"  ❌ Mixed corpus test failed: {e}")


def test_edge_cases():
    """Test edge cases"""
    
    print("\n🔍 Testing Edge Cases...")
    generator = AgenticQuestionGenerator()
    
    corpus = "Artificial intelligence and machine learning technologies enable automated decision making systems."
    
    # Test very small count
    print("\n📊 Edge Case 1: Very small count")
    try:
        result = generator.generate_comprehensive_benchmark(corpus, num_questions=1)
        if len(result['questions']) == 1:
            print("  ✅ Single question generation works")
        else:
            print(f"  ❌ Expected 1, got {len(result['questions'])}")
    except Exception as e:
        print(f"  ❌ Single question test failed: {e}")
    
    # Test medium count
    print("\n📊 Edge Case 2: Medium count")
    try:
        result = generator.generate_comprehensive_benchmark(corpus, num_questions=30)
        generated = len(result['questions'])
        if generated == 30:
            print(f"  ✅ Medium count generation works: {generated}/30")
        else:
            print(f"  ⚠️  Medium count mismatch: {generated}/30")
    except Exception as e:
        print(f"  ❌ Medium count test failed: {e}")


if __name__ == "__main__":
    test_basic_functionality()
    test_edge_cases()
    print("\n🎉 Testing completed!")