#!/usr/bin/env python3
"""Test that mathematical vs non-mathematical content is classified correctly"""

from docs_to_eval.utils.config import analyze_corpus_content, EvaluationType
from docs_to_eval.core.classification import EvaluationTypeClassifier
from docs_to_eval.core.verification import VerificationOrchestrator

def test_classification():
    """Test classification of different content types"""
    
    # Test cases with expected classification
    test_cases = [
        {
            "name": "Mathematical content",
            "content": """
            Solve the following problems:
            1. Calculate 25 + 37 = 62
            2. Find 15% of 200
            3. What is sqrt(144)?
            4. Solve for x: 2x + 5 = 15
            """,
            "expected": EvaluationType.MATHEMATICAL
        },
        {
            "name": "Etruscan history (non-mathematical)",
            "content": """
            The Etruscan civilization flourished in ancient Italy before the rise of Rome.
            Etruscan art and culture influenced Roman society significantly.
            The goddess Menrva was important in Etruscan religion.
            Etruscan language remains partially undeciphered.
            """,
            "expected": EvaluationType.DOMAIN_KNOWLEDGE
        },
        {
            "name": "Code content",
            "content": """
            def fibonacci(n):
                if n <= 1:
                    return n
                return fibonacci(n-1) + fibonacci(n-2)
            
            import numpy as np
            class DataProcessor:
                def process(self, data):
                    return np.mean(data)
            """,
            "expected": EvaluationType.CODE_GENERATION
        },
        {
            "name": "Generic Q&A",
            "content": """
            What is democracy?
            Define the concept of freedom.
            Explain how photosynthesis works.
            What are the key aspects of climate change?
            """,
            "expected": EvaluationType.DOMAIN_KNOWLEDGE
        }
    ]
    
    print("=" * 80)
    print("TESTING CONTENT CLASSIFICATION")
    print("=" * 80)
    
    for test in test_cases:
        print(f"\n📝 Test: {test['name']}")
        print(f"Expected: {test['expected']}")
        
        # Analyze content
        scores = analyze_corpus_content(test['content'])
        
        # Get top classification
        if scores:
            top_type = max(scores.items(), key=lambda x: x[1])
            print(f"Detected: {top_type[0]} (score: {top_type[1]:.3f})")
            
            # Show all scores
            print("All scores:")
            for eval_type, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                if score > 0:
                    print(f"  - {eval_type}: {score:.3f}")
            
            # Check if correct
            if top_type[0] == test['expected']:
                print("✅ PASS: Correctly classified")
            else:
                print(f"❌ FAIL: Expected {test['expected']}, got {top_type[0]}")
        else:
            print("❌ FAIL: No classification scores returned")


def test_verification_routing():
    """Test that verification routes to correct method based on eval type"""
    
    print("\n" + "=" * 80)
    print("TESTING VERIFICATION ROUTING")
    print("=" * 80)
    
    orchestrator = VerificationOrchestrator(use_mixed=True)
    
    test_cases = [
        {
            "name": "Mathematical problem",
            "prediction": "5",
            "ground_truth": "5",
            "eval_type": "mathematical",
            "question": "What is 2 + 3?",
            "expected_method": "math_verify_strict"
        },
        {
            "name": "Domain knowledge",
            "prediction": "Etruscan was an ancient civilization",
            "ground_truth": "Etruscan refers to the ancient civilization of Etruria",
            "eval_type": "domain_knowledge",
            "question": "What is Etruscan?",
            "expected_method_contains": ["mixed", "semantic", "similarity"]
        },
        {
            "name": "Wrong math answer",
            "prediction": "6",
            "ground_truth": "5",
            "eval_type": "mathematical",
            "question": "What is 2 + 3?",
            "expected_score": 0.0
        }
    ]
    
    for test in test_cases:
        print(f"\n📝 Test: {test['name']}")
        print(f"Eval type: {test['eval_type']}")
        print(f"Question: {test['question']}")
        
        result = orchestrator.verify(
            prediction=test['prediction'],
            ground_truth=test['ground_truth'],
            eval_type=test['eval_type'],
            question=test['question']
        )
        
        print(f"Method used: {result.method}")
        print(f"Score: {result.score}")
        
        if 'expected_method' in test:
            if result.method == test['expected_method']:
                print(f"✅ PASS: Correct method used")
            else:
                print(f"❌ FAIL: Expected method {test['expected_method']}, got {result.method}")
        
        if 'expected_method_contains' in test:
            if any(keyword in result.method.lower() for keyword in test['expected_method_contains']):
                print(f"✅ PASS: Method contains expected keyword")
            else:
                print(f"❌ FAIL: Method doesn't contain any of {test['expected_method_contains']}")
        
        if 'expected_score' in test:
            if abs(result.score - test['expected_score']) < 0.01:
                print(f"✅ PASS: Correct score")
            else:
                print(f"❌ FAIL: Expected score {test['expected_score']}, got {result.score}")


def main():
    """Run all tests"""
    test_classification()
    test_verification_routing()
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
