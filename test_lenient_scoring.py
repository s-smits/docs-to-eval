#!/usr/bin/env python3
"""
Test script to verify that the new lenient scoring system works properly
"""

import sys
sys.path.append('/Users/air/Developer/docs-to-eval')

from docs_to_eval.core.verification import NonDeterministicVerifier
from docs_to_eval.utils.similarity import (
    token_overlap_similarity, 
    character_overlap_similarity, 
    ngram_similarity,
    calculate_multi_similarity
)

def test_similarity_improvements():
    """Test that similarity scoring is now more lenient"""
    
    # Example from your message - similar archaeological findings content
    expected_answer = """Archaeological findings in tombs show women's importance through frescoes, sarcophagi, urns, and possessions. They are depicted in social activities. Spindle whorls and spring scales indicate manual work by women."""
    
    llm_response = """Archaeological findings in tombs (like frescoes, sarcophagi, urns, and possessions) show women's importance in society. They highlight depictions of women in social activities like banquets, dances, and reclining alongside men. Jewelry, clothing, and personal items indicate wealth and status."""
    
    print("=== Testing Individual Similarity Measures ===")
    
    # Test token overlap (now using Dice coefficient)
    token_score = token_overlap_similarity(expected_answer, llm_response)
    print(f"Token Overlap (Dice): {token_score:.3f}")
    
    # Test character overlap (now using Dice coefficient)  
    char_score = character_overlap_similarity(expected_answer, llm_response)
    print(f"Character Overlap (Dice): {char_score:.3f}")
    
    # Test n-gram similarity (now using Dice coefficient)
    ngram_score = ngram_similarity(expected_answer, llm_response)
    print(f"N-gram Similarity (Dice): {ngram_score:.3f}")
    
    print("\n=== Testing Multi-Similarity ===")
    multi_scores = calculate_multi_similarity(expected_answer, llm_response)
    for method, score in multi_scores.items():
        print(f"{method}: {score:.3f}")
    
    print("\n=== Testing Full Verification (with boost) ===")
    
    # Test the full verification system
    result = NonDeterministicVerifier.semantic_similarity_mock(llm_response, expected_answer)
    
    print(f"Final Score (boosted): {result.score:.3f}")
    print(f"Raw Score (pre-boost): {result.metrics.get('raw_score', 'N/A'):.3f}")
    print(f"Boost Applied: {result.details.get('boost_applied', False)}")
    
    print("\n=== Expected Results ===")
    print("With the new lenient scoring:")
    print("- Token overlap should be 0.4-0.7 range (was ~0.28 with Jaccard)")
    print("- Final boosted score should be 0.6-0.8 range")
    print("- This reflects the substantial factual overlap between the answers")
    
    return result.score

def test_additional_examples():
    """Test with additional examples to verify scoring range"""
    
    test_cases = [
        {
            "name": "High overlap case",
            "expected": "The Etruscans were an ancient civilization in Italy known for their art and culture.",
            "response": "Etruscans were an ancient Italian civilization famous for their artistic achievements and cultural contributions."
        },
        {
            "name": "Medium overlap case", 
            "expected": "Etruscan tombs contain frescoes depicting banquets and social life.",
            "response": "Archaeological evidence shows Etruscan burial sites with wall paintings of feasts and daily activities."
        },
        {
            "name": "Lower overlap case",
            "expected": "The Etruscan language used a unique alphabet derived from Greek.",
            "response": "Etruscan writing system was influenced by ancient Greek scripts but had distinctive characteristics."
        }
    ]
    
    print("\n" + "="*60)
    print("ADDITIONAL TEST CASES")
    print("="*60)
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        print(f"Expected: {case['expected']}")
        print(f"Response:  {case['response']}")
        
        result = NonDeterministicVerifier.semantic_similarity_mock(case['response'], case['expected'])
        print(f"Final Score: {result.score:.3f}")
        print(f"Raw Score: {result.metrics.get('raw_score', 0):.3f}")

if __name__ == "__main__":
    print("Testing the new lenient similarity scoring system...\n")
    
    main_score = test_similarity_improvements()
    test_additional_examples()
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Main test case score: {main_score:.3f}")
    
    if main_score >= 0.6:
        print("✅ SUCCESS: Scoring is now more lenient!")
        print("The score properly reflects the substantial factual overlap.")
    elif main_score >= 0.4:
        print("⚠️  PARTIAL: Some improvement, but could be more lenient.")
    else:
        print("❌ ISSUE: Scoring is still too strict.")