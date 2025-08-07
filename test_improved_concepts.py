#!/usr/bin/env python3
"""
Test the improved concept extraction system
"""

from docs_to_eval.core.evaluation import extract_key_concepts

def test_concept_extraction():
    """Test improved concept extraction with various text types"""
    
    # Test case 1: Technical/domain-specific text
    technical_text = """
    Machine learning algorithms use neural networks to process data.
    Deep learning models utilize convolutional layers and backpropagation.
    TensorFlow and PyTorch are popular frameworks for building AI systems.
    The transformer architecture revolutionized natural language processing.
    """
    
    print("🧪 Testing Technical Text:")
    concepts = extract_key_concepts(technical_text, max_concepts=10)
    print(f"Concepts: {concepts}")
    print(f"Quality check: {all(len(c) > 3 and c not in ['such', 'some', 'other'] for c in concepts)}")
    print()
    
    # Test case 2: Historical text 
    historical_text = """
    The Roman Empire was established in 27 BC when Augustus became emperor.
    Ancient Rome controlled vast territories across Europe, Africa, and Asia.
    Julius Caesar played a crucial role in the transformation of the Republic.
    The fall of Constantinople in 1453 marked the end of Byzantine civilization.
    """
    
    print("📚 Testing Historical Text:")
    concepts = extract_key_concepts(historical_text, max_concepts=10)
    print(f"Concepts: {concepts}")
    print(f"Quality check: {all(len(c) > 3 and c not in ['such', 'some', 'other'] for c in concepts)}")
    print()
    
    # Test case 3: Generic/problematic text (should filter out generic terms)
    generic_text = """
    This is some text about various things and other elements.
    There are different approaches to such problems in general.
    Many people have different opinions about these particular issues.
    Some methods are better than other techniques in certain situations.
    """
    
    print("⚠️  Testing Generic Text (should filter out bad concepts):")
    concepts = extract_key_concepts(generic_text, max_concepts=10)
    print(f"Concepts: {concepts}")
    bad_concepts = ['such', 'some', 'other', 'different', 'various', 'certain', 'particular', 'general']
    quality_check = not any(bad in concepts for bad in bad_concepts)
    print(f"Quality check (no bad concepts): {quality_check}")
    print()
    
    # Test case 4: Mathematical text
    math_text = """
    To solve quadratic equations, use the formula x = (-b ± √(b²-4ac))/2a.
    Linear algebra involves matrices, vectors, and eigenvalues.
    Calculus studies derivatives, integrals, and limits of functions.
    Probability theory analyzes random events and statistical distributions.
    """
    
    print("🔢 Testing Mathematical Text:")
    concepts = extract_key_concepts(math_text, max_concepts=10)
    print(f"Concepts: {concepts}")
    print(f"Quality check: {all(len(c) > 3 and c not in ['such', 'some', 'other'] for c in concepts)}")
    print()
    
    # Test case 5: Mixed content with proper nouns
    mixed_text = """
    Python is a programming language developed by Guido van Rossum.
    The Django framework enables rapid web development.
    GitHub provides version control and collaboration features.
    Machine learning libraries like scikit-learn simplify data analysis.
    """
    
    print("🔥 Testing Mixed Text with Proper Nouns:")
    concepts = extract_key_concepts(mixed_text, max_concepts=10)
    print(f"Concepts: {concepts}")
    proper_nouns = [c for c in concepts if c[0].isupper()]
    print(f"Proper nouns found: {proper_nouns}")
    print(f"Quality check: {all(len(c) > 3 and c not in ['such', 'some', 'other'] for c in concepts)}")
    
    return all([
        len([c for c in extract_key_concepts(technical_text) if c in ['such', 'some', 'other']]) == 0,
        len([c for c in extract_key_concepts(historical_text) if c in ['such', 'some', 'other']]) == 0,
        len([c for c in extract_key_concepts(generic_text) if c in ['such', 'some', 'other']]) == 0,
        len([c for c in extract_key_concepts(math_text) if c in ['such', 'some', 'other']]) == 0,
        len([c for c in extract_key_concepts(mixed_text) if c in ['such', 'some', 'other']]) == 0,
    ])

if __name__ == "__main__":
    print("🚀 Testing Improved Concept Extraction")
    print("=" * 50)
    
    success = test_concept_extraction()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! No more generic concepts like 'such', 'some', 'other'")
    else:
        print("❌ Some tests failed - generic concepts still being extracted")
        
    print("\n🎯 This should fix the poor question quality issues in your evaluation!")