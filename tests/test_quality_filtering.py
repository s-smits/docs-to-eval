#!/usr/bin/env python3
"""
Comprehensive tests for quality filtering and exact count maintenance
"""

import pytest
import asyncio
from typing import List, Dict, Any

from docs_to_eval.core.agentic import AgenticQuestionGenerator, QuestionItem
from docs_to_eval.core.evaluation import EvaluationType


class TestQualityFiltering:
    """Test suite for quality filtering and count maintenance"""

    def setup_method(self):
        """Set up test fixtures"""
        self.generator = AgenticQuestionGenerator()
        
        # High-quality corpus for testing
        self.quality_corpus = """
        Quantum computing represents a revolutionary paradigm in computational science, leveraging quantum mechanical phenomena
        such as superposition and entanglement to process information. Unlike classical computers that use bits representing
        either 0 or 1, quantum computers utilize quantum bits (qubits) that can exist in superposition states, allowing
        simultaneous representation of both 0 and 1. This fundamental difference enables quantum algorithms to solve certain
        computational problems exponentially faster than classical algorithms. Key applications include cryptography,
        optimization problems, and molecular simulation. The quantum advantage is particularly pronounced in problems
        involving large-scale factorization, database searching, and simulation of quantum systems.
        """
        
        # Low-quality corpus for testing edge cases
        self.poor_corpus = """
        This is simple. Things are good. Yes. No. Maybe. 
        Some stuff happens. It works. The end.
        """
        
        # Mixed quality corpus
        self.mixed_corpus = """
        Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning paradigms.
        Supervised learning uses labeled training data to learn mappings from inputs to outputs. 
        It's good. Works well. Yes.
        Unsupervised learning discovers hidden patterns in data without explicit labels, employing techniques such as
        clustering, dimensionality reduction, and anomaly detection to reveal underlying structures.
        Things happen. The end.
        """

    def test_exact_count_maintenance_quality_corpus(self):
        """Test that exactly the requested number of questions is returned with quality corpus"""
        test_counts = [5, 10, 25, 50]
        
        for count in test_counts:
            result = self.generator.generate_comprehensive_benchmark(
                self.quality_corpus, 
                num_questions=count, 
                eval_type=EvaluationType.DOMAIN_KNOWLEDGE
            )
            
            assert len(result['questions']) == count, \
                f"Expected {count} questions, got {len(result['questions'])}"
            
            # Verify all questions meet minimum quality
            for q in result['questions']:
                assert q['quality_score'] >= 0.4, \
                    f"Question quality {q['quality_score']} below minimum threshold"
                assert len(q['question'].split()) >= 5, \
                    f"Question too short: {q['question']}"
                assert len(q['answer'].split()) >= 2, \
                    f"Answer too short: {q['answer']}"

    def test_exact_count_maintenance_poor_corpus(self):
        """Test count maintenance with low-quality corpus that requires regeneration"""
        result = self.generator.generate_comprehensive_benchmark(
            self.poor_corpus, 
            num_questions=10, 
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        # Should still attempt to generate 10 questions
        generated_count = len(result['questions'])
        
        # May generate fewer than requested due to poor corpus, but should try hard
        assert generated_count >= 5, \
            f"Generated too few questions ({generated_count}) from poor corpus"
        
        # All generated questions should meet minimum standards
        for q in result['questions']:
            assert q['quality_score'] >= 0.1, \
                f"Question quality {q['quality_score']} below absolute minimum"

    def test_quality_filtering_removes_simple_questions(self):
        """Test that simple/low-quality questions are filtered out"""
        # Create mock low-quality questions
        low_quality_questions = [
            QuestionItem("What?", "Yes.", quality_score=0.2),
            QuestionItem("Define it", "Thing", quality_score=0.3),
            QuestionItem("What is placeholder?", "Insert answer here", quality_score=0.1),
            QuestionItem("...", "TODO", quality_score=0.05),
        ]
        
        # Create mock high-quality questions
        high_quality_questions = [
            QuestionItem(
                "How do quantum algorithms achieve exponential speedup over classical algorithms?",
                "Quantum algorithms leverage superposition and entanglement to explore multiple solution paths simultaneously.",
                quality_score=0.8
            ),
            QuestionItem(
                "What are the primary challenges in scaling quantum computing systems?",
                "Quantum decoherence, error correction, and maintaining qubit stability at scale.",
                quality_score=0.7
            )
        ]
        
        all_questions = low_quality_questions + high_quality_questions
        
        # Test quality filtering
        filtered = self.generator._apply_quality_filter(all_questions, 0.4)
        
        # Should only keep high-quality questions
        assert len(filtered) == 2, f"Expected 2 questions after filtering, got {len(filtered)}"
        assert all(q.quality_score >= 0.4 for q in filtered), "Low quality questions not filtered"

    def test_duplicate_removal(self):
        """Test that duplicate and similar questions are removed"""
        duplicate_questions = [
            QuestionItem("What is quantum computing?", "A computational paradigm", quality_score=0.8),
            QuestionItem("What is quantum computing systems?", "Computational systems using quantum mechanics", quality_score=0.7),
            QuestionItem("How does machine learning work?", "Using algorithms to learn patterns", quality_score=0.6),
            QuestionItem("Explain machine learning algorithms", "Algorithms that learn from data", quality_score=0.6),
        ]
        
        unique = self.generator._remove_duplicates(duplicate_questions)
        
        # Should remove similar questions
        assert len(unique) == 2, f"Expected 2 unique questions, got {len(unique)}"

    def test_simple_question_detection(self):
        """Test detection of overly simple questions"""
        simple_questions = [
            QuestionItem("What?", "Yes", quality_score=0.5),  # Too short
            QuestionItem("Define placeholder", "TODO fill in", quality_score=0.5),  # Placeholder
            QuestionItem("What is it", "Thing", quality_score=0.5),  # Generic + short
        ]
        
        complex_questions = [
            QuestionItem(
                "How do quantum entanglement protocols enable secure communication channels?",
                "Quantum entanglement creates correlated qubit pairs that detect eavesdropping attempts.",
                quality_score=0.7
            )
        ]
        
        for q in simple_questions:
            assert self.generator._is_too_simple(q), f"Failed to detect simple question: {q.question}"
        
        for q in complex_questions:
            assert not self.generator._is_too_simple(q), f"Incorrectly flagged complex question: {q.question}"

    def test_regeneration_logic(self):
        """Test that additional questions are generated when needed"""
        # Start with insufficient high-quality questions
        initial_questions = [
            QuestionItem("What?", "Yes", quality_score=0.2),  # Will be filtered
            QuestionItem("Good question about quantum computing algorithms?", "Detailed answer", quality_score=0.8),
        ]
        
        # Test the filtering with regeneration
        analysis = self.generator._analyze_corpus_structure(self.quality_corpus)
        
        result = self.generator._filter_and_ensure_count(
            initial_questions, 
            analysis, 
            target_count=5,
            corpus_text=self.quality_corpus,
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        # Should have regenerated to meet target count
        assert len(result) == 5, f"Expected 5 questions after regeneration, got {len(result)}"
        assert all(q.quality_score >= 0.4 for q in result), "Regenerated questions below quality threshold"

    def test_oversampling_strategy(self):
        """Test that oversampling generates sufficient candidates"""
        target_count = 20
        oversample_factor = 2.5
        expected_generated = int(target_count * oversample_factor)
        
        analysis = self.generator._analyze_corpus_structure(self.quality_corpus)
        
        questions = self.generator._generate_with_oversampling(
            self.quality_corpus,
            analysis,
            expected_generated,
            EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        # Should generate approximately the expected count (within reasonable range)
        assert len(questions) >= expected_generated * 0.8, \
            f"Undersampled: expected ~{expected_generated}, got {len(questions)}"
        assert len(questions) <= expected_generated * 1.2, \
            f"Oversampled excessively: expected ~{expected_generated}, got {len(questions)}"

    def test_mixed_corpus_handling(self):
        """Test handling of mixed quality corpus"""
        result = self.generator.generate_comprehensive_benchmark(
            self.mixed_corpus, 
            num_questions=15, 
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        # Should generate exactly 15 questions
        assert len(result['questions']) == 15, \
            f"Expected 15 questions from mixed corpus, got {len(result['questions'])}"
        
        # Should have reasonable quality distribution
        quality_scores = [q['quality_score'] for q in result['questions']]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        assert avg_quality >= 0.45, \
            f"Average quality too low: {avg_quality}"

    def test_quality_statistics(self):
        """Test that quality statistics are correctly computed"""
        result = self.generator.generate_comprehensive_benchmark(
            self.quality_corpus, 
            num_questions=20, 
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        stats = result['quality_stats']
        
        # Verify required statistics are present
        assert 'avg_quality' in stats
        assert 'min_quality' in stats
        assert 'max_quality' in stats
        assert 'total_questions' in stats
        
        # Verify statistical consistency
        assert stats['total_questions'] == 20
        assert stats['min_quality'] <= stats['avg_quality'] <= stats['max_quality']
        assert stats['avg_quality'] >= 0.4  # Should maintain minimum quality

    def test_edge_case_very_small_count(self):
        """Test handling of very small question counts"""
        result = self.generator.generate_comprehensive_benchmark(
            self.quality_corpus, 
            num_questions=1, 
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        assert len(result['questions']) == 1, "Failed to generate exactly 1 question"
        assert result['questions'][0]['quality_score'] >= 0.4, "Single question below quality threshold"

    def test_edge_case_large_count(self):
        """Test handling of large question counts"""
        result = self.generator.generate_comprehensive_benchmark(
            self.quality_corpus, 
            num_questions=100, 
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        # Should generate exactly 100 questions (may take time with regeneration)
        assert len(result['questions']) == 100, \
            f"Expected 100 questions for large count test, got {len(result['questions'])}"
        
        # Should maintain quality standards
        quality_scores = [q['quality_score'] for q in result['questions']]
        avg_quality = sum(quality_scores) / len(quality_scores)
        assert avg_quality >= 0.4, f"Large count generation quality too low: {avg_quality}"


class TestIntegrationScenarios:
    """Integration tests with real-world scenarios"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.generator = AgenticQuestionGenerator()

    def test_academic_paper_corpus(self):
        """Test with academic paper-style corpus"""
        academic_corpus = """
        Recent advances in transformer architectures have demonstrated remarkable capabilities in natural language processing tasks.
        The attention mechanism, introduced by Vaswani et al., allows models to selectively focus on relevant parts of input sequences.
        Self-attention computes attention weights by taking dot products of query, key, and value matrices derived from input embeddings.
        Multi-head attention performs attention computation in parallel across multiple representation subspaces.
        The transformer architecture's parallelizability makes it particularly suitable for training on modern hardware accelerators.
        Position encoding schemes inject information about token positions since attention operations are permutation-invariant.
        """
        
        result = self.generator.generate_comprehensive_benchmark(
            academic_corpus, 
            num_questions=30, 
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        assert len(result['questions']) == 30, "Academic corpus failed to generate exact count"
        
        # Academic content should generate high-quality questions
        quality_scores = [q['quality_score'] for q in result['questions']]
        avg_quality = sum(quality_scores) / len(quality_scores)
        assert avg_quality >= 0.6, f"Academic corpus quality too low: {avg_quality}"

    def test_technical_documentation_corpus(self):
        """Test with technical documentation style corpus"""
        tech_corpus = """
        The Docker containerization platform enables application packaging with dependencies into lightweight, portable containers.
        Docker images are built from Dockerfiles containing layer-by-layer instructions for constructing the container environment.
        Container orchestration platforms like Kubernetes manage deployment, scaling, and networking of containerized applications.
        Kubernetes pods represent the smallest deployable units, typically containing one or more tightly coupled containers.
        Services provide stable network interfaces to groups of pods, enabling load balancing and service discovery.
        ConfigMaps and Secrets manage configuration data and sensitive information separately from application code.
        """
        
        result = self.generator.generate_comprehensive_benchmark(
            tech_corpus, 
            num_questions=25, 
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        assert len(result['questions']) == 25, "Technical corpus failed to generate exact count"
        
        # Should have good diversity in question types
        categories = set(q['category'] for q in result['questions'])
        assert len(categories) >= 2, "Technical corpus should generate diverse question categories"

    def test_mathematical_content_corpus(self):
        """Test with mathematical content"""
        math_corpus = """
        Linear algebra forms the foundation of many machine learning algorithms and computational methods.
        Vector spaces are algebraic structures consisting of vectors that can be added together and multiplied by scalars.
        Matrix operations including multiplication, inversion, and eigendecomposition are fundamental to linear transformations.
        Eigenvalues and eigenvectors provide insights into the geometric properties of linear transformations.
        The singular value decomposition (SVD) factorizes matrices into products of unitary and diagonal matrices.
        Principal component analysis (PCA) uses eigendecomposition to reduce dimensionality while preserving variance.
        """
        
        result = self.generator.generate_comprehensive_benchmark(
            math_corpus, 
            num_questions=20, 
            eval_type=EvaluationType.MATHEMATICAL
        )
        
        assert len(result['questions']) == 20, "Mathematical corpus failed to generate exact count"
        
        # Mathematical content should maintain high quality
        for q in result['questions']:
            assert q['quality_score'] >= 0.4, f"Mathematical question below threshold: {q['question']}"


def run_stress_tests():
    """Run stress tests with various challenging scenarios"""
    generator = AgenticQuestionGenerator()
    
    print("🔄 Running stress tests...")
    
    # Test 1: Very poor corpus
    poor_corpus = "A. B. C. Yes. No. The end."
    try:
        result = generator.generate_comprehensive_benchmark(poor_corpus, num_questions=10)
        print(f"✅ Poor corpus test: Generated {len(result['questions'])} questions")
    except Exception as e:
        print(f"❌ Poor corpus test failed: {e}")
    
    # Test 2: Large count with good corpus
    good_corpus = """
    Artificial intelligence encompasses machine learning, deep learning, neural networks, natural language processing,
    computer vision, robotics, expert systems, and knowledge representation. Each subfield addresses different aspects
    of intelligent behavior and problem-solving capabilities.
    """ * 10  # Repeat for more content
    
    try:
        result = generator.generate_comprehensive_benchmark(good_corpus, num_questions=75)
        print(f"✅ Large count test: Generated {len(result['questions'])}/{75} questions")
    except Exception as e:
        print(f"❌ Large count test failed: {e}")
    
    # Test 3: Repetitive corpus
    repetitive_corpus = "Machine learning is good. Machine learning works well. Machine learning helps. " * 20
    try:
        result = generator.generate_comprehensive_benchmark(repetitive_corpus, num_questions=15)
        print(f"✅ Repetitive corpus test: Generated {len(result['questions'])} questions")
        
        # Check for diversity despite repetitive input
        questions = [q['question'] for q in result['questions']]
        unique_questions = len(set(questions))
        print(f"   Unique questions: {unique_questions}/{len(questions)}")
        
    except Exception as e:
        print(f"❌ Repetitive corpus test failed: {e}")


if __name__ == "__main__":
    # Run basic tests
    test_class = TestQualityFiltering()
    test_class.setup_method()
    
    print("🧪 Running quality filtering tests...")
    
    # Run individual tests
    test_methods = [
        test_class.test_exact_count_maintenance_quality_corpus,
        test_class.test_quality_filtering_removes_simple_questions,
        test_class.test_duplicate_removal,
        test_class.test_simple_question_detection,
        test_class.test_oversampling_strategy,
    ]
    
    for test_method in test_methods:
        try:
            test_method()
            print(f"✅ {test_method.__name__}")
        except Exception as e:
            print(f"❌ {test_method.__name__}: {e}")
    
    # Run integration tests
    print("\n🔄 Running integration tests...")
    integration_class = TestIntegrationScenarios()
    integration_class.setup_method()
    
    integration_methods = [
        integration_class.test_academic_paper_corpus,
        integration_class.test_technical_documentation_corpus,
        integration_class.test_mathematical_content_corpus,
    ]
    
    for test_method in integration_methods:
        try:
            test_method()
            print(f"✅ {test_method.__name__}")
        except Exception as e:
            print(f"❌ {test_method.__name__}: {e}")
    
    # Run stress tests
    print("\n💪 Running stress tests...")
    run_stress_tests()
    
    print("\n🎉 Test suite completed!")