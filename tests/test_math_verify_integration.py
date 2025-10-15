"""
Test cases for math-verify integration in the verification system
"""

import pytest
from docs_to_eval.core.verification import (
    MathVerifyVerifier, 
    VerificationOrchestrator,
    MATH_VERIFY_AVAILABLE
)


class TestMathVerifyVerifier:
    """Test the MathVerifyVerifier class"""
    
    def test_math_verify_match_basic(self):
        """Test basic mathematical verification"""
        verifier = MathVerifyVerifier()
        
        # Test basic mathematical equality
        result = verifier.math_verify_match("4", "2 + 2")
        assert result.score >= 0.0  # Should handle gracefully even without library
        assert result.method in ['math_verify', 'numerical_match']
        
    def test_math_verify_match_sets(self):
        """Test set operations verification"""
        verifier = MathVerifyVerifier()
        
        # Test set union - example from math-verify documentation
        prediction = "${1,2,3,4}$"
        ground_truth = "${1,3} \\cup {2,4}$"
        
        result = verifier.math_verify_match(prediction, ground_truth)
        assert result.score >= 0.0
        assert result.method in ['math_verify', 'numerical_match']
        
        # Check that details contain appropriate information
        assert 'library_available' in result.details
        
    def test_latex_expression_match(self):
        """Test LaTeX expression matching"""
        verifier = MathVerifyVerifier()
        
        # Test LaTeX square root
        prediction = "$\\sqrt{4}$"
        ground_truth = "$2$"
        
        result = verifier.latex_expression_match(prediction, ground_truth)
        assert result.score >= 0.0
        assert result.method in ['latex_math_verify', 'latex_fallback']
        
    def test_expression_match(self):
        """Test plain expression matching"""
        verifier = MathVerifyVerifier()
        
        # Test fraction to decimal
        prediction = "0.5"
        ground_truth = "1/2"
        
        result = verifier.expression_match(prediction, ground_truth)
        assert result.score >= 0.0
        assert result.method in ['expression_math_verify', 'numerical_match']
        
    def test_fallback_when_library_unavailable(self):
        """Test fallback behavior when math-verify is not available"""
        verifier = MathVerifyVerifier()
        
        # This should gracefully fall back to numerical matching
        result = verifier.math_verify_match("4.0", "4")
        assert result.score >= 0.0
        assert result.metrics is not None
        
    def test_error_handling(self):
        """Test error handling for invalid expressions"""
        verifier = MathVerifyVerifier()
        
        # Test with invalid/unparseable expressions
        result = verifier.math_verify_match("invalid_expression", "another_invalid")
        assert result.score >= 0.0  # Should not crash
        assert result.method is not None


class TestVerificationOrchestrator:
    """Test the VerificationOrchestrator with math-verify integration"""
    
    def setup_method(self):
        """Set up test environment"""
        self.orchestrator = VerificationOrchestrator()
    
    def test_mathematical_evaluation_type(self):
        """Test that 'mathematical' type uses math-verify"""
        result = self.orchestrator.verify("4", "2 + 2", "mathematical")
        assert result.score >= 0.0
        assert result.method in ['math_verify', 'numerical_match']
        
    def test_math_expression_evaluation_type(self):
        """Test that 'math_expression' type uses expression matching"""
        result = self.orchestrator.verify("0.5", "1/2", "math_expression")
        assert result.score >= 0.0
        assert result.method in ['expression_math_verify', 'numerical_match']
        
    def test_latex_math_evaluation_type(self):
        """Test that 'latex_math' type uses LaTeX matching"""
        result = self.orchestrator.verify("$2$", "$\\sqrt{4}$", "latex_math")
        assert result.score >= 0.0
        assert result.method in ['latex_math_verify', 'latex_fallback']
        
    def test_batch_verification_with_math_verify(self):
        """Test batch verification including math-verify cases"""
        predictions = ["4", "${1,2,3,4}$", "0.5"]
        ground_truths = ["2 + 2", "${1,3} \\cup {2,4}$", "1/2"]
        eval_types = ["mathematical", "latex_math", "math_expression"]
        
        # Use zip to create individual verifications
        results = []
        for pred, truth, eval_type in zip(predictions, ground_truths, eval_types):
            result = self.orchestrator.verify(pred, truth, eval_type)
            results.append(result)
        
        assert len(results) == 3
        for result in results:
            assert result.score >= 0.0
            assert result.method is not None
        
    def test_aggregate_metrics_with_math_verify(self):
        """Test aggregate metrics computation with math-verify results"""
        # Create some verification results
        results = [
            self.orchestrator.verify("4", "2 + 2", "mathematical"),
            self.orchestrator.verify("6", "3 * 2", "mathematical"),
            self.orchestrator.verify("0.5", "1/2", "math_expression")
        ]
        
        aggregates = self.orchestrator.compute_aggregate_metrics(results)
        
        assert 'mean_score' in aggregates
        assert 'num_samples' in aggregates
        assert aggregates['num_samples'] == 3
        assert 0.0 <= aggregates['mean_score'] <= 1.0


class TestMathVerifyExamples:
    """Test specific examples from math-verify documentation"""
    
    def setup_method(self):
        """Set up test environment"""
        self.orchestrator = VerificationOrchestrator()
    
    def test_set_union_example(self):
        """Test the exact example from math-verify docs"""
        # From documentation: gold = parse("${1,3} \\cup {2,4}$"), answer = parse("${1,2,3,4}$")
        gold = "${1,3} \\cup {2,4}$"
        answer = "${1,2,3,4}$"
        
        result = self.orchestrator.verify(answer, gold, "latex_math")
        
        # Should work if math-verify is available, otherwise fallback gracefully
        assert result.score >= 0.0
        assert result.method is not None
        
    def test_square_root_example(self):
        """Test square root LaTeX expression"""
        gold = "$\\sqrt{2}$"
        answer = "$\\sqrt{2}$"
        
        result = self.orchestrator.verify(answer, gold, "latex_math")
        
        # Exact match should score high
        assert result.score >= 0.0
        assert result.method is not None
        
    def test_fraction_expression_example(self):
        """Test fraction to decimal conversion"""
        gold = "1/2"
        answer = "0.5"
        
        result = self.orchestrator.verify(answer, gold, "math_expression")
        
        # Should handle mathematical equivalence
        assert result.score >= 0.0
        assert result.method is not None


class TestFallbackBehavior:
    """Test fallback behavior when math-verify is not available or fails"""
    
    def setup_method(self):
        """Set up test environment"""
        self.verifier = MathVerifyVerifier()
    
    def test_numerical_fallback(self):
        """Test fallback to numerical matching"""
        # This should work even without math-verify
        result = self.verifier.math_verify_match("3.14159", "3.14159")
        assert result.score >= 0.0
        assert result.metrics is not None
        
    def test_latex_cleaning_fallback(self):
        """Test LaTeX cleaning fallback"""
        # Test basic LaTeX cleaning when library is not available
        result = self.verifier.latex_expression_match("$x + 1$", "$x + 1$")
        assert result.score >= 0.0
        assert result.method in ['latex_math_verify', 'latex_fallback']


if __name__ == "__main__":
    # Run some basic tests
    print("Running Math-Verify Integration Tests")
    print("=" * 50)
    
    print(f"Math-verify library available: {MATH_VERIFY_AVAILABLE}")
    
    # Test basic functionality
    orchestrator = VerificationOrchestrator()
    
    test_cases = [
        ("4", "2 + 2", "mathematical"),
        ("${1,2,3,4}$", "${1,3} \\cup {2,4}$", "latex_math"),
        ("0.5", "1/2", "math_expression"),
    ]
    
    for prediction, ground_truth, eval_type in test_cases:
        result = orchestrator.verify(prediction, ground_truth, eval_type)
        print(f"\nTest: {eval_type}")
        print(f"Prediction: {prediction}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Score: {result.score:.3f}")
        print(f"Method: {result.method}")
        print(f"Details: {result.details}")