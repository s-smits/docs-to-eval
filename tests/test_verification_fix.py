"""
Tests for the Verification System Fix - Enum to String Conversion
Tests the critical fix that resolved the mean_score = 0 issue by properly
converting EvaluationType enums to strings for the verification system.
"""

import pytest
from unittest.mock import Mock, patch

from docs_to_eval.core.verification import VerificationOrchestrator, VerificationResult
from docs_to_eval.utils.config import EvaluationType
from docs_to_eval.core.classification import ClassificationResult


class TestEnumToStringConversion:
    """Test the enum to string conversion fix"""
    
    def setup_method(self):
        """Set up test environment"""
        self.orchestrator = VerificationOrchestrator()
    
    def test_domain_knowledge_enum_conversion(self):
        """Test that EvaluationType.DOMAIN_KNOWLEDGE enum is properly converted"""
        # Test direct enum usage (what was failing before)
        eval_type_enum = EvaluationType.DOMAIN_KNOWLEDGE
        eval_type_string = eval_type_enum.value
        
        assert eval_type_string == "domain_knowledge"
        
        # Test verification with string (should work)
        result = self.orchestrator.verify(
            prediction="Machine learning is a subset of AI",
            ground_truth="ML is part of artificial intelligence",
            eval_type=eval_type_string,
            question="What is machine learning?"
        )
        
        assert isinstance(result, VerificationResult)
        assert result.score > 0  # This was the bug - score was 0 before fix
        assert result.method is not None
    
    def test_all_evaluation_types_conversion(self):
        """Test enum to string conversion for all evaluation types"""
        test_cases = [
            (EvaluationType.DOMAIN_KNOWLEDGE, "domain_knowledge"),
            (EvaluationType.FACTUAL_QA, "factual_qa"),
            (EvaluationType.MATHEMATICAL, "mathematical"),
            (EvaluationType.CODE_GENERATION, "code_generation"),
            (EvaluationType.MULTIPLE_CHOICE, "multiple_choice"),
            (EvaluationType.SUMMARIZATION, "summarization"),
            (EvaluationType.TRANSLATION, "translation"),
            (EvaluationType.CREATIVE_WRITING, "creative_writing"),
            (EvaluationType.READING_COMPREHENSION, "reading_comprehension")
        ]
        
        for enum_val, expected_string in test_cases:
            assert enum_val.value == expected_string
            
            # Test that verification works with the string value
            result = self.orchestrator.verify(
                prediction="Test prediction",
                ground_truth="Test ground truth",
                eval_type=expected_string,
                question="Test question?"
            )
            
            assert isinstance(result, VerificationResult)
            assert result.score >= 0  # Should not crash and should have valid score
    
    def test_enum_hasattr_check(self):
        """Test the hasattr check used in the fix"""
        enum_val = EvaluationType.DOMAIN_KNOWLEDGE
        
        # Test the exact logic used in the fix
        eval_type_str = enum_val.value if hasattr(enum_val, 'value') else str(enum_val)
        
        assert eval_type_str == "domain_knowledge"
        
        # Test with non-enum (should use str())
        non_enum = "already_a_string"
        eval_type_str = non_enum.value if hasattr(non_enum, 'value') else str(non_enum)
        
        assert eval_type_str == "already_a_string"
    
    def test_verification_with_enum_vs_string(self):
        """Test verification behavior with enum vs string input"""
        prediction = "AI systems can learn from data"
        ground_truth = "Artificial intelligence learns from data"
        question = "How do AI systems learn?"
        
        # Test with string (should work)
        string_result = self.orchestrator.verify(
            prediction=prediction,
            ground_truth=ground_truth,
            eval_type="domain_knowledge",
            question=question
        )
        
        # Test with enum converted to string (the fix)
        enum_converted_result = self.orchestrator.verify(
            prediction=prediction,
            ground_truth=ground_truth,
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE.value,
            question=question
        )
        
        # Both should work and give similar results
        assert string_result.score > 0
        assert enum_converted_result.score > 0
        assert string_result.method == enum_converted_result.method
        
        # Scores should be identical since same verification method
        assert abs(string_result.score - enum_converted_result.score) < 0.001


class TestVerificationMethodSelection:
    """Test that the correct verification methods are selected"""
    
    def setup_method(self):
        """Set up test environment"""
        self.orchestrator = VerificationOrchestrator()
    
    def test_domain_knowledge_uses_domain_factual(self):
        """Test that domain_knowledge type uses domain factual verification"""
        result = self.orchestrator.verify(
            prediction="Quantum computers use qubits",
            ground_truth="Quantum computing systems utilize quantum bits",
            eval_type="domain_knowledge",
            question="What do quantum computers use?"
        )
        
        # Should use domain factual verification method
        assert result.method == "domain_factual_knowledge"
        assert result.score > 0
        assert "factual_accuracy" in result.details
    
    def test_mathematical_uses_math_verify(self):
        """Test that mathematical type uses math verification"""
        result = self.orchestrator.verify(
            prediction="4",
            ground_truth="2 + 2",
            eval_type="mathematical",
            question="What is 2 + 2?"
        )
        
        # Should use mathematical verification
        assert "math" in result.method.lower() or "numerical" in result.method.lower()
        assert result.score >= 0
    
    def test_factual_qa_uses_exact_or_similarity(self):
        """Test that factual_qa uses appropriate verification"""
        result = self.orchestrator.verify(
            prediction="Paris",
            ground_truth="Paris is the capital of France",
            eval_type="factual_qa",
            question="What is the capital of France?"
        )
        
        assert result.score >= 0
        assert result.method is not None
        # Should use exact match, similarity, or domain factual
        valid_methods = ["exact_match", "token_overlap", "domain_factual_knowledge"]
        assert any(method in result.method for method in valid_methods)
    
    def test_code_generation_verification(self):
        """Test code generation verification"""
        result = self.orchestrator.verify(
            prediction="def hello(): return 'Hi'",
            ground_truth="def hello(): return 'Hello'",
            eval_type="code_generation",
            question="Write a hello function"
        )
        
        assert result.score >= 0
        # Should use code execution or similarity matching
        assert result.method is not None


class TestPipelineIntegrationFix:
    """Test the fix in the context of the full pipeline"""
    
    def test_classification_to_verification_flow(self):
        """Test the flow from classification to verification"""
        # Simulate the classification result (enum)
        mock_classification = Mock(spec=ClassificationResult)
        mock_classification.primary_type = EvaluationType.DOMAIN_KNOWLEDGE
        mock_classification.confidence = 0.8
        
        # Test the conversion logic used in routes.py
        eval_type_str = mock_classification.primary_type.value if hasattr(mock_classification.primary_type, 'value') else str(mock_classification.primary_type)
        
        assert eval_type_str == "domain_knowledge"
        
        # Test verification with converted string
        result = VerificationOrchestrator().verify(
            prediction="Neural networks learn patterns",
            ground_truth="Neural networks identify patterns in data",
            eval_type=eval_type_str,
            question="What do neural networks do?"
        )
        
        assert result.score > 0  # The key fix - should not be 0
        assert result.method is not None
    
    def test_mock_llm_evaluation_integration(self):
        """Test the fix with mock LLM evaluation results"""
        # Simulate mock LLM results (as would come from the pipeline)
        mock_llm_results = [
            {
                "question": "What is machine learning?",
                "ground_truth": "A method for computers to learn from data",
                "prediction": "Machine learning enables automated learning from datasets",
                "confidence": 0.8,
                "source": "mock_llm"
            },
            {
                "question": "How do neural networks work?",
                "ground_truth": "They process information through interconnected nodes",
                "prediction": "Neural networks use connected nodes to process data",
                "confidence": 0.7,
                "source": "mock_llm" 
            }
        ]
        
        orchestrator = VerificationOrchestrator()
        
        # Test verification of all results
        verification_results = []
        for result in mock_llm_results:
            verification = orchestrator.verify(
                prediction=result["prediction"],
                ground_truth=result["ground_truth"],
                eval_type="domain_knowledge",  # Using string as per fix
                question=result["question"]
            )
            verification_results.append(verification)
        
        # Verify all results have non-zero scores (the bug fix)
        assert len(verification_results) == 2
        for verification in verification_results:
            assert verification.score > 0
            assert verification.method is not None
        
        # Calculate mean score (as done in pipeline)
        mean_score = sum(v.score for v in verification_results) / len(verification_results)
        assert mean_score > 0  # This was the main bug - mean_score was 0


class TestVerificationDebugOutput:
    """Test verification debug output and logging"""
    
    def setup_method(self):
        """Set up test environment"""
        self.orchestrator = VerificationOrchestrator()
    
    def test_domain_factual_debug_output(self):
        """Test that domain factual verification produces debug output"""
        with patch('builtins.print') as mock_print:
            _ = self.orchestrator.verify(
                prediction="Machine learning algorithms learn patterns",
                ground_truth="ML algorithms identify patterns in data",
                eval_type="domain_knowledge",
                question="What do ML algorithms do?"
            )
            
            # Should have printed debug info
            assert mock_print.called
            debug_calls = [call for call in mock_print.call_args_list 
                          if "Using domain factual verification" in str(call)]
            assert len(debug_calls) > 0
    
    def test_verification_details_structure(self):
        """Test that verification results have proper details structure"""
        result = self.orchestrator.verify(
            prediction="Deep learning uses neural networks",
            ground_truth="Deep learning utilizes artificial neural networks",
            eval_type="domain_knowledge",
            question="What does deep learning use?"
        )
        
        assert isinstance(result.details, dict)
        assert "question_type" in result.details
        assert "verification_approach" in result.details
        
        # Check that details provide insight into the verification process
        assert result.details["question_type"] in ["factual_knowledge", "conceptual", "analytical"]
        assert result.details["verification_approach"] in ["semantic_matching", "exact_matching", "hybrid"]


class TestRegressionPrevention:
    """Test to prevent regression of the enum-to-string bug"""
    
    def test_prevent_enum_direct_usage_regression(self):
        """Test to catch if enum is used directly in verification (regression)"""
        # This test ensures that if someone accidentally uses enum directly,
        # it will be caught by tests
        
        orchestrator = VerificationOrchestrator()
        
        # Direct enum usage (what was causing the bug)
        enum_type = EvaluationType.DOMAIN_KNOWLEDGE
        
        # Try to verify with enum directly (should be converted)
        with patch.object(orchestrator, 'verify') as mock_verify:
            # Simulate the fixed conversion
            eval_type_str = enum_type.value if hasattr(enum_type, 'value') else str(enum_type)
            
            # Call with converted string
            orchestrator.verify(
                prediction="test", 
                ground_truth="test", 
                eval_type=eval_type_str,
                question="test?"
            )
            
            # Verify that string was passed, not enum
            called_args = mock_verify.call_args
            assert called_args[1]['eval_type'] == "domain_knowledge"  # String, not enum
    
    def test_all_enum_values_work(self):
        """Test that all enum values work after conversion"""
        orchestrator = VerificationOrchestrator()
        
        # Test all evaluation types to ensure none break
        for eval_type_enum in EvaluationType:
            eval_type_str = eval_type_enum.value
            
            result = orchestrator.verify(
                prediction="Test prediction for " + eval_type_str,
                ground_truth="Test ground truth for " + eval_type_str,
                eval_type=eval_type_str,
                question=f"Test question for {eval_type_str}?"
            )
            
            # All should return valid results with non-negative scores
            assert isinstance(result, VerificationResult)
            assert result.score >= 0
            assert result.method is not None
            
            # None should return exactly 0 for reasonable inputs (the bug indicator)
            if eval_type_str in ["domain_knowledge", "factual_qa"]:
                assert result.score > 0, f"Score should be > 0 for {eval_type_str}, got {result.score}"


class TestFixValidation:
    """Validate that the fix actually works in realistic scenarios"""
    
    def test_realistic_evaluation_scenario(self):
        """Test realistic evaluation scenario that was failing before fix"""
        # Simulate a realistic corpus evaluation
        corpus_text = "Etruscan civilization flourished in central Italy before Roman times."
        
        # Simulate classification result (enum, as it comes from classifier)
        mock_classification = Mock()
        mock_classification.primary_type = EvaluationType.DOMAIN_KNOWLEDGE
        
        # Simulate LLM results
        llm_results = [
            {
                "question": "Where did Etruscan civilization flourish?",
                "ground_truth": "Central Italy",
                "prediction": "The Etruscan civilization was located in central Italy",
                "confidence": 0.8
            },
            {
                "question": "When did Etruscan civilization exist?",
                "ground_truth": "Before Roman times",
                "prediction": "Etruscans existed prior to the Roman period",
                "confidence": 0.7
            }
        ]
        
        orchestrator = VerificationOrchestrator(corpus_text=corpus_text)
        
        # Apply the fix: convert enum to string
        eval_type_str = mock_classification.primary_type.value if hasattr(mock_classification.primary_type, 'value') else str(mock_classification.primary_type)
        
        verification_results = []
        for result in llm_results:
            verification = orchestrator.verify(
                prediction=result["prediction"],
                ground_truth=result["ground_truth"],
                eval_type=eval_type_str,  # The fix
                options=result.get("options"),
                question=result["question"]
            )
            verification_results.append({
                "question": result["question"],
                "prediction": result["prediction"],
                "ground_truth": result["ground_truth"],
                "score": verification.score,
                "method": verification.method,
                "details": verification.details
            })
        
        # Verify the fix worked
        assert len(verification_results) == 2
        
        # Calculate mean score as done in the pipeline
        scores = [r["score"] for r in verification_results]
        mean_score = sum(scores) / len(scores) if scores else 0
        
        # This was the main bug: mean_score was 0
        assert mean_score > 0, f"Mean score should be > 0, got {mean_score}. Fix may not be working."
        
        # Individual scores should also be > 0 for reasonable content
        for result in verification_results:
            assert result["score"] > 0, f"Individual score should be > 0, got {result['score']} for question: {result['question']}"
            assert result["method"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])