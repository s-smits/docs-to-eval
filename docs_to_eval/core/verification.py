"""
Verification systems for deterministic and non-deterministic evaluation
Implements exact match, similarity metrics, execution-based, and LLM-judge verification
"""

import re
import json
import math
from functools import reduce
from collections import Counter
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel

try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False

from ..utils.text_processing import normalize_answer, extract_numbers
from ..utils.similarity import calculate_similarity, calculate_multi_similarity


class VerificationResult(BaseModel):
    """Structured result from verification"""
    score: float
    metrics: Dict[str, float]
    method: str
    details: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.dict()


class DeterministicVerifier:
    """Handles exact match verification for deterministic answers"""
    
    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> VerificationResult:
        """Exact string match after normalization"""
        pred_normalized = normalize_answer(prediction)
        truth_normalized = normalize_answer(ground_truth)
        
        score = 1.0 if pred_normalized == truth_normalized else 0.0
        
        return VerificationResult(
            score=score,
            metrics={'exact_match': score},
            method='exact_match',
            details={
                'prediction_normalized': pred_normalized,
                'ground_truth_normalized': truth_normalized
            }
        )
    
    @staticmethod
    def numerical_match(prediction: str, ground_truth: str, tolerance: float = 0.05) -> VerificationResult:
        """Numerical matching with tolerance for floating point answers"""
        pred_numbers = extract_numbers(prediction)
        truth_numbers = extract_numbers(ground_truth)
        
        if not pred_numbers or not truth_numbers:
            # Fall back to exact match if no numbers found
            return DeterministicVerifier.exact_match(prediction, ground_truth)
        
        try:
            pred_val = float(pred_numbers[0])
            truth_val = float(truth_numbers[0])
            
            if abs(pred_val - truth_val) <= tolerance:
                score = 1.0
            else:
                # Partial credit based on relative error
                relative_error = abs(pred_val - truth_val) / max(abs(truth_val), 1)
                score = max(0.0, 1.0 - relative_error)
        except (ValueError, IndexError):
            score = 0.0
        
        return VerificationResult(
            score=score,
            metrics={'numerical_match': score, 'tolerance': tolerance},
            method='numerical_match',
            details={
                'predicted_numbers': pred_numbers,
                'ground_truth_numbers': truth_numbers
            }
        )
    
    @staticmethod
    def multiple_choice_match(prediction: str, ground_truth: str, options: Optional[List[str]] = None) -> VerificationResult:
        """Multiple choice answer verification"""
        # Extract letter choices (A, B, C, D)
        pred_choice = re.search(r'\b[ABCD]\b', prediction.upper())
        truth_choice = re.search(r'\b[ABCD]\b', ground_truth.upper())
        
        if pred_choice and truth_choice:
            score = 1.0 if pred_choice.group() == truth_choice.group() else 0.0
        else:
            # Fallback to exact match
            score = 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0
        
        return VerificationResult(
            score=score,
            metrics={'multiple_choice': score},
            method='multiple_choice',
            details={
                'predicted_choice': pred_choice.group() if pred_choice else None,
                'ground_truth_choice': truth_choice.group() if truth_choice else None,
                'options': options
            }
        )
    
    @staticmethod
    def code_execution_match(prediction: str, ground_truth: str, test_cases: Optional[List[Dict]] = None) -> VerificationResult:
        """Code execution verification (mock implementation)"""
        # This is a simplified mock implementation
        # In production, this would execute code in a sandboxed environment
        
        # Check for basic code patterns
        code_patterns = [
            r'def\s+\w+\s*\(',  # Function definition
            r'class\s+\w+\s*:',  # Class definition
            r'return\s+',  # Return statement
            r'import\s+\w+',  # Import statement
        ]
        
        pred_has_code = any(re.search(pattern, prediction) for pattern in code_patterns)
        truth_has_code = any(re.search(pattern, ground_truth) for pattern in code_patterns)
        
        if pred_has_code and truth_has_code:
            # Simple similarity check for code structure
            score = calculate_similarity(prediction, ground_truth, method="token_overlap")
        else:
            score = 0.5  # Partial credit if one has code structure
        
        return VerificationResult(
            score=score,
            metrics={'code_execution': score, 'syntax_check': float(pred_has_code)},
            method='code_execution',
            details={
                'has_code_patterns': pred_has_code,
                'test_cases_passed': 0,  # Mock value
                'syntax_errors': []  # Mock value
            }
        )


class NonDeterministicVerifier:
    """Handles similarity-based verification for non-deterministic answers"""
    
    @staticmethod
    def token_overlap_similarity(prediction: str, ground_truth: str) -> VerificationResult:
        """Token overlap similarity"""
        score = calculate_similarity(prediction, ground_truth, method="token_overlap")
        
        return VerificationResult(
            score=score,
            metrics={'token_overlap': score},
            method='token_overlap_similarity',
            details={
                'prediction_tokens': prediction.split(),
                'ground_truth_tokens': ground_truth.split()
            }
        )
    
    @staticmethod
    def ngram_similarity(prediction: str, ground_truth: str, n: int = 2) -> VerificationResult:
        """N-gram similarity"""
        score = calculate_similarity(prediction, ground_truth, method="ngram")
        
        return VerificationResult(
            score=score,
            metrics={f'{n}gram_similarity': score},
            method='ngram_similarity',
            details={'n': n}
        )
    
    @staticmethod
    def rouge_l_similarity(prediction: str, ground_truth: str) -> VerificationResult:
        """ROUGE-L similarity"""
        score = calculate_similarity(prediction, ground_truth, method="rouge_l")
        
        return VerificationResult(
            score=score,
            metrics={'rouge_l': score},
            method='rouge_l_similarity',
            details={}
        )
    
    @staticmethod
    def semantic_similarity_mock(prediction: str, ground_truth: str) -> VerificationResult:
        """Mock semantic similarity (placeholder for real embeddings)"""
        # Use multiple similarity measures and average them
        similarities = calculate_multi_similarity(prediction, ground_truth)
        
        # Weighted average of different similarity measures
        weights = {
            'token_overlap': 0.3,
            'ngram': 0.2,
            'rouge_l': 0.3,
            'character_overlap': 0.2
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for method, weight in weights.items():
            if method in similarities:
                weighted_score += similarities[method] * weight
                total_weight += weight
        
        score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        return VerificationResult(
            score=score,
            metrics={'semantic_similarity': score, **similarities},
            method='semantic_similarity',
            details={'method': 'multi_similarity_average'}
        )


class MathVerifyVerifier:
    """Mathematical expression verifier using math-verify library"""
    
    @staticmethod
    def math_verify_match(prediction: str, ground_truth: str) -> VerificationResult:
        """Smart mathematical verification with multiple strategies"""
        
        # First try to extract and compare numbers for basic mathematical answers
        pred_numbers = extract_numbers(prediction)
        truth_numbers = extract_numbers(ground_truth)
        
        # Handle percentage and fraction cases
        if pred_numbers and truth_numbers:
            try:
                pred_val = float(pred_numbers[0])
                truth_val = float(truth_numbers[0])
                
                # Check for direct match with tolerance
                if abs(pred_val - truth_val) <= max(0.1, abs(truth_val) * 0.01):  # 1% or 0.1 tolerance
                    return VerificationResult(
                        score=1.0,
                        metrics={'numerical_match': 1.0},
                        method='smart_numerical',
                        details={'values_compared': (pred_val, truth_val), 'match_type': 'direct'}
                    )
                
                # Check for percentage/decimal conversion issues
                if abs(pred_val * 100 - truth_val) <= 0.1:  # pred is decimal, truth is percentage
                    return VerificationResult(
                        score=1.0,
                        metrics={'percentage_conversion': 1.0},
                        method='smart_percentage',
                        details={'conversion': f"{pred_val} → {pred_val * 100}% (matches {truth_val}%)"}
                    )
                
                if abs(truth_val * 100 - pred_val) <= 0.1:  # truth is decimal, pred is percentage
                    return VerificationResult(
                        score=1.0,
                        metrics={'decimal_conversion': 1.0},
                        method='smart_decimal',
                        details={'conversion': f"{truth_val} → {truth_val * 100}% (matches {pred_val}%)"}
                    )
                    
            except (ValueError, IndexError):
                pass
        
        # Try math-verify library if available
        if MATH_VERIFY_AVAILABLE:
            try:
                gold_parsed = parse(ground_truth)
                answer_parsed = parse(prediction)
                
                is_match = verify(gold_parsed, answer_parsed)
                if is_match:
                    return VerificationResult(
                        score=1.0,
                        metrics={'math_verify_match': 1.0},
                        method='math_verify',
                        details={'library_available': True, 'exact_match': True}
                    )
                    
            except Exception as e:
                pass
        
        # Fallback to numerical matching with higher tolerance
        fallback_result = DeterministicVerifier.numerical_match(prediction, ground_truth, tolerance=0.1)
        fallback_result.method = 'fallback_numerical'
        return fallback_result
    
    @staticmethod
    def latex_expression_match(prediction: str, ground_truth: str) -> VerificationResult:
        """Specialized LaTeX expression matching"""
        if not MATH_VERIFY_AVAILABLE:
            # Simple LaTeX pattern matching fallback
            pred_clean = re.sub(r'[{}$\\]', '', prediction).strip()
            truth_clean = re.sub(r'[{}$\\]', '', ground_truth).strip()
            score = 1.0 if pred_clean == truth_clean else 0.0
            
            return VerificationResult(
                score=score,
                metrics={'latex_match': score},
                method='latex_fallback',
                details={'library_available': False}
            )
        
        try:
            from math_verify import LatexExtractionConfig
            
            # Use only LaTeX extraction for specialized LaTeX matching
            gold_parsed = parse(ground_truth, extraction_config=[LatexExtractionConfig()])
            answer_parsed = parse(prediction, extraction_config=[LatexExtractionConfig()])
            
            is_match = verify(gold_parsed, answer_parsed)
            score = 1.0 if is_match else 0.0
            
            return VerificationResult(
                score=score,
                metrics={'latex_expression_match': score},
                method='latex_math_verify',
                details={
                    'gold_latex': str(gold_parsed),
                    'answer_latex': str(answer_parsed),
                    'extraction_config': 'LatexExtractionConfig',
                    'exact_match': is_match
                }
            )
            
        except Exception as e:
            # Fallback to basic LaTeX cleaning
            pred_clean = re.sub(r'[{}$\\]', '', prediction).strip()
            truth_clean = re.sub(r'[{}$\\]', '', ground_truth).strip()
            score = 1.0 if pred_clean == truth_clean else 0.0
            
            return VerificationResult(
                score=score,
                metrics={'latex_match': score},
                method='latex_fallback',
                details={
                    'math_verify_error': str(e),
                    'fallback_used': True
                }
            )
    
    @staticmethod
    def expression_match(prediction: str, ground_truth: str) -> VerificationResult:
        """Plain mathematical expression matching"""
        if not MATH_VERIFY_AVAILABLE:
            return DeterministicVerifier.numerical_match(prediction, ground_truth)
        
        try:
            from math_verify import ExprExtractionConfig
            
            # Use only expression extraction for plain mathematical expressions
            gold_parsed = parse(ground_truth, extraction_config=[ExprExtractionConfig()])
            answer_parsed = parse(prediction, extraction_config=[ExprExtractionConfig()])
            
            is_match = verify(gold_parsed, answer_parsed)
            score = 1.0 if is_match else 0.0
            
            return VerificationResult(
                score=score,
                metrics={'expression_match': score},
                method='expression_math_verify',
                details={
                    'gold_expr': str(gold_parsed),
                    'answer_expr': str(answer_parsed),
                    'extraction_config': 'ExprExtractionConfig',
                    'exact_match': is_match
                }
            )
            
        except Exception as e:
            # Fallback to numerical matching
            fallback_result = DeterministicVerifier.numerical_match(prediction, ground_truth)
            fallback_result.details.update({
                'math_verify_error': str(e),
                'fallback_used': True
            })
            return fallback_result


class LLMJudgeVerifier:
    """Uses LLM to judge quality of responses"""
    
    def __init__(self, llm=None):
        self.llm = llm or MockLLMJudge()
    
    def evaluate_quality(self, prediction: str, ground_truth: str, 
                        criteria: List[str] = ['relevance', 'accuracy', 'completeness']) -> VerificationResult:
        """Evaluate response quality using LLM judge"""
        
        prompt = self._create_judge_prompt(prediction, ground_truth, criteria)
        judgment = self.llm.judge(prompt)
        scores = self._parse_judgment(judgment, criteria)
        
        # Overall score is average of criteria scores
        overall_score = sum(scores.values()) / len(scores) if scores else 0.5
        
        return VerificationResult(
            score=overall_score,
            metrics=scores,
            method='llm_judge',
            details={
                'criteria': criteria,
                'judgment': judgment,
                'llm_model': getattr(self.llm, 'model_name', 'unknown')
            }
        )
    
    def _create_judge_prompt(self, prediction: str, ground_truth: str, criteria: List[str]) -> str:
        """Create prompt for LLM judge"""
        criteria_desc = ', '.join(criteria)
        
        return f"""
Please evaluate the following response based on {criteria_desc}.

Ground Truth Answer: {ground_truth}

Predicted Answer: {prediction}

For each criterion, provide a score from 0.0 to 1.0:
{chr(10).join([f'- {criterion}: [score]' for criterion in criteria])}

Provide your evaluation in the following format:
{chr(10).join([f'{criterion}: [score] - [brief explanation]' for criterion in criteria])}
        """.strip()
    
    def _parse_judgment(self, judgment: str, criteria: List[str]) -> Dict[str, float]:
        """Parse LLM judgment into scores"""
        scores = {}
        
        for criterion in criteria:
            # Look for pattern: "criterion: score"
            pattern = rf'{criterion}:\s*([0-9.]+)'
            match = re.search(pattern, judgment, re.IGNORECASE)
            
            if match:
                try:
                    score = float(match.group(1))
                    scores[criterion] = min(1.0, max(0.0, score))
                except ValueError:
                    scores[criterion] = 0.5  # Default score
            else:
                scores[criterion] = 0.5  # Default score
        
        return scores


class MockLLMJudge:
    """Mock LLM judge for testing"""
    
    def __init__(self):
        self.model_name = "MockJudge-v1"
    
    def judge(self, prompt: str) -> str:
        """Generate mock judgment based on prompt analysis"""
        # Simple heuristic-based mock judgment
        if 'mathematical' in prompt.lower() or 'calculate' in prompt.lower():
            return """
relevance: 0.9 - Response addresses the mathematical question
accuracy: 0.8 - Calculation appears correct
completeness: 0.7 - Could provide more detailed steps
            """.strip()
        
        elif 'code' in prompt.lower() or 'function' in prompt.lower():
            return """
relevance: 0.85 - Code addresses the programming task
accuracy: 0.75 - Implementation looks reasonable
completeness: 0.8 - Includes necessary components
            """.strip()
        
        else:
            return """
relevance: 0.8 - Response is related to the question
accuracy: 0.7 - Information appears reasonable
completeness: 0.75 - Covers main points adequately
            """.strip()


class VerificationOrchestrator:
    """Orchestrates different verification methods based on evaluation type"""
    
    def __init__(self, corpus_text: str = ""):
        self.deterministic_verifier = DeterministicVerifier()
        self.non_deterministic_verifier = NonDeterministicVerifier()
        self.llm_judge = LLMJudgeVerifier()
        self.math_verify_verifier = MathVerifyVerifier()
        
        # Import and initialize domain-specific verifier
        try:
            from .domain_verification import DomainSpecificVerifier
            self.domain_verifier = DomainSpecificVerifier(corpus_text)
        except ImportError:
            self.domain_verifier = None
    
    def verify(self, prediction: str, ground_truth: str, eval_type: str, 
              options: Optional[List[str]] = None, question: str = "") -> VerificationResult:
        """Main verification method that routes to appropriate verifier"""
        
        # Use domain-specific verification for better accuracy
        if self.domain_verifier and eval_type in ['mathematical', 'factual_qa', 'domain_knowledge']:
            # Determine if question is mathematical based on content, not just eval_type
            has_numbers = bool(extract_numbers(question) or extract_numbers(ground_truth))
            has_math_keywords = any(keyword in question.lower() for keyword in 
                                  ['calculate', 'compute', 'volume', 'percentage', 'ratio', 'years', 'divide', 'multiply'])
            
            if has_numbers and has_math_keywords:
                # This is a mathematical question regardless of eval_type
                print(f"[DEBUG] Using domain mathematical verification for: {question[:50]}...")
                domain_result = self.domain_verifier.verify_mathematical_reasoning(
                    prediction, ground_truth, question
                )
            else:
                print(f"[DEBUG] Using domain factual verification for: {question[:50]}...")
                domain_result = self.domain_verifier.verify_factual_knowledge(
                    prediction, ground_truth, question
                )
            
            # Convert domain result to standard VerificationResult
            return VerificationResult(
                score=domain_result.score,
                metrics={
                    'overall_score': domain_result.score,
                    'reasoning_quality': domain_result.reasoning_quality,
                    'factual_accuracy': domain_result.factual_accuracy,
                    'mathematical_accuracy': domain_result.mathematical_accuracy,
                    'contextual_relevance': domain_result.contextual_relevance
                },
                method=domain_result.method_used,
                details=domain_result.details
            )
        
        # Fallback to original verification methods
        if eval_type in ['mathematical', 'math_expression', 'latex_math', 'factual_qa', 'multiple_choice', 'domain_knowledge']:
            if eval_type == 'mathematical':
                # Use math-verify for enhanced mathematical verification
                return self.math_verify_verifier.math_verify_match(prediction, ground_truth)
            elif eval_type == 'math_expression':
                # Use math-verify for plain mathematical expressions
                return self.math_verify_verifier.expression_match(prediction, ground_truth)
            elif eval_type == 'latex_math':
                # Use math-verify for LaTeX mathematical expressions
                return self.math_verify_verifier.latex_expression_match(prediction, ground_truth)
            elif eval_type == 'multiple_choice':
                return self.deterministic_verifier.multiple_choice_match(prediction, ground_truth, options)
            else:
                return self.deterministic_verifier.exact_match(prediction, ground_truth)
        
        elif eval_type == 'code_generation':
            return self.deterministic_verifier.code_execution_match(prediction, ground_truth)
        
        elif eval_type == 'numerical':
            return self.deterministic_verifier.numerical_match(prediction, ground_truth)
        
        elif eval_type == 'exact':
            return self.deterministic_verifier.exact_match(prediction, ground_truth)
        
        elif eval_type == 'similarity':
            return self.non_deterministic_verifier.semantic_similarity_mock(prediction, ground_truth)
        
        elif eval_type == 'domain_factual':
            # Use similarity for domain factual knowledge with higher tolerance
            result = self.non_deterministic_verifier.semantic_similarity_mock(prediction, ground_truth)
            # Boost scores for domain factual to be more lenient
            result.score = min(1.0, result.score * 1.5)  # 50% boost
            result.method = 'domain_factual_similarity'
            return result
        
        elif eval_type in ['summarization', 'translation', 'reading_comprehension']:
            return self.non_deterministic_verifier.semantic_similarity_mock(prediction, ground_truth)
        
        elif eval_type == 'creative_writing':
            return self.llm_judge.evaluate_quality(prediction, ground_truth)
        
        else:
            # Default to exact match
            return self.deterministic_verifier.exact_match(prediction, ground_truth)
    
    def verify_batch(self, predictions: List[str], ground_truths: List[str], 
                    eval_type: str, options_list: Optional[List[List[str]]] = None) -> List[VerificationResult]:
        """Verify multiple predictions at once"""
        results = []
        
        for i, (pred, truth) in enumerate(zip(predictions, ground_truths)):
            options = options_list[i] if options_list else None
            result = self.verify(pred, truth, eval_type, options)
            results.append(result)
        
        return results
    
    def compute_aggregate_metrics(self, verification_results: List[VerificationResult]) -> Dict[str, Any]:
        """Compute aggregate metrics from verification results"""
        if not verification_results:
            return {}
        
        scores = [result.score for result in verification_results]
        
        # Collect all metric values
        all_metrics = {}
        for result in verification_results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Compute aggregates
        aggregates = {
            'mean_score': sum(scores) / len(scores),
            'median_score': sorted(scores)[len(scores) // 2],
            'min_score': min(scores),
            'max_score': max(scores),
            'num_samples': len(scores)
        }
        
        # Add metric-specific aggregates
        for metric_name, values in all_metrics.items():
            aggregates[f'{metric_name}_mean'] = sum(values) / len(values)
            aggregates[f'{metric_name}_std'] = self._calculate_std(values)
        
        return aggregates
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)


if __name__ == "__main__":
    # Test the verification system
    orchestrator = VerificationOrchestrator()
    
    # Test cases
    test_cases = [
        ("2 + 2 = 4", "4", "mathematical"),
        ("${1,3} \\cup {2,4}$", "${1,2,3,4}$", "latex_math"),
        ("1/2", "0.5", "math_expression"),
        ("Paris", "Paris is the capital of France", "factual_qa"),
        ("The story was about adventure", "A tale of adventure and discovery", "creative_writing"),
        ("def hello(): print('hi')", "def greet(): print('hello')", "code_generation")
    ]
    
    print("Testing Verification System:")
    print("=" * 50)
    
    for pred, truth, eval_type in test_cases:
        result = orchestrator.verify(pred, truth, eval_type)
        print(f"\nEval Type: {eval_type}")
        print(f"Prediction: {pred}")
        print(f"Ground Truth: {truth}")
        print(f"Score: {result.score:.3f}")
        print(f"Method: {result.method}")
        print(f"Metrics: {result.metrics}")
    
    # Test aggregate metrics
    results = [orchestrator.verify(pred, truth, eval_type) for pred, truth, eval_type in test_cases]
    aggregates = orchestrator.compute_aggregate_metrics(results)
    
    print(f"\nAggregate Metrics:")
    for metric, value in aggregates.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")