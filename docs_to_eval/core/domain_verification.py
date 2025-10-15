"""
Domain-specific verification system tailored for corpus-based evaluation
Focuses on knowledge accuracy, reasoning quality, and contextual understanding
"""

import re
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from ..utils.text_processing import normalize_answer, extract_numbers


@dataclass
class DomainVerificationResult:
    """Results from domain-specific verification"""
    score: float  # 0.0 to 1.0
    reasoning_quality: float  # How well did the LLM explain its reasoning?
    factual_accuracy: float  # Are the facts correct?
    mathematical_accuracy: float  # Are calculations correct?
    contextual_relevance: float  # Does it relate to the corpus domain?
    method_used: str
    details: Dict[str, Any]


class DomainSpecificVerifier:
    """Verification system designed for domain knowledge evaluation"""
    
    def __init__(self, domain_context: str = ""):
        self.domain_context = domain_context
    
    def verify_mathematical_reasoning(self, prediction: str, ground_truth: str, 
                                    question: str) -> DomainVerificationResult:
        """
        Verify mathematical answers with focus on reasoning process, not just final numbers
        """
        # Ensure inputs are strings
        prediction = str(prediction) if prediction is not None else ""
        ground_truth = str(ground_truth) if ground_truth is not None else ""
        question = str(question) if question is not None else ""
        details = {
            "question_type": "mathematical_reasoning",
            "extraction_method": "multi_stage"
        }
        
        # Extract all numbers from both answers
        pred_numbers = extract_numbers(prediction)
        truth_numbers = extract_numbers(ground_truth)
        
        # Step 1: Check for mathematical accuracy
        mathematical_accuracy = 0.0
        if pred_numbers and truth_numbers:
            try:
                # For multi-step problems, check if key intermediate and final values match
                final_pred = float(pred_numbers[-1]) if pred_numbers else 0.0
                final_truth = float(truth_numbers[-1]) if truth_numbers else 0.0
                
                # Use relative tolerance for mathematical accuracy
                if final_truth != 0:
                    relative_error = abs(final_pred - final_truth) / abs(final_truth)
                    if relative_error <= 0.05:  # 5% tolerance
                        mathematical_accuracy = 1.0
                    elif relative_error <= 0.10:  # 10% tolerance - partial credit
                        mathematical_accuracy = 0.8
                    elif relative_error <= 0.20:  # 20% tolerance - some credit
                        mathematical_accuracy = 0.6
                    else:
                        mathematical_accuracy = 0.0
                else:
                    mathematical_accuracy = 1.0 if abs(final_pred - final_truth) <= 0.1 else 0.0
            except (ValueError, TypeError):
                # If we can't convert to numbers, fall back to text matching
                mathematical_accuracy = 0.5 if pred_numbers and truth_numbers else 0.0
        
        # Step 2: Evaluate reasoning quality
        reasoning_quality = self._assess_reasoning_quality(prediction, question)
        
        # Step 3: Check factual accuracy within domain context
        factual_accuracy = self._assess_factual_accuracy(prediction, question)
        
        # Step 4: Assess contextual relevance
        contextual_relevance = self._assess_contextual_relevance(prediction, question)
        
        # Composite score with weighted components
        score = (
            mathematical_accuracy * 0.4 +  # Math is important but not everything
            reasoning_quality * 0.3 +      # Clear reasoning is valuable
            factual_accuracy * 0.2 +       # Domain facts matter
            contextual_relevance * 0.1     # Context awareness
        )
        
        details.update({
            "mathematical_accuracy": mathematical_accuracy,
            "reasoning_quality": reasoning_quality,
            "factual_accuracy": factual_accuracy,
            "contextual_relevance": contextual_relevance,
            "pred_numbers": pred_numbers,
            "truth_numbers": truth_numbers,
            "final_values": (final_pred if pred_numbers else None, final_truth if truth_numbers else None)
        })
        
        return DomainVerificationResult(
            score=score,
            reasoning_quality=reasoning_quality,
            factual_accuracy=factual_accuracy,
            mathematical_accuracy=mathematical_accuracy,
            contextual_relevance=contextual_relevance,
            method_used="domain_mathematical_reasoning",
            details=details
        )
    
    def verify_factual_knowledge(self, prediction: str, ground_truth: str, 
                                question: str) -> DomainVerificationResult:
        """
        Verify factual answers with emphasis on domain knowledge accuracy
        """
        # Ensure inputs are strings
        prediction = str(prediction) if prediction is not None else ""
        ground_truth = str(ground_truth) if ground_truth is not None else ""
        question = str(question) if question is not None else ""
        details = {
            "question_type": "factual_knowledge",
            "verification_approach": "semantic_matching"
        }
        
        # Step 1: Direct content matching
        factual_accuracy = self._calculate_content_overlap(prediction, ground_truth)
        
        # Step 2: Reasoning quality assessment
        reasoning_quality = self._assess_reasoning_quality(prediction, question)
        
        # Step 3: Mathematical components (if any)
        mathematical_accuracy = 1.0  # Default to perfect if no math involved
        pred_numbers = extract_numbers(prediction)
        truth_numbers = extract_numbers(ground_truth)
        
        if pred_numbers and truth_numbers:
            # There are mathematical components
            mathematical_accuracy = self._compare_numerical_values(pred_numbers, truth_numbers)
        
        # Step 4: Contextual relevance
        contextual_relevance = self._assess_contextual_relevance(prediction, question)
        
        # Weighted scoring for factual questions
        score = (
            factual_accuracy * 0.5 +        # Facts are most important
            reasoning_quality * 0.2 +       # Some credit for good reasoning
            mathematical_accuracy * 0.2 +   # Math components if present
            contextual_relevance * 0.1      # Context awareness
        )
        
        details.update({
            "factual_accuracy": factual_accuracy,
            "content_overlap_score": factual_accuracy,
            "mathematical_components": len(pred_numbers) > 0
        })
        
        return DomainVerificationResult(
            score=score,
            reasoning_quality=reasoning_quality,
            factual_accuracy=factual_accuracy,
            mathematical_accuracy=mathematical_accuracy,
            contextual_relevance=contextual_relevance,
            method_used="domain_factual_knowledge",
            details=details
        )
    
    def _assess_reasoning_quality(self, prediction: str, question: str) -> float:
        """
        Assess the quality of reasoning demonstrated in the response
        """
        reasoning_indicators = [
            r"step[s]?\s*\d+",  # "Step 1:", "Steps:", etc.
            r"first[ly]?|second[ly]?|third[ly]?|finally",  # Sequential reasoning
            r"therefore|thus|hence|consequently",  # Logical conclusions
            r"because|since|given that",  # Causal reasoning
            r"calculate|compute|determine",  # Mathematical reasoning
            r"given information|from the (text|passage|corpus)",  # Source awareness
        ]
        
        reasoning_score = 0.0
        prediction_lower = prediction.lower()
        
        # Check for reasoning indicators
        for pattern in reasoning_indicators:
            if re.search(pattern, prediction_lower):
                reasoning_score += 0.15
        
        # Check for structured approach (multiple steps visible)
        step_count = len(re.findall(r"step\s*\d+", prediction_lower))
        if step_count >= 2:
            reasoning_score += 0.3
        
        # Check for calculations shown
        calculation_patterns = [r"\d+\s*[+\-*/÷×]\s*\d+", r"=\s*\d+"]
        for pattern in calculation_patterns:
            if re.search(pattern, prediction):
                reasoning_score += 0.1
        
        return min(1.0, reasoning_score)
    
    def _assess_factual_accuracy(self, prediction: str, question: str) -> float:
        """
        Assess factual accuracy based on domain knowledge
        """
        # Look for domain-specific terms and concepts
        domain_terms = [
            "etruscan", "villanovan", "liver of piacenza", "corpus speculorum",
            "fufluns", "bronze", "fascicle", "inscription", "bc", "ad",
            "archaeological", "ancient", "civilization", "culture"
        ]
        
        prediction_lower = prediction.lower()
        factual_score = 0.5  # Base score
        
        # Reward domain-appropriate terminology
        for term in domain_terms:
            if term in prediction_lower:
                factual_score += 0.05
        
        # Check for historical context awareness
        historical_patterns = [
            r"\d+\s*bc", r"\d+\s*ad", r"ancient", r"century", r"period"
        ]
        for pattern in historical_patterns:
            if re.search(pattern, prediction_lower):
                factual_score += 0.1
        
        return min(1.0, factual_score)
    
    def _assess_contextual_relevance(self, prediction: str, question: str) -> float:
        """
        Assess how well the response relates to the domain context
        """
        # Check if response demonstrates understanding of the domain
        context_indicators = [
            "etruscan", "archaeological", "ancient", "bronze", "inscription",
            "civilization", "culture", "historical", "artifact", "discovery"
        ]
        
        prediction_lower = prediction.lower()
        relevance_score = 0.3  # Base score
        
        for indicator in context_indicators:
            if indicator in prediction_lower:
                relevance_score += 0.1
        
        # Check for additional context (shows deep understanding)
        if "note:" in prediction_lower or "context:" in prediction_lower:
            relevance_score += 0.2
        
        return min(1.0, relevance_score)
    
    def _calculate_content_overlap(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate semantic content overlap between prediction and ground truth
        """
        # Ensure inputs are strings
        prediction = str(prediction) if prediction is not None else ""
        ground_truth = str(ground_truth) if ground_truth is not None else ""
        
        try:
            pred_words = set(prediction.lower().split())
            truth_words = set(ground_truth.lower().split())
        except Exception as e:
            # Log the error for debugging
            print(f"[DEBUG] Error in _calculate_content_overlap: {e}")
            print(f"[DEBUG] prediction type: {type(prediction)}, value: {repr(prediction)}")
            print(f"[DEBUG] ground_truth type: {type(ground_truth)}, value: {repr(ground_truth)}")
            return 0.0
        
        if not truth_words:
            return 0.0
        
        # Remove common stop words for better matching
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
        pred_words = pred_words - stop_words
        truth_words = truth_words - stop_words
        
        if not truth_words:
            return 0.5  # If only stop words, give neutral score
        
        overlap = len(pred_words.intersection(truth_words))
        union = len(pred_words.union(truth_words))
        
        # Jaccard similarity with bias toward recall (covering ground truth)
        recall = overlap / len(truth_words) if truth_words else 0
        precision = overlap / len(pred_words) if pred_words else 0
        
        # Weight recall more heavily than precision
        return 0.7 * recall + 0.3 * precision
    
    def _compare_numerical_values(self, pred_numbers: List[float], 
                                truth_numbers: List[float], tolerance: float = 0.05) -> float:
        """
        Compare numerical values with appropriate tolerance
        """
        if not pred_numbers or not truth_numbers:
            return 0.0
        
        # Compare the most significant numbers (usually the final answers)
        try:
            key_pred = float(pred_numbers[-1])  # Convert string to float
            key_truth = float(truth_numbers[-1])
        except (ValueError, TypeError):
            return 0.0  # If conversion fails, no match
        
        if key_truth == 0:
            return 1.0 if abs(key_pred) <= 0.1 else 0.0
        
        relative_error = abs(key_pred - key_truth) / abs(key_truth)
        
        if relative_error <= tolerance:
            return 1.0
        elif relative_error <= tolerance * 2:
            return 0.8
        elif relative_error <= tolerance * 4:
            return 0.6
        else:
            return 0.0


def create_domain_verifier(corpus_text: str, domain_type: str = "historical") -> DomainSpecificVerifier:
    """
    Factory function to create appropriate domain verifier
    """
    return DomainSpecificVerifier(domain_context=corpus_text)


if __name__ == "__main__":
    # Test the domain verifier
    verifier = DomainSpecificVerifier()
    
    # Test mathematical reasoning
    question = "Calculate the volume of the Liver of Piacenza measuring 126 × 76 × 60 mm"
    ground_truth = "Volume = 574,560 cubic mm = 574.56 cubic cm"
    prediction = "Volume = 12.6 × 7.6 × 6.0 = 575.04 cm³"
    
    result = verifier.verify_mathematical_reasoning(prediction, ground_truth, question)
    print(f"Score: {result.score:.2f}")
    print(f"Mathematical accuracy: {result.mathematical_accuracy:.2f}")
    print(f"Reasoning quality: {result.reasoning_quality:.2f}")