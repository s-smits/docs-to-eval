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
        Verify mathematical answers with focus on reasoning process, not just final numbers.
        Enhanced for complex reasoning questions requiring synthesis and inference.
        """
        details = {
            "question_type": "mathematical_reasoning",
            "extraction_method": "multi_stage_enhanced"
        }
        
        # Extract all numbers from both answers
        pred_numbers = extract_numbers(prediction)
        truth_numbers = extract_numbers(ground_truth)
        
        # Step 1: Check for mathematical accuracy (enhanced for complex questions)
        mathematical_accuracy = 0.0
        final_pred = None
        final_truth = None
        
        if pred_numbers and truth_numbers:
            try:
                # For multi-step problems, check if key intermediate and final values match
                final_pred = float(pred_numbers[-1]) if pred_numbers else 0.0
                final_truth = float(truth_numbers[-1]) if truth_numbers else 0.0
                
                # Enhanced tolerance system for sophisticated questions
                if final_truth != 0:
                    relative_error = abs(final_pred - final_truth) / abs(final_truth)
                    if relative_error <= 0.02:  # 2% tolerance for exact calculations
                        mathematical_accuracy = 1.0
                    elif relative_error <= 0.05:  # 5% tolerance - high accuracy
                        mathematical_accuracy = 0.95
                    elif relative_error <= 0.10:  # 10% tolerance - good accuracy
                        mathematical_accuracy = 0.85
                    elif relative_error <= 0.20:  # 20% tolerance - partial credit
                        mathematical_accuracy = 0.70
                    elif relative_error <= 0.50:  # 50% tolerance - some understanding
                        mathematical_accuracy = 0.40
                    else:
                        mathematical_accuracy = 0.0
                else:
                    mathematical_accuracy = 1.0 if abs(final_pred - final_truth) <= 0.1 else 0.0
                    
                # Check for intermediate calculations accuracy
                if len(pred_numbers) > 1 and len(truth_numbers) > 1:
                    intermediate_matches = 0
                    for p_num in pred_numbers[:-1]:  # Exclude final answer
                        for t_num in truth_numbers[:-1]:
                            try:
                                if abs(float(p_num) - float(t_num)) / max(abs(float(t_num)), 1) <= 0.10:
                                    intermediate_matches += 1
                                    break
                            except (ValueError, TypeError):
                                continue
                    
                    # Bonus for showing correct intermediate steps
                    if intermediate_matches > 0:
                        mathematical_accuracy = min(1.0, mathematical_accuracy + 0.05 * intermediate_matches)
                        details["intermediate_steps_correct"] = intermediate_matches
                        
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
    
    def verify_advanced_reasoning(self, prediction: str, ground_truth: str, 
                                question: str, complexity_layer: str = "synthesis") -> DomainVerificationResult:
        """
        Verify advanced reasoning questions with sophisticated multi-layered analysis.
        Designed for questions requiring synthesis, inference, ambiguity handling, and extrapolation.
        """
        details = {
            "question_type": "advanced_reasoning",
            "complexity_layer": complexity_layer,
            "verification_approach": "multi_dimensional_enhanced"
        }
        
        # Step 1: Enhanced reasoning quality assessment
        reasoning_quality = self._assess_reasoning_quality(prediction, question)
        
        # Step 2: Complexity-specific assessment
        complexity_score = self._assess_complexity_handling(prediction, complexity_layer)
        
        # Step 3: Synthesis and integration assessment
        synthesis_score = self._assess_synthesis_quality(prediction, question)
        
        # Step 4: Content accuracy (more flexible for sophisticated questions)
        content_accuracy = self._assess_advanced_content_accuracy(prediction, ground_truth, question)
        
        # Step 5: Mathematical components (if present)
        mathematical_accuracy = 1.0  # Default to perfect if no math
        pred_numbers = extract_numbers(prediction)
        truth_numbers = extract_numbers(ground_truth)
        
        if pred_numbers and truth_numbers:
            mathematical_accuracy = self._compare_numerical_values(pred_numbers, truth_numbers, tolerance=0.15)
        
        # Step 6: Contextual and domain integration
        contextual_relevance = self._assess_contextual_relevance(prediction, question)
        
        # Adaptive scoring based on complexity layer
        if complexity_layer == "synthesis":
            # Emphasize integration and connection-making
            score = (
                reasoning_quality * 0.25 +      # Clear reasoning process
                synthesis_score * 0.30 +       # Synthesis across sources/concepts
                content_accuracy * 0.25 +      # Content accuracy
                complexity_score * 0.10 +      # Complexity handling
                contextual_relevance * 0.10    # Domain awareness
            )
        elif complexity_layer == "inference":
            # Emphasize logical reasoning and evidence-based conclusions
            score = (
                reasoning_quality * 0.35 +      # Strong reasoning is critical
                content_accuracy * 0.25 +      # Accurate inferences
                complexity_score * 0.20 +      # Inference complexity
                mathematical_accuracy * 0.10 + # Math if present
                contextual_relevance * 0.10    # Domain grounding
            )
        elif complexity_layer == "ambiguity":
            # Emphasize handling uncertainty and alternative interpretations
            score = (
                complexity_score * 0.30 +      # Ambiguity handling
                reasoning_quality * 0.25 +     # Clear reasoning under uncertainty
                content_accuracy * 0.20 +      # Reasonable interpretations
                synthesis_score * 0.15 +       # Multiple perspectives
                contextual_relevance * 0.10    # Domain awareness
            )
        elif complexity_layer == "extrapolation":
            # Emphasize extending beyond given information
            score = (
                reasoning_quality * 0.30 +      # Logical extension
                complexity_score * 0.25 +      # Extrapolation quality
                content_accuracy * 0.20 +      # Grounded extrapolation
                mathematical_accuracy * 0.15 + # Quantitative projections
                contextual_relevance * 0.10    # Domain-appropriate extensions
            )
        else:
            # Default balanced scoring
            score = (
                reasoning_quality * 0.30 +
                content_accuracy * 0.25 +
                complexity_score * 0.20 +
                synthesis_score * 0.15 +
                contextual_relevance * 0.10
            )
        
        details.update({
            "reasoning_quality": reasoning_quality,
            "complexity_score": complexity_score,
            "synthesis_score": synthesis_score,
            "content_accuracy": content_accuracy,
            "mathematical_accuracy": mathematical_accuracy,
            "contextual_relevance": contextual_relevance,
            "scoring_weights": {
                "reasoning": 0.30, "content": 0.25, "complexity": 0.20, 
                "synthesis": 0.15, "context": 0.10
            }
        })
        
        return DomainVerificationResult(
            score=score,
            reasoning_quality=reasoning_quality,
            factual_accuracy=content_accuracy,
            mathematical_accuracy=mathematical_accuracy,
            contextual_relevance=contextual_relevance,
            method_used=f"advanced_reasoning_{complexity_layer}",
            details=details
        )
    
    def _assess_reasoning_quality(self, prediction: str, question: str) -> float:
        """
        Enhanced assessment of reasoning quality for sophisticated questions requiring
        synthesis, inference, ambiguity handling, and extrapolation
        """
        reasoning_indicators = [
            r"step[s]?\s*\d+",  # "Step 1:", "Steps:", etc.
            r"first[ly]?|second[ly]?|third[ly]?|finally",  # Sequential reasoning
            r"therefore|thus|hence|consequently",  # Logical conclusions
            r"because|since|given that",  # Causal reasoning
            r"calculate|compute|determine",  # Mathematical reasoning
            r"given information|from the (text|passage|corpus)",  # Source awareness
        ]
        
        # Advanced reasoning indicators for sophisticated questions
        advanced_reasoning = [
            r"synthesis|synthesiz[e|ing]",  # Synthesis across sources
            r"infer[ence]?|implied?|suggest[s]?",  # Inference beyond facts
            r"assumption|assuming|if we assume",  # Handling assumptions
            r"compare[d]?|contrast[ed]?|versus|vs\.",  # Comparative analysis
            r"context[ual]?|considering|taking into account",  # Contextual awareness
            r"methodology|approach|method",  # Methodological reasoning
            r"alternative[ly]?|however|on the other hand",  # Considering alternatives
            r"extrapolat[e|ing]|project[ing]?|trend",  # Extrapolation
            r"interdisciplinary|cross-field|multiple perspectives",  # Integration
            r"uncertainty|ambiguity|unclear|may be|could be"  # Handling ambiguity
        ]
        
        # Complexity layer indicators
        complexity_indicators = [
            r"what-if|hypothetical|scenario",  # Scenario analysis
            r"implications?|consequences?|significance",  # Broader implications
            r"recalculate|re-evaluate|adjust[ed]?",  # Dynamic reasoning
            r"factor[s]?\s+in|account\s+for|consider[ing]?",  # Multi-factor analysis
            r"evidence suggests?|data indicates?|research shows?",  # Evidence-based reasoning
        ]
        
        reasoning_score = 0.0
        prediction_lower = prediction.lower()
        
        # Base reasoning indicators (0.1 each, max 0.6)
        for pattern in reasoning_indicators:
            if re.search(pattern, prediction_lower):
                reasoning_score += 0.1
        
        # Advanced reasoning indicators (0.15 each, max 0.45)
        advanced_count = 0
        for pattern in advanced_reasoning:
            if re.search(pattern, prediction_lower):
                reasoning_score += 0.15
                advanced_count += 1
                if advanced_count >= 3:  # Cap at 3 for balance
                    break
        
        # Complexity handling (0.1 each, max 0.3) 
        complexity_count = 0
        for pattern in complexity_indicators:
            if re.search(pattern, prediction_lower):
                reasoning_score += 0.1
                complexity_count += 1
                if complexity_count >= 3:
                    break
        
        # Check for structured multi-step approach
        step_count = len(re.findall(r"step\s*\d+", prediction_lower))
        if step_count >= 2:
            reasoning_score += 0.2
        elif step_count >= 4:  # Extra credit for very detailed reasoning
            reasoning_score += 0.3
        
        # Check for calculations and formulas shown
        calculation_patterns = [r"\d+\s*[+\-*/÷×]\s*\d+", r"=\s*\d+", r"formula|equation"]
        calculation_count = 0
        for pattern in calculation_patterns:
            if re.search(pattern, prediction):
                calculation_count += 1
                reasoning_score += 0.05
        
        # Bonus for connecting multiple concepts (synthesis)
        connection_words = ["connects?", "relates?", "links?", "integrat[e|ing]", "combines?"]
        for word in connection_words:
            if re.search(word, prediction_lower):
                reasoning_score += 0.1
                break
        
        # Check for domain-specific sophisticated language
        sophisticated_terms = [
            "archaeological", "methodological", "interdisciplinary", "chronological",
            "contextual", "comparative", "analytical", "interpretive", "theoretical"
        ]
        sophisticated_count = sum(1 for term in sophisticated_terms if term in prediction_lower)
        if sophisticated_count >= 2:
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
        pred_words = set(prediction.lower().split())
        truth_words = set(ground_truth.lower().split())
        
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
    
    def _compare_numerical_values(self, pred_numbers: List[str], 
                                truth_numbers: List[str], tolerance: float = 0.05) -> float:
        """
        Compare numerical values with appropriate tolerance
        """
        if not pred_numbers or not truth_numbers:
            return 0.0
        
        try:
            # Compare the most significant numbers (usually the final answers)
            key_pred = float(pred_numbers[-1])  # Last number is usually the final answer
            key_truth = float(truth_numbers[-1])
            
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
        
        except (ValueError, TypeError, IndexError):
            # If we can't convert to numbers, return 0 (no match)
            return 0.0
    
    def _assess_complexity_handling(self, prediction: str, complexity_layer: str) -> float:
        """
        Assess how well the response handles the specific complexity layer
        """
        prediction_lower = prediction.lower()
        score = 0.0
        
        if complexity_layer == "synthesis":
            # Look for evidence of synthesis across sources/concepts
            synthesis_indicators = [
                r"combin[e|ing]", r"integrat[e|ing]", r"connect[s|ing]", 
                r"synthesis", r"together", r"across", r"multiple"
            ]
            for pattern in synthesis_indicators:
                if re.search(pattern, prediction_lower):
                    score += 0.2
            
        elif complexity_layer == "inference":
            # Look for inferential reasoning
            inference_indicators = [
                r"infer[s]?", r"suggest[s]?", r"implies?", r"indicates?",
                r"conclude[s]?", r"deduce[s]?", r"evidence points?"
            ]
            for pattern in inference_indicators:
                if re.search(pattern, prediction_lower):
                    score += 0.25
        
        elif complexity_layer == "ambiguity":
            # Look for acknowledgment of uncertainty or multiple interpretations
            ambiguity_indicators = [
                r"uncertain", r"unclear", r"ambiguous", r"multiple interpretations?",
                r"could be", r"might be", r"possibly", r"alternative[ly]?",
                r"however", r"on the other hand", r"but"
            ]
            for pattern in ambiguity_indicators:
                if re.search(pattern, prediction_lower):
                    score += 0.2
        
        elif complexity_layer == "extrapolation":
            # Look for extension beyond given information
            extrapolation_indicators = [
                r"extrapolat[e|ing]", r"project[ing]?", r"extend[ing]?",
                r"beyond", r"future", r"trend", r"pattern", r"likely"
            ]
            for pattern in extrapolation_indicators:
                if re.search(pattern, prediction_lower):
                    score += 0.25
        
        return min(1.0, score)
    
    def _assess_synthesis_quality(self, prediction: str, question: str) -> float:
        """
        Assess the quality of synthesis and integration in the response
        """
        prediction_lower = prediction.lower()
        score = 0.0
        
        # Look for explicit synthesis language
        synthesis_words = [
            "synthesis", "integrate", "combine", "connect", "relate",
            "together", "across", "between", "link", "merge"
        ]
        synthesis_count = sum(1 for word in synthesis_words if word in prediction_lower)
        score += min(0.3, synthesis_count * 0.1)
        
        # Look for comparative analysis
        comparison_words = ["compare", "contrast", "versus", "vs", "difference", "similar"]
        if any(word in prediction_lower for word in comparison_words):
            score += 0.2
        
        # Look for multiple perspective indicators
        perspective_words = ["perspective", "viewpoint", "approach", "angle", "aspect"]
        if any(word in prediction_lower for word in perspective_words):
            score += 0.15
        
        # Look for temporal integration (across time periods)
        temporal_words = ["period", "era", "century", "over time", "throughout", "during"]
        temporal_count = sum(1 for word in temporal_words if word in prediction_lower)
        if temporal_count >= 2:
            score += 0.2
        
        # Look for causal connections
        causal_words = ["because", "due to", "leads to", "results in", "causes", "influenced by"]
        if any(word in prediction_lower for word in causal_words):
            score += 0.15
        
        return min(1.0, score)
    
    def _assess_advanced_content_accuracy(self, prediction: str, ground_truth: str, question: str) -> float:
        """
        Assess content accuracy for advanced questions with more flexible matching
        """
        # For advanced questions, we're more interested in reasonable interpretations
        # than exact matches, especially for synthesis and inference questions
        
        if not ground_truth or ground_truth.lower().startswith("multi-part response"):
            # Ground truth is a description of expected reasoning approach
            # Focus on whether the response addresses the key components
            components_mentioned = 0
            expected_components = [
                "quantitative", "historical", "analysis", "comparison", 
                "inference", "evidence", "calculation", "interpretation"
            ]
            
            prediction_lower = prediction.lower()
            for component in expected_components:
                if component in prediction_lower:
                    components_mentioned += 1
            
            return min(1.0, components_mentioned / 4.0)  # Need at least 4 components for full score
        
        else:
            # Use the existing content overlap method
            return self._calculate_content_overlap(prediction, ground_truth)


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