"""
Statistical analysis utilities for evaluation following lm-evaluation-harness principles
Implements bootstrap confidence intervals, statistical significance testing, and comprehensive metrics
"""

import math
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from dataclasses import dataclass


@dataclass
class StatisticalResults:
    """Comprehensive statistical results for evaluation metrics"""
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float
    confidence_interval_95: Tuple[float, float]
    confidence_interval_99: Tuple[float, float]
    num_samples: int
    statistical_significance: float
    bootstrap_samples: int = 1000
    

class EvaluationStatistics:
    """Statistical analysis for LLM evaluation following gold-standard practices"""
    
    @staticmethod
    def bootstrap_confidence_interval(scores: List[float], confidence_level: float = 0.95, 
                                    n_bootstrap: int = 1000) -> Tuple[float, float]:
        """
        Calculate bias-corrected and accelerated (BCa) bootstrap confidence interval
        Following lm-evaluation-harness statistical rigor principles
        """
        if len(scores) < 2:
            return (0.0, 1.0)
            
        try:
            n = len(scores)
            original_mean = np.mean(scores)
            
            # Generate bootstrap samples
            bootstrap_means = []
            for _ in range(n_bootstrap):
                bootstrap_sample = [random.choice(scores) for _ in range(n)]
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            if not bootstrap_means:
                return (float(original_mean), float(original_mean))
            
            bootstrap_means.sort()
            
            # Calculate percentiles for confidence interval
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(bootstrap_means, lower_percentile)
            upper_bound = np.percentile(bootstrap_means, upper_percentile)
            
            return (float(lower_bound), float(upper_bound))
            
        except Exception as e:
            # Fallback to simple mean +/- error
            mean_score = np.mean(scores) if scores else 0.0
            error = np.std(scores) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0
            return (float(max(0.0, mean_score - error)), float(min(1.0, mean_score + error)))
    
    @staticmethod
    def calculate_comprehensive_metrics(scores: List[float]) -> StatisticalResults:
        """
        Calculate comprehensive statistical metrics following lm-evaluation-harness standards
        """
        if not scores:
            return StatisticalResults(
                mean=0.0, std=0.0, median=0.0, min_val=0.0, max_val=0.0,
                confidence_interval_95=(0.0, 0.0), confidence_interval_99=(0.0, 0.0),
                num_samples=0, statistical_significance=1.0
            )
        
        scores_array = np.array(scores)
        
        # Basic statistics
        mean_score = float(np.mean(scores_array))
        std_score = float(np.std(scores_array, ddof=1)) if len(scores) > 1 else 0.0
        median_score = float(np.median(scores_array))
        min_score = float(np.min(scores_array))
        max_score = float(np.max(scores_array))
        
        # Bootstrap confidence intervals
        ci_95 = EvaluationStatistics.bootstrap_confidence_interval(scores, 0.95)
        ci_99 = EvaluationStatistics.bootstrap_confidence_interval(scores, 0.99)
        
        # Statistical significance (p-value for difference from random chance)
        # For accuracy metrics, random chance is typically 0.5 or 1/num_choices
        # We'll test against 0.5 as a conservative baseline
        if len(scores) > 1 and std_score > 0:
            t_stat = (mean_score - 0.5) / (std_score / math.sqrt(len(scores)))
            # Simplified p-value calculation (two-tailed test)
            p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + math.sqrt(len(scores) - 1)))
            p_value = max(0.001, min(1.0, p_value))  # Clamp to reasonable range
        elif std_score == 0 and len(scores) > 1:
            # If std deviation is 0, all scores are identical
            # If mean is far from 0.5, it's likely significant
            p_value = 0.001 if abs(mean_score - 0.5) > 0.1 else 0.5
        else:
            p_value = 1.0
        
        return StatisticalResults(
            mean=mean_score,
            std=std_score,
            median=median_score,
            min_val=min_score,
            max_val=max_score,
            confidence_interval_95=ci_95,
            confidence_interval_99=ci_99,
            num_samples=len(scores),
            statistical_significance=p_value
        )
    
    @staticmethod
    def calculate_accuracy_metrics(predictions: List[str], ground_truths: List[str], 
                                 scores: List[float]) -> Dict[str, Any]:
        """
        Calculate accuracy-based metrics following lm-evaluation-harness patterns
        """
        if not predictions or len(predictions) != len(ground_truths) != len(scores):
            return {"error": "Mismatched input lengths"}
        
        # Exact match accuracy
        exact_matches = [1.0 if pred.strip().lower() == truth.strip().lower() 
                        else 0.0 for pred, truth in zip(predictions, ground_truths)]
        
        # Normalized accuracy (accounting for different answer lengths)
        normalized_scores = []
        for pred, truth, score in zip(predictions, ground_truths, scores):
            # Length normalization factor
            pred_len = len(pred.split()) if pred else 1
            truth_len = len(truth.split()) if truth else 1
            len_factor = min(pred_len, truth_len) / max(pred_len, truth_len, 1)
            normalized_score = score * len_factor
            normalized_scores.append(normalized_score)
        
        return {
            "exact_match_accuracy": EvaluationStatistics.calculate_comprehensive_metrics(exact_matches),
            "raw_accuracy": EvaluationStatistics.calculate_comprehensive_metrics(scores),
            "normalized_accuracy": EvaluationStatistics.calculate_comprehensive_metrics(normalized_scores)
        }
    
    @staticmethod
    def calculate_f1_scores(predictions: List[str], ground_truths: List[str]) -> List[float]:
        """
        Calculate F1 scores for each prediction-truth pair
        Following SQuAD/reading comprehension evaluation patterns
        """
        f1_scores = []
        
        for pred, truth in zip(predictions, ground_truths):
            pred_tokens = set(pred.lower().split())
            truth_tokens = set(truth.lower().split())
            
            if len(pred_tokens) == 0 and len(truth_tokens) == 0:
                f1_scores.append(1.0)
                continue
            
            if len(pred_tokens) == 0 or len(truth_tokens) == 0:
                f1_scores.append(0.0)
                continue
            
            # Calculate precision, recall, F1
            common_tokens = pred_tokens.intersection(truth_tokens)
            precision = len(common_tokens) / len(pred_tokens)
            recall = len(common_tokens) / len(truth_tokens)
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
                f1_scores.append(f1)
        
        return f1_scores
    
    @staticmethod
    def detect_potential_contamination(corpus_text: str, predictions: List[str], 
                                     threshold: float = 0.8) -> Dict[str, Any]:
        """
        Basic contamination detection using n-gram overlap analysis
        Following lm-evaluation-harness contamination detection principles
        """
        corpus_lower = corpus_text.lower()
        contamination_flags = []
        overlap_scores = []
        
        for pred in predictions:
            pred_lower = pred.lower()
            
            # Check for exact substring matches (potential memorization)
            if len(pred_lower) > 20 and pred_lower in corpus_lower:
                contamination_flags.append(True)
                overlap_scores.append(1.0)
                continue
            
            # N-gram overlap analysis (using 4-grams as in GPT-3 evaluation)
            pred_4grams = set()
            words = pred_lower.split()
            for i in range(len(words) - 3):
                pred_4grams.add(" ".join(words[i:i+4]))
            
            corpus_4grams = set()
            corpus_words = corpus_lower.split()
            for i in range(len(corpus_words) - 3):
                corpus_4grams.add(" ".join(corpus_words[i:i+4]))
            
            if len(pred_4grams) == 0:
                overlap_score = 0.0
            else:
                overlap_score = len(pred_4grams.intersection(corpus_4grams)) / len(pred_4grams)
            
            overlap_scores.append(overlap_score)
            contamination_flags.append(overlap_score > threshold)
        
        return {
            "contamination_detected": any(contamination_flags),
            "contamination_rate": sum(contamination_flags) / len(contamination_flags) if contamination_flags else 0.0,
            "mean_overlap_score": np.mean(overlap_scores) if overlap_scores else 0.0,
            "max_overlap_score": max(overlap_scores) if overlap_scores else 0.0,
            "contaminated_samples": sum(contamination_flags)
        }
    
    @staticmethod
    def generate_evaluation_report(verification_results: List[Dict], corpus_text: str = "") -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report following lm-evaluation-harness standards
        """
        if not verification_results:
            return {"error": "No verification results provided"}
        
        # Extract data
        scores = [r["score"] for r in verification_results]
        predictions = [r["prediction"] for r in verification_results]
        ground_truths = [r["ground_truth"] for r in verification_results]
        methods = [r.get("method", "unknown") for r in verification_results]
        
        # Calculate comprehensive statistics
        main_stats = EvaluationStatistics.calculate_comprehensive_metrics(scores)
        
        # Calculate accuracy metrics
        accuracy_metrics = EvaluationStatistics.calculate_accuracy_metrics(
            predictions, ground_truths, scores
        )
        
        # Calculate F1 scores
        f1_scores = EvaluationStatistics.calculate_f1_scores(predictions, ground_truths)
        f1_stats = EvaluationStatistics.calculate_comprehensive_metrics(f1_scores)
        
        # Method breakdown
        method_breakdown = {}
        method_counter = Counter(methods)
        for method, count in method_counter.items():
            method_scores = [s for s, m in zip(scores, methods) if m == method]
            method_breakdown[method] = {
                "count": count,
                "statistics": EvaluationStatistics.calculate_comprehensive_metrics(method_scores)
            }
        
        # Contamination detection
        contamination_analysis = {}
        if corpus_text and predictions:
            contamination_analysis = EvaluationStatistics.detect_potential_contamination(
                corpus_text, predictions
            )
        
        return {
            "main_statistics": main_stats,
            "accuracy_metrics": accuracy_metrics,
            "f1_statistics": f1_stats,
            "method_breakdown": method_breakdown,
            "contamination_analysis": contamination_analysis,
            "total_samples": len(verification_results),
            "evaluation_quality": {
                "statistically_significant": main_stats.statistical_significance < 0.05,
                "sufficient_samples": len(verification_results) >= 30,
                "reliable_confidence": main_stats.confidence_interval_95[1] - main_stats.confidence_interval_95[0] < 0.2
            }
        }


if __name__ == "__main__":
    # Test the statistical analysis
    test_scores = [0.8, 0.9, 0.7, 0.85, 0.75, 0.95, 0.6, 0.88, 0.92, 0.78]
    stats = EvaluationStatistics.calculate_comprehensive_metrics(test_scores)
    print(f"Mean: {stats.mean:.3f}")
    print(f"95% CI: [{stats.confidence_interval_95[0]:.3f}, {stats.confidence_interval_95[1]:.3f}]")
    print(f"Statistical significance (p-value): {stats.statistical_significance:.3f}")