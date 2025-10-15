"""
Advanced evaluation utilities following lm-evaluation-harness best practices
Implements bootstrap confidence intervals, ensemble verification, and statistical rigor
"""

import random
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .similarity import calculate_multi_similarity
from .text_processing import normalize_answer


class BootstrapStatistics:
    """Bootstrap confidence interval calculation following lm-eval-harness standards"""
    
    @staticmethod
    def bootstrap_confidence_interval(
        scores: List[float], 
        confidence_level: float = 0.95, 
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for scores
        
        Args:
            scores: List of evaluation scores
            confidence_level: Confidence level (0.95 for 95% CI)  
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not scores or len(scores) < 2:
            return (0.0, 0.0)
        
        # Generate bootstrap samples
        bootstrap_means = []
        n_scores = len(scores)
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_sample = [random.choice(scores) for _ in range(n_scores)]
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # Calculate percentiles for confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        
        return (float(lower_bound), float(upper_bound))
    
    @staticmethod
    def calculate_statistical_significance(scores1: List[float], scores2: List[float]) -> Dict[str, float]:
        """
        Calculate statistical significance between two score distributions
        
        Returns:
            Dictionary with p_value and effect_size
        """
        if not scores1 or not scores2:
            return {"p_value": 1.0, "effect_size": 0.0}
        
        # Simple two-sample t-test approximation
        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        std1, std2 = np.std(scores1, ddof=1), np.std(scores2, ddof=1)
        n1, n2 = len(scores1), len(scores2)
        
        if std1 == 0 and std2 == 0:
            return {"p_value": 1.0 if mean1 == mean2 else 0.0, "effect_size": 0.0}
        
        # Pooled standard error
        pooled_se = math.sqrt((std1**2 / n1) + (std2**2 / n2))
        
        if pooled_se == 0:
            return {"p_value": 1.0, "effect_size": 0.0}
        
        # t-statistic
        t_stat = (mean1 - mean2) / pooled_se
        
        # Degrees of freedom (Welch's approximation)
        df = ((std1**2 / n1) + (std2**2 / n2))**2 / (
            (std1**2 / n1)**2 / (n1 - 1) + (std2**2 / n2)**2 / (n2 - 1)
        )
        
        # Effect size (Cohen's d)
        pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        effect_size = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        
        # Approximate p-value (simplified)
        p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + math.sqrt(df))) if df > 0 else 1.0
        
        return {"p_value": float(p_value), "effect_size": float(effect_size)}


class EnsembleVerifier:
    """Ensemble verification using multiple methods following lm-eval-harness patterns"""
    
    def __init__(self):
        self.method_weights = {
            'semantic': 0.4,      # Real embeddings - highest weight
            'token_overlap': 0.25, # Dice coefficient - good for factual overlap
            'ngram': 0.15,        # Dice coefficient - captures phrase similarity  
            'rouge_l': 0.15,      # F1-based - good for content coverage
            'character_overlap': 0.05  # Basic fallback
        }
    
    def verify_with_ensemble(self, prediction: str, ground_truth: str) -> Dict[str, Any]:
        """
        Verify using ensemble of methods with statistical confidence
        
        Args:
            prediction: Model's predicted answer
            ground_truth: Expected/reference answer
            
        Returns:
            Dictionary with scores, confidence intervals, and method breakdown
        """
        # Calculate all similarity scores
        similarities = calculate_multi_similarity(prediction, ground_truth)
        
        # Apply length normalization (lm-eval-harness acc_norm style)
        normalized_similarities = self._apply_length_normalization(
            similarities, prediction, ground_truth
        )
        
        # Calculate weighted ensemble score
        ensemble_score = self._calculate_ensemble_score(normalized_similarities)
        
        # Apply final boost following our lenient scoring approach
        final_score = self._apply_final_boost(ensemble_score)
        
        # Calculate confidence metrics
        method_scores = list(normalized_similarities.values())
        ci_95 = BootstrapStatistics.bootstrap_confidence_interval(method_scores, 0.95)
        ci_99 = BootstrapStatistics.bootstrap_confidence_interval(method_scores, 0.99)
        
        return {
            'final_score': final_score,
            'ensemble_score': ensemble_score,
            'individual_scores': normalized_similarities,
            'confidence_interval_95': ci_95,
            'confidence_interval_99': ci_99,
            'method_agreement': self._calculate_method_agreement(normalized_similarities),
            'length_penalty_applied': self._had_length_penalty(prediction, ground_truth),
            'boost_applied': final_score > ensemble_score
        }
    
    def _apply_length_normalization(
        self, 
        similarities: Dict[str, float], 
        pred: str, 
        gt: str
    ) -> Dict[str, float]:
        """Apply length normalization like lm-eval-harness acc_norm"""
        pred_len = len(pred.split())
        gt_len = len(gt.split())
        
        if pred_len == 0 or gt_len == 0:
            return similarities
        
        # Length factor: penalize very different lengths
        length_factor = min(pred_len, gt_len) / max(pred_len, gt_len)
        
        # Apply gentle length normalization (not too harsh)
        length_multiplier = 0.85 + 0.15 * length_factor
        
        normalized = {}
        for method, score in similarities.items():
            # Semantic similarity gets less length penalty (it's more meaning-focused)
            if method == 'semantic':
                normalized[method] = score * (0.95 + 0.05 * length_factor)
            else:
                normalized[method] = score * length_multiplier
        
        return normalized
    
    def _calculate_ensemble_score(self, similarities: Dict[str, float]) -> float:
        """Calculate weighted ensemble score"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for method, weight in self.method_weights.items():
            if method in similarities:
                weighted_sum += similarities[method] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _apply_final_boost(self, score: float) -> float:
        """Apply final leniency boost"""
        if score > 0.15:
            # Power transformation boost for reasonable scores
            boosted = score + (score ** 0.7) * 0.25
            return min(1.0, boosted)
        return score
    
    def _calculate_method_agreement(self, similarities: Dict[str, float]) -> float:
        """Calculate agreement between different similarity methods"""
        scores = list(similarities.values())
        if len(scores) <= 1:
            return 1.0
        
        # Calculate coefficient of variation (lower = more agreement)  
        mean_score = np.mean(scores)
        if mean_score == 0:
            return 1.0
        
        cv = np.std(scores) / mean_score
        # Convert to agreement score (0-1, where 1 = perfect agreement)
        agreement = max(0.0, 1.0 - cv)
        return agreement
    
    def _had_length_penalty(self, pred: str, gt: str) -> bool:
        """Check if length penalty was applied"""
        pred_len = len(pred.split())
        gt_len = len(gt.split())
        
        if pred_len == 0 or gt_len == 0:
            return False
        
        length_ratio = min(pred_len, gt_len) / max(pred_len, gt_len)
        return length_ratio < 0.8  # Penalty applied if length difference > 20%


class AdvancedEvaluationFramework:
    """Advanced evaluation framework implementing lm-eval-harness best practices"""
    
    def __init__(self):
        self.ensemble_verifier = EnsembleVerifier()
        self.evaluation_history = []
    
    def evaluate_response_advanced(
        self, 
        prediction: str, 
        ground_truth: str,
        eval_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Advanced evaluation with statistical rigor and ensemble verification
        
        Args:
            prediction: Model's predicted answer
            ground_truth: Expected/reference answer  
            eval_context: Optional context (question type, domain, etc.)
            
        Returns:
            Comprehensive evaluation results
        """
        # Normalize answers
        pred_norm = normalize_answer(prediction)
        gt_norm = normalize_answer(ground_truth)
        
        # Exact match check
        exact_match = pred_norm == gt_norm
        
        # Ensemble verification
        ensemble_results = self.ensemble_verifier.verify_with_ensemble(prediction, ground_truth)
        
        # Create evaluation result
        result = {
            'prediction': prediction,
            'ground_truth': ground_truth,
            'exact_match': exact_match,
            'similarity_score': ensemble_results['final_score'],
            'raw_ensemble_score': ensemble_results['ensemble_score'], 
            'individual_methods': ensemble_results['individual_scores'],
            'confidence_intervals': {
                '95%': ensemble_results['confidence_interval_95'],
                '99%': ensemble_results['confidence_interval_99']
            },
            'method_agreement': ensemble_results['method_agreement'],
            'evaluation_metadata': {
                'length_penalty_applied': ensemble_results['length_penalty_applied'],
                'boost_applied': ensemble_results['boost_applied'],
                'eval_context': eval_context or {}
            }
        }
        
        # Store in history for batch analysis
        self.evaluation_history.append(result)
        
        return result
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get statistical analysis of recent evaluations"""
        if not self.evaluation_history:
            return {}
        
        scores = [r['similarity_score'] for r in self.evaluation_history]
        exact_matches = [r['exact_match'] for r in self.evaluation_history]
        
        return {
            'num_evaluations': len(self.evaluation_history),
            'mean_similarity_score': np.mean(scores),
            'median_similarity_score': np.median(scores),
            'std_similarity_score': np.std(scores),
            'exact_match_rate': np.mean(exact_matches),
            'confidence_interval_95': BootstrapStatistics.bootstrap_confidence_interval(scores, 0.95),
            'score_distribution': {
                'min': np.min(scores),
                'max': np.max(scores),
                'q25': np.percentile(scores, 25),
                'q75': np.percentile(scores, 75)
            }
        }
    
    def clear_history(self):
        """Clear evaluation history"""
        self.evaluation_history = []