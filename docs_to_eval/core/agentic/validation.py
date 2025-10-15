"""
Comprehensive validation system for agentic benchmark generation
Implements deterministic/non-deterministic guardrails and quality control
"""

import asyncio
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import logging

from .models import (
    EnhancedBenchmarkItem,
    AnswerType,
    DifficultyLevel,
    validate_deterministic_answer_type
)
from ..evaluation import EvaluationType, is_deterministic
from ..verification import VerificationOrchestrator


logger = logging.getLogger(__name__)


class DeterministicGuardRail:
    """
    Enforces deterministic vs non-deterministic split consistency
    Ensures items marked as deterministic can pass exact-match verification
    """
    
    def __init__(self):
        self.verifier = VerificationOrchestrator()
        self.validation_cache = {}
    
    async def validate_deterministic_consistency(
        self, 
        items: List[EnhancedBenchmarkItem]
    ) -> Dict[str, Any]:
        """
        Validate that deterministic items truly pass deterministic verification
        
        Args:
            items: List of benchmark items to validate
            
        Returns:
            Comprehensive validation report
        """
        
        validation_results = {
            'total_items': len(items),
            'deterministic_items': [],
            'non_deterministic_items': [],
            'consistency_violations': [],
            'passed_validation': 0,
            'failed_validation': 0,
            'warnings': []
        }
        
        for i, item in enumerate(items):
            item_result = await self._validate_single_item(item, i)
            
            if item.metadata.deterministic:
                validation_results['deterministic_items'].append(item_result)
            else:
                validation_results['non_deterministic_items'].append(item_result)
            
            if item_result['consistent']:
                validation_results['passed_validation'] += 1
            else:
                validation_results['failed_validation'] += 1
                validation_results['consistency_violations'].append(item_result)
        
        # Add summary statistics
        validation_results.update(self._calculate_summary_stats(validation_results))
        
        return validation_results
    
    async def _validate_single_item(self, item: EnhancedBenchmarkItem, index: int) -> Dict[str, Any]:
        """Validate consistency of a single benchmark item"""
        
        item_result = {
            'index': index,
            'question': item.question[:50] + '...',
            'eval_type': item.eval_type,
            'marked_deterministic': item.metadata.deterministic,
            'answer_type': getattr(item, 'expected_answer_type', 'unknown'),
            'consistent': True,
            'issues': [],
            'verification_score': 0.0
        }
        
        # Check if answer type matches deterministic classification
        if hasattr(item, 'expected_answer_type'):
            should_be_deterministic = validate_deterministic_answer_type(item.expected_answer_type)
            
            if item.metadata.deterministic != should_be_deterministic:
                item_result['consistent'] = False
                item_result['issues'].append(
                    f"Answer type {item.expected_answer_type} suggests deterministic={should_be_deterministic}, "
                    f"but item marked as {item.metadata.deterministic}"
                )
        
        # For items marked as deterministic, verify they pass exact match
        if item.metadata.deterministic:
            try:
                verification_result = await self._test_deterministic_verification(item)
                item_result['verification_score'] = verification_result.score
                
                if verification_result.score < 0.95:  # Should be nearly perfect
                    item_result['consistent'] = False
                    item_result['issues'].append(
                        f"Deterministic item fails exact verification (score: {verification_result.score:.3f})"
                    )
            except Exception as e:
                item_result['consistent'] = False
                item_result['issues'].append(f"Verification error: {str(e)}")
        
        # Check eval type consistency
        eval_type_deterministic = is_deterministic(item.eval_type)
        if eval_type_deterministic and not item.metadata.deterministic:
            item_result['issues'].append(
                f"Eval type {item.eval_type} is typically deterministic but item marked as non-deterministic"
            )
        
        return item_result
    
    async def _test_deterministic_verification(self, item: EnhancedBenchmarkItem):
        """Test if item passes deterministic verification"""
        
        # Cache key for verification results
        cache_key = f"{item.question}:{item.answer}:{item.eval_type}"
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        # Mock perfect prediction for deterministic test
        mock_prediction = item.answer
        
        # Map eval types to verification types
        eval_type_map = {
            EvaluationType.MATHEMATICAL: 'mathematical',
            EvaluationType.FACTUAL_QA: 'factual_qa',
            EvaluationType.CODE_GENERATION: 'code_generation',
            EvaluationType.MULTIPLE_CHOICE: 'multiple_choice',
            EvaluationType.DOMAIN_KNOWLEDGE: 'factual_qa'
        }
        
        verification_type = eval_type_map.get(item.eval_type, 'factual_qa')
        
        result = self.verifier.verify(
            mock_prediction, 
            item.answer, 
            verification_type, 
            getattr(item, 'options', None)
        )
        
        # Cache result
        self.validation_cache[cache_key] = result
        return result
    
    def _calculate_summary_stats(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for validation report"""
        
        total = validation_results['total_items']
        if total == 0:
            return {'consistency_rate': 1.0, 'deterministic_ratio': 0.0}
        
        consistency_rate = validation_results['passed_validation'] / total
        deterministic_count = len(validation_results['deterministic_items'])
        deterministic_ratio = deterministic_count / total
        
        return {
            'consistency_rate': consistency_rate,
            'deterministic_ratio': deterministic_ratio,
            'avg_verification_score': self._calculate_avg_verification_score(validation_results)
        }
    
    def _calculate_avg_verification_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate average verification score"""
        scores = []
        
        for item_result in validation_results['deterministic_items']:
            if item_result.get('verification_score', 0) > 0:
                scores.append(item_result['verification_score'])
        
        return sum(scores) / len(scores) if scores else 0.0


class QualityController:
    """
    Comprehensive quality control system for benchmark items
    Implements multi-dimensional quality assessment
    """
    
    def __init__(self, min_quality_score: float = 0.6):
        self.min_quality_score = min_quality_score
        self.quality_dimensions = [
            'clarity',
            'difficulty_appropriateness',
            'context_relevance',
            'answer_quality',
            'uniqueness'
        ]
    
    async def assess_batch_quality(
        self, 
        items: List[EnhancedBenchmarkItem]
    ) -> Dict[str, Any]:
        """
        Assess quality of a batch of benchmark items
        
        Args:
            items: List of benchmark items to assess
            
        Returns:
            Comprehensive quality assessment report
        """
        
        quality_report = {
            'total_items': len(items),
            'quality_scores': [],
            'dimension_scores': defaultdict(list),
            'high_quality_items': [],
            'low_quality_items': [],
            'quality_distribution': {},
            'recommendations': []
        }
        
        # Assess each item
        for i, item in enumerate(items):
            item_quality = await self._assess_item_quality(item, i)
            quality_report['quality_scores'].append(item_quality['overall_score'])
            
            # Collect dimension scores
            for dimension, score in item_quality['dimension_scores'].items():
                quality_report['dimension_scores'][dimension].append(score)
            
            # Categorize items
            if item_quality['overall_score'] >= 0.8:
                quality_report['high_quality_items'].append(item_quality)
            elif item_quality['overall_score'] < self.min_quality_score:
                quality_report['low_quality_items'].append(item_quality)
        
        # Calculate aggregated metrics
        quality_report.update(self._calculate_quality_aggregates(quality_report))
        
        # Generate recommendations
        quality_report['recommendations'] = self._generate_quality_recommendations(quality_report)
        
        return quality_report
    
    async def _assess_item_quality(self, item: EnhancedBenchmarkItem, index: int) -> Dict[str, Any]:
        """Assess quality of a single benchmark item"""
        
        dimension_scores = {}
        
        # Clarity assessment
        dimension_scores['clarity'] = self._assess_clarity(item.question, item.answer)
        
        # Difficulty appropriateness
        dimension_scores['difficulty_appropriateness'] = self._assess_difficulty_appropriateness(item)
        
        # Context relevance
        dimension_scores['context_relevance'] = self._assess_context_relevance(item)
        
        # Answer quality
        dimension_scores['answer_quality'] = self._assess_answer_quality(item)
        
        # Uniqueness (placeholder - would need item comparison for full implementation)
        dimension_scores['uniqueness'] = 0.8  # Default high uniqueness
        
        # Calculate weighted overall score
        weights = {
            'clarity': 0.25,
            'difficulty_appropriateness': 0.20,
            'context_relevance': 0.20,
            'answer_quality': 0.25,
            'uniqueness': 0.10
        }
        
        overall_score = sum(
            dimension_scores[dim] * weights[dim] 
            for dim in self.quality_dimensions
        )
        
        return {
            'index': index,
            'question_preview': item.question[:50] + '...',
            'overall_score': overall_score,
            'dimension_scores': dimension_scores,
            'meets_threshold': overall_score >= self.min_quality_score
        }
    
    def _assess_clarity(self, question: str, answer: str) -> float:
        """Assess clarity of question and answer"""
        score = 1.0
        
        # Question clarity checks
        if len(question.strip()) < 10:
            score -= 0.3
        if not question.endswith('?') and '?' not in question:
            score -= 0.2
        if question.count('?') > 2:
            score -= 0.1  # Too many questions
        
        # Answer clarity checks
        if len(answer.strip()) < 3:
            score -= 0.3
        if answer.strip().endswith('...'):
            score -= 0.2  # Incomplete answer
        
        return max(0.0, score)
    
    def _assess_difficulty_appropriateness(self, item: EnhancedBenchmarkItem) -> float:
        """Assess if difficulty level matches question complexity"""
        
        # Complexity indicators
        complexity_factors = {
            'question_length': len(item.question) > 80,
            'has_context': item.context is not None and len(item.context) > 100,
            'has_options': item.options is not None and len(item.options) > 2,
            'multi_step_reasoning': len(getattr(item, 'reasoning_chain', [])) > 2,
            'adversarial_techniques': len(item.metadata.adversarial_techniques) > 0
        }
        
        complexity_score = sum(complexity_factors.values()) / len(complexity_factors)
        
        # Expected complexity for difficulty levels
        expected_complexity = {
            DifficultyLevel.BASIC: 0.2,
            DifficultyLevel.INTERMEDIATE: 0.5,
            DifficultyLevel.HARD: 0.7,
            DifficultyLevel.EXPERT: 0.9
        }
        
        expected = expected_complexity.get(item.metadata.difficulty, 0.5)
        difference = abs(complexity_score - expected)
        
        return max(0.0, 1.0 - difference * 1.5)
    
    def _assess_context_relevance(self, item: EnhancedBenchmarkItem) -> float:
        """Assess relevance between question and context"""
        
        if not item.context:
            return 0.7  # Neutral score for missing context
        
        # Simple token overlap assessment
        question_tokens = set(item.question.lower().split())
        context_tokens = set(item.context.lower().split())
        
        if not question_tokens or not context_tokens:
            return 0.5
        
        overlap = len(question_tokens.intersection(context_tokens))
        union = len(question_tokens.union(context_tokens))
        
        relevance_score = overlap / union if union > 0 else 0.5
        return min(1.0, relevance_score * 2)  # Scale up
    
    def _assess_answer_quality(self, item: EnhancedBenchmarkItem) -> float:
        """Assess quality of the answer"""
        
        answer = item.answer.strip()
        if not answer:
            return 0.0
        
        score = 1.0
        
        # Length appropriateness
        if len(answer) < 3:
            score -= 0.4
        elif len(answer.split()) == 1 and item.eval_type != EvaluationType.MULTIPLE_CHOICE:
            score -= 0.2  # Single word answers often too simple
        
        # Check for placeholder text
        placeholders = ['...', 'TODO', 'placeholder', 'insert', 'example']
        if any(placeholder in answer.lower() for placeholder in placeholders):
            score -= 0.5
        
        # Type-specific checks
        if hasattr(item, 'expected_answer_type'):
            answer_type = item.expected_answer_type
            
            if answer_type == AnswerType.NUMERIC_EXACT:
                import re
                if not re.search(r'\d', answer):
                    score -= 0.4
            elif answer_type == AnswerType.CODE:
                code_indicators = ['def', 'function', 'return', 'class', 'import']
                if not any(indicator in answer.lower() for indicator in code_indicators):
                    score -= 0.3
        
        return max(0.0, score)
    
    def _calculate_quality_aggregates(self, quality_report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate quality metrics"""
        
        scores = quality_report['quality_scores']
        if not scores:
            return {'avg_quality': 0.0, 'quality_std': 0.0}
        
        avg_quality = sum(scores) / len(scores)
        quality_std = (sum((s - avg_quality) ** 2 for s in scores) / len(scores)) ** 0.5
        
        # Quality distribution
        distribution = {
            'excellent': len([s for s in scores if s >= 0.9]),
            'good': len([s for s in scores if 0.7 <= s < 0.9]),
            'acceptable': len([s for s in scores if 0.6 <= s < 0.7]),
            'poor': len([s for s in scores if s < 0.6])
        }
        
        # Dimension averages
        dimension_averages = {}
        for dimension, scores_list in quality_report['dimension_scores'].items():
            if scores_list:
                dimension_averages[dimension] = sum(scores_list) / len(scores_list)
        
        return {
            'avg_quality': avg_quality,
            'quality_std': quality_std,
            'quality_distribution': distribution,
            'dimension_averages': dimension_averages,
            'pass_rate': len([s for s in scores if s >= self.min_quality_score]) / len(scores)
        }
    
    def _generate_quality_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality assessment"""
        
        recommendations = []
        
        # Overall quality recommendations
        avg_quality = quality_report.get('avg_quality', 0.0)
        if avg_quality < 0.7:
            recommendations.append("Overall quality is below target. Consider reviewing generation parameters.")
        
        # Dimension-specific recommendations
        dimension_averages = quality_report.get('dimension_averages', {})
        
        for dimension, avg_score in dimension_averages.items():
            if avg_score < 0.6:
                if dimension == 'clarity':
                    recommendations.append("Improve question clarity by ensuring proper punctuation and structure.")
                elif dimension == 'difficulty_appropriateness':
                    recommendations.append("Review difficulty calibration - questions may not match stated difficulty.")
                elif dimension == 'context_relevance':
                    recommendations.append("Improve context selection to better support questions.")
                elif dimension == 'answer_quality':
                    recommendations.append("Enhance answer quality by avoiding placeholders and ensuring completeness.")
        
        # Distribution-based recommendations
        distribution = quality_report.get('quality_distribution', {})
        if distribution.get('poor', 0) > distribution.get('excellent', 0):
            recommendations.append("High proportion of poor-quality items. Consider stricter validation thresholds.")
        
        return recommendations


class ComprehensiveValidator:
    """
    Main validation orchestrator that combines all validation systems
    Provides comprehensive validation for agentic benchmark generation
    """
    
    def __init__(self, min_quality_score: float = 0.6):
        self.deterministic_guard = DeterministicGuardRail()
        self.quality_controller = QualityController(min_quality_score)
    
    async def validate_benchmark_batch(
        self, 
        items: List[EnhancedBenchmarkItem]
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of benchmark batch
        
        Args:
            items: List of benchmark items to validate
            
        Returns:
            Complete validation report
        """
        
        logger.info(f"Starting comprehensive validation of {len(items)} items")
        
        # Run all validation systems in parallel
        deterministic_validation, quality_assessment = await asyncio.gather(
            self.deterministic_guard.validate_deterministic_consistency(items),
            self.quality_controller.assess_batch_quality(items)
        )
        
        # Combine results
        comprehensive_report = {
            'validation_timestamp': asyncio.get_event_loop().time(),
            'total_items_validated': len(items),
            'deterministic_validation': deterministic_validation,
            'quality_assessment': quality_assessment,
            'overall_pass_rate': self._calculate_overall_pass_rate(deterministic_validation, quality_assessment),
            'actionable_recommendations': self._combine_recommendations(deterministic_validation, quality_assessment)
        }
        
        logger.info(f"Validation complete. Overall pass rate: {comprehensive_report['overall_pass_rate']:.3f}")
        
        return comprehensive_report
    
    def _calculate_overall_pass_rate(self, det_validation: Dict, quality_assessment: Dict) -> float:
        """Calculate overall pass rate combining both validation systems"""
        
        det_pass_rate = det_validation.get('consistency_rate', 0.0)
        quality_pass_rate = quality_assessment.get('pass_rate', 0.0)
        
        # Weighted combination (deterministic consistency is critical)
        return det_pass_rate * 0.6 + quality_pass_rate * 0.4
    
    def _combine_recommendations(self, det_validation: Dict, quality_assessment: Dict) -> List[str]:
        """Combine recommendations from all validation systems"""
        
        recommendations = []
        
        # Deterministic validation recommendations
        if det_validation.get('consistency_rate', 1.0) < 0.9:
            recommendations.append("CRITICAL: Fix deterministic consistency issues before deployment")
        
        if det_validation.get('deterministic_ratio', 0.5) < 0.3:
            recommendations.append("Consider increasing proportion of deterministic questions")
        
        # Quality recommendations
        recommendations.extend(quality_assessment.get('recommendations', []))
        
        # Add priorities
        prioritized_recommendations = []
        for rec in recommendations:
            if 'CRITICAL' in rec:
                prioritized_recommendations.insert(0, rec)  # High priority first
            else:
                prioritized_recommendations.append(rec)
        
        return prioritized_recommendations
    
    async def validate_and_filter(
        self, 
        items: List[EnhancedBenchmarkItem], 
        strict_mode: bool = True
    ) -> Tuple[List[EnhancedBenchmarkItem], Dict[str, Any]]:
        """
        Validate items and filter out those that don't meet standards
        
        Args:
            items: Items to validate and filter
            strict_mode: If True, apply strict filtering criteria
            
        Returns:
            Tuple of (filtered_items, validation_report)
        """
        
        validation_report = await self.validate_benchmark_batch(items)
        
        # Filter items based on validation results
        filtered_items = []
        
        det_items = {item['index']: item for item in validation_report['deterministic_validation']['deterministic_items']}
        non_det_items = {item['index']: item for item in validation_report['deterministic_validation']['non_deterministic_items']}
        # Build map of items meeting quality threshold (kept for potential future filtering logic)
        _ = {item['index']: item for item in validation_report['quality_assessment']['high_quality_items'] + 
             [item for item in validation_report['quality_assessment'].get('quality_scores', []) 
              if isinstance(item, dict) and item.get('meets_threshold', False)]}
        
        for i, item in enumerate(items):
            should_include = True
            
            # Check deterministic consistency
            det_result = det_items.get(i) or non_det_items.get(i, {})
            if strict_mode and not det_result.get('consistent', True):
                should_include = False
            
            # Check quality threshold
            quality_scores = validation_report['quality_assessment']['quality_scores']
            if i < len(quality_scores):
                quality_score = quality_scores[i]
                if strict_mode and quality_score < self.quality_controller.min_quality_score:
                    should_include = False
            
            if should_include:
                filtered_items.append(item)
        
        # Update validation report with filtering results
        validation_report['filtering'] = {
            'original_count': len(items),
            'filtered_count': len(filtered_items),
            'retention_rate': len(filtered_items) / len(items) if items else 0,
            'strict_mode': strict_mode
        }
        
        return filtered_items, validation_report