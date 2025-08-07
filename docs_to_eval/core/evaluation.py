"""
Core evaluation framework for domain-specific LLM benchmarking
Supports both deterministic and non-deterministic evaluation methods
"""

import json
import re
import random
from functools import reduce
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
from enum import Enum

# Import enums from central config location
from ..utils.config import EvaluationType, VerificationMethod


class EvaluationConfig(BaseModel):
    """Configuration for an evaluation type"""
    deterministic: bool
    verification: VerificationMethod
    metrics: List[str]
    
    class Config:
        use_enum_values = True


class BenchmarkItem(BaseModel):
    """A single benchmark question/answer pair"""
    question: str
    answer: str
    context: Optional[str] = None
    options: Optional[List[str]] = None
    eval_type: EvaluationType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class EvaluationResult(BaseModel):
    """Result of evaluating a single item"""
    prediction: str
    ground_truth: str
    score: float = Field(ge=0, le=1)
    metrics: Dict[str, float]
    eval_type: EvaluationType
    verified: bool = True
    
    class Config:
        use_enum_values = True


# Evaluation Types Configuration
EVAL_TYPES = {
    EvaluationType.MATHEMATICAL: EvaluationConfig(
        deterministic=True,
        verification=VerificationMethod.EXACT_MATCH,
        metrics=['accuracy', 'pass_rate']
    ),
    EvaluationType.CODE_GENERATION: EvaluationConfig(
        deterministic=True,
        verification=VerificationMethod.EXECUTION,
        metrics=['pass_rate', 'syntax_correctness']
    ),
    EvaluationType.FACTUAL_QA: EvaluationConfig(
        deterministic=True,
        verification=VerificationMethod.EXACT_MATCH,
        metrics=['accuracy', 'f1_score']
    ),
    EvaluationType.MULTIPLE_CHOICE: EvaluationConfig(
        deterministic=True,
        verification=VerificationMethod.EXACT_MATCH,
        metrics=['accuracy', 'normalized_accuracy']
    ),
    EvaluationType.SUMMARIZATION: EvaluationConfig(
        deterministic=False,
        verification=VerificationMethod.SIMILARITY,
        metrics=['rouge_l', 'bleu', 'semantic_similarity']
    ),
    EvaluationType.TRANSLATION: EvaluationConfig(
        deterministic=False,
        verification=VerificationMethod.SIMILARITY,
        metrics=['bleu', 'chrf', 'semantic_similarity']
    ),
    EvaluationType.CREATIVE_WRITING: EvaluationConfig(
        deterministic=False,
        verification=VerificationMethod.LLM_JUDGE,
        metrics=['coherence', 'creativity', 'fluency']
    ),
    EvaluationType.COMMONSENSE_REASONING: EvaluationConfig(
        deterministic=True,
        verification=VerificationMethod.EXACT_MATCH,
        metrics=['accuracy']
    ),
    EvaluationType.READING_COMPREHENSION: EvaluationConfig(
        deterministic=False,
        verification=VerificationMethod.SIMILARITY,
        metrics=['f1_score', 'exact_match', 'semantic_similarity']
    ),
    EvaluationType.DOMAIN_KNOWLEDGE: EvaluationConfig(
        deterministic=True,
        verification=VerificationMethod.EXACT_MATCH,
        metrics=['accuracy', 'precision', 'recall']
    )
}


# Functional utilities
def get_deterministic_types() -> List[EvaluationType]:
    """Get list of deterministic evaluation types"""
    return [k for k, v in EVAL_TYPES.items() if v.deterministic]


def get_non_deterministic_types() -> List[EvaluationType]:
    """Get list of non-deterministic evaluation types"""
    return [k for k, v in EVAL_TYPES.items() if not v.deterministic]


def is_deterministic(eval_type: EvaluationType) -> bool:
    """Check if evaluation type is deterministic"""
    return EVAL_TYPES.get(eval_type, EVAL_TYPES[EvaluationType.DOMAIN_KNOWLEDGE]).deterministic


def get_verification_method(eval_type: EvaluationType) -> VerificationMethod:
    """Get verification method for evaluation type"""
    return EVAL_TYPES.get(eval_type, EVAL_TYPES[EvaluationType.DOMAIN_KNOWLEDGE]).verification


def get_metrics_for_type(eval_type: EvaluationType) -> List[str]:
    """Get metrics for evaluation type"""
    return EVAL_TYPES.get(eval_type, EVAL_TYPES[EvaluationType.DOMAIN_KNOWLEDGE]).metrics


# Corpus analysis functions
def analyze_corpus_content(corpus_text: str) -> Dict[str, float]:
    """Analyze corpus content to determine evaluation types"""
    patterns = {
        EvaluationType.MATHEMATICAL: [
            r'\d+\s*[\+\-\*\/]\s*\d+',
            r'equation|formula|calculate|solve',
            r'equals?|=',
            r'\b\d+%|\$\d+',
        ],
        EvaluationType.CODE_GENERATION: [
            r'def\s+\w+\(|function\s+\w+\(',
            r'import\s+\w+|from\s+\w+\s+import',
            r'class\s+\w+:|public\s+class',
            r'if\s*\(|while\s*\(|for\s*\(',
        ],
        EvaluationType.FACTUAL_QA: [
            r'\b(who|what|when|where|why|how)\b',
            r'\b(president|capital|founded|invented)\b',
            r'\b\d{4}\b',  # years
            r'\b(company|organization|person)\b',
        ],
        EvaluationType.DOMAIN_KNOWLEDGE: [
            r'\b(definition|concept|theory|principle)\b',
            r'\b(according to|research shows|studies indicate)\b',
            r'\b(methodology|approach|framework)\b',
        ],
        EvaluationType.SUMMARIZATION: [
            r'\b(summary|abstract|overview|conclusion)\b',
            r'\b(in summary|to summarize|in conclusion)\b',
            r'key points|main ideas|highlights',
        ]
    }
    
    scores = {}
    word_count = len(corpus_text.split())
    
    for eval_type, pattern_list in patterns.items():
        score = 0
        for pattern in pattern_list:
            matches = len(re.findall(pattern, corpus_text, re.IGNORECASE))
            score += matches
        
        # Normalize by text length
        scores[eval_type] = score / max(word_count / 100, 1)
    
    return scores


def determine_primary_eval_type(corpus_text: str) -> EvaluationType:
    """Determine primary evaluation type for corpus"""
    scores = analyze_corpus_content(corpus_text)
    if not scores:
        return EvaluationType.DOMAIN_KNOWLEDGE  # default
    
    primary_type = max(scores.items(), key=lambda x: x[1])[0]
    return primary_type


# Benchmark generation utilities
def sample_corpus_segments(corpus_text: str, num_segments: int = 10, segment_length: int = 500) -> List[str]:
    """Sample segments from corpus for benchmark generation"""
    words = corpus_text.split()
    segments = []
    
    if len(words) <= segment_length:
        return [corpus_text]
    
    for _ in range(num_segments):
        start_idx = random.randint(0, max(0, len(words) - segment_length))
        segment = ' '.join(words[start_idx:start_idx + segment_length])
        segments.append(segment)
    
    return segments


def extract_key_concepts(corpus_text: str, max_concepts: int = 20) -> List[str]:
    """Extract key concepts from corpus"""
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{4,}\b', corpus_text.lower())
    word_freq = defaultdict(int)
    
    # Common stop words to filter out
    stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'would', 'could', 'should'}
    
    for word in words:
        if word not in stop_words:
            word_freq[word] += 1
    
    # Get top concepts
    concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_concepts]
    return [concept[0] for concept in concepts]


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark generation"""
    eval_type: EvaluationType
    num_questions: int = Field(gt=0, le=1000, default=50)
    temperature: float = Field(ge=0, le=2, default=0.7)
    max_tokens: int = Field(gt=0, le=4096, default=512)
    verification_threshold: float = Field(ge=0, le=1, default=0.8)
    key_concepts: List[str] = Field(default_factory=list)
    corpus_segments: List[str] = Field(default_factory=list)
    
    # Finetune test set configuration
    finetune_test_set_enabled: bool = Field(default=True)
    finetune_test_set_percentage: float = Field(ge=0.1, le=0.5, default=0.2)
    finetune_random_seed: int = Field(ge=0, default=42)
    
    class Config:
        use_enum_values = True


class FinetuneTestSet(BaseModel):
    """Data structure for finetune test set"""
    test_questions: List[Dict[str, Any]] = Field(default_factory=list, description="Questions reserved for finetune testing")
    train_questions: List[Dict[str, Any]] = Field(default_factory=list, description="Questions for finetuning")
    test_set_size: int = Field(ge=0, description="Number of questions in test set")
    train_set_size: int = Field(ge=0, description="Number of questions in training set")
    test_percentage: float = Field(ge=0, le=1, description="Actual percentage of questions in test set")
    random_seed: int = Field(ge=0, description="Random seed used for split")
    split_timestamp: str = Field(description="Timestamp when split was created")
    
    class Config:
        json_schema_extra = {
            "example": {
                "test_set_size": 10,
                "train_set_size": 40,
                "test_percentage": 0.2,
                "random_seed": 42,
                "split_timestamp": "2025-08-05T12:00:00Z"
            }
        }


class BenchmarkWithFinetuneSet(BaseModel):
    """Benchmark results with finetune test set separation"""
    config: BenchmarkConfig
    all_questions: List[Dict[str, Any]] = Field(default_factory=list, description="All generated questions")
    finetune_test_set: Optional[FinetuneTestSet] = Field(None, description="Finetune train/test split")
    generation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about generation process")
    
    def get_train_questions(self) -> List[Dict[str, Any]]:
        """Get questions for finetuning (training set)"""
        if self.finetune_test_set:
            return self.finetune_test_set.train_questions
        return self.all_questions
    
    def get_test_questions(self) -> List[Dict[str, Any]]:
        """Get questions for evaluating finetuned model (test set)"""
        if self.finetune_test_set:
            return self.finetune_test_set.test_questions
        return []
    
    def get_finetune_summary(self) -> Dict[str, Any]:
        """Get summary of finetune test set configuration"""
        if not self.finetune_test_set:
            return {
                "enabled": False,
                "total_questions": len(self.all_questions),
                "train_questions": len(self.all_questions),
                "test_questions": 0,
                "test_percentage": 0.0
            }
        
        return {
            "enabled": True,
            "total_questions": len(self.all_questions),
            "train_questions": self.finetune_test_set.train_set_size,
            "test_questions": self.finetune_test_set.test_set_size,
            "test_percentage": self.finetune_test_set.test_percentage,
            "random_seed": self.finetune_test_set.random_seed
        }


def create_finetune_test_set(
    questions: List[Dict[str, Any]], 
    test_percentage: float = 0.2, 
    random_seed: int = 42
) -> FinetuneTestSet:
    """
    Create a finetune test set by splitting questions into train/test sets
    
    Args:
        questions: List of all generated questions
        test_percentage: Percentage of questions to reserve for testing (0.2 = 20%)
        random_seed: Random seed for reproducible splits
        
    Returns:
        FinetuneTestSet with train/test split
    """
    import random
    from datetime import datetime
    
    if not questions:
        return FinetuneTestSet(
            test_questions=[],
            train_questions=[],
            test_set_size=0,
            train_set_size=0,
            test_percentage=0.0,
            random_seed=random_seed,
            split_timestamp=datetime.utcnow().isoformat() + "Z"
        )
    
    # Set random seed for reproducible splits
    random.seed(random_seed)
    
    # Shuffle questions to ensure random distribution
    shuffled_questions = questions.copy()
    random.shuffle(shuffled_questions)
    
    # Calculate split sizes
    total_questions = len(questions)
    test_size = max(1, int(total_questions * test_percentage))  # At least 1 test question
    train_size = total_questions - test_size
    
    # Ensure we don't exceed available questions
    if test_size >= total_questions:
        test_size = max(1, total_questions // 2)  # Use half if percentage is too high
        train_size = total_questions - test_size
    
    # Split the questions
    test_questions = shuffled_questions[:test_size]
    train_questions = shuffled_questions[test_size:]
    
    # Calculate actual percentage
    actual_percentage = test_size / total_questions if total_questions > 0 else 0.0
    
    return FinetuneTestSet(
        test_questions=test_questions,
        train_questions=train_questions,
        test_set_size=test_size,
        train_set_size=train_size,
        test_percentage=actual_percentage,
        random_seed=random_seed,
        split_timestamp=datetime.utcnow().isoformat() + "Z"
    )


def create_benchmark_with_finetune_set(
    questions: List[Dict[str, Any]], 
    config: BenchmarkConfig,
    generation_metadata: Optional[Dict[str, Any]] = None
) -> BenchmarkWithFinetuneSet:
    """
    Create a benchmark with optional finetune test set
    
    Args:
        questions: All generated questions
        config: Benchmark configuration
        generation_metadata: Additional metadata about generation process
        
    Returns:
        BenchmarkWithFinetuneSet with optional train/test split
    """
    finetune_test_set = None
    
    if config.finetune_test_set_enabled and len(questions) > 1:
        finetune_test_set = create_finetune_test_set(
            questions=questions,
            test_percentage=config.finetune_test_set_percentage,
            random_seed=config.finetune_random_seed
        )
    
    return BenchmarkWithFinetuneSet(
        config=config,
        all_questions=questions,
        finetune_test_set=finetune_test_set,
        generation_metadata=generation_metadata or {}
    )


def generate_benchmark_config(corpus_text: str, num_questions: int = 50) -> BenchmarkConfig:
    """Generate benchmark configuration from corpus"""
    primary_type = determine_primary_eval_type(corpus_text)
    key_concepts = extract_key_concepts(corpus_text)
    segments = sample_corpus_segments(corpus_text, num_segments=min(20, num_questions))
    
    is_deterministic_eval = is_deterministic(primary_type)
    
    return BenchmarkConfig(
        eval_type=primary_type,
        num_questions=num_questions,
        temperature=0.0 if is_deterministic_eval else 0.7,
        key_concepts=key_concepts,
        corpus_segments=segments
    )


class EvaluationFramework:
    """Main framework class for evaluation orchestration"""
    
    def __init__(self):
        self.benchmarks: List[BenchmarkItem] = []
        self.results: List[EvaluationResult] = []
        self.config: Optional[BenchmarkConfig] = None
    
    def create_benchmark_from_corpus(self, corpus_text: str, num_questions: int = 50) -> BenchmarkConfig:
        """Create benchmark configuration from corpus"""
        config = generate_benchmark_config(corpus_text, num_questions)
        self.config = config
        return config
    
    def add_benchmark_item(self, item: Union[BenchmarkItem, dict]):
        """Add benchmark item to framework"""
        if isinstance(item, dict):
            item = BenchmarkItem(**item)
        self.benchmarks.append(item)
    
    def evaluate_response(self, prediction: str, ground_truth: str, eval_type: EvaluationType) -> EvaluationResult:
        """Evaluate a single response"""
        from ..utils.text_processing import normalize_answer
        
        verification_method = get_verification_method(eval_type)
        
        if verification_method == VerificationMethod.EXACT_MATCH:
            score = 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0
        elif verification_method == VerificationMethod.SIMILARITY:
            # Placeholder for similarity computation
            score = 0.8  # Mock score
        elif verification_method == VerificationMethod.EXECUTION:
            # Placeholder for code execution
            score = 1.0  # Mock score
        else:
            score = 0.5  # Default
        
        metrics = {metric: score for metric in get_metrics_for_type(eval_type)}
        
        result = EvaluationResult(
            prediction=prediction,
            ground_truth=ground_truth,
            score=score,
            metrics=metrics,
            eval_type=eval_type,
            verified=True if is_deterministic(eval_type) else False
        )
        
        self.results.append(result)
        return result
    
    def compute_aggregate_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics across all results"""
        if not self.results:
            return {}
        
        metrics_by_type = defaultdict(list)
        for result in self.results:
            eval_type = result.eval_type
            for metric, value in result.metrics.items():
                metrics_by_type[f"{eval_type}_{metric}"].append(value)
        
        aggregated = {}
        for metric_name, values in metrics_by_type.items():
            aggregated[metric_name] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
        
        return aggregated
    
    def get_benchmark_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        return {
            'config': self.config.dict() if self.config else {},
            'num_benchmarks': len(self.benchmarks),
            'num_results': len(self.results),
            'aggregate_metrics': self.compute_aggregate_metrics(),
            'eval_type_distribution': self._get_eval_type_distribution()
        }
    
    def _get_eval_type_distribution(self) -> Dict[str, int]:
        """Get distribution of evaluation types"""
        distribution = defaultdict(int)
        for item in self.benchmarks:
            eval_type = item.eval_type
            distribution[eval_type] += 1
        return dict(distribution)


if __name__ == "__main__":
    # Example usage
    framework = EvaluationFramework()
    
    sample_corpus = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms
    that can learn from data. The basic principle involves training models on datasets
    to make predictions or decisions. Common algorithms include linear regression,
    decision trees, and neural networks. Performance is typically measured using
    metrics like accuracy, precision, and recall.
    """
    
    config = framework.create_benchmark_from_corpus(sample_corpus, num_questions=50)
    print("Generated benchmark configuration:")
    print(config.json(indent=2))