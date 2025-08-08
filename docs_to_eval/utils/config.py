"""
Configuration management with Pydantic models
"""

import json
import yaml
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
from functools import lru_cache


class EvaluationType(str, Enum):
    """Supported evaluation types"""
    MATHEMATICAL = "mathematical"
    CODE_GENERATION = "code_generation"
    FACTUAL_QA = "factual_qa"
    MULTIPLE_CHOICE = "multiple_choice"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CREATIVE_WRITING = "creative_writing"
    COMMONSENSE_REASONING = "commonsense_reasoning"
    READING_COMPREHENSION = "reading_comprehension"
    DOMAIN_KNOWLEDGE = "domain_knowledge"


class VerificationMethod(str, Enum):
    """Supported verification methods"""
    EXACT_MATCH = "exact_match"
    EXECUTION = "execution"
    SIMILARITY = "similarity"
    LLM_JUDGE = "llm_judge"


class SimilarityMethod(str, Enum):
    """Supported similarity calculation methods"""
    TOKEN_OVERLAP = "token_overlap"
    CHARACTER_OVERLAP = "character_overlap"
    LEVENSHTEIN = "levenshtein"
    NGRAM = "ngram"
    ROUGE_L = "rouge_l"
    BLEU = "bleu"
    SEMANTIC = "semantic"


class LLMConfig(BaseModel):
    """Configuration for LLM interface"""
    model_name: str = "openai/gpt-5-mini"
    temperature: float = Field(ge=0, le=2, default=0.7)
    max_tokens: int = Field(gt=0, le=131072, default=32768)
    timeout: int = Field(gt=0, default=30)
    max_retries: int = Field(ge=0, default=3)
    api_key: Optional[str] = None
    base_url: Optional[str] = "https://openrouter.ai/api/v1"
    provider: str = "openrouter"  # openrouter, openai, anthropic
    site_url: Optional[str] = "https://docs-to-eval.ai"  # For OpenRouter
    app_name: Optional[str] = "docs-to-eval"  # For OpenRouter
    mock_mode: bool = False  # Use mock LLMs when no API key is available
    
    def __init__(self, **kwargs):
        # Auto-load API key from environment if not provided
        import os
        if not kwargs.get('api_key'):
            kwargs['api_key'] = os.getenv('OPENROUTER_API_KEY')
        super().__init__(**kwargs)
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('Temperature must be between 0 and 2')
        return v
    
    @validator('base_url')
    def set_base_url_by_provider(cls, v, values):
        provider = values.get('provider', 'openrouter')
        if provider == 'openrouter':
            return "https://openrouter.ai/api/v1"
        elif provider == 'openai':
            return "https://api.openai.com/v1"
        elif provider == 'anthropic':
            return "https://api.anthropic.com"
        return v


class GenerationConfig(BaseModel):
    """Configuration for benchmark generation"""
    num_questions: int = Field(gt=0, le=1000, default=50)
    use_agentic: bool = True
    difficulty_levels: List[str] = Field(default=["basic", "intermediate", "advanced", "expert"])
    question_categories: List[str] = Field(default=["factual", "analytical", "synthesis", "application"])
    max_context_length: int = Field(gt=0, default=128000)
    min_question_length: int = Field(gt=0, default=10)
    max_question_length: int = Field(gt=0, default=500)
    quality_threshold: float = Field(ge=0, le=1, default=0.5)
    
    # Finetune test set configuration
    finetune_test_set_enabled: bool = Field(default=True, description="Enable creation of finetune test set")
    finetune_test_set_percentage: float = Field(ge=0.1, le=0.5, default=0.2, description="Percentage of questions to reserve for finetune testing (20% = 0.2)")
    finetune_random_seed: int = Field(ge=0, default=42, description="Random seed for reproducible train/test splits")


class VerificationConfig(BaseModel):
    """Configuration for response verification"""
    method: VerificationMethod = VerificationMethod.EXACT_MATCH
    similarity_method: SimilarityMethod = SimilarityMethod.TOKEN_OVERLAP
    similarity_threshold: float = Field(ge=0, le=1, default=0.8)
    exact_match_normalize: bool = True
    code_execution_timeout: int = Field(gt=0, default=10)
    llm_judge_criteria: List[str] = Field(default=["relevance", "accuracy", "completeness"])
    
    # Mixed verification settings
    use_mixed_verification: bool = Field(default=True, description="Enable intelligent mixed verification methods")
    mixed_verification_weights: bool = Field(default=True, description="Use weighted scoring for multiple verification methods")
    fuzzy_match_threshold: float = Field(ge=0, le=1, default=0.7, description="Threshold for fuzzy string matching")


class ChunkingConfig(BaseModel):
    """Configuration for text chunking with chonkie integration"""
    enable_chonkie: bool = Field(default=True, description="Enable chonkie library for smart chunking")
    chunking_strategy: str = Field(default="semantic", description="Chunking strategy: semantic, recursive, sentence")
    force_chunker: Optional[str] = Field(default=None, description="Force specific chunker: semantic, recursive, sentence")
    
    # Token-aware chunking (recommended for LLM contexts)
    use_token_chunking: bool = Field(default=True, description="Use token-aware chunking for LLM contexts")
    target_token_size: int = Field(default=3000, description="Target chunk size in tokens (~3k for good context)")
    min_token_size: int = Field(default=2000, description="Minimum chunk size in tokens")
    max_token_size: int = Field(default=4000, description="Maximum chunk size in tokens")
    overlap_tokens: int = Field(default=300, description="Overlap between chunks in tokens")
    
    # Character-based fallback settings
    target_chunk_size: int = Field(default=12000, description="Target chunk size in characters (fallback)")
    min_chunk_size: int = Field(default=8000, description="Minimum chunk size in characters")
    max_chunk_size: int = Field(default=16000, description="Maximum chunk size in characters")
    overlap_size: int = Field(default=1200, description="Overlap between chunks in characters")


class ReportingConfig(BaseModel):
    """Configuration for report generation"""
    include_individual_results: bool = True
    max_individual_results: int = Field(gt=0, default=200)
    include_visualizations: bool = True
    export_formats: List[str] = Field(default=["json", "html"])
    save_intermediate_results: bool = True


class SystemConfig(BaseModel):
    """Overall system configuration"""
    run_id: Optional[str] = None
    output_dir: str = "output"
    log_level: str = "INFO"
    max_concurrent_requests: int = Field(gt=0, default=5)
    enable_caching: bool = True
    cache_dir: str = "cache"
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()


class EvaluationConfig(BaseModel):
    """Complete evaluation configuration"""
    eval_type: EvaluationType = EvaluationType.DOMAIN_KNOWLEDGE
    llm: LLMConfig = Field(default_factory=LLMConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    
    class Config:
        use_enum_values = True
    
    @validator('verification')
    def validate_verification_method(cls, v, values):
        if 'eval_type' in values:
            eval_type = values['eval_type']
            # Automatically set appropriate verification method based on eval type
            if eval_type in [EvaluationType.MATHEMATICAL, EvaluationType.FACTUAL_QA, 
                           EvaluationType.MULTIPLE_CHOICE, EvaluationType.DOMAIN_KNOWLEDGE]:
                v.method = VerificationMethod.EXACT_MATCH
            elif eval_type == EvaluationType.CODE_GENERATION:
                v.method = VerificationMethod.EXECUTION
            elif eval_type in [EvaluationType.SUMMARIZATION, EvaluationType.TRANSLATION,
                             EvaluationType.READING_COMPREHENSION]:
                v.method = VerificationMethod.SIMILARITY
            elif eval_type == EvaluationType.CREATIVE_WRITING:
                v.method = VerificationMethod.LLM_JUDGE
        return v


def load_config(config_path: Union[str, Path]) -> EvaluationConfig:
    """Load configuration from file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    return EvaluationConfig(**config_data)


def save_config(config: EvaluationConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to file"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            yaml.dump(config.dict(), f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config.dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def create_default_config() -> EvaluationConfig:
    """Create default configuration"""
    return EvaluationConfig()


def validate_config(config: EvaluationConfig) -> List[str]:
    """Validate configuration and return list of warnings/issues"""
    warnings = []
    
    # Check LLM configuration
    if config.llm.temperature > 1.0 and config.eval_type in [
        EvaluationType.MATHEMATICAL, EvaluationType.CODE_GENERATION, EvaluationType.FACTUAL_QA
    ]:
        warnings.append(f"High temperature ({config.llm.temperature}) for deterministic evaluation type {config.eval_type}")
    
    # Check generation configuration
    if config.generation.num_questions > 500:
        warnings.append(f"Large number of questions ({config.generation.num_questions}) may take significant time")
    
    # Check verification configuration
    if (config.verification.method == VerificationMethod.SIMILARITY and 
        config.verification.similarity_threshold < 0.5):
        warnings.append(f"Low similarity threshold ({config.verification.similarity_threshold}) may accept poor matches")
    
    # Check system configuration
    if config.system.max_concurrent_requests > 10:
        warnings.append(f"High concurrency ({config.system.max_concurrent_requests}) may hit API rate limits")
    
    return warnings


def merge_configs(base_config: EvaluationConfig, override_config: Dict[str, Any]) -> EvaluationConfig:
    """Merge configuration with overrides"""
    base_dict = base_config.dict()
    
    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    merged_dict = deep_merge(base_dict, override_config)
    return EvaluationConfig(**merged_dict)


def config_from_args(args: Dict[str, Any]) -> EvaluationConfig:
    """Create configuration from command line arguments"""
    config = create_default_config()
    
    # Map common CLI args to config structure
    arg_mapping = {
        'eval_type': ['eval_type'],
        'num_questions': ['generation', 'num_questions'],
        'temperature': ['llm', 'temperature'],
        'max_tokens': ['llm', 'max_tokens'],
        'output_dir': ['system', 'output_dir'],
        'log_level': ['system', 'log_level'],
        'similarity_threshold': ['verification', 'similarity_threshold'],
        'use_agentic': ['generation', 'use_agentic'],
    }
    
    config_dict = config.dict()
    
    for arg_name, config_path in arg_mapping.items():
        if arg_name in args and args[arg_name] is not None:
            # Navigate to the nested config location
            current = config_dict
            for key in config_path[:-1]:
                current = current[key]
            current[config_path[-1]] = args[arg_name]
    
    return EvaluationConfig(**config_dict)


class ConfigManager:
    """Manage configuration with environment variables and file overrides"""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        self.config_file = Path(config_file) if config_file else None
        self.config = self._load_config()
    
    def _load_config(self) -> EvaluationConfig:
        """Load configuration from file or create default"""
        if self.config_file and self.config_file.exists():
            return load_config(self.config_file)
        else:
            return create_default_config()
    
    def update_from_args(self, args: Dict[str, Any]) -> None:
        """Update configuration from command line arguments"""
        override_config = config_from_args(args)
        self.config = merge_configs(self.config, override_config.dict())
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables"""
        import os
        
        env_mapping = {
            'DOCS_TO_EVAL_LOG_LEVEL': ['system', 'log_level'],
            'DOCS_TO_EVAL_OUTPUT_DIR': ['system', 'output_dir'],
            'DOCS_TO_EVAL_API_KEY': ['llm', 'api_key'],
            'OPENROUTER_API_KEY': ['llm', 'api_key'],  # Support standard OpenRouter env var
            'DOCS_TO_EVAL_MODEL_NAME': ['llm', 'model_name'],
            'DOCS_TO_EVAL_MAX_TOKENS': ['llm', 'max_tokens'],
            'DOCS_TO_EVAL_TEMPERATURE': ['llm', 'temperature'],
            'DOCS_TO_EVAL_PROVIDER': ['llm', 'provider'],
            'DOCS_TO_EVAL_BASE_URL': ['llm', 'base_url'],
        }
        
        # Set PyTorch device fallback for Apple Silicon MPS compatibility
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        
        config_dict = self.config.dict()
        
        for env_var, config_path in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Type conversion
                if config_path[-1] in ['max_tokens', 'timeout', 'max_retries', 'num_questions']:
                    value = int(value)
                elif config_path[-1] in ['temperature', 'similarity_threshold', 'quality_threshold']:
                    value = float(value)
                elif config_path[-1] in ['use_agentic', 'enable_caching', 'include_visualizations']:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                
                # Navigate to the nested config location
                current = config_dict
                for key in config_path[:-1]:
                    current = current[key]
                current[config_path[-1]] = value
        
        self.config = EvaluationConfig(**config_dict)
    
    def save(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file"""
        save_path = Path(output_path) if output_path else self.config_file
        if save_path:
            save_config(self.config, save_path)
    
    def get_config(self) -> EvaluationConfig:
        """Get current configuration"""
        return self.config


if __name__ == "__main__":
    # Test configuration system
    config = create_default_config()
    print("Default config:")
    print(config.json(indent=2))
    
    # Test validation
    warnings = validate_config(config)
    if warnings:
        print("\nConfiguration warnings:")
        for warning in warnings:
            print(f"- {warning}")
    
    # Test config manager
    manager = ConfigManager()
    manager.update_from_env()
    print("\nConfig after environment update:")
    print(manager.get_config().json(indent=2))


# Provide backward-compatible re-export for tests expecting BenchmarkConfig in utils.config
try:
    # Import from core.evaluation where BenchmarkConfig is defined
    from ..core.evaluation import BenchmarkConfig as _BenchmarkConfig
    BenchmarkConfig = _BenchmarkConfig  # type: ignore
except Exception:
    # If import fails during partial environments, define a minimal placeholder to avoid import errors
    class BenchmarkConfig(BaseModel):  # type: ignore
        eval_type: EvaluationType
        num_questions: int = Field(gt=0, le=1000, default=50)


# Define a basic EvaluationConfig class for EVAL_TYPES
class BasicEvaluationConfig:
    def __init__(self, deterministic: bool, verification: VerificationMethod, metrics: List[str]):
        self.deterministic = deterministic
        self.verification = verification
        self.metrics = metrics
    
    def dict(self):
        return {
            'deterministic': self.deterministic,
            'verification': self.verification,
            'metrics': self.metrics
        }


# Evaluation Types Configuration
EVAL_TYPES = {
    EvaluationType.MATHEMATICAL: BasicEvaluationConfig(
        deterministic=True,
        verification=VerificationMethod.EXACT_MATCH,
        metrics=['accuracy', 'pass_rate']
    ),
    EvaluationType.CODE_GENERATION: BasicEvaluationConfig(
        deterministic=True,
        verification=VerificationMethod.EXECUTION,
        metrics=['pass_rate', 'syntax_correctness']
    ),
    EvaluationType.FACTUAL_QA: BasicEvaluationConfig(
        deterministic=True,
        verification=VerificationMethod.EXACT_MATCH,
        metrics=['accuracy', 'f1_score']
    ),
    EvaluationType.MULTIPLE_CHOICE: BasicEvaluationConfig(
        deterministic=True,
        verification=VerificationMethod.EXACT_MATCH,
        metrics=['accuracy', 'normalized_accuracy']
    ),
    EvaluationType.SUMMARIZATION: BasicEvaluationConfig(
        deterministic=False,
        verification=VerificationMethod.SIMILARITY,
        metrics=['rouge_l', 'bleu', 'semantic_similarity']
    ),
    EvaluationType.TRANSLATION: BasicEvaluationConfig(
        deterministic=False,
        verification=VerificationMethod.SIMILARITY,
        metrics=['bleu', 'chrf', 'semantic_similarity']
    ),
    EvaluationType.CREATIVE_WRITING: BasicEvaluationConfig(
        deterministic=False,
        verification=VerificationMethod.LLM_JUDGE,
        metrics=['coherence', 'creativity', 'fluency']
    ),
    EvaluationType.COMMONSENSE_REASONING: BasicEvaluationConfig(
        deterministic=True,
        verification=VerificationMethod.EXACT_MATCH,
        metrics=['accuracy']
    ),
    EvaluationType.READING_COMPREHENSION: BasicEvaluationConfig(
        deterministic=False,
        verification=VerificationMethod.SIMILARITY,
        metrics=['f1_score', 'exact_match', 'semantic_similarity']
    ),
    EvaluationType.DOMAIN_KNOWLEDGE: BasicEvaluationConfig(
        deterministic=True,
        verification=VerificationMethod.EXACT_MATCH,
        metrics=['accuracy', 'precision', 'recall']
    )
}


# ========================================
# OPTIMIZED CLASSIFICATION SYSTEM
# ========================================

# Module-level compiled patterns (lazy-loaded for efficiency)
_CLASSIFICATION_PATTERNS = None


def _init_classification_patterns():
    """Initialize and compile classification patterns once"""
    global _CLASSIFICATION_PATTERNS
    if _CLASSIFICATION_PATTERNS is not None:
        return _CLASSIFICATION_PATTERNS
    
    # Define patterns with weights for intelligent scoring
    # Format: (pattern, weight, description)
    patterns = {
        EvaluationType.MATHEMATICAL: [
            # High confidence signals (weight >= 8)
            (r'\d+\s*[\+\-\*\/]\s*\d+\s*=', 10.0, "complete equations"),
            (r'\\frac\{.*?\}\{.*?\}', 9.0, "LaTeX fractions"),
            (r'\$[^\$]+\$', 8.0, "LaTeX inline math"),
            (r'solve\s+for\s+[a-z]\b', 8.0, "explicit solve problems"),
            
            # Medium confidence signals (weight 4-7)
            (r'\b(calculate|compute|equation|formula)\b', 6.0, "math keywords"),
            (r'\b(integral|derivative|matrix|vector)\b', 6.0, "advanced math"),
            (r'\b(sin|cos|tan|log|ln|sqrt)\b', 5.0, "math functions"),
            (r'\d+\s*[\+\-\*\/\^]\s*\d+', 5.0, "arithmetic operations"),
            
            # Low confidence signals (weight 1-3)
            (r'\b(sum|product|mean|median)\b', 3.0, "basic math terms"),
            (r'\b\d+\.?\d*%', 2.0, "percentages"),
        ],
        
        EvaluationType.CODE_GENERATION: [
            # High confidence signals
            (r'```[a-z]+\n', 10.0, "code blocks with language"),
            (r'def\s+\w+\([^)]*\):', 10.0, "Python functions"),
            (r'function\s+\w+\([^)]*\)\s*\{', 9.0, "JavaScript functions"),
            (r'class\s+\w+[:\({]', 9.0, "class definitions"),
            
            # Medium confidence signals
            (r'import\s+\w+|from\s+\w+\s+import', 7.0, "imports"),
            (r'\b(return|print|console\.log)\b', 5.0, "code keywords"),
            (r'if\s*\([^)]+\)|for\s*\([^)]+\)', 5.0, "control flow"),
            
            # Low confidence signals
            (r'\b(var|let|const|int|string|bool)\b', 3.0, "type keywords"),
        ],
        
        EvaluationType.FACTUAL_QA: [
            # High confidence signals
            (r'\b(who|what|when|where|why|how)\s+(is|was|did|does)\b', 8.0, "question patterns"),
            (r'\b(capital of|president of|founded in|invented by)\b', 8.0, "factual phrases"),
            
            # Medium confidence signals
            (r'\b(born|died|founded|established|created)\s+in\s+\d{4}', 6.0, "historical facts"),
            (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', 4.0, "proper names"),
            
            # Low confidence signals
            (r'\b\d{4}\b', 2.0, "years"),
        ],
        
        EvaluationType.DOMAIN_KNOWLEDGE: [
            # High confidence signals
            (r'\b(definition|concept|theory|principle)\s+of\b', 7.0, "academic terms"),
            
            # Medium confidence signals
            (r'\b(according to|research shows|studies indicate)\b', 5.0, "research language"),
            (r'\b(methodology|framework|approach|model)\b', 4.0, "domain terms"),
            
            # Low confidence signals
            (r'\b(analysis|evaluation|assessment)\b', 2.0, "analytical terms"),
        ]
    }
    
    # Compile all patterns
    _CLASSIFICATION_PATTERNS = {}
    for eval_type, pattern_list in patterns.items():
        compiled = []
        for pattern, weight, desc in pattern_list:
            try:
                compiled.append((re.compile(pattern, re.IGNORECASE), weight, desc))
            except re.error:
                # Skip invalid patterns
                continue
        _CLASSIFICATION_PATTERNS[eval_type] = compiled
    
    return _CLASSIFICATION_PATTERNS

@lru_cache(maxsize=128)
def analyze_corpus_content(corpus_text: str) -> Dict[str, float]:
    """
    Optimized two-tier classification system with early exit for obvious cases.
    Returns a dictionary mapping evaluation types to confidence scores.
    """
    if not corpus_text or len(corpus_text.strip()) == 0:
        return {EvaluationType.DOMAIN_KNOWLEDGE: 0.5}
    
    # Initialize patterns if needed
    patterns = _init_classification_patterns()
    
    # TIER 1: Fast path - check first 2000 chars for strong signals
    sample_text = corpus_text[:2000] if len(corpus_text) > 2000 else corpus_text
    
    # Score each evaluation type
    scores = {}
    max_score = 0
    
    for eval_type, pattern_list in patterns.items():
        score = 0
        high_confidence_hits = 0
        
        for pattern, weight, desc in pattern_list:
            # Search in sample for speed
            if pattern.search(sample_text):
                score += weight
                if weight >= 8.0:  # High confidence signal
                    high_confidence_hits += 1
        
        scores[eval_type] = score
        
        # Track best match
        if score > max_score:
            max_score = score
        
        # EARLY EXIT: If we have very strong signals, return immediately
        if high_confidence_hits >= 2 or score >= 15.0:
            return {eval_type: 1.0}
    
    # TIER 2: If no strong signal, do full analysis on best candidates
    if max_score < 8.0:
        # Analyze full text for top 2 scoring types
        top_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
        
        for eval_type, initial_score in top_types:
            full_score = _analyze_full_text(corpus_text, patterns[eval_type])
            scores[eval_type] = full_score
    
    # Normalize scores to 0-1 range
    if max(scores.values()) > 0:
        max_val = max(scores.values())
        scores = {k: v / max_val for k, v in scores.items()}
    else:
        # Default fallback
        scores = {EvaluationType.DOMAIN_KNOWLEDGE: 0.5}
    
    return scores


def _analyze_full_text(text: str, pattern_list: List[Tuple]) -> float:
    """
    Analyze full text with given patterns.
    Returns weighted score based on pattern matches.
    """
    score = 0
    text_length = len(text.split())
    
    for pattern, weight, desc in pattern_list:
        # Count all matches in full text
        matches = len(pattern.findall(text))
        
        # Apply weight with diminishing returns for multiple matches
        if matches > 0:
            # First match gets full weight, subsequent matches get less
            weighted_matches = weight * (1 + 0.5 * min(matches - 1, 5))
            score += weighted_matches
    
    # Normalize by text length (per 100 words)
    if text_length > 0:
        score = score * (100 / text_length)
    
    return score

def classify_corpus_simple(corpus_text: str) -> EvaluationType:
    """
    Simple helper to get the most likely evaluation type for a corpus.
    Returns a single EvaluationType instead of score dictionary.
    """
    scores = analyze_corpus_content(corpus_text)
    if not scores:
        return EvaluationType.DOMAIN_KNOWLEDGE
    
    # Return the type with highest score
    return max(scores.items(), key=lambda x: x[1])[0]


def get_classification_confidence(corpus_text: str) -> Tuple[EvaluationType, float]:
    """
    Get the classification result with confidence score.
    Returns: (evaluation_type, confidence_score)
    """
    scores = analyze_corpus_content(corpus_text)
    if not scores:
        return EvaluationType.DOMAIN_KNOWLEDGE, 0.5
    
    best_type = max(scores.items(), key=lambda x: x[1])
    return best_type[0], best_type[1]


def explain_classification(corpus_text: str, detailed: bool = False) -> Dict[str, Any]:
    """
    Explain why a corpus was classified a certain way.
    Useful for debugging and understanding classification decisions.
    """
    patterns = _init_classification_patterns()
    sample_text = corpus_text[:2000] if len(corpus_text) > 2000 else corpus_text
    
    results = {}
    for eval_type, pattern_list in patterns.items():
        matches = []
        total_score = 0
        
        for pattern, weight, desc in pattern_list:
            if pattern.search(sample_text):
                total_score += weight
                if detailed:
                    # Find actual matches for explanation
                    found = pattern.findall(sample_text)[:3]  # First 3 matches
                    matches.append({
                        'description': desc,
                        'weight': weight,
                        'examples': found[:2] if found else []
                    })
        
        results[eval_type.value] = {
            'score': total_score,
            'matches': matches if detailed else len(matches),
            'confidence': 'high' if total_score >= 15 else 'medium' if total_score >= 8 else 'low'
        }
    
    # Add final classification
    classification = classify_corpus_simple(corpus_text)
    results['final_classification'] = classification.value
    
    return results


# ========================================
# BACKWARD COMPATIBILITY
# ========================================

def analyze_corpus_content_comprehensive(corpus_text: str) -> Dict[str, float]:
    """Backward compatibility wrapper"""
    return analyze_corpus_content(corpus_text)


# Legacy function name for backward compatibility
def analyze_corpus_content_original(corpus_text: str) -> Dict[str, float]:
    """Original analyze_corpus_content implementation (for reference/testing)"""
    patterns = {
        EvaluationType.MATHEMATICAL: [
            # Basic arithmetic operations
            r'\d+\s*[\+\-\*\/\^]\s*\d+',
            r'\d+\s*[\+\-\*\/\^]\s*\(\d+\)',
            r'\(\d+\s*[\+\-\*\/\^]\s*\d+\)',
            
            # Mathematical terms and keywords
            r'\b(equation|formula|calculate|solve|compute)\b',
            r'\b(equals?|equal to|=)\b',
            r'\b(sum|product|difference|quotient|fraction)\b',
            r'\b(root|square root|cube root|sqrt)\b',
            r'\b(exponent|power|logarithm|log|ln)\b',
            r'\b(integral|derivative|differentiate|integrate)\b',
            r'\b(trigonometric|sin|cos|tan|sine|cosine|tangent)\b',
            r'\b(matrix|vector|determinant|eigenvalue|eigenvector)\b',
            r'\b(probability|statistics|variance|standard deviation|mean|median)\b',
            r'\b(theorem|proof|lemma|corollary|axiom)\b',
            r'\b(geometry|algebra|calculus|arithmetic)\b',
            
            # Mathematical symbols and expressions
            r'\b\d+%|\$\d+',
            r'[<>=≤≥≠≈∞]',
            r'±|∓|×|÷|∑|∏|∫|∂|∇|Δ|π|θ|φ|α|β|γ',
            
            # LaTeX mathematical expressions
            r'\\frac\{.*?\}\{.*?\}',
            r'\\sqrt\{.*?\}',
            r'\\sum_\{.*?\}\^\{.*?\}',
            r'\\int_\{.*?\}\^\{.*?\}',
            r'\\lim_\{.*?\}',
            r'\$.*?\$',
            r'\\\(.*?\\\)',
            r'\\\[.*?\\\]',
            r'\\begin\{(equation|align|matrix|pmatrix|bmatrix)\}',
            
            # Fractions and ratios
            r'\d+/\d+',
            r'\d+:\d+',
            r'\b(ratio|proportion|percentage)\b',
            
            # Mathematical sets and logic
            r'\{.*?\}',  # Set notation
            r'\\cup|\\cap|\\subset|\\superset|\\in|\\notin',
            r'\\forall|\\exists|\\neg|\\land|\\lor|\\implies',
            
            # Advanced mathematics
            r'\b(polynomial|exponential|factorial|binomial)\b',
            r'\b(limit|continuity|convergence|divergence)\b',
            r'\b(linear|quadratic|cubic|quartic)\b',
            r'n!|\d+!',
            r'x\^\d+|x\^n|x\^[a-z]',
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