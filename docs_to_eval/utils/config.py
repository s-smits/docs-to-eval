"""
Configuration management with Pydantic models
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


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
    model_name: str = "gpt-3.5-turbo"
    temperature: float = Field(ge=0, le=2, default=0.7)
    max_tokens: int = Field(gt=0, le=8192, default=512)
    timeout: int = Field(gt=0, default=30)
    max_retries: int = Field(ge=0, default=3)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('Temperature must be between 0 and 2')
        return v


class GenerationConfig(BaseModel):
    """Configuration for benchmark generation"""
    num_questions: int = Field(gt=0, le=1000, default=50)
    use_agentic: bool = True
    difficulty_levels: List[str] = Field(default=["basic", "intermediate", "advanced", "expert"])
    question_categories: List[str] = Field(default=["factual", "analytical", "synthesis", "application"])
    max_context_length: int = Field(gt=0, default=2000)
    min_question_length: int = Field(gt=0, default=10)
    max_question_length: int = Field(gt=0, default=500)
    quality_threshold: float = Field(ge=0, le=1, default=0.5)


class VerificationConfig(BaseModel):
    """Configuration for response verification"""
    method: VerificationMethod = VerificationMethod.EXACT_MATCH
    similarity_method: SimilarityMethod = SimilarityMethod.TOKEN_OVERLAP
    similarity_threshold: float = Field(ge=0, le=1, default=0.8)
    exact_match_normalize: bool = True
    code_execution_timeout: int = Field(gt=0, default=10)
    llm_judge_criteria: List[str] = Field(default=["relevance", "accuracy", "completeness"])


class ReportingConfig(BaseModel):
    """Configuration for report generation"""
    include_individual_results: bool = True
    max_individual_results: int = Field(gt=0, default=100)
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
            'DOCS_TO_EVAL_MODEL_NAME': ['llm', 'model_name'],
            'DOCS_TO_EVAL_MAX_TOKENS': ['llm', 'max_tokens'],
            'DOCS_TO_EVAL_TEMPERATURE': ['llm', 'temperature'],
        }
        
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