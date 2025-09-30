"""
Pydantic models for agentic benchmark generation protocol
Defines strict schemas for inter-agent communication and validation

Updated for Pydantic V2 best practices with field_validator, model_validator, and ConfigDict
"""

from typing import Dict, List, Any, Optional, Union
from typing_extensions import Self
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum
import json
from datetime import datetime

from ..evaluation import EvaluationType


class DifficultyLevel(str, Enum):
    """Difficulty levels for benchmark items"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate" 
    HARD = "hard"
    EXPERT = "expert"


class AnswerType(str, Enum):
    """Expected answer types for deterministic classification"""
    NUMERIC_EXACT = "numeric_exact"
    NUMERIC_TOLERANCE = "numeric_tolerance" 
    STRING_EXACT = "string_exact"
    CODE = "code"
    MULTIPLE_CHOICE = "multiple_choice"
    FREE_TEXT = "free_text"
    BOOLEAN = "boolean"


class AgentResponse(BaseModel):
    """Base response structure for all agents"""
    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)
    
    agent_name: str
    agent_version: str = "v1"
    success: bool = True
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConceptExtractionResult(BaseModel):
    """Result from ConceptMiner agent"""
    model_config = ConfigDict(validate_assignment=True)
    
    key_concepts: List[str] = Field(min_length=1, max_length=50)
    supporting_snippets: Dict[str, str] = Field(default_factory=dict)
    concept_importance_scores: Dict[str, float] = Field(default_factory=dict)
    chunk_ids: List[str] = Field(default_factory=list)
    
    @field_validator('concept_importance_scores', mode='after')
    @classmethod
    def validate_scores(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure all scores are between 0 and 1"""
        for concept, score in v.items():
            if not 0 <= score <= 1:
                raise ValueError(f"Score for {concept} must be between 0 and 1, got {score}")
        return v


class BenchmarkDraft(BaseModel):
    """Raw idea from QuestionWriter before adversarial enhancement"""
    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)
    
    question: str = Field(max_length=200)
    answer: str
    concept: str
    context_snippet: str = Field(max_length=800)
    chunk_id: Optional[str] = None
    expected_answer_type: AnswerType
    reasoning_chain: List[str] = Field(default_factory=list)
    difficulty_estimate: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    
    @field_validator('question', mode='after')
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()
    
    @field_validator('answer', mode='after')
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Answer cannot be empty")
        return v.strip()


class BenchmarkCandidate(BaseModel):
    """Enhanced item after Adversary processing"""
    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)
    
    question: str = Field(max_length=150)
    answer: str
    context: Optional[str] = Field(default=None, max_length=500)
    options: Optional[List[str]] = Field(default=None, max_length=6)
    concept: str
    expected_answer_type: AnswerType
    difficulty: DifficultyLevel
    reasoning_chain: List[str] = Field(default_factory=list)
    adversarial_techniques: List[str] = Field(default_factory=list)
    distractor_rationale: Optional[str] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('question', mode='after')
    @classmethod
    def validate_question_length(cls, v: str) -> str:
        """Enforce maximum question length"""
        if len(v) > 150:
            return v[:150]
        return v
    
    @model_validator(mode='after')
    def validate_multiple_choice(self) -> Self:
        """Ensure multiple choice has proper options"""
        if self.expected_answer_type == AnswerType.MULTIPLE_CHOICE:
            if not self.options or len(self.options) < 2:
                raise ValueError("Multiple choice must have at least 2 options")
        return self


class BenchmarkMetadata(BaseModel):
    """Enhanced metadata for full provenance tracking"""
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()}
    )
    
    source: str = "agentic_v2"
    deterministic: bool = True
    difficulty: DifficultyLevel
    agents_used: List[str] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    generation_timestamp: datetime = Field(default_factory=datetime.now)
    corpus_chunk_id: Optional[str] = None
    concept_importance: float = Field(default=0.5, ge=0, le=1)
    validation_score: Optional[float] = Field(default=None, ge=0, le=1)
    adversarial_techniques: List[str] = Field(default_factory=list)


class EnhancedBenchmarkItem(BaseModel):
    """Enhanced BenchmarkItem with full agentic metadata"""
    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)
    
    question: str = Field(max_length=150)
    answer: str
    context: Optional[str] = Field(default=None, max_length=500)
    options: Optional[List[str]] = None
    eval_type: EvaluationType
    metadata: BenchmarkMetadata
    
    # Additional agentic fields
    expected_answer_type: AnswerType
    reasoning_chain: List[str] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)
    
    def to_standard_benchmark_item(self) -> Dict[str, Any]:
        """Convert to standard benchmark format for compatibility"""
        return {
            "question": self.question,
            "answer": self.answer,
            "context": self.context,
            "options": self.options,
            "eval_type": self.eval_type,
            "metadata": {
                **self.metadata.model_dump(),
                "expected_answer_type": self.expected_answer_type,
                "reasoning_chain": self.reasoning_chain,
                "variables": self.variables
            }
        }
    
    @field_validator('question', mode='after')
    @classmethod
    def validate_question_format(cls, v: str) -> str:
        """Ensure question is properly formatted"""
        v = v.strip()
        if len(v) < 5:
            raise ValueError("Question is too short (minimum 5 characters)")
        if len(v) > 150:
            v = v[:150]
        # Ensure proper punctuation
        if not v.endswith('?') and not v.endswith('.'):
            v += '?'
        return v


class ValidationResult(BaseModel):
    """Result from Validator agent"""
    model_config = ConfigDict(validate_assignment=True)
    
    accepted: bool
    score: float = Field(ge=0, le=1)
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    deterministic_check_passed: bool = True
    verification_method_used: str
    
    @field_validator('score', mode='after')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure score is within valid range"""
        if not 0 <= v <= 1:
            raise ValueError(f"Validation score must be between 0 and 1, got {v}")
        return v


class AgentConfig(BaseModel):
    """Configuration for individual agents"""
    model_config = ConfigDict(validate_assignment=True)
    
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, gt=0, le=4096)
    timeout_seconds: float = Field(default=30.0, gt=0)
    retry_attempts: int = Field(default=3, ge=0, le=10)
    model_name: Optional[str] = None
    
    # Agent-specific configs
    concept_extraction_max: int = Field(default=20, gt=0, le=50)
    adversarial_intensity: float = Field(default=0.7, ge=0, le=1)
    validation_threshold: float = Field(default=0.6, ge=0, le=1)


class PipelineConfig(BaseModel):
    """Configuration for the entire agentic pipeline"""
    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)
    
    difficulty: DifficultyLevel = DifficultyLevel.HARD
    num_questions: int = Field(default=50, gt=0, le=1000)
    oversample_factor: float = Field(default=3.0, ge=1.0, le=10.0)
    parallel_batch_size: int = Field(default=5, gt=0, le=20)
    max_retry_cycles: int = Field(default=2, ge=0, le=5)
    
    # Quality control
    min_validation_score: float = Field(default=0.6, ge=0, le=1)
    enforce_deterministic_split: bool = True
    
    # Agent configurations
    agent_configs: Dict[str, AgentConfig] = Field(default_factory=dict)


# Utility functions

def validate_deterministic_answer_type(answer_type: AnswerType) -> bool:
    """Check if an answer type should be treated as deterministic"""
    deterministic_types = {
        AnswerType.NUMERIC_EXACT,
        AnswerType.STRING_EXACT,
        AnswerType.CODE,
        AnswerType.MULTIPLE_CHOICE,
        AnswerType.BOOLEAN
    }
    return answer_type in deterministic_types


def create_enhanced_metadata(
    difficulty: DifficultyLevel,
    agents_used: List[str],
    deterministic: bool,
    **kwargs
) -> BenchmarkMetadata:
    """Factory function to create BenchmarkMetadata with validation"""
    return BenchmarkMetadata(
        difficulty=difficulty,
        agents_used=agents_used,
        deterministic=deterministic,
        **kwargs
    )


def export_schemas() -> Dict[str, Dict[str, Any]]:
    """Export JSON schemas for all models"""
    return {
        "BenchmarkDraft": BenchmarkDraft.model_json_schema(),
        "BenchmarkCandidate": BenchmarkCandidate.model_json_schema(),
        "EnhancedBenchmarkItem": EnhancedBenchmarkItem.model_json_schema(),
        "BenchmarkMetadata": BenchmarkMetadata.model_json_schema(),
        "ValidationResult": ValidationResult.model_json_schema(),
        "PipelineConfig": PipelineConfig.model_json_schema(),
    }