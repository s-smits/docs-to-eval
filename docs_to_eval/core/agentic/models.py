"""
Pydantic models for agentic benchmark generation protocol
Defines strict schemas for inter-agent communication and validation
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum
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
    agent_name: str
    agent_version: str = "v1"
    success: bool = True
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class ConceptExtractionResult(BaseModel):
    """Result from ConceptMiner agent"""
    key_concepts: List[str] = Field(min_items=1, max_items=50)
    supporting_snippets: Dict[str, str] = Field(default_factory=dict)
    concept_importance_scores: Dict[str, float] = Field(default_factory=dict)
    chunk_ids: List[str] = Field(default_factory=list)
    
    @validator('concept_importance_scores')
    def validate_scores(cls, v):
        """Ensure all scores are between 0 and 1"""
        for concept, score in v.items():
            if not 0 <= score <= 1:
                raise ValueError(f"Score for {concept} must be between 0 and 1")
        return v


class BenchmarkDraft(BaseModel):
    """Raw idea from QuestionWriter before adversarial enhancement"""
    question: str = Field(max_length=200)
    answer: str
    concept: str
    context_snippet: str = Field(max_length=800)
    chunk_id: Optional[str] = None
    expected_answer_type: AnswerType
    reasoning_chain: List[str] = Field(default_factory=list)
    difficulty_estimate: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    
    class Config:
        use_enum_values = True
    
    @validator('question')
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()
    
    @validator('answer')
    def answer_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Answer cannot be empty")
        return v.strip()


class BenchmarkCandidate(BaseModel):
    """Enhanced item after Adversary processing"""
    question: str = Field(max_length=150)
    answer: str
    context: Optional[str] = Field(max_length=500)
    options: Optional[List[str]] = Field(default=None, max_items=6)
    concept: str
    expected_answer_type: AnswerType
    difficulty: DifficultyLevel
    reasoning_chain: List[str] = Field(default_factory=list)
    adversarial_techniques: List[str] = Field(default_factory=list)
    distractor_rationale: Optional[str] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
    
    @validator('question')
    def validate_question_length(cls, v):
        if len(v) > 150:
            raise ValueError("Question too long after adversarial processing")
        return v.strip()
    
    @validator('options')
    def validate_multiple_choice(cls, v, values):
        """Ensure multiple choice items have valid options"""
        answer_type = values.get('expected_answer_type')
        if answer_type == AnswerType.MULTIPLE_CHOICE:
            if not v or len(v) < 2:
                raise ValueError("Multiple choice questions need at least 2 options")
            if len(v) > 6:
                raise ValueError("Too many options for multiple choice")
        return v


class BenchmarkMetadata(BaseModel):
    """Enhanced metadata for full provenance tracking"""
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
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class EnhancedBenchmarkItem(BaseModel):
    """Enhanced BenchmarkItem with full agentic metadata"""
    question: str = Field(max_length=150)
    answer: str
    context: Optional[str] = Field(max_length=500)
    options: Optional[List[str]] = None
    eval_type: EvaluationType
    metadata: BenchmarkMetadata
    
    # Additional agentic fields
    expected_answer_type: AnswerType
    reasoning_chain: List[str] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
    
    def to_standard_benchmark_item(self) -> Dict[str, Any]:
        """Convert to standard BenchmarkItem format for compatibility"""
        return {
            'question': self.question,
            'answer': self.answer,
            'context': self.context,
            'options': self.options,
            'eval_type': self.eval_type,
            'metadata': self.metadata.dict()
        }
    
    @validator('question')
    def validate_question_format(cls, v):
        """Ensure question meets formatting requirements"""
        v = v.strip()
        # Remove hard character limit to allow domain-specific detailed questions
        # Domain-specific questions need more space for entities, dates, and context
        if not v.endswith('?') and not any(word in v.lower() for word in ['what', 'how', 'why', 'when', 'where', 'which']):
            # Add question mark if it's clearly a question
            if any(word in v.lower() for word in ['is', 'are', 'can', 'could', 'would', 'should']):
                v += '?'
        return v


class ValidationResult(BaseModel):
    """Result from Validator agent"""
    accepted: bool
    score: float = Field(ge=0, le=1)
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    deterministic_check_passed: bool = True
    verification_method_used: str
    
    @validator('score')
    def validate_score_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Validation score must be between 0 and 1")
        return v


class AgentConfig(BaseModel):
    """Configuration for individual agents"""
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2000, gt=0, le=4096)
    timeout_seconds: float = Field(default=30.0, gt=0)
    retry_attempts: int = Field(default=2, ge=0, le=10)
    model_name: Optional[str] = None
    
    # Agent-specific configs
    concept_extraction_max: int = Field(default=20, gt=0, le=50)
    adversarial_intensity: float = Field(default=0.7, ge=0, le=1)
    validation_threshold: float = Field(default=0.6, ge=0, le=1)


class PipelineConfig(BaseModel):
    """Configuration for the entire agentic pipeline"""
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
    
    class Config:
        use_enum_values = True


# Utility functions for model validation and conversion

def validate_deterministic_answer_type(answer_type: AnswerType) -> bool:
    """Check if answer type should be deterministic"""
    deterministic_types = {
        AnswerType.NUMERIC_EXACT,
        AnswerType.NUMERIC_TOLERANCE,
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
    """Factory function for creating BenchmarkMetadata"""
    return BenchmarkMetadata(
        difficulty=difficulty,
        agents_used=agents_used,
        deterministic=deterministic,
        provenance=kwargs.get('provenance', {}),
        **{k: v for k, v in kwargs.items() if k != 'provenance'}
    )


# JSON Schema export for external validation
def export_schemas() -> Dict[str, Dict[str, Any]]:
    """Export all schemas as JSON Schema for external validation"""
    return {
        'BenchmarkDraft': BenchmarkDraft.schema(),
        'BenchmarkCandidate': BenchmarkCandidate.schema(),
        'EnhancedBenchmarkItem': EnhancedBenchmarkItem.schema(),
        'ValidationResult': ValidationResult.schema(),
        'PipelineConfig': PipelineConfig.schema()
    }