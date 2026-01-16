"""
Agentic benchmark generation system with lm-evaluation-harness integration
Provides automated, intelligent benchmark creation with specialized agents
"""

from .models import (
    BenchmarkDraft,
    BenchmarkCandidate,
    EnhancedBenchmarkItem,
    AgentResponse,
    BenchmarkMetadata,
    ValidationResult,
    AgentConfig,
    PipelineConfig,
    DifficultyLevel,
    AnswerType
)

# Legacy 5-agent system (deprecated, kept for compatibility)
from .agents import (
    ConceptMiner,
    QuestionWriter,
    Adversary,
    Refiner,
    Validator
)
from .orchestrator import AgenticBenchmarkOrchestrator
from .generator import AgenticBenchmarkGenerator
from ..agentic_question_generator import AgenticQuestionGenerator, QuestionItem

# Streamlined 3-agent system (recommended for production)
from .streamlined_agents import (
    ConceptExtractor,
    QuestionGenerator,
    QualityValidator
)
from .streamlined_orchestrator import StreamlinedOrchestrator

# Validation utilities
from .validation import (
    DeterministicGuardRail,
    QualityController,
    ComprehensiveValidator
)

# Backward-compatibility imports (historical public API)
# Provide AgenticQuestionGenerator and QuestionItem at package level
from ..agentic_question_generator import (
    AgenticQuestionGenerator,
    QuestionItem,
)

# LM-Evaluation-Harness Integration
from .lm_eval_exporter import (
    export_agentic_benchmark_to_lm_eval,
    validate_lm_eval_export
)
from .lm_eval_utils import (
    generate_and_export_benchmark,
    quick_export_demo
)

__all__ = [
    # Core models
    'BenchmarkDraft',
    'BenchmarkCandidate',
    'EnhancedBenchmarkItem', 
    'AgentResponse',
    'BenchmarkMetadata',
    'ValidationResult',
    'AgentConfig',
    'PipelineConfig',
    'DifficultyLevel',
    'AnswerType',
    
    # Legacy Agents (deprecated)
    'ConceptMiner',
    'QuestionWriter',
    'Adversary',
    'Refiner',
    'Validator',
    
    # Streamlined Agents (recommended)
    'ConceptExtractor',
    'QuestionGenerator', 
    'QualityValidator',
    
    # Orchestration
    'AgenticBenchmarkOrchestrator',  # Legacy
    'AgenticBenchmarkGenerator',     # Legacy
     'AgenticQuestionGenerator',      # Legacy compatibility
     'QuestionItem',                  # Legacy compatibility
    'StreamlinedOrchestrator',        # Recommended
    
    # Validation
    'DeterministicGuardRail',
    'QualityController',
    'ComprehensiveValidator',
    
     # Legacy public API (back-compat)
     'AgenticQuestionGenerator',
     'QuestionItem',
     
    # LM-Eval Integration
    'export_agentic_benchmark_to_lm_eval',
    'validate_lm_eval_export',
    'generate_and_export_benchmark',
    'quick_export_demo'
]