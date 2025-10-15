"""
Custom exceptions for the docs-to-eval system
Provides clear, actionable error messages for different failure modes
"""

from typing import Optional, Dict, Any, List


class DocsToEvalError(Exception):
    """Base exception for all docs-to-eval errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# Corpus and Input Errors

class CorpusError(DocsToEvalError):
    """Raised when there are issues with the input corpus"""
    pass


class EmptyCorpusError(CorpusError):
    """Raised when corpus text is empty or too short"""
    
    def __init__(self, length: int, minimum: int = 100):
        super().__init__(
            f"Corpus is too short: {length} characters (minimum: {minimum})",
            {"actual_length": length, "minimum_length": minimum}
        )


class InvalidCorpusFormatError(CorpusError):
    """Raised when corpus format is invalid"""
    pass


# Classification Errors

class ClassificationError(DocsToEvalError):
    """Raised when corpus classification fails"""
    pass


class AmbiguousClassificationError(ClassificationError):
    """Raised when classification confidence is too low"""
    
    def __init__(self, confidence: float, threshold: float = 0.5):
        super().__init__(
            f"Classification confidence too low: {confidence:.2f} (threshold: {threshold})",
            {"confidence": confidence, "threshold": threshold}
        )


# Generation Errors

class GenerationError(DocsToEvalError):
    """Raised when benchmark generation fails"""
    pass


class InsufficientQuestionsError(GenerationError):
    """Raised when unable to generate enough questions"""
    
    def __init__(self, generated: int, requested: int):
        super().__init__(
            f"Generated {generated} questions, but {requested} were requested",
            {"generated": generated, "requested": requested}
        )


class ConceptExtractionError(GenerationError):
    """Raised when concept extraction fails"""
    
    def __init__(self, reason: str):
        super().__init__(f"Failed to extract concepts: {reason}")


# Validation Errors

class ValidationError(DocsToEvalError):
    """Raised when validation fails"""
    pass


class QualityThresholdError(ValidationError):
    """Raised when generated items don't meet quality threshold"""
    
    def __init__(self, avg_score: float, threshold: float, items_count: int):
        super().__init__(
            f"Quality score {avg_score:.2f} below threshold {threshold} ({items_count} items)",
            {"avg_score": avg_score, "threshold": threshold, "items_count": items_count}
        )


class DeterministicConsistencyError(ValidationError):
    """Raised when deterministic items fail consistency checks"""
    
    def __init__(self, failed_items: List[int], total_items: int):
        super().__init__(
            f"{len(failed_items)} of {total_items} deterministic items failed consistency checks",
            {"failed_items": failed_items, "total_items": total_items}
        )


# LLM Interface Errors

class LLMError(DocsToEvalError):
    """Raised when LLM operations fail"""
    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out"""
    
    def __init__(self, timeout: float, operation: str):
        super().__init__(
            f"LLM operation '{operation}' timed out after {timeout}s",
            {"timeout": timeout, "operation": operation}
        )


class LLMRateLimitError(LLMError):
    """Raised when hitting LLM rate limits"""
    
    def __init__(self, retry_after: Optional[float] = None):
        message = "LLM rate limit exceeded"
        if retry_after:
            message += f", retry after {retry_after}s"
        super().__init__(message, {"retry_after": retry_after})


class LLMResponseParseError(LLMError):
    """Raised when unable to parse LLM response"""
    
    def __init__(self, response: str, expected_format: str):
        super().__init__(
            f"Failed to parse LLM response as {expected_format}",
            {"response_preview": response[:100], "expected_format": expected_format}
        )


# Verification Errors

class VerificationError(DocsToEvalError):
    """Raised when verification fails"""
    pass


class MathVerificationError(VerificationError):
    """Raised when mathematical verification fails"""
    pass


class CodeExecutionError(VerificationError):
    """Raised when code execution verification fails"""
    
    def __init__(self, error: str, code: str):
        super().__init__(
            f"Code execution failed: {error}",
            {"error": error, "code_preview": code[:100]}
        )


# Configuration Errors

class ConfigurationError(DocsToEvalError):
    """Raised when configuration is invalid"""
    pass


class InvalidEvalTypeError(ConfigurationError):
    """Raised when evaluation type is invalid or unsupported"""
    
    def __init__(self, eval_type: str, supported_types: List[str]):
        super().__init__(
            f"Unsupported evaluation type: {eval_type}",
            {"eval_type": eval_type, "supported_types": supported_types}
        )


class InvalidConfigValueError(ConfigurationError):
    """Raised when a configuration value is invalid"""
    
    def __init__(self, key: str, value: Any, reason: str):
        super().__init__(
            f"Invalid configuration for '{key}': {reason}",
            {"key": key, "value": value, "reason": reason}
        )


# Pipeline Errors

class PipelineError(DocsToEvalError):
    """Raised when pipeline execution fails"""
    pass


class PipelineHealthCheckError(PipelineError):
    """Raised when pipeline health check fails"""
    
    def __init__(self, issues: List[str], warnings: List[str]):
        super().__init__(
            f"Pipeline health check failed with {len(issues)} issues",
            {"issues": issues, "warnings": warnings}
        )


class AgentError(PipelineError):
    """Raised when an agent fails"""
    
    def __init__(self, agent_name: str, operation: str, reason: str):
        super().__init__(
            f"Agent '{agent_name}' failed during {operation}: {reason}",
            {"agent_name": agent_name, "operation": operation, "reason": reason}
        )


# Export Errors

class ExportError(DocsToEvalError):
    """Raised when export fails"""
    pass


class LMEvalExportError(ExportError):
    """Raised when lm-eval export fails"""
    pass


class FileWriteError(ExportError):
    """Raised when file writing fails"""
    
    def __init__(self, path: str, reason: str):
        super().__init__(
            f"Failed to write file '{path}': {reason}",
            {"path": path, "reason": reason}
        )


# Utility functions for error handling

def wrap_exception(
    func,
    exception_type: type[DocsToEvalError],
    message: str,
    **details
):
    """
    Decorator to wrap exceptions with custom error types
    
    Usage:
        @wrap_exception(GenerationError, "Question generation failed")
        def generate_question(...):
            ...
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DocsToEvalError:
            # Re-raise our own exceptions
            raise
        except Exception as e:
            # Wrap external exceptions
            raise exception_type(f"{message}: {str(e)}", details) from e
    return wrapper


async def wrap_exception_async(
    func,
    exception_type: type[DocsToEvalError],
    message: str,
    **details
):
    """Async version of wrap_exception"""
    try:
        return await func
    except DocsToEvalError:
        raise
    except Exception as e:
        raise exception_type(f"{message}: {str(e)}", details) from e
