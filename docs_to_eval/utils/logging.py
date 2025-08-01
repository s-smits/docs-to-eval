"""
Logging utilities for the evaluation framework
"""

import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager


class StructuredLogger:
    """Structured logger with JSON output support"""
    
    def __init__(self, name: str, level: str = "INFO", output_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatter
        formatter = StructuredFormatter()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if output_file:
            file_handler = logging.FileHandler(output_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        self._log(logging.INFO, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data"""
        self._log(logging.ERROR, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data"""
        self._log(logging.WARNING, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method"""
        extra = {
            'structured_data': kwargs,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.log(level, message, extra=extra)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': getattr(record, 'timestamp', datetime.now().isoformat()),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add structured data if present
        if hasattr(record, 'structured_data'):
            log_data.update(record.structured_data)
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class EvaluationLogger:
    """Specialized logger for evaluation operations"""
    
    def __init__(self, run_id: str, output_dir: str = "logs"):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        log_file = self.output_dir / f"evaluation_{run_id}.log"
        self.logger = StructuredLogger("evaluation", output_file=str(log_file))
        
        self.start_time = datetime.now()
        self.phase_times = {}
        self.current_phase = None
    
    def start_phase(self, phase_name: str):
        """Start a new evaluation phase"""
        if self.current_phase:
            self.end_phase()
        
        self.current_phase = phase_name
        self.phase_times[phase_name] = {'start': datetime.now()}
        
        self.logger.info(f"Starting phase: {phase_name}", 
                        run_id=self.run_id, 
                        phase=phase_name)
    
    def end_phase(self):
        """End the current evaluation phase"""
        if not self.current_phase:
            return
        
        end_time = datetime.now()
        start_time = self.phase_times[self.current_phase]['start']
        duration = (end_time - start_time).total_seconds()
        
        self.phase_times[self.current_phase]['end'] = end_time
        self.phase_times[self.current_phase]['duration'] = duration
        
        self.logger.info(f"Completed phase: {self.current_phase}", 
                        run_id=self.run_id, 
                        phase=self.current_phase,
                        duration_seconds=duration)
        
        self.current_phase = None
    
    def log_corpus_info(self, corpus_stats: Dict[str, Any]):
        """Log corpus information"""
        self.logger.info("Corpus loaded", 
                        run_id=self.run_id,
                        corpus_stats=corpus_stats)
    
    def log_classification(self, classification_result: Dict[str, Any]):
        """Log classification results"""
        self.logger.info("Corpus classified", 
                        run_id=self.run_id,
                        classification=classification_result)
    
    def log_benchmark_generation(self, num_questions: int, generation_config: Dict[str, Any]):
        """Log benchmark generation"""
        self.logger.info("Benchmark generated", 
                        run_id=self.run_id,
                        num_questions=num_questions,
                        config=generation_config)
    
    def log_evaluation_progress(self, completed: int, total: int, current_question: str = ""):
        """Log evaluation progress"""
        progress = completed / total if total > 0 else 0
        
        self.logger.info("Evaluation progress", 
                        run_id=self.run_id,
                        completed=completed,
                        total=total,
                        progress_percent=round(progress * 100, 1),
                        current_question=current_question[:100])
    
    def log_verification_results(self, results_summary: Dict[str, Any]):
        """Log verification results"""
        self.logger.info("Verification completed", 
                        run_id=self.run_id,
                        results=results_summary)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        self.logger.error(f"Error in {context}: {str(error)}", 
                         run_id=self.run_id,
                         error_type=type(error).__name__,
                         context=context,
                         exc_info=True)
    
    def log_performance_stats(self, stats: Dict[str, Any]):
        """Log performance statistics"""
        self.logger.info("Performance statistics", 
                        run_id=self.run_id,
                        stats=stats)
    
    def get_run_summary(self) -> Dict[str, Any]:
        """Get summary of the entire run"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        return {
            'run_id': self.run_id,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration_seconds': total_duration,
            'phases': {
                phase: {
                    'start': times['start'].isoformat(),
                    'end': times.get('end', '').isoformat() if times.get('end') else None,
                    'duration_seconds': times.get('duration', 0)
                }
                for phase, times in self.phase_times.items()
            }
        }


def setup_logging(level: str = "INFO", output_dir: str = "logs") -> None:
    """Setup global logging configuration"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{output_dir}/application.log")
        ]
    )


def get_logger(name: str, level: str = "INFO") -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name, level)


@contextmanager
def evaluation_context(run_id: str, output_dir: str = "logs"):
    """Context manager for evaluation logging"""
    logger = EvaluationLogger(run_id, output_dir)
    
    try:
        logger.logger.info("Evaluation started", run_id=run_id)
        yield logger
    except Exception as e:
        logger.log_error(e, "evaluation_context")
        raise
    finally:
        if logger.current_phase:
            logger.end_phase()
        
        summary = logger.get_run_summary()
        logger.logger.info("Evaluation completed", **summary)


class ProgressTracker:
    """Track progress of long-running operations"""
    
    def __init__(self, total: int, logger: Optional[StructuredLogger] = None, 
                 log_interval: int = 10):
        self.total = total
        self.completed = 0
        self.logger = logger
        self.log_interval = log_interval
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1, message: str = ""):
        """Update progress"""
        self.completed += increment
        
        if self.logger and self.completed % self.log_interval == 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.completed / elapsed if elapsed > 0 else 0
            eta = (self.total - self.completed) / rate if rate > 0 else 0
            
            self.logger.info("Progress update",
                           completed=self.completed,
                           total=self.total,
                           progress_percent=round((self.completed / self.total) * 100, 1),
                           rate_per_second=round(rate, 2),
                           eta_seconds=round(eta, 1),
                           message=message)
    
    def finish(self):
        """Mark as finished"""
        self.completed = self.total
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.logger:
            self.logger.info("Progress completed",
                           total=self.total,
                           duration_seconds=round(elapsed, 2),
                           rate_per_second=round(self.total / elapsed if elapsed > 0 else 0, 2))


if __name__ == "__main__":
    # Test the logging system
    setup_logging("DEBUG")
    
    logger = get_logger("test")
    logger.info("Test message", test_data={"key": "value"})
    
    # Test evaluation logger
    with evaluation_context("test_run_123") as eval_logger:
        eval_logger.start_phase("test_phase")
        eval_logger.log_corpus_info({"chars": 1000, "words": 200})
        eval_logger.end_phase()
        
        # Test progress tracker
        tracker = ProgressTracker(100, logger)
        for i in range(100):
            tracker.update(message=f"Processing item {i}")
        tracker.finish()