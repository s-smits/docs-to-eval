"""
Core data transformation layer for lm-evaluation-harness integration
Converts agentic benchmark items to harness-compatible formats
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import re
import json
from pathlib import Path

from .models import EnhancedBenchmarkItem, AnswerType, DifficultyLevel
from ..evaluation import EvaluationType


class HarnessOutputType(str, Enum):
    """Output types supported by lm-evaluation-harness"""
    MULTIPLE_CHOICE = "multiple_choice"
    GENERATE_UNTIL = "generate_until" 
    LOGLIKELIHOOD = "loglikelihood"
    LOGLIKELIHOOD_ROLLING = "loglikelihood_rolling"


@dataclass
class HarnessMetricConfig:
    """Configuration for a single metric in lm-evaluation-harness"""
    metric: str
    aggregation: str = "mean"
    higher_is_better: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "aggregation": self.aggregation,
            "higher_is_better": self.higher_is_better
        }


@dataclass
class HarnessTaskConfig:
    """Complete task configuration for lm-evaluation-harness"""
    task: str
    dataset_name: str
    dataset_path: str
    output_type: HarnessOutputType
    doc_to_text: str
    doc_to_target: str
    metric_list: List[HarnessMetricConfig]
    
    # Optional fields
    doc_to_choice: Optional[str] = None
    description: Optional[str] = None
    training_split: Optional[str] = None
    validation_split: Optional[str] = None
    test_split: str = "test"
    should_decontaminate: bool = False
    doc_to_decontamination_query: Optional[str] = None
    generation_kwargs: Optional[Dict[str, Any]] = None
    process_results: Optional[str] = None
    filter_list: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for YAML export"""
        result = {
            "task": self.task,
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "output_type": self.output_type.value,
            "doc_to_text": self.doc_to_text,
            "doc_to_target": self.doc_to_target,
            "test_split": self.test_split,
            "metric_list": [metric.to_dict() for metric in self.metric_list]
        }
        
        # Add optional fields if they exist
        optional_fields = [
            "doc_to_choice", "description", "training_split", "validation_split",
            "should_decontaminate", "doc_to_decontamination_query", 
            "generation_kwargs", "process_results", "filter_list", "metadata"
        ]
        
        for field in optional_fields:
            value = getattr(self, field)
            if value is not None:
                result[field] = value
                
        return result


class AnswerTypeMapper:
    """Maps agentic AnswerType to harness output types and metrics"""
    
    @staticmethod
    def get_output_type(answer_type: AnswerType) -> HarnessOutputType:
        """Map AnswerType to HarnessOutputType"""
        
        mapping = {
            AnswerType.MULTIPLE_CHOICE: HarnessOutputType.MULTIPLE_CHOICE,
            AnswerType.NUMERIC_EXACT: HarnessOutputType.GENERATE_UNTIL,
            AnswerType.NUMERIC_TOLERANCE: HarnessOutputType.GENERATE_UNTIL,
            AnswerType.STRING_EXACT: HarnessOutputType.GENERATE_UNTIL,
            AnswerType.CODE: HarnessOutputType.GENERATE_UNTIL,
            AnswerType.BOOLEAN: HarnessOutputType.GENERATE_UNTIL,
            AnswerType.FREE_TEXT: HarnessOutputType.GENERATE_UNTIL
        }
        
        return mapping.get(answer_type, HarnessOutputType.GENERATE_UNTIL)
    
    @staticmethod
    def get_metrics(answer_type: AnswerType, eval_type: EvaluationType) -> List[HarnessMetricConfig]:
        """Get appropriate metrics for answer type and evaluation type"""
        
        if answer_type == AnswerType.MULTIPLE_CHOICE:
            return [
                HarnessMetricConfig("acc", "mean", True),
                HarnessMetricConfig("acc_norm", "mean", True)
            ]
        
        elif answer_type in [AnswerType.NUMERIC_EXACT, AnswerType.STRING_EXACT, AnswerType.BOOLEAN]:
            return [
                HarnessMetricConfig("exact_match", "mean", True)
            ]
        
        elif answer_type == AnswerType.NUMERIC_TOLERANCE:
            return [
                HarnessMetricConfig("exact_match", "mean", True),
                HarnessMetricConfig("quasi_exact_match", "mean", True)
            ]
        
        elif answer_type == AnswerType.CODE:
            if eval_type == EvaluationType.CODE_GENERATION:
                return [
                    HarnessMetricConfig("exact_match", "mean", True),
                    HarnessMetricConfig("bleu", "mean", True)
                ]
            else:
                return [HarnessMetricConfig("exact_match", "mean", True)]
        
        else:  # FREE_TEXT
            if eval_type in [EvaluationType.READING_COMPREHENSION, EvaluationType.DOMAIN_KNOWLEDGE]:
                return [
                    HarnessMetricConfig("bleu", "mean", True),
                    HarnessMetricConfig("rouge", "mean", True)
                ]
            else:
                return [HarnessMetricConfig("bleu", "mean", True)]
    
    @staticmethod
    def get_generation_kwargs(answer_type: AnswerType) -> Optional[Dict[str, Any]]:
        """Get generation parameters for generate_until tasks"""
        
        if answer_type == AnswerType.MULTIPLE_CHOICE:
            return None  # Not used for multiple choice
        
        base_kwargs = {
            "temperature": 0.0,
            "do_sample": False
        }
        
        if answer_type in [AnswerType.NUMERIC_EXACT, AnswerType.NUMERIC_TOLERANCE]:
            return {
                **base_kwargs,
                "until": ["\\n", ".", "?", " "],
                "max_gen_toks": 50
            }
        
        elif answer_type in [AnswerType.STRING_EXACT, AnswerType.BOOLEAN]:
            return {
                **base_kwargs,
                "until": ["\\n", ".", "?"],
                "max_gen_toks": 100
            }
        
        elif answer_type == AnswerType.CODE:
            return {
                **base_kwargs,
                "until": ["\\n\\n", "# End", "```"],
                "max_gen_toks": 512
            }
        
        else:  # FREE_TEXT
            return {
                **base_kwargs,
                "until": ["\\n\\n"],
                "max_gen_toks": 256
            }


class TemplateGenerator:
    """Generates Jinja2 templates for harness tasks"""
    
    @staticmethod
    def create_doc_to_text(
        item: EnhancedBenchmarkItem, 
        answer_type: AnswerType
    ) -> str:
        """Generate doc_to_text template based on item structure"""
        
        # Start with base template
        if item.context:
            template = "Context: {{context}}\\n\\nQuestion: {{question}}"
        else:
            template = "Question: {{question}}"
        
        # Add answer type specific formatting
        if answer_type == AnswerType.MULTIPLE_CHOICE:
            template += "\\nA. {{choices[0]}}\\nB. {{choices[1]}}\\nC. {{choices[2]}}\\nD. {{choices[3]}}\\nAnswer:"
        
        elif answer_type == AnswerType.CODE:
            template += "\\n\\nProvide your code solution:"
        
        elif answer_type in [AnswerType.NUMERIC_EXACT, AnswerType.NUMERIC_TOLERANCE]:
            template += "\\n\\nProvide the numerical answer:"
        
        elif answer_type == AnswerType.BOOLEAN:
            template += "\\n\\nAnswer with True or False:"
        
        else:
            template += "\\n\\nAnswer:"
        
        return template
    
    @staticmethod
    def create_doc_to_target(answer_type: AnswerType) -> str:
        """Generate doc_to_target template"""
        
        if answer_type == AnswerType.MULTIPLE_CHOICE:
            # For MC, we need the choice index, not the text
            return "{{answer_index}}"
        else:
            return "{{answer}}"
    
    @staticmethod
    def create_doc_to_choice(answer_type: AnswerType) -> Optional[str]:
        """Generate doc_to_choice template for multiple choice"""
        
        if answer_type == AnswerType.MULTIPLE_CHOICE:
            return "{{choices}}"
        return None


class DatasetItemTransformer:
    """Transforms agentic items to harness dataset format"""
    
    @staticmethod
    def transform_item(item: EnhancedBenchmarkItem) -> Dict[str, Any]:
        """Transform single item to harness dataset format"""
        
        # Get answer type
        answer_type = getattr(item, 'expected_answer_type', AnswerType.FREE_TEXT)
        if isinstance(answer_type, str):
            try:
                answer_type = AnswerType(answer_type)
            except ValueError:
                answer_type = AnswerType.FREE_TEXT
        
        # Base transformation
        transformed = {
            "question": item.question,
            "answer": item.answer,
        }
        
        # Add context if present
        if item.context:
            transformed["context"] = item.context
        
        # Handle multiple choice
        if answer_type == AnswerType.MULTIPLE_CHOICE and item.options:
            transformed["choices"] = item.options
            
            # Convert answer to index if it's currently text
            if isinstance(item.answer, str) and item.answer in item.options:
                transformed["answer_index"] = item.options.index(item.answer)
            else:
                # Try to find the answer by partial match
                answer_index = DatasetItemTransformer._find_answer_index(item.answer, item.options)
                transformed["answer_index"] = answer_index
        
        # Add metadata
        transformed["metadata"] = {
            "difficulty": item.metadata.difficulty.value if hasattr(item.metadata.difficulty, 'value') else item.metadata.difficulty,
            "deterministic": item.metadata.deterministic,
            "eval_type": item.eval_type.value if hasattr(item.eval_type, 'value') else str(item.eval_type),
            "answer_type": answer_type.value if hasattr(answer_type, 'value') else str(answer_type),
            "validation_score": item.metadata.validation_score,
            "agents_used": item.metadata.agents_used,
            "adversarial_techniques": item.metadata.adversarial_techniques,
            "source": "agentic_benchmark_factory"
        }
        
        # Add reasoning chain if available
        if hasattr(item, 'reasoning_chain') and item.reasoning_chain:
            transformed["reasoning_chain"] = item.reasoning_chain
        
        # Add variables if available
        if hasattr(item, 'variables') and item.variables:
            transformed["variables"] = item.variables
        
        return transformed
    
    @staticmethod
    def _find_answer_index(answer: str, choices: List[str]) -> int:
        """Find the index of the correct answer in choices"""
        
        # Direct match
        if answer in choices:
            return choices.index(answer)
        
        # Partial match (case insensitive)
        answer_lower = answer.lower().strip()
        for i, choice in enumerate(choices):
            if answer_lower in choice.lower() or choice.lower() in answer_lower:
                return i
        
        # Pattern match for single letter answers (A, B, C, D)
        letter_match = re.search(r'\b([ABCD])\b', answer.upper())
        if letter_match:
            letter = letter_match.group(1)
            letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            if letter in letter_to_index and letter_to_index[letter] < len(choices):
                return letter_to_index[letter]
        
        # Default to first option if no match found
        return 0
    
    @staticmethod
    def transform_batch(items: List[EnhancedBenchmarkItem]) -> List[Dict[str, Any]]:
        """Transform batch of items to harness dataset format"""
        return [DatasetItemTransformer.transform_item(item) for item in items]


class TaskConfigurationGenerator:
    """Generates complete task configurations for harness"""
    
    def __init__(self):
        self.answer_type_mapper = AnswerTypeMapper()
        self.template_generator = TemplateGenerator()
    
    def generate_task_config(
        self,
        items: List[EnhancedBenchmarkItem],
        task_name: str,
        answer_type: AnswerType,
        eval_type: EvaluationType
    ) -> HarnessTaskConfig:
        """Generate complete task configuration"""
        
        if not items:
            raise ValueError("Cannot generate config for empty item list")
        
        # Get representative item for template generation
        sample_item = items[0]
        
        # Determine output type and metrics
        output_type = self.answer_type_mapper.get_output_type(answer_type)
        metrics = self.answer_type_mapper.get_metrics(answer_type, eval_type)
        generation_kwargs = self.answer_type_mapper.get_generation_kwargs(answer_type)
        
        # Generate templates
        doc_to_text = self.template_generator.create_doc_to_text(sample_item, answer_type)
        doc_to_target = self.template_generator.create_doc_to_target(answer_type)
        doc_to_choice = self.template_generator.create_doc_to_choice(answer_type)
        
        # Create configuration
        config = HarnessTaskConfig(
            task=task_name,
            dataset_name=task_name,
            dataset_path=f"{task_name}_dataset.jsonl",
            output_type=output_type,
            doc_to_text=doc_to_text,
            doc_to_target=doc_to_target,
            metric_list=metrics,
            doc_to_choice=doc_to_choice,
            description=f"Agentic benchmark for {eval_type.value if hasattr(eval_type, 'value') else str(eval_type)} evaluation ({answer_type.value if hasattr(answer_type, 'value') else str(answer_type)} answers)",
            should_decontaminate=True,
            doc_to_decontamination_query="{{question}}",
            generation_kwargs=generation_kwargs,
            metadata={
                "version": "2.0",
                "source": "agentic_benchmark_factory",
                "eval_type": eval_type.value if hasattr(eval_type, 'value') else str(eval_type),
                "answer_type": answer_type.value if hasattr(answer_type, 'value') else str(answer_type),
                "items_count": len(items),
                "difficulty_distribution": self._get_difficulty_distribution(items),
                "deterministic_ratio": sum(1 for item in items if item.metadata.deterministic) / len(items)
            }
        )
        
        return config
    
    def _get_difficulty_distribution(self, items: List[EnhancedBenchmarkItem]) -> Dict[str, int]:
        """Calculate difficulty distribution"""
        distribution = {}
        for item in items:
            difficulty = item.metadata.difficulty.value if hasattr(item.metadata.difficulty, 'value') else str(item.metadata.difficulty)
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution


class HarnessTransformer:
    """Main transformer class for converting agentic items to harness format"""
    
    def __init__(self):
        self.dataset_transformer = DatasetItemTransformer()
        self.config_generator = TaskConfigurationGenerator()
    
    def transform_to_harness_format(
        self,
        items: List[EnhancedBenchmarkItem],
        task_name: str
    ) -> Dict[str, Any]:
        """
        Transform agentic benchmark items to complete harness format
        
        Args:
            items: List of agentic benchmark items
            task_name: Base name for the task
            
        Returns:
            Dictionary containing task configs and datasets grouped by type
        """
        
        if not items:
            raise ValueError("No items provided for transformation")
        
        # Group items by eval_type and answer_type
        grouped_items = self._group_items(items)
        
        result = {
            "tasks": {},
            "datasets": {},
            "metadata": {
                "total_items": len(items),
                "total_tasks": len(grouped_items),
                "base_task_name": task_name,
                "transformation_timestamp": None  # Will be set by exporter
            }
        }
        
        # Process each group
        for (eval_type, answer_type), group_items in grouped_items.items():
            eval_type_str = eval_type.value if hasattr(eval_type, 'value') else str(eval_type)
            answer_type_str = answer_type.value if hasattr(answer_type, 'value') else str(answer_type)
            group_task_name = f"{task_name}_{eval_type_str}_{answer_type_str}"
            
            # Generate task configuration
            task_config = self.config_generator.generate_task_config(
                group_items, group_task_name, answer_type, eval_type
            )
            
            # Transform dataset
            dataset_items = self.dataset_transformer.transform_batch(group_items)
            
            result["tasks"][group_task_name] = task_config
            result["datasets"][group_task_name] = dataset_items
        
        return result
    
    def _group_items(
        self, 
        items: List[EnhancedBenchmarkItem]
    ) -> Dict[Tuple[EvaluationType, AnswerType], List[EnhancedBenchmarkItem]]:
        """Group items by evaluation type and answer type"""
        
        grouped = {}
        
        for item in items:
            eval_type = item.eval_type
            answer_type = getattr(item, 'expected_answer_type', AnswerType.FREE_TEXT)
            
            # Convert string answer types to enum
            if isinstance(answer_type, str):
                try:
                    answer_type = AnswerType(answer_type)
                except ValueError:
                    answer_type = AnswerType.FREE_TEXT
            
            group_key = (eval_type, answer_type)
            
            if group_key not in grouped:
                grouped[group_key] = []
            grouped[group_key].append(item)
        
        return grouped