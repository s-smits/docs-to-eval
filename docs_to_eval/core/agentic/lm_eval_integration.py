"""
LM-Evaluation-Harness Integration Module
Exports agentic benchmark items as YAML tasks compatible with lm-evaluation-harness
"""

import yaml
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import zipfile
from datetime import datetime

from .models import EnhancedBenchmarkItem, AnswerType
from ..evaluation import EvaluationType


class LMEvalHarnessExporter:
    """
    Converts agentic benchmark items to lm-evaluation-harness compatible YAML tasks
    """
    
    def __init__(self):
        self.task_counter = 0
        
    def export_benchmark_to_harness(
        self, 
        items: List[EnhancedBenchmarkItem], 
        task_name: str,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Export benchmark items as lm-evaluation-harness task files
        
        Args:
            items: List of enhanced benchmark items
            task_name: Name for the task (will be used in YAML)
            output_dir: Directory to save files (if None, returns data only)
            
        Returns:
            Dictionary containing all task configurations and metadata
        """
        
        if not items:
            raise ValueError("No items provided for export")
        
        # Group items by evaluation type and answer type for optimal task structure
        grouped_items = self._group_items_for_export(items)
        
        exported_tasks = {}
        task_files = {}
        
        for group_key, group_items in grouped_items.items():
            eval_type, answer_type = group_key
            group_task_name = f"{task_name}_{eval_type.value}_{answer_type.value}"
            
            # Create harness-compatible task configuration
            task_config = self._create_task_config(group_items, group_task_name, eval_type, answer_type)
            
            # Create dataset file
            dataset_content = self._create_dataset_content(group_items)
            
            exported_tasks[group_task_name] = {
                'config': task_config,
                'dataset': dataset_content,
                'metadata': self._create_task_metadata(group_items, group_task_name)
            }
            
            if output_dir:
                task_files.update(self._save_task_files(
                    task_config, dataset_content, group_task_name, output_dir
                ))
        
        # Create summary configuration
        summary_config = self._create_summary_config(exported_tasks, task_name)
        
        if output_dir:
            # Save summary and create ZIP
            summary_file = output_dir / f"{task_name}_summary.yaml"
            with open(summary_file, 'w') as f:
                yaml.dump(summary_config, f, default_flow_style=False, sort_keys=False)
            
            task_files['summary'] = str(summary_file)
            
            # Create ZIP package
            zip_path = self._create_task_package(task_files, task_name, output_dir)
            task_files['package'] = str(zip_path)
        
        return {
            'tasks': exported_tasks,
            'summary': summary_config,
            'files': task_files,
            'export_metadata': {
                'export_time': datetime.now().isoformat(),
                'total_items': len(items),
                'total_tasks': len(exported_tasks),
                'evaluation_types': list(set(item.eval_type.value for item in items)),
                'answer_types': list(set(getattr(item, 'expected_answer_type', 'unknown') for item in items))
            }
        }
    
    def _group_items_for_export(self, items: List[EnhancedBenchmarkItem]) -> Dict[tuple, List[EnhancedBenchmarkItem]]:
        """Group items by eval type and answer type for optimal task organization"""
        
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
    
    def _create_task_config(
        self, 
        items: List[EnhancedBenchmarkItem], 
        task_name: str,
        eval_type: EvaluationType,
        answer_type: AnswerType
    ) -> Dict[str, Any]:
        """Create lm-evaluation-harness task configuration YAML"""
        
        # Determine output type based on answer type
        output_type, metric_config = self._get_output_type_and_metrics(answer_type, eval_type)
        
        # Base configuration
        config = {
            'task': task_name,
            'dataset_name': task_name,
            'dataset_path': f'{task_name}_dataset.jsonl',
            'output_type': output_type,
            'training_split': None,
            'validation_split': None,
            'test_split': 'test',
            'doc_to_text': self._create_doc_to_text_template(items[0], answer_type),
            'doc_to_target': self._create_doc_to_target_template(answer_type),
            'description': f"Agentic benchmark for {eval_type.value} evaluation using {answer_type.value} answers",
            'metadata': {
                'version': '2.0',
                'source': 'agentic_benchmark_factory',
                'eval_type': eval_type.value,
                'answer_type': answer_type.value,
                'difficulty_distribution': self._get_difficulty_distribution(items),
                'total_items': len(items)
            }
        }
        
        # Add type-specific configurations
        config.update(metric_config)
        
        # Add multiple choice specific configs
        if answer_type == AnswerType.MULTIPLE_CHOICE:
            config.update({
                'doc_to_choice': '{{choices}}',
                'should_decontaminate': True,
                'doc_to_decontamination_query': '{{question}}'
            })
        
        # Add context if present
        if any(item.context for item in items):
            config['doc_to_text'] = "Context: {{context}}\\n\\nQuestion: {{question}}"
        
        return config
    
    def _get_output_type_and_metrics(self, answer_type: AnswerType, eval_type: EvaluationType) -> tuple:
        """Determine output type and metrics configuration"""
        
        if answer_type == AnswerType.MULTIPLE_CHOICE:
            return 'multiple_choice', {
                'metric_list': [
                    {'metric': 'acc', 'aggregation': 'mean', 'higher_is_better': True},
                    {'metric': 'acc_norm', 'aggregation': 'mean', 'higher_is_better': True}
                ]
            }
        
        elif answer_type in [AnswerType.NUMERIC_EXACT, AnswerType.NUMERIC_TOLERANCE]:
            return 'generate_until', {
                'metric_list': [
                    {'metric': 'exact_match', 'aggregation': 'mean', 'higher_is_better': True}
                ],
                'generation_kwargs': {
                    'until': ['\\n', '.', '?'],
                    'max_gen_toks': 50
                }
            }
        
        elif answer_type == AnswerType.CODE:
            return 'generate_until', {
                'metric_list': [
                    {'metric': 'exact_match', 'aggregation': 'mean', 'higher_is_better': True},
                    {'metric': 'bleu', 'aggregation': 'mean', 'higher_is_better': True}
                ],
                'generation_kwargs': {
                    'until': ['\\n\\n', '# End'],
                    'max_gen_toks': 512
                },
                'process_results': '!function utils.code_postprocess'
            }
        
        elif answer_type == AnswerType.STRING_EXACT:
            return 'generate_until', {
                'metric_list': [
                    {'metric': 'exact_match', 'aggregation': 'mean', 'higher_is_better': True},
                    {'metric': 'f1', 'aggregation': 'mean', 'higher_is_better': True}
                ],
                'generation_kwargs': {
                    'until': ['\\n', '.'],
                    'max_gen_toks': 100
                }
            }
        
        else:  # FREE_TEXT and others
            return 'generate_until', {
                'metric_list': [
                    {'metric': 'bleu', 'aggregation': 'mean', 'higher_is_better': True},
                    {'metric': 'rouge', 'aggregation': 'mean', 'higher_is_better': True}
                ],
                'generation_kwargs': {
                    'until': ['\\n\\n'],
                    'max_gen_toks': 256
                }
            }
    
    def _create_doc_to_text_template(self, sample_item: EnhancedBenchmarkItem, answer_type: AnswerType) -> str:
        """Create doc_to_text template based on item structure"""
        
        # Base template
        if sample_item.context:
            template = "Context: {{context}}\\n\\nQuestion: {{question}}"
        else:
            template = "Question: {{question}}"
        
        # Add answer type specific prompting
        if answer_type == AnswerType.MULTIPLE_CHOICE:
            template += "\\nA. {{choices[0]}}\\nB. {{choices[1]}}\\nC. {{choices[2]}}\\nD. {{choices[3]}}\\nAnswer:"
        elif answer_type == AnswerType.CODE:
            template += "\\n\\nProvide your code solution:"
        elif answer_type in [AnswerType.NUMERIC_EXACT, AnswerType.NUMERIC_TOLERANCE]:
            template += "\\n\\nProvide the numerical answer:"
        else:
            template += "\\n\\nAnswer:"
        
        return template
    
    def _create_doc_to_target_template(self, answer_type: AnswerType) -> str:
        """Create doc_to_target template"""
        
        if answer_type == AnswerType.MULTIPLE_CHOICE:
            return "{{answer}}"
        else:
            return "{{answer}}"
    
    def _create_dataset_content(self, items: List[EnhancedBenchmarkItem]) -> List[Dict[str, Any]]:
        """Create JSONL dataset content"""
        
        dataset = []
        
        for item in items:
            # Base data
            data_item = {
                'question': item.question,
                'answer': item.answer,
                'context': item.context,
                'metadata': {
                    'difficulty': item.metadata.difficulty.value,
                    'deterministic': item.metadata.deterministic,
                    'agents_used': item.metadata.agents_used,
                    'validation_score': item.metadata.validation_score,
                    'adversarial_techniques': item.metadata.adversarial_techniques
                }
            }
            
            # Add multiple choice options if present
            if item.options:
                data_item['choices'] = item.options
                # For multiple choice, answer should be the index
                if isinstance(item.answer, str) and item.answer in item.options:
                    data_item['answer'] = item.options.index(item.answer)
            
            # Add variables for randomization
            if hasattr(item, 'variables') and item.variables:
                data_item['variables'] = item.variables
            
            # Add reasoning chain
            if hasattr(item, 'reasoning_chain') and item.reasoning_chain:
                data_item['reasoning_chain'] = item.reasoning_chain
            
            dataset.append(data_item)
        
        return dataset
    
    def _create_task_metadata(self, items: List[EnhancedBenchmarkItem], task_name: str) -> Dict[str, Any]:
        """Create comprehensive task metadata"""
        
        return {
            'task_name': task_name,
            'version': '2.0',
            'description': f'Agentic benchmark with {len(items)} items',
            'items_count': len(items),
            'difficulty_distribution': self._get_difficulty_distribution(items),
            'deterministic_ratio': sum(1 for item in items if item.metadata.deterministic) / len(items),
            'average_validation_score': sum(item.metadata.validation_score or 0 for item in items) / len(items),
            'agent_techniques_used': list(set(
                technique for item in items 
                for technique in item.metadata.adversarial_techniques
            )),
            'creation_timestamp': datetime.now().isoformat(),
            'source': 'agentic_benchmark_factory'
        }
    
    def _get_difficulty_distribution(self, items: List[EnhancedBenchmarkItem]) -> Dict[str, int]:
        """Get distribution of difficulty levels"""
        
        distribution = {}
        for item in items:
            difficulty = item.metadata.difficulty.value
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        
        return distribution
    
    def _create_summary_config(self, exported_tasks: Dict, base_task_name: str) -> Dict[str, Any]:
        """Create summary configuration for all exported tasks"""
        
        return {
            'task_group': base_task_name,
            'description': f'Agentic benchmark group: {base_task_name}',
            'version': '2.0',
            'tasks': list(exported_tasks.keys()),
            'total_items': sum(
                task_data['metadata']['total_items'] 
                for task_data in exported_tasks.values()
            ),
            'evaluation_types': list(set(
                task_data['metadata']['eval_type'] 
                for task_data in exported_tasks.values()
            )),
            'usage': {
                'command': f'lm_eval --model hf --model_args pretrained=<model> --tasks {base_task_name}',
                'description': 'Run evaluation on the complete agentic benchmark suite'
            },
            'export_info': {
                'generated_at': datetime.now().isoformat(),
                'source': 'agentic_benchmark_factory',
                'format_version': '2.0'
            }
        }
    
    def _save_task_files(
        self, 
        task_config: Dict, 
        dataset_content: List[Dict], 
        task_name: str, 
        output_dir: Path
    ) -> Dict[str, str]:
        """Save task configuration and dataset files"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files_created = {}
        
        # Save task YAML
        yaml_file = output_dir / f"{task_name}.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(task_config, f, default_flow_style=False, sort_keys=False)
        files_created['yaml'] = str(yaml_file)
        
        # Save dataset JSONL
        jsonl_file = output_dir / f"{task_name}_dataset.jsonl"
        with open(jsonl_file, 'w') as f:
            for item in dataset_content:
                f.write(json.dumps(item) + '\\n')
        files_created['dataset'] = str(jsonl_file)
        
        # Save metadata JSON
        metadata_file = output_dir / f"{task_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(task_config.get('metadata', {}), f, indent=2)
        files_created['metadata'] = str(metadata_file)
        
        return files_created
    
    def _create_task_package(self, task_files: Dict[str, str], task_name: str, output_dir: Path) -> Path:
        """Create ZIP package of all task files"""
        
        zip_path = output_dir / f"{task_name}_lm_eval_harness.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_type, file_path in task_files.items():
                if file_type != 'package':  # Don't include the zip in itself
                    zf.write(file_path, Path(file_path).name)
            
            # Add README
            readme_content = self._create_readme_content(task_name)
            zf.writestr('README.md', readme_content)
        
        return zip_path
    
    def _create_readme_content(self, task_name: str) -> str:
        """Create README content for the task package"""
        
        return f"""# {task_name} - Agentic Benchmark for LM-Evaluation-Harness

This package contains benchmark tasks generated by the Agentic Benchmark Factory,
compatible with EleutherAI's lm-evaluation-harness.

## Contents

- `*.yaml` - Task configuration files
- `*_dataset.jsonl` - Dataset files in JSONL format
- `*_metadata.json` - Task metadata and statistics
- `{task_name}_summary.yaml` - Summary configuration

## Usage

1. Extract files to your lm-evaluation-harness tasks directory
2. Run evaluation:

```bash
lm_eval --model hf \\
        --model_args pretrained=<your_model> \\
        --tasks {task_name} \\
        --device cuda \\
        --batch_size 8
```

## Task Structure

Each task follows the lm-evaluation-harness format:
- **Multiple Choice**: Uses `acc` and `acc_norm` metrics
- **Generative**: Uses appropriate metrics (BLEU, ROUGE, exact match)
- **Code**: Includes code-specific evaluation
- **Mathematical**: Numerical answer verification

## Quality Assurance

All questions generated through:
- ✅ Multi-agent pipeline (ConceptMiner → QuestionWriter → Adversary → Refiner → Validator)
- ✅ Deterministic vs non-deterministic classification
- ✅ Quality scoring and filtering
- ✅ Adversarial enhancement for increased difficulty

## Metadata

Each item includes:
- Difficulty level (basic/intermediate/hard/expert)
- Deterministic classification
- Agent provenance and techniques used
- Quality validation scores
- Generation timestamps

Generated by Agentic Benchmark Factory v2.0
"""


def add_harness_export_methods():
    """Add lm-evaluation-harness export methods to existing classes"""
    
    def to_harness_task(self) -> Dict[str, Any]:
        """Export single item as harness-compatible task configuration"""
        exporter = LMEvalHarnessExporter()
        
        # Determine output type
        answer_type = getattr(self, 'expected_answer_type', AnswerType.FREE_TEXT)
        if isinstance(answer_type, str):
            try:
                answer_type = AnswerType(answer_type)
            except ValueError:
                answer_type = AnswerType.FREE_TEXT
        
        output_type, metric_config = exporter._get_output_type_and_metrics(answer_type, self.eval_type)
        
        return {
            'dataset_name': f'item_{id(self)}',
            'doc_to_text': exporter._create_doc_to_text_template(self, answer_type),
            'doc_to_target': exporter._create_doc_to_target_template(answer_type),
            'output_type': output_type,
            **metric_config,
            'metadata': {
                'source': 'agentic_benchmark_factory',
                'difficulty': self.metadata.difficulty.value,
                'deterministic': self.metadata.deterministic,
                'validation_score': self.metadata.validation_score
            }
        }
    
    # Add method to EnhancedBenchmarkItem
    EnhancedBenchmarkItem.to_harness_task = to_harness_task


# Initialize the method additions
add_harness_export_methods()


# Utility functions for easy integration

def export_benchmark_to_lm_eval(
    items: List[EnhancedBenchmarkItem],
    task_name: str,
    output_dir: str = "./lm_eval_exports"
) -> Dict[str, Any]:
    """
    Convenience function to export benchmark to lm-evaluation-harness format
    
    Args:
        items: List of enhanced benchmark items
        task_name: Name for the exported task
        output_dir: Directory to save exported files
        
    Returns:
        Export report with file paths and metadata
    """
    
    exporter = LMEvalHarnessExporter()
    output_path = Path(output_dir)
    
    return exporter.export_benchmark_to_harness(items, task_name, output_path)


def create_lm_eval_config_only(
    items: List[EnhancedBenchmarkItem],
    task_name: str
) -> Dict[str, Any]:
    """
    Create lm-evaluation-harness configuration without saving files
    
    Args:
        items: List of enhanced benchmark items
        task_name: Name for the task
        
    Returns:
        Task configurations and dataset content
    """
    
    exporter = LMEvalHarnessExporter()
    return exporter.export_benchmark_to_harness(items, task_name, output_dir=None)