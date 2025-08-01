"""
File export system for lm-evaluation-harness integration
Handles YAML task generation, JSONL dataset creation, and file packaging
"""

import yaml
import json
import shutil
import zipfile
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

from .lm_eval_transform import HarnessTransformer, HarnessTaskConfig
from .models import EnhancedBenchmarkItem


logger = logging.getLogger(__name__)


class YAMLTaskExporter:
    """Exports task configurations as YAML files compatible with lm-evaluation-harness"""
    
    def __init__(self, preserve_order: bool = True):
        self.preserve_order = preserve_order
    
    def export_task_yaml(
        self, 
        task_config: HarnessTaskConfig, 
        output_path: Path
    ) -> Path:
        """
        Export single task configuration as YAML file
        
        Args:
            task_config: Task configuration to export
            output_path: Path where to save the YAML file
            
        Returns:
            Path to the created YAML file
        """
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dictionary
        config_dict = task_config.to_dict()
        
        # Custom YAML representation for better formatting
        formatted_config = self._format_config_for_yaml(config_dict)
        
        # Write YAML with custom formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                formatted_config, 
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=120,
                indent=2
            )
        
        logger.info(f"Exported task YAML: {output_path}")
        return output_path
    
    def export_task_group_yaml(
        self,
        task_configs: Dict[str, HarnessTaskConfig],
        group_name: str,
        output_path: Path
    ) -> Path:
        """
        Export task group configuration with multiple tasks
        
        Args:
            task_configs: Dictionary of task name -> config
            group_name: Name of the task group
            output_path: Path where to save the group YAML
            
        Returns:
            Path to the created group YAML file
        """
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create group configuration
        group_config = {
            "group": group_name,
            "task": list(task_configs.keys()),
            "description": f"Agentic benchmark group: {group_name}",
            "metadata": {
                "version": "2.0",
                "source": "agentic_benchmark_factory",
                "total_tasks": len(task_configs),
                "created_at": datetime.now().isoformat()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                group_config,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                indent=2
            )
        
        logger.info(f"Exported task group YAML: {output_path}")
        return output_path
    
    def _format_config_for_yaml(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Format configuration dictionary for optimal YAML output"""
        
        # Create ordered dictionary with preferred field ordering
        ordered_config = {}
        
        # Primary fields first
        primary_fields = [
            "task", "dataset_name", "dataset_path", "output_type",
            "doc_to_text", "doc_to_target", "doc_to_choice"
        ]
        
        for field in primary_fields:
            if field in config:
                ordered_config[field] = config[field]
        
        # Metric list with special formatting
        if "metric_list" in config:
            ordered_config["metric_list"] = config["metric_list"]
        
        # Generation kwargs with special formatting
        if "generation_kwargs" in config and config["generation_kwargs"]:
            ordered_config["generation_kwargs"] = config["generation_kwargs"]
        
        # Configuration fields
        config_fields = [
            "training_split", "validation_split", "test_split",
            "should_decontaminate", "doc_to_decontamination_query",
            "process_results", "filter_list"
        ]
        
        for field in config_fields:
            if field in config and config[field] is not None:
                ordered_config[field] = config[field]
        
        # Description and metadata last
        if "description" in config:
            ordered_config["description"] = config["description"]
        
        if "metadata" in config and config["metadata"]:
            ordered_config["metadata"] = config["metadata"]
        
        return ordered_config


class JSONLDatasetExporter:
    """Exports datasets as JSONL files for lm-evaluation-harness"""
    
    def export_dataset_jsonl(
        self,
        dataset_items: List[Dict[str, Any]],
        output_path: Path
    ) -> Path:
        """
        Export dataset items as JSONL file
        
        Args:
            dataset_items: List of dataset items to export
            output_path: Path where to save the JSONL file
            
        Returns:
            Path to the created JSONL file
        """
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset_items:
                # Ensure proper JSON formatting
                json_line = json.dumps(item, ensure_ascii=False, separators=(',', ':'))
                f.write(json_line + '\\n')
        
        logger.info(f"Exported dataset JSONL: {output_path} ({len(dataset_items)} items)")
        return output_path
    
    def validate_dataset_format(self, dataset_items: List[Dict[str, Any]]) -> List[str]:
        """
        Validate dataset format for common issues
        
        Args:
            dataset_items: Dataset items to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        
        errors = []
        
        if not dataset_items:
            errors.append("Dataset is empty")
            return errors
        
        required_fields = {"question", "answer"}
        
        for i, item in enumerate(dataset_items):
            item_errors = []
            
            # Check required fields
            missing_fields = required_fields - set(item.keys())
            if missing_fields:
                item_errors.append(f"Missing required fields: {missing_fields}")
            
            # Check field types
            if "question" in item and not isinstance(item["question"], str):
                item_errors.append("Question must be a string")
            
            if "answer" in item and not isinstance(item["answer"], (str, int, float)):
                item_errors.append("Answer must be a string or number")
            
            # Check multiple choice consistency
            if "choices" in item:
                if not isinstance(item["choices"], list):
                    item_errors.append("Choices must be a list")
                elif len(item["choices"]) < 2:
                    item_errors.append("Multiple choice must have at least 2 choices")
                
                if "answer_index" in item:
                    if not isinstance(item["answer_index"], int):
                        item_errors.append("Answer index must be an integer")
                    elif not (0 <= item["answer_index"] < len(item["choices"])):
                        item_errors.append("Answer index out of range")
            
            # Add item-specific errors
            if item_errors:
                errors.extend([f"Item {i}: {error}" for error in item_errors])
        
        return errors


class LMEvalHarnessExporter:
    """Main exporter class for complete lm-evaluation-harness integration"""
    
    def __init__(self):
        self.transformer = HarnessTransformer()
        self.yaml_exporter = YAMLTaskExporter()
        self.jsonl_exporter = JSONLDatasetExporter()
    
    def export_complete_benchmark(
        self,
        items: List[EnhancedBenchmarkItem],
        task_name: str,
        output_dir: Path,
        create_package: bool = True
    ) -> Dict[str, Any]:
        """
        Export complete benchmark with all files and packaging
        
        Args:
            items: List of agentic benchmark items
            task_name: Base name for the exported benchmark
            output_dir: Directory where to save all files
            create_package: Whether to create a ZIP package
            
        Returns:
            Export report with file paths and metadata
        """
        
        if not items:
            raise ValueError("No items provided for export")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting export of {len(items)} items to {output_dir}")
        
        # Transform items to harness format
        transformed_data = self.transformer.transform_to_harness_format(items, task_name)
        
        export_report = {
            "task_name": task_name,
            "output_directory": str(output_dir),
            "export_timestamp": datetime.now().isoformat(),
            "total_items": len(items),
            "tasks_created": {},
            "files_created": [],
            "validation_results": {},
            "package_path": None
        }
        
        # Export each task
        for task_name_full, task_config in transformed_data["tasks"].items():
            task_report = self._export_single_task(
                task_config,
                transformed_data["datasets"][task_name_full],
                task_name_full,
                output_dir
            )
            
            export_report["tasks_created"][task_name_full] = task_report
            export_report["files_created"].extend(task_report["files"])
        
        # Create task group configuration
        if len(transformed_data["tasks"]) > 1:
            group_yaml_path = self.yaml_exporter.export_task_group_yaml(
                transformed_data["tasks"],
                task_name,
                output_dir / f"{task_name}_group.yaml"
            )
            export_report["files_created"].append(str(group_yaml_path))
        
        # Create documentation
        readme_path = self._create_documentation(
            task_name,
            transformed_data,
            output_dir
        )
        export_report["files_created"].append(str(readme_path))
        
        # Create package if requested
        if create_package:
            package_path = self._create_package(
                task_name,
                output_dir,
                export_report["files_created"]
            )
            export_report["package_path"] = str(package_path)
        
        logger.info(f"Export completed: {len(export_report['tasks_created'])} tasks, "
                   f"{len(export_report['files_created'])} files")
        
        return export_report
    
    def _export_single_task(
        self,
        task_config: HarnessTaskConfig,
        dataset_items: List[Dict[str, Any]],
        task_name: str,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Export single task with YAML config and JSONL dataset"""
        
        task_report = {
            "task_name": task_name,
            "items_count": len(dataset_items),
            "files": [],
            "validation_errors": []
        }
        
        # Validate dataset
        validation_errors = self.jsonl_exporter.validate_dataset_format(dataset_items)
        if validation_errors:
            task_report["validation_errors"] = validation_errors
            logger.warning(f"Dataset validation errors for {task_name}: {validation_errors}")
        
        # Export YAML configuration
        yaml_path = self.yaml_exporter.export_task_yaml(
            task_config,
            output_dir / f"{task_name}.yaml"
        )
        task_report["files"].append(str(yaml_path))
        
        # Export JSONL dataset
        jsonl_path = self.jsonl_exporter.export_dataset_jsonl(
            dataset_items,
            output_dir / f"{task_name}_dataset.jsonl"
        )
        task_report["files"].append(str(jsonl_path))
        
        # Export metadata
        metadata_path = self._export_task_metadata(
            task_config,
            task_name,
            output_dir
        )
        task_report["files"].append(str(metadata_path))
        
        return task_report
    
    def _export_task_metadata(
        self,
        task_config: HarnessTaskConfig,
        task_name: str,
        output_dir: Path
    ) -> Path:
        """Export task metadata as JSON file"""
        
        metadata_path = output_dir / f"{task_name}_metadata.json"
        
        metadata = {
            "task_name": task_name,
            "export_timestamp": datetime.now().isoformat(),
            "task_config": task_config.to_dict(),
            "source": "agentic_benchmark_factory",
            "version": "2.0"
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata_path
    
    def _create_documentation(
        self,
        task_name: str,
        transformed_data: Dict[str, Any],
        output_dir: Path
    ) -> Path:
        """Create README documentation for the exported benchmark"""
        
        readme_path = output_dir / "README.md"
        
        # Collect statistics
        total_items = sum(len(dataset) for dataset in transformed_data["datasets"].values())
        task_types = list(transformed_data["tasks"].keys())
        
        readme_content = f"""# {task_name} - Agentic Benchmark for LM-Evaluation-Harness

This benchmark was generated by the Agentic Benchmark Factory and is compatible with EleutherAI's lm-evaluation-harness.

## Overview

- **Total Items**: {total_items}
- **Tasks**: {len(task_types)}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Source**: Agentic Benchmark Factory v2.0

## Tasks Included

{self._format_task_list(transformed_data["tasks"])}

## Quick Start

1. **Install lm-evaluation-harness**:
   ```bash
   pip install lm-eval
   ```

2. **Run evaluation**:
   ```bash
   lm_eval --model hf \\
           --model_args pretrained=<your_model> \\
           --tasks {task_name} \\
           --device cuda \\
           --batch_size 8
   ```

3. **Run specific subtask**:
   ```bash
   lm_eval --model hf \\
           --model_args pretrained=<your_model> \\
           --tasks {task_types[0] if task_types else task_name} \\
           --device cuda
   ```

## File Structure

```
{task_name}/
├── README.md                    # This file
├── {task_name}_group.yaml      # Task group configuration (if multiple tasks)
{self._format_file_list(task_types)}
```

## Task Details

{self._format_task_details(transformed_data["tasks"])}

## Quality Assurance

All questions in this benchmark were generated through a multi-agent pipeline:

- ✅ **ConceptMiner**: Extracts key concepts from source corpus
- ✅ **QuestionWriter**: Creates well-grounded questions using chain-of-thought
- ✅ **Adversary**: Enhances difficulty with adversarial techniques
- ✅ **Refiner**: Enforces formatting and style guidelines
- ✅ **Validator**: Quality control and deterministic validation

Each item includes comprehensive metadata with:
- Difficulty level (basic/intermediate/hard/expert)
- Deterministic vs non-deterministic classification
- Quality validation scores
- Agent provenance and techniques used
- Generation timestamps

## Support

For issues or questions about this benchmark:
- Review the task configurations in the `.yaml` files
- Check the dataset format in the `.jsonl` files
- Refer to lm-evaluation-harness documentation: https://github.com/EleutherAI/lm-evaluation-harness

Generated by Agentic Benchmark Factory v2.0
"""

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        return readme_path
    
    def _format_task_list(self, tasks: Dict[str, HarnessTaskConfig]) -> str:
        """Format task list for documentation"""
        lines = []
        for task_name, config in tasks.items():
            metadata = config.metadata or {}
            item_count = metadata.get('items_count', 'unknown')
            answer_type = metadata.get('answer_type', 'unknown')
            lines.append(f"- **{task_name}**: {item_count} items ({answer_type} answers)")
        return '\\n'.join(lines)
    
    def _format_file_list(self, task_types: List[str]) -> str:
        """Format file list for documentation"""
        lines = []
        for task_name in task_types:
            lines.extend([
                f"├── {task_name}.yaml            # Task configuration",
                f"├── {task_name}_dataset.jsonl   # Dataset items",
                f"├── {task_name}_metadata.json   # Task metadata"
            ])
        return '\\n'.join(lines)
    
    def _format_task_details(self, tasks: Dict[str, HarnessTaskConfig]) -> str:
        """Format detailed task information"""
        lines = []
        for task_name, config in tasks.items():
            metadata = config.metadata or {}
            lines.extend([
                f"### {task_name}",
                f"",
                f"- **Output Type**: {config.output_type.value}",
                f"- **Items**: {metadata.get('items_count', 'unknown')}",
                f"- **Answer Type**: {metadata.get('answer_type', 'unknown')}",
                f"- **Evaluation Type**: {metadata.get('eval_type', 'unknown')}",
                f"- **Deterministic Ratio**: {metadata.get('deterministic_ratio', 'unknown'):.1%}" if isinstance(metadata.get('deterministic_ratio'), (int, float)) else f"- **Deterministic Ratio**: {metadata.get('deterministic_ratio', 'unknown')}",
                f"- **Metrics**: {', '.join(m.metric for m in config.metric_list)}",
                f""
            ])
        return '\\n'.join(lines)
    
    def _create_package(
        self,
        task_name: str,
        output_dir: Path,
        file_list: List[str]
    ) -> Path:
        """Create ZIP package of all exported files"""
        
        package_path = output_dir / f"{task_name}_lm_eval_harness.zip"
        
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in file_list:
                file_path = Path(file_path)
                if file_path.exists():
                    # Store with relative path
                    arcname = file_path.name
                    zf.write(file_path, arcname)
        
        logger.info(f"Created package: {package_path}")
        return package_path


# Convenience functions for easy usage

def export_agentic_benchmark_to_lm_eval(
    items: List[EnhancedBenchmarkItem],
    task_name: str,
    output_dir: str = "./lm_eval_exports",
    create_package: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to export agentic benchmark to lm-evaluation-harness format
    
    Args:
        items: List of agentic benchmark items
        task_name: Name for the exported benchmark
        output_dir: Directory where to save exported files
        create_package: Whether to create a ZIP package
        
    Returns:
        Export report with file paths and metadata
    """
    
    exporter = LMEvalHarnessExporter()
    return exporter.export_complete_benchmark(
        items, task_name, Path(output_dir), create_package
    )


def validate_lm_eval_export(export_dir: Path) -> Dict[str, Any]:
    """
    Validate an existing lm-eval export directory
    
    Args:
        export_dir: Directory containing exported files
        
    Returns:
        Validation report
    """
    
    export_dir = Path(export_dir)
    validation_report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "files_found": [],
        "tasks_found": []
    }
    
    if not export_dir.exists():
        validation_report["valid"] = False
        validation_report["errors"].append(f"Export directory does not exist: {export_dir}")
        return validation_report
    
    # Find YAML task files
    yaml_files = list(export_dir.glob("*.yaml"))
    jsonl_files = list(export_dir.glob("*_dataset.jsonl"))
    
    validation_report["files_found"] = [str(f) for f in yaml_files + jsonl_files]
    
    # Validate each task
    for yaml_file in yaml_files:
        if yaml_file.name.endswith("_group.yaml"):
            continue  # Skip group files
        
        try:
            with open(yaml_file, 'r') as f:
                task_config = yaml.safe_load(f)
            
            # Basic validation
            required_fields = ["task", "dataset_path", "output_type", "doc_to_text", "doc_to_target"]
            missing_fields = [field for field in required_fields if field not in task_config]
            
            if missing_fields:
                validation_report["errors"].append(
                    f"Task {yaml_file.name}: Missing required fields: {missing_fields}"
                )
                validation_report["valid"] = False
            else:
                validation_report["tasks_found"].append(task_config["task"])
            
            # Check if corresponding dataset file exists
            dataset_path = export_dir / task_config.get("dataset_path", "")
            if not dataset_path.exists():
                validation_report["errors"].append(
                    f"Task {yaml_file.name}: Dataset file not found: {dataset_path}"
                )
                validation_report["valid"] = False
        
        except Exception as e:
            validation_report["errors"].append(f"Task {yaml_file.name}: YAML parsing error: {e}")
            validation_report["valid"] = False
    
    return validation_report