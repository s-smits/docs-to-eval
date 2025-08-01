"""
Utility functions and CLI interface for lm-evaluation-harness integration
Provides convenient functions for common export and validation operations
"""

import argparse
import json
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from .lm_eval_exporter import export_agentic_benchmark_to_lm_eval, validate_lm_eval_export
from .models import EnhancedBenchmarkItem
from .generator import AgenticBenchmarkGenerator
from ..evaluation import EvaluationType


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_agentic_benchmark_items(file_path: Path) -> List[EnhancedBenchmarkItem]:
    """
    Load agentic benchmark items from JSON file
    
    Args:
        file_path: Path to JSON file containing benchmark items
        
    Returns:
        List of EnhancedBenchmarkItem objects
    """
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = []
    
    # Handle different input formats
    if isinstance(data, dict):
        if 'items' in data:
            # Standard benchmark format
            raw_items = data['items']
        elif 'benchmark_items' in data:
            # Alternative format
            raw_items = data['benchmark_items']
        else:
            # Assume the dict itself is an item
            raw_items = [data]
    elif isinstance(data, list):
        # List of items
        raw_items = data
    else:
        raise ValueError("Invalid benchmark file format")
    
    # Convert to EnhancedBenchmarkItem objects
    for raw_item in raw_items:
        try:
            # This is a simplified conversion - in practice, you'd need proper deserialization
            # For now, we'll assume the items are already in the correct format or create mock items
            if isinstance(raw_item, dict):
                # Create a mock EnhancedBenchmarkItem for testing
                # In practice, this would be proper deserialization
                item = create_mock_enhanced_item(raw_item)
                items.append(item)
        except Exception as e:
            logger.warning(f"Failed to load item: {e}")
            continue
    
    logger.info(f"Loaded {len(items)} benchmark items from {file_path}")
    return items


def create_mock_enhanced_item(raw_item: Dict[str, Any]) -> EnhancedBenchmarkItem:
    """
    Create a mock EnhancedBenchmarkItem for testing purposes
    In production, this would be proper deserialization
    """
    from .models import BenchmarkMetadata, DifficultyLevel, AnswerType
    
    # Extract basic fields
    question = raw_item.get('question', 'Sample question?')
    answer = raw_item.get('answer', 'Sample answer')
    context = raw_item.get('context')
    options = raw_item.get('options')
    eval_type = EvaluationType(raw_item.get('eval_type', 'domain_knowledge'))
    
    # Create metadata
    metadata = BenchmarkMetadata(
        difficulty=DifficultyLevel(raw_item.get('difficulty', 'intermediate')),
        deterministic=raw_item.get('deterministic', True),
        agents_used=['MockAgent'],
        validation_score=raw_item.get('validation_score', 0.8)
    )
    
    # Determine answer type
    expected_answer_type = AnswerType.FREE_TEXT
    if options:
        expected_answer_type = AnswerType.MULTIPLE_CHOICE
    elif isinstance(answer, (int, float)):
        expected_answer_type = AnswerType.NUMERIC_EXACT
    
    return EnhancedBenchmarkItem(
        question=question,
        answer=answer,
        context=context,
        options=options,
        eval_type=eval_type,
        metadata=metadata,
        expected_answer_type=expected_answer_type,
        reasoning_chain=[],
        variables={}
    )


def generate_and_export_benchmark(
    corpus_text: str,
    eval_type: EvaluationType,
    task_name: str,
    num_questions: int = 50,
    output_dir: str = "./lm_eval_exports",
    create_package: bool = True,
    llm_pool: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generate agentic benchmark and export to lm-evaluation-harness format in one step
    
    Args:
        corpus_text: Source text for benchmark generation
        eval_type: Type of evaluation to generate
        task_name: Name for the exported benchmark
        num_questions: Number of questions to generate
        output_dir: Directory where to save exported files
        create_package: Whether to create a ZIP package
        llm_pool: LLM pool for agentic generation (optional)
        
    Returns:
        Combined generation and export report
    """
    
    logger.info(f"Generating and exporting {num_questions} {eval_type.value} questions...")
    
    # Generate agentic benchmark
    generator = AgenticBenchmarkGenerator(eval_type, llm_pool)
    standard_items = generator.generate_benchmark(corpus_text, num_questions)
    
    # Convert to enhanced items (simplified conversion for demo)
    enhanced_items = []
    for item in standard_items:
        enhanced_item = create_mock_enhanced_item(item)
        enhanced_items.append(enhanced_item)
    
    # Export to lm-eval format
    export_report = export_agentic_benchmark_to_lm_eval(
        enhanced_items, task_name, output_dir, create_package
    )
    
    # Combine reports
    combined_report = {
        "generation": {
            "items_generated": len(standard_items),
            "generator_type": "agentic",
            "eval_type": eval_type.value
        },
        "export": export_report,
        "success": True
    }
    
    logger.info(f"Successfully generated and exported benchmark: {task_name}")
    return combined_report


def quick_export_demo():
    """
    Quick demonstration of the export functionality
    Creates a sample benchmark and exports it
    """
    
    logger.info("Running quick export demo...")
    
    # Sample corpus
    sample_corpus = """
    Machine learning is a method of data analysis that automates analytical model building.
    It is a branch of artificial intelligence based on the idea that systems can learn from data,
    identify patterns and make decisions with minimal human intervention.
    
    The core principle of machine learning involves algorithms that can receive input data
    and use statistical analysis to predict an output value within an acceptable range.
    As new data is fed to these systems, they learn and optimize their operations to improve
    performance, developing 'intelligence' over time.
    """
    
    try:
        # Generate and export
        report = generate_and_export_benchmark(
            corpus_text=sample_corpus,
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
            task_name="ml_demo_benchmark",
            num_questions=5,
            output_dir="./demo_export",
            create_package=True
        )
        
        print("\\n" + "="*60)
        print("DEMO EXPORT SUCCESSFUL!")
        print("="*60)
        print(f"Generated: {report['generation']['items_generated']} items")
        print(f"Tasks created: {len(report['export']['tasks_created'])}")
        print(f"Files created: {len(report['export']['files_created'])}")
        if report['export']['package_path']:
            print(f"Package: {report['export']['package_path']}")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return False


def cli_export_command():
    """Command-line interface for exporting benchmarks"""
    
    parser = argparse.ArgumentParser(
        description="Export agentic benchmarks to lm-evaluation-harness format"
    )
    
    # Positional arguments
    parser.add_argument(
        "action",
        choices=["export", "validate", "demo"],
        help="Action to perform"
    )
    
    # Export arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input file (JSON benchmark or text corpus)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./lm_eval_exports",
        help="Output directory (default: ./lm_eval_exports)"
    )
    
    parser.add_argument(
        "--name", "-n",
        type=str,
        required=False,
        help="Task name for export"
    )
    
    parser.add_argument(
        "--eval-type", "-e",
        type=str,
        choices=[t.value for t in EvaluationType],
        default="domain_knowledge",
        help="Evaluation type (default: domain_knowledge)"
    )
    
    parser.add_argument(
        "--num-questions", "-q",
        type=int,
        default=50,
        help="Number of questions to generate (default: 50)"
    )
    
    parser.add_argument(
        "--no-package",
        action="store_true",
        help="Don't create ZIP package"
    )
    
    # Validation arguments
    parser.add_argument(
        "--export-dir",
        type=str,
        help="Directory to validate (for validate action)"
    )
    
    # General arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.action == "demo":
            success = quick_export_demo()
            sys.exit(0 if success else 1)
        
        elif args.action == "export":
            if not args.input:
                parser.error("--input is required for export action")
            
            if not args.name:
                # Generate name from input file
                input_path = Path(args.input)
                args.name = input_path.stem
            
            input_path = Path(args.input)
            
            if input_path.suffix.lower() == '.json':
                # Load existing benchmark items
                items = load_agentic_benchmark_items(input_path)
                report = export_agentic_benchmark_to_lm_eval(
                    items, args.name, args.output, not args.no_package
                )
                
                print(f"Exported {len(items)} items to {args.output}")
                
            else:
                # Generate from corpus
                corpus_text = input_path.read_text(encoding='utf-8')
                eval_type = EvaluationType(args.eval_type)
                
                report = generate_and_export_benchmark(
                    corpus_text, eval_type, args.name, args.num_questions,
                    args.output, not args.no_package
                )
                
                print(f"Generated and exported {args.num_questions} questions to {args.output}")
            
            # Print summary
            print(f"Tasks created: {len(report.get('export', report).get('tasks_created', {}))}")
            if report.get('export', report).get('package_path'):
                print(f"Package: {report.get('export', report)['package_path']}")
        
        elif args.action == "validate":
            if not args.export_dir:
                parser.error("--export-dir is required for validate action")
            
            validation_report = validate_lm_eval_export(Path(args.export_dir))
            
            print(f"Validation: {'PASSED' if validation_report['valid'] else 'FAILED'}")
            print(f"Tasks found: {len(validation_report['tasks_found'])}")
            print(f"Files found: {len(validation_report['files_found'])}")
            
            if validation_report['errors']:
                print("\\nErrors:")
                for error in validation_report['errors']:
                    print(f"  - {error}")
            
            if validation_report['warnings']:
                print("\\nWarnings:")
                for warning in validation_report['warnings']:
                    print(f"  - {warning}")
            
            sys.exit(0 if validation_report['valid'] else 1)
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def create_integration_example():
    """Create a comprehensive example showing how to use the integration"""
    
    example_code = '''
"""
Example: Complete workflow for agentic benchmark → lm-evaluation-harness
"""

from docs_to_eval.core.agentic.lm_eval_utils import (
    generate_and_export_benchmark,
    export_agentic_benchmark_to_lm_eval
)
from docs_to_eval.core.evaluation import EvaluationType
from docs_to_eval.core.agentic import AgenticBenchmarkGenerator

# 1. Generate agentic benchmark from corpus
corpus = """
    Your domain-specific corpus text here.
    This should contain the knowledge you want to test.
"""

# Option A: Generate and export in one step
report = generate_and_export_benchmark(
    corpus_text=corpus,
    eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
    task_name="my_benchmark",
    num_questions=100,
    output_dir="./my_lm_eval_tasks"
)

# Option B: Generate first, then export separately
generator = AgenticBenchmarkGenerator(EvaluationType.DOMAIN_KNOWLEDGE)
items = generator.generate_benchmark_async(corpus, num_questions=100)

export_report = export_agentic_benchmark_to_lm_eval(
    items=items,
    task_name="my_benchmark",
    output_dir="./my_lm_eval_tasks",
    create_package=True
)

# 2. Use with lm-evaluation-harness
# The exported files can now be used directly:
#
# lm_eval --model hf \\
#        --model_args pretrained=microsoft/DialoGPT-medium \\
#        --tasks my_benchmark \\
#        --device cuda \\
#        --batch_size 8 \\
#        --output_path ./results

print("Export completed!")
print(f"Package created: {export_report['package_path']}")
print(f"Tasks: {list(export_report['tasks_created'].keys())}")
'''

    example_path = Path("lm_eval_integration_example.py")
    with open(example_path, 'w') as f:
        f.write(example_code)
    
    logger.info(f"Created integration example: {example_path}")
    return example_path


# CLI entry point
def main():
    """Main entry point for CLI"""
    cli_export_command()


if __name__ == "__main__":
    main()