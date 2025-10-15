"""
Comprehensive tests for lm-evaluation-harness export functionality
Tests the complete pipeline from agentic generation to harness export
"""

import asyncio
import json
import pytest
from pathlib import Path
import tempfile
import shutil

from docs_to_eval.core.agentic.generator import AgenticBenchmarkGenerator
from docs_to_eval.core.agentic.lm_eval_exporter import (
    export_agentic_benchmark_to_lm_eval,
    validate_lm_eval_export,
    LMEvalHarnessExporter
)
from docs_to_eval.core.agentic.lm_eval_transform import HarnessTransformer
from docs_to_eval.core.evaluation import EvaluationType
from docs_to_eval.core.agentic.models import (
    EnhancedBenchmarkItem,
    BenchmarkMetadata,
    DifficultyLevel,
    AnswerType
)


class TestLMEvalExport:
    """Test suite for lm-evaluation-harness export"""
    
    @pytest.fixture
    def sample_corpus(self):
        """Sample corpus for testing"""
        return """
        Machine learning is a method of data analysis that automates analytical model building.
        It is a branch of artificial intelligence based on the idea that systems can learn from data,
        identify patterns and make decisions with minimal human intervention.
        
        Deep learning is a subset of machine learning that uses neural networks with multiple layers.
        These networks can automatically learn hierarchical representations of data without manual
        feature engineering. Common architectures include CNNs for image recognition and RNNs for
        sequence processing.
        
        Supervised learning uses labeled training data to learn a mapping from inputs to outputs.
        Unsupervised learning finds hidden patterns in data without labeled examples. Reinforcement
        learning uses rewards and penalties to learn optimal actions in an environment.
        """
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp(prefix="test_lm_eval_")
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_enhanced_items(self):
        """Create sample enhanced benchmark items for testing"""
        items = []
        
        # Multiple choice item
        items.append(EnhancedBenchmarkItem(
            question="What is machine learning?",
            answer="A method of data analysis that automates analytical model building",
            context="Machine learning is a method of data analysis...",
            options=[
                "A method of data analysis that automates analytical model building",
                "A type of computer hardware",
                "A programming language",
                "A database system"
            ],
            eval_type=EvaluationType.MULTIPLE_CHOICE,
            expected_answer_type=AnswerType.MULTIPLE_CHOICE,
            metadata=BenchmarkMetadata(
                difficulty=DifficultyLevel.INTERMEDIATE,
                deterministic=True,
                agents_used=["ConceptMiner", "QuestionWriter", "Validator"],
                validation_score=0.85
            ),
            reasoning_chain=["Identify key concept", "Extract definition", "Generate options"]
        ))
        
        # Free text item
        items.append(EnhancedBenchmarkItem(
            question="Explain the difference between supervised and unsupervised learning.",
            answer="Supervised learning uses labeled training data to learn input-output mappings, while unsupervised learning finds hidden patterns without labeled examples.",
            context="Supervised learning uses labeled training data...",
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
            expected_answer_type=AnswerType.FREE_TEXT,
            metadata=BenchmarkMetadata(
                difficulty=DifficultyLevel.HARD,
                deterministic=False,
                agents_used=["ConceptMiner", "QuestionWriter", "Adversary", "Validator"],
                validation_score=0.92,
                adversarial_techniques=["multi_hop_reasoning"]
            ),
            reasoning_chain=["Compare concepts", "Contrast approaches", "Synthesize explanation"]
        ))
        
        # Numeric item
        items.append(EnhancedBenchmarkItem(
            question="How many types of machine learning are mentioned?",
            answer="3",
            context="Supervised learning, unsupervised learning, and reinforcement learning",
            eval_type=EvaluationType.FACTUAL_QA,
            expected_answer_type=AnswerType.NUMERIC_EXACT,
            metadata=BenchmarkMetadata(
                difficulty=DifficultyLevel.BASIC,
                deterministic=True,
                agents_used=["ConceptMiner", "QuestionWriter", "Validator"],
                validation_score=0.95
            )
        ))
        
        return items
    
    def test_jsonl_format(self, sample_enhanced_items, temp_output_dir):
        """Test that JSONL export has correct format (newlines not escaped)"""
        exporter = LMEvalHarnessExporter()
        
        report = exporter.export_complete_benchmark(
            sample_enhanced_items,
            "test_benchmark",
            temp_output_dir,
            create_package=False
        )
        
        # Find a JSONL file
        jsonl_files = list(temp_output_dir.glob("*_dataset.jsonl"))
        assert len(jsonl_files) > 0, "No JSONL files created"
        
        # Read and verify format
        with open(jsonl_files[0], 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check that lines are actually separated by newlines
            lines = content.strip().split('\n')
            assert len(lines) > 0, "JSONL file is empty"
            
            # Each line should be valid JSON
            for i, line in enumerate(lines):
                try:
                    obj = json.loads(line)
                    assert isinstance(obj, dict), f"Line {i} is not a dict"
                    assert 'question' in obj, f"Line {i} missing 'question'"
                    assert 'answer' in obj, f"Line {i} missing 'answer'"
                except json.JSONDecodeError as e:
                    pytest.fail(f"Line {i} is not valid JSON: {e}\nContent: {line[:100]}")
            
            # Verify no escaped newlines in the file
            assert '\\n' not in content, "Found escaped newlines (\\\\n) in JSONL - should be actual newlines"
    
    def test_yaml_template_newlines(self, sample_enhanced_items, temp_output_dir):
        """Test that YAML templates use actual newlines, not escaped ones"""
        exporter = LMEvalHarnessExporter()
        
        report = exporter.export_complete_benchmark(
            sample_enhanced_items,
            "test_benchmark",
            temp_output_dir,
            create_package=False
        )
        
        # Find YAML files
        yaml_files = list(temp_output_dir.glob("*.yaml"))
        yaml_files = [f for f in yaml_files if not f.name.endswith("_group.yaml")]
        
        assert len(yaml_files) > 0, "No YAML task files created"
        
        # Read and check templates
        import yaml
        for yaml_file in yaml_files:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Check doc_to_text template
            if 'doc_to_text' in config:
                template = config['doc_to_text']
                # Should have actual newlines in the YAML (which will show as \n in the string)
                assert '\n' in template or 'Question' in template, \
                    f"Template should contain newlines or question text: {template}"
    
    def test_export_validation(self, sample_enhanced_items, temp_output_dir):
        """Test that exported files pass validation"""
        exporter = LMEvalHarnessExporter()
        
        report = exporter.export_complete_benchmark(
            sample_enhanced_items,
            "test_benchmark",
            temp_output_dir,
            create_package=False
        )
        
        # Validate the export
        validation_report = validate_lm_eval_export(temp_output_dir)
        
        assert validation_report['valid'], \
            f"Export validation failed: {validation_report['errors']}"
        assert len(validation_report['tasks_found']) > 0, "No tasks found"
        assert len(validation_report['files_found']) > 0, "No files found"
    
    def test_multiple_answer_types(self, sample_enhanced_items, temp_output_dir):
        """Test that different answer types are correctly grouped and exported"""
        exporter = LMEvalHarnessExporter()
        
        report = exporter.export_complete_benchmark(
            sample_enhanced_items,
            "test_benchmark",
            temp_output_dir,
            create_package=False
        )
        
        # Should create separate tasks for different answer types
        assert len(report['tasks_created']) >= 2, \
            "Should create multiple tasks for different answer types"
        
        # Check that we have the expected answer types
        task_names = list(report['tasks_created'].keys())
        answer_types_in_tasks = [name.split('_')[-1] for name in task_names]
        
        assert 'multiple_choice' in answer_types_in_tasks, \
            "Should have multiple choice task"
        assert 'free_text' in answer_types_in_tasks or 'numeric_exact' in answer_types_in_tasks, \
            "Should have free text or numeric task"
    
    def test_generation_kwargs(self, sample_enhanced_items, temp_output_dir):
        """Test that generation_kwargs are correctly set"""
        exporter = LMEvalHarnessExporter()
        
        report = exporter.export_complete_benchmark(
            sample_enhanced_items,
            "test_benchmark",
            temp_output_dir,
            create_package=False
        )
        
        # Read YAML files and check generation_kwargs
        import yaml
        yaml_files = list(temp_output_dir.glob("*.yaml"))
        yaml_files = [f for f in yaml_files if not f.name.endswith("_group.yaml")]
        
        for yaml_file in yaml_files:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config.get('output_type') == 'generate_until':
                assert 'generation_kwargs' in config, \
                    f"generate_until tasks should have generation_kwargs: {yaml_file.name}"
                
                kwargs = config['generation_kwargs']
                assert 'until' in kwargs, "generation_kwargs should have 'until'"
                assert isinstance(kwargs['until'], list), "'until' should be a list"
                
                # Check that 'until' tokens are not escaped
                for token in kwargs['until']:
                    assert '\\' not in token, \
                        f"'until' tokens should not have escaped characters: {token}"
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_corpus, temp_output_dir):
        """Test complete pipeline from generation to export"""
        # Generate benchmark
        generator = AgenticBenchmarkGenerator(
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE
        )
        
        items = await generator.generate_benchmark_async(
            sample_corpus,
            num_questions=5
        )
        
        assert len(items) > 0, "Generator should produce items"
        
        # Export to lm-eval format
        report = export_agentic_benchmark_to_lm_eval(
            items,
            "test_ml_benchmark",
            str(temp_output_dir),
            create_package=True
        )
        
        assert report['total_items'] > 0, "Should export items"
        assert len(report['tasks_created']) > 0, "Should create tasks"
        assert len(report['files_created']) > 0, "Should create files"
        assert report['package_path'] is not None, "Should create package"
        
        # Verify package exists
        package_path = Path(report['package_path'])
        assert package_path.exists(), "Package file should exist"
        assert package_path.suffix == '.zip', "Package should be ZIP file"
    
    def test_readme_generation(self, sample_enhanced_items, temp_output_dir):
        """Test that README is properly generated"""
        exporter = LMEvalHarnessExporter()
        
        report = exporter.export_complete_benchmark(
            sample_enhanced_items,
            "test_benchmark",
            temp_output_dir,
            create_package=False
        )
        
        readme_path = temp_output_dir / "README.md"
        assert readme_path.exists(), "README.md should be created"
        
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        # Check for essential sections
        assert "# test_benchmark" in readme_content, "Should have title"
        assert "Quick Start" in readme_content, "Should have quick start"
        assert "lm_eval" in readme_content, "Should mention lm_eval command"
        assert "Quality Assurance" in readme_content, "Should have quality section"


def test_transformer_grouping():
    """Test that items are correctly grouped by eval_type and answer_type"""
    from docs_to_eval.core.agentic.lm_eval_transform import HarnessTransformer
    
    items = [
        EnhancedBenchmarkItem(
            question="Q1",
            answer="A1",
            eval_type=EvaluationType.FACTUAL_QA,
            expected_answer_type=AnswerType.STRING_EXACT,
            metadata=BenchmarkMetadata(
                difficulty=DifficultyLevel.BASIC,
                deterministic=True,
                agents_used=[]
            )
        ),
        EnhancedBenchmarkItem(
            question="Q2",
            answer="A2",
            eval_type=EvaluationType.FACTUAL_QA,
            expected_answer_type=AnswerType.FREE_TEXT,
            metadata=BenchmarkMetadata(
                difficulty=DifficultyLevel.BASIC,
                deterministic=False,
                agents_used=[]
            )
        ),
        EnhancedBenchmarkItem(
            question="Q3",
            answer="A3",
            eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
            expected_answer_type=AnswerType.FREE_TEXT,
            metadata=BenchmarkMetadata(
                difficulty=DifficultyLevel.HARD,
                deterministic=False,
                agents_used=[]
            )
        ),
    ]
    
    transformer = HarnessTransformer()
    result = transformer.transform_to_harness_format(items, "test")
    
    # Should create 3 tasks (2 eval types × different answer types)
    assert len(result['tasks']) == 3, f"Expected 3 tasks, got {len(result['tasks'])}"
    assert len(result['datasets']) == 3, "Should have 3 datasets"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
