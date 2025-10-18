#!/usr/bin/env python3
"""
Comprehensive System Validation Script
Validates all components of the docs-to-eval system
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.manual import RESULTS_DIR

from docs_to_eval.core.agentic.generator import AgenticBenchmarkGenerator
from docs_to_eval.core.agentic.lm_eval_exporter import (
    export_agentic_benchmark_to_lm_eval,
    validate_lm_eval_export
)
from docs_to_eval.core.agentic.validation import ComprehensiveValidator
from docs_to_eval.core.evaluation import EvaluationType
from docs_to_eval.core.classification import EvaluationTypeClassifier
from docs_to_eval.core.agentic.models import DifficultyLevel


class SystemValidator:
    """Validates the entire docs-to-eval system"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'overall_status': 'unknown'
        }
        self.test_corpus = """
        Machine learning is a method of data analysis that automates analytical 
        model building. It is a branch of artificial intelligence based on the 
        idea that systems can learn from data, identify patterns and make 
        decisions with minimal human intervention.
        
        Deep learning is a subset of machine learning that uses neural networks 
        with multiple layers. These networks can automatically learn hierarchical 
        representations of data without manual feature engineering. Common 
        architectures include CNNs for image recognition and RNNs for sequence 
        processing.
        
        Supervised learning uses labeled training data to learn input-output 
        mappings. Unsupervised learning finds hidden patterns without labels. 
        Reinforcement learning uses rewards to learn optimal actions.
        """
    
    def print_section(self, title: str, char: str = "="):
        """Print formatted section header"""
        print(f"\n{char * 70}")
        print(f"{title:^70}")
        print(f"{char * 70}\n")
    
    def print_test(self, name: str, status: str, details: str = ""):
        """Print test result"""
        status_symbols = {
            'PASS': '✅',
            'FAIL': '❌',
            'WARN': '⚠️',
            'INFO': 'ℹ️'
        }
        symbol = status_symbols.get(status, '•')
        print(f"{symbol} {name:.<50} {status}")
        if details:
            print(f"   {details}")
    
    async def test_classification(self) -> Dict[str, Any]:
        """Test corpus classification"""
        self.print_section("Testing Classification System", "-")
        
        result = {'passed': 0, 'failed': 0, 'details': []}
        
        try:
            classifier = EvaluationTypeClassifier()
            classification = classifier.classify_corpus(self.test_corpus)
            
            # Test 1: Classification returns valid type
            if classification.primary_type in EvaluationType:
                self.print_test("Classification type valid", "PASS")
                result['passed'] += 1
            else:
                self.print_test("Classification type valid", "FAIL",
                              f"Invalid type: {classification.primary_type}")
                result['failed'] += 1
            
            # Test 2: Confidence score in valid range
            if 0 <= classification.confidence <= 1:
                self.print_test("Confidence score valid", "PASS",
                              f"Score: {classification.confidence:.3f}")
                result['passed'] += 1
            else:
                self.print_test("Confidence score valid", "FAIL",
                              f"Invalid: {classification.confidence}")
                result['failed'] += 1
            
            # Test 3: Pipeline config generated
            if 'verification' in classification.pipeline:
                self.print_test("Pipeline config generated", "PASS")
                result['passed'] += 1
            else:
                self.print_test("Pipeline config generated", "FAIL")
                result['failed'] += 1
            
            result['classification_result'] = {
                'primary_type': str(classification.primary_type),
                'confidence': classification.confidence,
                'secondary_types': [str(t) for t in classification.secondary_types]
            }
            
        except Exception as e:
            self.print_test("Classification system", "FAIL", str(e))
            result['failed'] += 1
            result['error'] = str(e)
        
        return result
    
    async def test_agentic_generation(self) -> Dict[str, Any]:
        """Test agentic benchmark generation"""
        self.print_section("Testing Agentic Generation", "-")
        
        result = {'passed': 0, 'failed': 0, 'details': []}
        
        try:
            generator = AgenticBenchmarkGenerator(
                eval_type=EvaluationType.DOMAIN_KNOWLEDGE
            )
            
            # Generate small batch for testing
            items = await generator.generate_benchmark_async(
                self.test_corpus,
                num_questions=5
            )
            
            # Test 1: Items generated
            if len(items) > 0:
                self.print_test("Items generated", "PASS",
                              f"Generated {len(items)} items")
                result['passed'] += 1
            else:
                self.print_test("Items generated", "FAIL",
                              "No items generated")
                result['failed'] += 1
            
            # Test 2: Items have required fields
            required_fields = ['question', 'answer', 'metadata']
            all_valid = True
            for i, item in enumerate(items):
                for field in required_fields:
                    if not hasattr(item, field):
                        all_valid = False
                        self.print_test(f"Item {i} has {field}", "FAIL")
                        result['failed'] += 1
            
            if all_valid:
                self.print_test("All items have required fields", "PASS")
                result['passed'] += 1
            
            # Test 3: Metadata quality
            avg_quality = sum(
                item.metadata.validation_score or 0 
                for item in items
            ) / len(items) if items else 0
            
            if avg_quality >= 0.5:
                self.print_test("Average quality score", "PASS",
                              f"Score: {avg_quality:.3f}")
                result['passed'] += 1
            else:
                self.print_test("Average quality score", "WARN",
                              f"Low score: {avg_quality:.3f}")
            
            result['generation_stats'] = {
                'items_generated': len(items),
                'avg_quality': avg_quality,
                'difficulties': list(set(
                    item.metadata.difficulty.value if hasattr(item.metadata.difficulty, 'value')
                    else str(item.metadata.difficulty)
                    for item in items
                ))
            }
            
            # Store items for later tests
            self.generated_items = items
            
        except Exception as e:
            self.print_test("Agentic generation", "FAIL", str(e))
            result['failed'] += 1
            result['error'] = str(e)
            import traceback
            traceback.print_exc()
        
        return result
    
    async def test_validation(self) -> Dict[str, Any]:
        """Test validation system"""
        self.print_section("Testing Validation System", "-")
        
        result = {'passed': 0, 'failed': 0, 'details': []}
        
        try:
            if not hasattr(self, 'generated_items'):
                self.print_test("Validation system", "FAIL",
                              "No items to validate (generation failed)")
                result['failed'] += 1
                return result
            
            validator = ComprehensiveValidator(min_quality_score=0.6)
            
            # Test validation
            validation_report = await validator.validate_benchmark_batch(
                self.generated_items
            )
            
            # Test 1: Validation completes
            if validation_report:
                self.print_test("Validation completes", "PASS")
                result['passed'] += 1
            else:
                self.print_test("Validation completes", "FAIL")
                result['failed'] += 1
            
            # Test 2: Pass rate reasonable
            pass_rate = validation_report.get('overall_pass_rate', 0)
            if pass_rate >= 0.5:
                self.print_test("Overall pass rate", "PASS",
                              f"Rate: {pass_rate:.1%}")
                result['passed'] += 1
            else:
                self.print_test("Overall pass rate", "WARN",
                              f"Low rate: {pass_rate:.1%}")
            
            # Test 3: Deterministic validation works
            det_validation = validation_report.get('deterministic_validation', {})
            if 'consistency_rate' in det_validation:
                self.print_test("Deterministic validation", "PASS",
                              f"Consistency: {det_validation['consistency_rate']:.1%}")
                result['passed'] += 1
            else:
                self.print_test("Deterministic validation", "FAIL")
                result['failed'] += 1
            
            result['validation_stats'] = {
                'overall_pass_rate': pass_rate,
                'consistency_rate': det_validation.get('consistency_rate', 0),
                'avg_quality': validation_report.get('quality_assessment', {}).get('avg_quality', 0)
            }
            
        except Exception as e:
            self.print_test("Validation system", "FAIL", str(e))
            result['failed'] += 1
            result['error'] = str(e)
            import traceback
            traceback.print_exc()
        
        return result
    
    async def test_lm_eval_export(self) -> Dict[str, Any]:
        """Test lm-evaluation-harness export"""
        self.print_section("Testing LM-Eval Export", "-")
        
        result = {'passed': 0, 'failed': 0, 'details': []}
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="validation_test_")
        
        try:
            if not hasattr(self, 'generated_items'):
                self.print_test("LM-Eval export", "FAIL",
                              "No items to export (generation failed)")
                result['failed'] += 1
                return result
            
            # Export
            export_report = export_agentic_benchmark_to_lm_eval(
                self.generated_items,
                "test_benchmark",
                temp_dir,
                create_package=True
            )
            
            # Test 1: Export completes
            if export_report:
                self.print_test("Export completes", "PASS")
                result['passed'] += 1
            else:
                self.print_test("Export completes", "FAIL")
                result['failed'] += 1
            
            # Test 2: Files created
            files_created = export_report.get('files_created', [])
            if len(files_created) > 0:
                self.print_test("Files created", "PASS",
                              f"{len(files_created)} files")
                result['passed'] += 1
            else:
                self.print_test("Files created", "FAIL")
                result['failed'] += 1
            
            # Test 3: JSONL format validation
            jsonl_files = [f for f in files_created if f.endswith('.jsonl')]
            jsonl_valid = True
            
            for jsonl_file in jsonl_files:
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                        # Validate each line is valid JSON
                        for i, line in enumerate(lines):
                            line = line.strip()
                            if not line:
                                continue
                            
                            # Parse JSON
                            obj = json.loads(line)
                            
                            # Check required fields
                            if 'question' not in obj:
                                raise ValueError(f"Line {i}: missing 'question'")
                            if 'answer' not in obj:
                                raise ValueError(f"Line {i}: missing 'answer'")
                
                except Exception as e:
                    self.print_test("JSONL format", "FAIL",
                                  f"{Path(jsonl_file).name}: {e}")
                    jsonl_valid = False
                    result['failed'] += 1
                    break
            
            if jsonl_valid and jsonl_files:
                self.print_test("JSONL format valid", "PASS")
                result['passed'] += 1
            
            # Test 4: YAML validation
            yaml_files = [f for f in files_created 
                         if f.endswith('.yaml') and not f.endswith('_group.yaml')]
            
            if len(yaml_files) > 0:
                import yaml
                yaml_valid = True
                
                for yaml_file in yaml_files:
                    try:
                        with open(yaml_file, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f)
                        
                        # Check required fields
                        required = ['task', 'dataset_path', 'output_type',
                                  'doc_to_text', 'doc_to_target']
                        missing = [f for f in required if f not in config]
                        
                        if missing:
                            self.print_test("YAML structure", "FAIL",
                                          f"{Path(yaml_file).name} missing: {missing}")
                            yaml_valid = False
                            result['failed'] += 1
                            break
                    
                    except Exception as e:
                        self.print_test("YAML parsing", "FAIL",
                                      f"{Path(yaml_file).name}: {e}")
                        yaml_valid = False
                        result['failed'] += 1
                        break
                
                if yaml_valid:
                    self.print_test("YAML configs valid", "PASS")
                    result['passed'] += 1
            
            # Test 5: Package created
            package_path = export_report.get('package_path')
            if package_path and Path(package_path).exists():
                self.print_test("Package created", "PASS",
                              Path(package_path).name)
                result['passed'] += 1
            else:
                self.print_test("Package created", "FAIL")
                result['failed'] += 1
            
            # Test 6: Validation passes
            validation_report = validate_lm_eval_export(Path(temp_dir))
            
            if validation_report['valid']:
                self.print_test("Export validation passes", "PASS")
                result['passed'] += 1
            else:
                self.print_test("Export validation passes", "FAIL",
                              f"Errors: {validation_report['errors']}")
                result['failed'] += 1
            
            result['export_stats'] = {
                'files_created': len(files_created),
                'tasks_created': len(export_report.get('tasks_created', {})),
                'package_created': package_path is not None,
                'validation_passed': validation_report['valid']
            }
            
        except Exception as e:
            self.print_test("LM-Eval export", "FAIL", str(e))
            result['failed'] += 1
            result['error'] = str(e)
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return result
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests"""
        self.print_section("DOCS-TO-EVAL SYSTEM VALIDATION")
        
        print(f"Started: {self.results['timestamp']}")
        print(f"Test Corpus: {len(self.test_corpus)} characters\n")
        
        # Run tests sequentially
        self.results['tests']['classification'] = await self.test_classification()
        self.results['tests']['generation'] = await self.test_agentic_generation()
        self.results['tests']['validation'] = await self.test_validation()
        self.results['tests']['lm_eval_export'] = await self.test_lm_eval_export()
        
        # Calculate overall results
        total_passed = sum(t['passed'] for t in self.results['tests'].values())
        total_failed = sum(t['failed'] for t in self.results['tests'].values())
        total_tests = total_passed + total_failed
        
        self.results['summary'] = {
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'pass_rate': total_passed / total_tests if total_tests > 0 else 0
        }
        
        # Determine overall status
        if total_failed == 0:
            self.results['overall_status'] = 'PASS'
        elif total_passed > total_failed:
            self.results['overall_status'] = 'PARTIAL'
        else:
            self.results['overall_status'] = 'FAIL'
        
        # Print summary
        self.print_section("VALIDATION SUMMARY")
        
        print(f"Total Tests:  {total_tests}")
        print(f"Passed:       {total_passed} ✅")
        print(f"Failed:       {total_failed} ❌")
        print(f"Pass Rate:    {self.results['summary']['pass_rate']:.1%}")
        print(f"Overall:      {self.results['overall_status']}")
        
        # Save results
        results_file = RESULTS_DIR / "validation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📄 Full results saved to: {results_file}")
        
        return self.results


async def main():
    """Main validation function"""
    validator = SystemValidator()
    results = await validator.run_all_tests()
    
    # Exit with appropriate code
    if results['overall_status'] == 'PASS':
        sys.exit(0)
    elif results['overall_status'] == 'PARTIAL':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
