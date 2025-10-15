# LM-Evaluation-Harness Integration Guide

Complete guide for exporting agentic benchmarks to EleutherAI's lm-evaluation-harness format.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [File Format Specifications](#file-format-specifications)
- [Troubleshooting](#troubleshooting)
- [Advanced Features](#advanced-features)

## Overview

The lm-evaluation-harness integration allows you to:

1. ✅ Generate high-quality benchmarks using the agentic pipeline
2. ✅ Export to lm-evaluation-harness compatible YAML + JSONL format
3. ✅ Automatically group items by evaluation type and answer type
4. ✅ Generate proper Jinja2 templates for task configuration
5. ✅ Create complete packages ready for evaluation

### Key Components

| Component | Purpose | File |
|-----------|---------|------|
| `HarnessTransformer` | Core data transformation | `lm_eval_transform.py` |
| `LMEvalHarnessExporter` | File generation and packaging | `lm_eval_exporter.py` |
| `YAMLTaskExporter` | YAML task config export | `lm_eval_exporter.py` |
| `JSONLDatasetExporter` | JSONL dataset export | `lm_eval_exporter.py` |

## Quick Start

### 1. Generate and Export in One Step

```python
from docs_to_eval.core.agentic.lm_eval_utils import generate_and_export_benchmark
from docs_to_eval.core.evaluation import EvaluationType

corpus = """
    Machine learning is a method of data analysis that automates
    analytical model building. It uses neural networks, statistical
    algorithms, and computational methods to learn from data.
"""

# Generate and export
report = generate_and_export_benchmark(
    corpus_text=corpus,
    eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
    task_name="ml_benchmark",
    num_questions=50,
    output_dir="./lm_eval_tasks",
    create_package=True
)

print(f"Created {len(report['export']['tasks_created'])} tasks")
print(f"Package: {report['export']['package_path']}")
```

### 2. Generate First, Export Later

```python
from docs_to_eval.core.agentic.generator import AgenticBenchmarkGenerator
from docs_to_eval.core.agentic.lm_eval_exporter import export_agentic_benchmark_to_lm_eval

# Step 1: Generate
generator = AgenticBenchmarkGenerator(EvaluationType.DOMAIN_KNOWLEDGE)
items = await generator.generate_benchmark_async(corpus, num_questions=50)

# Step 2: Export
report = export_agentic_benchmark_to_lm_eval(
    items=items,
    task_name="ml_benchmark",
    output_dir="./lm_eval_tasks",
    create_package=True
)
```

### 3. Use with lm-evaluation-harness

```bash
# Install lm-evaluation-harness
pip install lm-eval

# Run evaluation
lm_eval --model hf \
        --model_args pretrained=microsoft/DialoGPT-medium \
        --tasks ml_benchmark \
        --device cuda \
        --batch_size 8 \
        --output_path ./results
```

## Architecture

### Data Flow

```
┌─────────────────┐
│ Corpus Text     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ AgenticBenchmarkGenerator   │
│ (ConceptMiner → Writer →    │
│  Adversary → Refiner →      │
│  Validator)                 │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ EnhancedBenchmarkItem[]     │
│ - question, answer          │
│ - eval_type, answer_type    │
│ - metadata, provenance      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ HarnessTransformer          │
│ - Group by type             │
│ - Generate templates        │
│ - Map metrics               │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ LMEvalHarnessExporter       │
│ - Create YAML configs       │
│ - Export JSONL datasets     │
│ - Generate README           │
│ - Package files             │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ lm-evaluation-harness       │
│ Ready Package               │
│ - task_name.yaml            │
│ - task_name_dataset.jsonl   │
│ - README.md                 │
│ - task_name.zip             │
└─────────────────────────────┘
```

### Answer Type Mapping

| AnswerType | HarnessOutputType | Primary Metrics | Stop Tokens |
|------------|------------------|-----------------|-------------|
| MULTIPLE_CHOICE | multiple_choice | acc, acc_norm | N/A |
| NUMERIC_EXACT | generate_until | exact_match | \n, ., ?, (space) |
| STRING_EXACT | generate_until | exact_match, f1 | \n, ., ? |
| CODE | generate_until | exact_match, bleu | \n\n, # End, ``` |
| FREE_TEXT | generate_until | bleu, rouge | \n\n |
| BOOLEAN | generate_until | exact_match | \n, ., ? |

## Usage Examples

### Example 1: Domain-Specific Benchmark

```python
import asyncio
from docs_to_eval.core.agentic import AgenticBenchmarkDemo

async def create_domain_benchmark():
    demo = AgenticBenchmarkDemo()
    
    corpus = """
    The Etruscan civilization flourished in central Italy before
    Roman times. Their language remains largely undeciphered, though
    we have many inscriptions from tombs and temples.
    """
    
    # Generate with specific difficulty
    from docs_to_eval.core.agentic.models import PipelineConfig, DifficultyLevel
    
    config = PipelineConfig(
        difficulty=DifficultyLevel.EXPERT,
        num_questions=30,
        min_validation_score=0.8
    )
    
    report = await demo.demonstrate_agentic_pipeline(
        corpus,
        EvaluationType.DOMAIN_KNOWLEDGE,
        num_questions=30
    )
    
    return report

asyncio.run(create_domain_benchmark())
```

### Example 2: Multiple Evaluation Types

```python
from docs_to_eval.core.agentic.lm_eval_utils import generate_and_export_benchmark

# Create benchmarks for different types
for eval_type in [EvaluationType.FACTUAL_QA, 
                  EvaluationType.DOMAIN_KNOWLEDGE,
                  EvaluationType.READING_COMPREHENSION]:
    
    report = generate_and_export_benchmark(
        corpus_text=corpus,
        eval_type=eval_type,
        task_name=f"benchmark_{eval_type.value}",
        num_questions=50,
        output_dir=f"./exports/{eval_type.value}"
    )
    
    print(f"✅ Created {eval_type.value} benchmark")
```

### Example 3: Custom Validation

```python
from docs_to_eval.core.agentic.validation import ComprehensiveValidator

# Generate items
items = await generator.generate_benchmark_async(corpus, num_questions=100)

# Validate with custom threshold
validator = ComprehensiveValidator(min_quality_score=0.75)
filtered_items, report = await validator.validate_and_filter(
    items,
    strict_mode=True
)

print(f"Retained {len(filtered_items)}/{len(items)} items")
print(f"Average quality: {report['quality_assessment']['avg_quality']:.3f}")

# Export only high-quality items
export_report = export_agentic_benchmark_to_lm_eval(
    filtered_items,
    "high_quality_benchmark",
    "./exports"
)
```

## File Format Specifications

### YAML Task Configuration

```yaml
# Example: ml_benchmark_domain_knowledge_free_text.yaml

task: ml_benchmark_domain_knowledge_free_text
dataset_name: ml_benchmark_domain_knowledge_free_text
dataset_path: ml_benchmark_domain_knowledge_free_text_dataset.jsonl
output_type: generate_until
test_split: test

doc_to_text: "Context: {{context}}\n\nQuestion: {{question}}\n\nAnswer:"
doc_to_target: "{{answer}}"

metric_list:
  - metric: bleu
    aggregation: mean
    higher_is_better: true
  - metric: rouge
    aggregation: mean
    higher_is_better: true

generation_kwargs:
  temperature: 0.0
  do_sample: false
  until: ["\n\n"]
  max_gen_toks: 256

should_decontaminate: true
doc_to_decontamination_query: "{{question}}"

description: "Agentic benchmark for domain_knowledge evaluation (free_text answers)"

metadata:
  version: "2.0"
  source: "agentic_benchmark_factory"
  eval_type: "domain_knowledge"
  answer_type: "free_text"
  items_count: 42
  deterministic_ratio: 0.35
```

### JSONL Dataset Format

```jsonl
{"question":"What is machine learning?","answer":"A method of data analysis that automates analytical model building","context":"Machine learning is a method...","metadata":{"difficulty":"intermediate","deterministic":false,"eval_type":"domain_knowledge","answer_type":"free_text","validation_score":0.85}}
{"question":"Explain neural networks","answer":"Neural networks are computational models inspired by biological neural networks...","context":"Deep learning uses neural networks...","metadata":{"difficulty":"hard","deterministic":false,"eval_type":"domain_knowledge","answer_type":"free_text","validation_score":0.92}}
```

**Critical Requirements:**
- ✅ Each line is a complete JSON object
- ✅ Lines separated by actual newlines (`\n`), NOT escaped (`\\n`)
- ✅ UTF-8 encoding
- ✅ No trailing newline after last object
- ✅ Required fields: `question`, `answer`
- ✅ Optional fields: `context`, `choices`, `metadata`, `reasoning_chain`

### Multiple Choice Handling

For multiple choice questions:

```json
{
  "question": "What is the primary advantage of deep learning?",
  "answer": "Automatic feature learning",
  "choices": [
    "Automatic feature learning",
    "Faster computation",
    "Less data required",
    "Simpler algorithms"
  ],
  "answer_index": 0
}
```

The YAML config uses:
```yaml
doc_to_choice: "{{choices}}"
doc_to_target: "{{answer_index}}"  # Not {{answer}}
```

## Troubleshooting

### Common Issues

#### 1. JSONL Parse Errors

**Symptom:** `JSONDecodeError` when loading dataset

**Causes:**
- Escaped newlines (`\\n`) instead of actual newlines
- Invalid JSON in a line
- Missing closing braces

**Fix:**
```python
# Verify JSONL format
import json

with open('dataset.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error on line {i}: {e}")
            print(f"Content: {line[:100]}")
```

#### 2. Template Rendering Fails

**Symptom:** Jinja2 error or incorrect model input

**Cause:** Escaped newlines in templates

**Fix:** Templates should use actual newlines:
```yaml
# ✅ Correct
doc_to_text: "Question: {{question}}\n\nAnswer:"

# ❌ Wrong
doc_to_text: "Question: {{question}}\\n\\nAnswer:"
```

#### 3. Validation Fails

**Symptom:** `validate_lm_eval_export()` returns errors

**Common errors:**
```python
{
    'valid': False,
    'errors': [
        'Task xyz.yaml: Missing required fields: [doc_to_target]',
        'Task xyz.yaml: Dataset file not found'
    ]
}
```

**Fix:** Ensure all required fields are present in YAML

#### 4. lm-eval Can't Find Tasks

**Symptom:** `TaskNotFoundError` when running lm_eval

**Solutions:**
1. Copy files to lm-eval tasks directory:
   ```bash
   cp -r ./exports/ml_benchmark/* ~/.cache/lm-evaluation-harness/tasks/
   ```

2. Or use `--include_path`:
   ```bash
   lm_eval --model hf \
           --tasks ml_benchmark \
           --include_path ./exports \
           ...
   ```

## Advanced Features

### Custom Metrics

You can specify custom metrics in the export:

```python
from docs_to_eval.core.agentic.lm_eval_transform import HarnessMetricConfig

custom_metrics = [
    HarnessMetricConfig("custom_score", "mean", True),
    HarnessMetricConfig("domain_accuracy", "mean", True)
]

# Modify task config after transformation
transformer = HarnessTransformer()
result = transformer.transform_to_harness_format(items, "task_name")

for task_config in result['tasks'].values():
    task_config.metric_list.extend(custom_metrics)
```

### Filtering by Difficulty

```python
from docs_to_eval.core.agentic.models import DifficultyLevel

# Filter items by difficulty
expert_items = [
    item for item in items 
    if item.metadata.difficulty == DifficultyLevel.EXPERT
]

# Export only expert-level questions
export_agentic_benchmark_to_lm_eval(
    expert_items,
    "expert_benchmark",
    "./exports"
)
```

### Deterministic/Non-Deterministic Split

```python
# Separate by deterministic type
deterministic_items = [i for i in items if i.metadata.deterministic]
non_deterministic_items = [i for i in items if not i.metadata.deterministic]

# Export separately
export_agentic_benchmark_to_lm_eval(
    deterministic_items,
    "deterministic_benchmark",
    "./exports/deterministic"
)

export_agentic_benchmark_to_lm_eval(
    non_deterministic_items,
    "creative_benchmark",
    "./exports/creative"
)
```

## CLI Usage

### Quick Export Demo

```bash
# Run the demo export
python -m docs_to_eval.core.agentic.lm_eval_utils demo
```

### Export Existing Benchmark

```bash
# From JSON file
python -m docs_to_eval.core.agentic.lm_eval_utils export \
    --input my_benchmark.json \
    --name my_task \
    --output ./lm_eval_tasks

# From corpus text
python -m docs_to_eval.core.agentic.lm_eval_utils export \
    --input corpus.txt \
    --name ml_benchmark \
    --eval-type domain_knowledge \
    --num-questions 100 \
    --output ./lm_eval_tasks
```

### Validate Export

```bash
python -m docs_to_eval.core.agentic.lm_eval_utils validate \
    --export-dir ./lm_eval_tasks
```

## Best Practices

### 1. Quality Over Quantity

```python
# Generate with high validation threshold
config = PipelineConfig(
    min_validation_score=0.8,  # High threshold
    oversample_factor=3.0,     # Generate 3x, keep best
    enforce_deterministic_split=True
)
```

### 2. Balanced Difficulty Distribution

```python
# Check difficulty distribution before export
from collections import Counter

difficulties = Counter(
    item.metadata.difficulty.value 
    for item in items
)

print("Difficulty distribution:", difficulties)

# Aim for balanced distribution
target = {
    'basic': 0.20,
    'intermediate': 0.40,
    'hard': 0.30,
    'expert': 0.10
}
```

### 3. Validate Before Export

```python
# Always validate before exporting
validator = ComprehensiveValidator(min_quality_score=0.7)
validation_report = await validator.validate_benchmark_batch(items)

if validation_report['overall_pass_rate'] < 0.8:
    print("⚠️  Many items failing validation. Review before export.")
else:
    export_agentic_benchmark_to_lm_eval(items, task_name, output_dir)
```

### 4. Document Your Benchmark

Include comprehensive metadata:

```python
# Add custom metadata to task config
task_config.metadata.update({
    'domain': 'machine_learning',
    'source_corpus': 'ML textbook chapters 1-5',
    'generation_date': datetime.now().isoformat(),
    'quality_threshold': 0.8,
    'human_review': True
})
```

## FAQ

**Q: Can I mix multiple evaluation types in one task?**

A: No. The exporter automatically groups items by `(eval_type, answer_type)` pairs and creates separate tasks for each group. This ensures proper metric selection and template generation.

**Q: How do I add custom stop tokens?**

A: Modify the `generation_kwargs` in the YAML file after export, or customize `AnswerTypeMapper.get_generation_kwargs()`.

**Q: What's the maximum number of questions per task?**

A: No hard limit, but for practical purposes, 100-500 questions per task is recommended. Larger benchmarks should be split into multiple tasks.

**Q: Can I use this with private LLMs?**

A: Yes! Just provide your own `llm_pool` when creating the `AgenticBenchmarkGenerator`:

```python
from my_llm import MyPrivateLLM

llm = MyPrivateLLM()
llm_pool = {
    'retriever': llm,
    'creator': llm,
    'adversary': llm,
    'refiner': llm
}

generator = AgenticBenchmarkGenerator(
    eval_type=eval_type,
    llm_pool=llm_pool
)
```

## Support

For issues or questions:
- Check the [troubleshooting section](#troubleshooting)
- Review example code in `docs_to_eval/core/agentic/demo.py`
- Consult lm-evaluation-harness docs: https://github.com/EleutherAI/lm-evaluation-harness

## Version History

- **v2.0** (Current): Complete rewrite with proper data transformation, fixed JSONL/YAML formatting
- **v1.0**: Initial integration (deprecated due to formatting bugs)
