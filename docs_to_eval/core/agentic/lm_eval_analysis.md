# LM-Evaluation-Harness Integration Analysis

## Overview
Detailed analysis of requirements for integrating agentic benchmark items with EleutherAI's lm-evaluation-harness framework.

## Core Architecture Requirements

### 1. Task YAML Configuration Structure

```yaml
# Required fields
task: task_name                    # Unique task identifier
dataset_name: dataset_name         # Name for the dataset
dataset_path: path/to/data.jsonl   # Path to JSONL dataset file
output_type: "multiple_choice"     # Output type (see section below)

# Template fields (Jinja2 format)
doc_to_text: "{{question}}"        # How to format input to model
doc_to_target: "{{answer}}"        # How to extract target answer
doc_to_choice: "{{choices}}"       # For multiple choice (optional)

# Metrics configuration
metric_list:
  - metric: "acc"
    aggregation: "mean"
    higher_is_better: true

# Optional fields
description: "Task description"
training_split: null
validation_split: null
test_split: "test"
should_decontaminate: true
doc_to_decontamination_query: "{{question}}"
```

### 2. Output Types and Their Requirements

#### A. Multiple Choice (`multiple_choice`)
- **Use case**: Questions with fixed set of answer options
- **Model interaction**: Model computes loglikelihood for each choice
- **Required fields**: `doc_to_choice`
- **Metrics**: `acc`, `acc_norm`
- **Dataset format**: Must include `choices` array

#### B. Generative (`generate_until`)
- **Use case**: Free-form text generation, numerical answers, code
- **Model interaction**: Model generates text until stopping criteria
- **Required fields**: `generation_kwargs`
- **Metrics**: `exact_match`, `bleu`, `rouge`, `f1`
- **Special configs**: `until` tokens, `max_gen_toks`

#### C. Loglikelihood (`loglikelihood`)
- **Use case**: Perplexity measurement, language modeling
- **Model interaction**: Compute likelihood of target sequence
- **Metrics**: `perplexity`, `word_perplexity`

### 3. Dataset Format (JSONL)

Each line is a JSON object with required fields:
```json
{
  "question": "What is machine learning?",
  "answer": "A method of data analysis...",
  "context": "Optional context text",
  "choices": ["Option A", "Option B", "Option C", "Option D"],
  "metadata": {
    "difficulty": "hard",
    "source": "agentic_factory"
  }
}
```

### 4. Metric Configuration Mapping

| Answer Type | Primary Metrics | Aggregation | Higher Better |
|-------------|----------------|-------------|---------------|
| Multiple Choice | `acc`, `acc_norm` | `mean` | `true` |
| Numeric Exact | `exact_match` | `mean` | `true` |
| String Exact | `exact_match`, `f1` | `mean` | `true` |
| Code | `exact_match`, `bleu` | `mean` | `true` |
| Free Text | `bleu`, `rouge` | `mean` | `true` |
| Boolean | `exact_match` | `mean` | `true` |

### 5. Template System Requirements

#### Jinja2 Template Variables
Templates use double braces `{{variable}}` format:
- `{{question}}` - The question text
- `{{answer}}` - The target answer
- `{{context}}` - Optional context
- `{{choices}}` - Array of choices (for MC)
- `{{choices[0]}}` - Individual choice access

#### Common Template Patterns

**Multiple Choice**:
```
Question: {{question}}
A. {{choices[0]}}
B. {{choices[1]}}
C. {{choices[2]}}
D. {{choices[3]}}
Answer:
```

**Context + Question**:
```
Context: {{context}}

Question: {{question}}

Answer:
```

**Code Generation**:
```
{{question}}

Provide your code solution:
```

### 6. Generation Configuration

For `generate_until` tasks:
```yaml
generation_kwargs:
  until: ["\n", ".", "?"]           # Stop tokens
  max_gen_toks: 256                 # Max tokens to generate
  temperature: 0.0                  # Sampling temperature
  do_sample: false                  # Deterministic generation
```

### 7. Special Configurations

#### Decontamination
```yaml
should_decontaminate: true
doc_to_decontamination_query: "{{question}}"
```

#### Task Groups
```yaml
group: "task_group_name"
group_alias: "short_name"
```

#### Custom Processing
```yaml
process_results: "!function utils.custom_processor"
filter_list:
  - name: "remove_whitespace"
    filter:
      - function: "remove_whitespace"
      - function: "take_first"
        take_first: 1
```

## Integration Requirements Analysis

### 1. Data Transformation Needs

#### From Agentic Items to Harness Format:
- **EnhancedBenchmarkItem** → **JSONL Dataset**
- **AnswerType enum** → **output_type + metrics**
- **Context/Options** → **Template variables**
- **Metadata** → **Additional dataset fields**

#### Key Mapping Challenges:
1. **Answer Type Detection**: Need robust mapping from AnswerType to output_type
2. **Template Generation**: Dynamic template creation based on item structure
3. **Metric Selection**: Automatic metric assignment based on answer type
4. **Option Handling**: Convert between string answers and choice indices

### 2. File Generation Requirements

#### YAML Task Files
- One YAML per task type/answer type combination
- Proper Jinja2 template formatting
- Correct metric configurations
- Metadata preservation

#### JSONL Dataset Files
- One line per benchmark item
- Proper JSON escaping
- Choice index conversion for MC
- Metadata embedding

#### Supporting Files
- `utils.py` for custom processing functions
- `README.md` with usage instructions
- `__init__.py` for task registration

### 3. Validation Requirements

#### Task Validation
- YAML syntax validation
- Required field presence
- Template variable consistency
- Metric compatibility with output type

#### Dataset Validation
- JSONL format correctness
- Required field presence in each item
- Choice array consistency for MC
- Answer format validation

### 4. Testing Requirements

#### Unit Tests
- Data transformation accuracy
- Template generation correctness
- Metric configuration validity
- File format compliance

#### Integration Tests
- Full export pipeline
- lm-eval harness compatibility
- Task execution verification
- Results consistency

### 5. CLI/API Requirements

#### Export Functions
```python
# High-level export
export_to_lm_eval(items, task_name, output_dir)

# Granular control
create_task_config(items, task_name)
create_dataset_jsonl(items, output_file)
validate_export(config_file, dataset_file)
```

#### CLI Commands
```bash
# Export benchmark
agentic-export --input benchmark.json --output ./lm_eval_tasks --name my_task

# Validate export
agentic-validate --task-dir ./lm_eval_tasks --task my_task

# Test with harness
agentic-test --task my_task --model gpt2
```

## Implementation Phases

### Phase 1: Core Data Transformation
1. AnswerType → output_type mapping
2. Basic template generation
3. JSONL dataset creation
4. Metric configuration mapping

### Phase 2: Advanced Features
1. Complex template system
2. Custom processing functions
3. Task grouping and organization
4. Decontamination support

### Phase 3: Validation & Testing
1. Format validation
2. lm-eval integration testing
3. Performance benchmarking
4. Error handling and recovery

### Phase 4: User Experience
1. CLI interface
2. Documentation generation
3. Example tasks and tutorials
4. Migration utilities

## Critical Success Factors

1. **Format Compliance**: Must perfectly match lm-eval harness expectations
2. **Template Accuracy**: Jinja2 templates must render correctly
3. **Metric Validity**: Metrics must be appropriate for task types
4. **Performance**: Export should be fast for large datasets
5. **Usability**: Simple API for common use cases
6. **Maintainability**: Code should handle format changes gracefully

## Risk Assessment

### High Risk
- Template generation complexity
- Metric configuration errors
- Format compatibility issues

### Medium Risk
- Performance with large datasets
- Custom processing function integration
- CLI interface complexity

### Low Risk
- Basic JSONL generation
- Simple YAML creation
- File organization

## Next Steps

1. Implement core data transformation (Phase 1)
2. Create template generation system
3. Build validation framework
4. Test with real lm-eval harness
5. Add CLI interface
6. Create comprehensive documentation