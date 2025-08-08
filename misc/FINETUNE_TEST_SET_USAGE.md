# Finetune Test Set Feature

The finetune test set feature automatically creates a train/test split of your generated questions to support finetuning workflows. This ensures you have a held-out test set to evaluate your finetuned model's performance.

## Overview

- **Purpose**: Create separate training and testing sets from generated questions
- **Default**: 20% of questions reserved for testing, 80% for training
- **Reproducible**: Uses configurable random seed for consistent splits
- **Example**: 50 questions → 40 for training + 10 for testing

## Configuration Options

| Parameter | Default | Range | Description |
|-----------|---------|--------|-------------|
| `finetune_test_set_enabled` | `true` | boolean | Enable/disable test set creation |
| `finetune_test_set_percentage` | `0.2` | 0.1-0.5 | Percentage for test set (20% = 0.2) |
| `finetune_random_seed` | `42` | ≥0 | Random seed for reproducible splits |

## API Usage

### 1. Creating Evaluation with Finetune Test Set

```json
POST /api/v1/evaluation/start
{
    "corpus_text": "Your training corpus...",
    "num_questions": 50,
    "use_agentic": true,
    "finetune_test_set_enabled": true,
    "finetune_test_set_percentage": 0.2,
    "finetune_random_seed": 42
}
```

### 2. Checking Results

The evaluation results will include a `finetune_test_set` section:

```json
{
    "run_id": "abc-123",
    "finetune_test_set": {
        "enabled": true,
        "total_questions": 50,
        "train_questions": 40,
        "test_questions": 10,
        "test_percentage": 0.2,
        "random_seed": 42,
        "split_timestamp": "2025-08-05T12:00:00Z"
    }
}
```

### 3. Accessing Train/Test Sets

```bash
# Get finetune test set overview
GET /api/v1/evaluation/{run_id}/finetune-test-set

# Get training questions (for finetuning)
GET /api/v1/evaluation/{run_id}/finetune-test-set/train

# Get test questions (for evaluation)
GET /api/v1/evaluation/{run_id}/finetune-test-set/test
```

## Workflow Example

### Step 1: Generate Questions with Test Set
```python
from docs_to_eval.core.evaluation import create_benchmark_with_finetune_set, BenchmarkConfig

# Configure benchmark with finetune test set
config = BenchmarkConfig(
    eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
    num_questions=50,
    finetune_test_set_enabled=True,
    finetune_test_set_percentage=0.2,  # 20% for testing
    finetune_random_seed=42
)

# Create benchmark with automatic train/test split
benchmark = create_benchmark_with_finetune_set(
    questions=generated_questions,
    config=config
)
```

### Step 2: Access Training Questions
```python
# Get questions for finetuning (80% of total)
train_questions = benchmark.get_train_questions()
print(f"Training on {len(train_questions)} questions")

# Use these for finetuning your model
for question in train_questions:
    # Format for your finetuning pipeline
    training_example = {
        "input": question["question"],
        "output": question["answer"]
    }
```

### Step 3: Access Test Questions
```python
# Get held-out questions for evaluation (20% of total)
test_questions = benchmark.get_test_questions()
print(f"Testing on {len(test_questions)} questions")

# Use these to evaluate your finetuned model
for question in test_questions:
    # Evaluate finetuned model performance
    model_response = your_finetuned_model(question["question"])
    accuracy = evaluate_response(model_response, question["answer"])
```

## Key Benefits

### 1. **Prevents Data Leakage**
- Test questions are completely separate from training
- Ensures valid evaluation of finetuned model performance

### 2. **Reproducible Splits**
- Same random seed produces identical train/test splits
- Critical for consistent model evaluation and comparison

### 3. **Configurable Split Ratios**
- Adjust test percentage based on your needs
- Minimum 10% test set ensures meaningful evaluation
- Maximum 50% test set prevents insufficient training data

### 4. **Automatic Integration**
- Works seamlessly with existing evaluation pipeline
- No additional setup required - just enable the feature

## Use Cases

### Scenario 1: Model Finetuning
- **Total Questions**: 100
- **Test Percentage**: 20% (20 questions)
- **Result**: 80 questions for finetuning, 20 for evaluation

### Scenario 2: Small Dataset
- **Total Questions**: 20
- **Test Percentage**: 30% (6 questions)
- **Result**: 14 questions for finetuning, 6 for evaluation

### Scenario 3: Large Dataset
- **Total Questions**: 500
- **Test Percentage**: 15% (75 questions)
- **Result**: 425 questions for finetuning, 75 for evaluation

## Integration Notes

### Current Evaluation vs Finetune Test Set

**Important**: The finetune test set is separate from the current evaluation system:

- **Current Evaluation**: Tests how well models answer questions initially
- **Finetune Test Set**: Reserved questions for testing finetuned models later

### Workflow Integration

```
1. Generate 50 questions
2. Create finetune split: 40 train + 10 test
3. Current evaluation: Test model on all 50 questions (normal evaluation)
4. Later: Finetune model on 40 train questions
5. Finally: Evaluate finetuned model on 10 test questions
```

This ensures you can:
1. Assess initial model performance (current evaluation)
2. Train an improved model (using train set)
3. Validate improvement (using held-out test set)

## Error Handling

The system gracefully handles edge cases:

- **Insufficient Questions**: Ensures at least 1 test question
- **High Test Percentage**: Automatically caps at 50% of total questions
- **Disabled Feature**: Returns all questions as training set when disabled
- **Failed Generation**: Provides empty sets with proper error messages

## Best Practices

1. **Use Consistent Seeds**: Same seed = same split for reproducibility
2. **Appropriate Test Size**: 20% is recommended for most use cases
3. **Document Your Splits**: Record the random seed used for future reference
4. **Validate Splits**: Check that train + test = total questions
5. **Keep Test Set Secret**: Never train on test questions to avoid overfitting