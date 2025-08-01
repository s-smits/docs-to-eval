# Agentic Benchmark Generation System

## Overview

The Agentic Benchmark Generation System transforms "docs-to-eval" into an intelligent benchmark factory that automatically produces HARD, well-formatted evaluation tasks while preserving the deterministic vs non-deterministic split.

## Architecture

### 1. Specialized Agents

- **ConceptMiner**: Extracts key concepts using hybrid RAG approach
- **QuestionWriter**: Creates initial questions with chain-of-thought reasoning  
- **Adversary**: Enhances difficulty with adversarial techniques
- **Refiner**: Enforces style guide and formatting requirements
- **Validator**: Quality control and deterministic guardrails

### 2. Task Protocol

Strict Pydantic models ensure validation at every step:

- `BenchmarkDraft`: Raw ideas from QuestionWriter
- `BenchmarkCandidate`: After adversarial enhancement  
- `EnhancedBenchmarkItem`: Final output with full metadata

### 3. Pipeline Workflow

```
Corpus Text → ConceptMiner → QuestionWriter → Adversary → Refiner → Validator → BenchmarkItem
```

Each step includes retry logic, quality checks, and full provenance tracking.

## Key Features

### 🎯 Intelligent Difficulty Enhancement

- Multi-hop reasoning requirements
- Plausible distractors for multiple choice
- Adversarial obfuscation while preserving groundedness
- Data-to-code transformations

### 🔒 Deterministic Guardrails

- Automatic classification of answer types
- Validation that deterministic items pass exact-match verification
- Enforced split between deterministic and non-deterministic evaluation

### ⚡ Async Pipeline Architecture

- Parallel processing of concept batches
- Rate limiting and error handling
- Comprehensive metrics and monitoring

### 📊 Quality Control

- Multi-dimensional quality assessment
- Automatic filtering of low-quality items
- Detailed validation reports with actionable recommendations

## Usage

### Basic Usage

```python
from docs_to_eval.core.agentic import AgenticBenchmarkGenerator
from docs_to_eval.core.evaluation import EvaluationType

# Create generator
generator = AgenticBenchmarkGenerator(
    eval_type=EvaluationType.DOMAIN_KNOWLEDGE,
    llm_pool=your_llm_pool
)

# Generate benchmark
items = generator.generate_benchmark(corpus_text, num_questions=50)
```

### Factory Integration

```python
from docs_to_eval.core.benchmarks import BenchmarkGeneratorFactory

# Use agentic generation through factory
generator = BenchmarkGeneratorFactory.create_generator(
    EvaluationType.DOMAIN_KNOWLEDGE, 
    use_agentic=True,
    llm_pool=llm_pool
)
```

### Advanced Pipeline Control

```python
from docs_to_eval.core.agentic import AgenticBenchmarkOrchestrator, PipelineConfig

config = PipelineConfig(
    difficulty=DifficultyLevel.HARD,
    min_validation_score=0.7,
    parallel_batch_size=5
)

orchestrator = AgenticBenchmarkOrchestrator(llm_pool, config)
items = await orchestrator.generate(corpus_text, eval_type, num_questions=100)
```

## Configuration

### Pipeline Configuration

```python
PipelineConfig(
    difficulty=DifficultyLevel.HARD,           # Target difficulty
    num_questions=50,                          # Number of questions
    oversample_factor=2.5,                     # Generate extra for selection
    parallel_batch_size=5,                     # Concurrent processing
    min_validation_score=0.6,                  # Quality threshold
    enforce_deterministic_split=True           # Strict type validation
)
```

### Agent Configuration

```python
AgentConfig(
    temperature=0.7,                           # LLM creativity
    max_tokens=512,                            # Response length
    retry_attempts=3,                          # Error recovery
    validation_threshold=0.6                   # Acceptance criteria
)
```

## Quality Metrics

The system tracks comprehensive metrics:

- **Generation Speed**: Items per second, agent processing time
- **Quality Scores**: Multi-dimensional assessment across clarity, difficulty, relevance
- **Validation Rates**: Deterministic consistency, overall pass rates
- **Pipeline Health**: Success rates, retry cycles, error tracking

## Deterministic vs Non-Deterministic

### Automatic Classification

The system automatically classifies questions based on:

- Answer type (numeric, code, multiple choice → deterministic)
- Evaluation type (mathematical, factual QA → typically deterministic)  
- Answer characteristics (short factual answers → deterministic)

### Validation Guardrails

- Items marked deterministic MUST pass exact-match verification
- Pipeline aborts if deterministic items fail self-checks
- Comprehensive reporting on classification accuracy

## Running the Demo

```bash
cd docs_to_eval/core/agentic
python demo.py
```

The demo showcases:
- Full pipeline execution across multiple evaluation types
- Quality validation and filtering
- Factory integration
- Performance metrics and recommendations

## Integration Points

### Existing Framework Compatibility

The agentic system integrates seamlessly with:

- `BenchmarkGeneratorFactory` - factory pattern integration
- `VerificationOrchestrator` - quality validation
- `EvaluationFramework` - downstream evaluation
- Standard `BenchmarkItem` format - backward compatibility

### LLM Interface Support

Compatible with any `BaseLLMInterface` implementation:

- OpenAI API
- Anthropic Claude
- Mock interfaces for testing
- Custom LLM providers

## Performance Characteristics

### Scaling

- **Small batches** (1-10 questions): ~5-15 seconds
- **Medium batches** (50 questions): ~30-60 seconds  
- **Large batches** (200+ questions): ~2-5 minutes

### Resource Usage

- Concurrent LLM calls: configurable batch size
- Memory: ~10MB per 100 items generated
- Network: depends on LLM provider rate limits

## Best Practices

1. **Start with pilot batches** to tune configuration
2. **Monitor quality metrics** and adjust thresholds
3. **Use appropriate difficulty levels** for your domain
4. **Implement proper LLM rate limiting**
5. **Cache concept extraction** for large corpora

## Troubleshooting

### Common Issues

- **Low quality scores**: Reduce `min_validation_score` or improve corpus quality
- **Deterministic failures**: Check answer format and eval type consistency  
- **Timeout errors**: Reduce `parallel_batch_size` or increase timeout
- **Rate limiting**: Adjust LLM pool configuration

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('docs_to_eval.core.agentic').setLevel(logging.DEBUG)
```

## Future Enhancements

- Vector similarity for concept mining
- Advanced adversarial techniques
- Multi-modal question generation
- Automated difficulty calibration
- Real-time quality monitoring dashboard