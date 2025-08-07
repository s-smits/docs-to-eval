# Streamlined Agentic Benchmark Generation System

## Overview

The streamlined agentic system is a production-ready, simplified version of the benchmark generation pipeline. It reduces complexity from 5 agents to 3 essential agents while maintaining high quality and diversity in question generation.

## Architecture

### 3-Agent Pipeline

```
ConceptExtractor → QuestionGenerator → QualityValidator
```

1. **ConceptExtractor**: Identifies key concepts from the corpus
2. **QuestionGenerator**: Creates diverse, high-quality questions  
3. **QualityValidator**: Ensures question quality and standards

## Key Features

### 🎯 Question Diversity

The system automatically rotates through 6 question types to ensure variety:

- **Factual**: Direct knowledge questions (What is X?)
- **Analytical**: Reasoning questions (Why is X important?)
- **Comparative**: Comparison questions (How does X differ from Y?)
- **Application**: Practical questions (How would you apply X?)
- **Evaluative**: Assessment questions (Evaluate the effectiveness of X)
- **Synthesis**: Integration questions (How do X and Y work together?)

### ⚡ Efficient Processing

- Parallel batch processing for speed
- Smart oversampling (generates 50% more, selects best)
- Graceful fallback mechanisms
- Progress tracking throughout

### 🔧 Flexible Configuration

```python
config = {
    'min_quality_score': 0.5,      # Minimum quality threshold
    'oversample_factor': 1.5,       # Generate extra for filtering
    'parallel_batch_size': 5,       # Process in batches
    'max_retries': 2,              # Retry failed generations
    'enable_diversity': True        # Ensure question variety
}
```

## Usage

### Basic Usage

```python
from docs_to_eval.core.agentic import StreamlinedOrchestrator
from docs_to_eval.core.agentic.models import DifficultyLevel

# Initialize orchestrator (with or without LLM)
orchestrator = StreamlinedOrchestrator(llm_interface)

# Generate questions
questions = await orchestrator.generate(
    corpus_text="Your document text here...",
    num_questions=20,
    eval_type="domain_knowledge",
    difficulty=DifficultyLevel.INTERMEDIATE
)
```

### Integration with UI/API

The system is fully integrated with the UI through `routes.py`:

```python
# In routes.py
questions = await generate_evaluation_questions(
    corpus_text, 
    num_questions, 
    eval_type,
    llm_config,
    tracker
)
```

## Comparison with Legacy System

| Aspect | Legacy (5 Agents) | Streamlined (3 Agents) |
|--------|------------------|------------------------|
| **Complexity** | High - 5 separate agents | Low - 3 focused agents |
| **Processing Time** | Slower due to many steps | Faster with parallel processing |
| **Question Diversity** | Complex adversarial techniques | Structured rotation through types |
| **Maintenance** | Difficult - many moving parts | Easy - clear responsibilities |
| **Fallback Quality** | Poor - random templates | Better - structured fallbacks |
| **Production Ready** | No - too complex | Yes - battle-tested |

## Agent Details

### ConceptExtractor

**Purpose**: Extract key concepts from corpus for question generation

**Capabilities**:
- LLM-based concept extraction (when available)
- Keyword-based fallback extraction
- Supporting snippet generation
- Importance scoring

### QuestionGenerator

**Purpose**: Generate diverse, high-quality questions

**Capabilities**:
- 6 different question types for diversity
- Template-based and LLM-based generation
- Automatic answer type detection
- Multiple choice option generation
- Difficulty-appropriate questions

### QualityValidator

**Purpose**: Ensure question quality meets standards

**Capabilities**:
- Multi-factor quality scoring
- LLM-based validation (when available)
- Heuristic-based checks
- Context relevance validation
- Answer completeness verification

## Quality Assurance

The system includes multiple quality checks:

1. **Question Clarity**: Proper formatting, punctuation, length
2. **Answer Quality**: Completeness, correctness, appropriate type
3. **Context Relevance**: Concept appears in context
4. **Difficulty Appropriateness**: Matches target difficulty
5. **LLM Validation**: Additional quality check when LLM available

## Fallback Mechanisms

The system gracefully degrades when LLM is unavailable:

1. **Concept Extraction**: Falls back to keyword extraction
2. **Question Generation**: Uses intelligent templates
3. **Quality Validation**: Uses heuristic scoring
4. **Final Fallback**: Simple but structured questions

## Performance

Typical performance metrics:

- **Processing Time**: ~0.5-2s per question (with LLM)
- **Acceptance Rate**: 70-90% (quality filtered)
- **Diversity Score**: 5-6 unique types per 12 questions
- **Quality Score**: Average 0.6-0.8 (on 0-1 scale)

## Configuration Options

### Difficulty Levels

- `BASIC`: Simple, straightforward questions
- `INTERMEDIATE`: Moderate complexity, some reasoning required
- `HARD`: Challenging, deep understanding needed  
- `EXPERT`: Complex, expert-level knowledge required

### Evaluation Types

- `factual_qa`: Direct factual questions
- `domain_knowledge`: Domain-specific understanding
- `mathematical`: Calculation and problem-solving
- `code_generation`: Programming questions
- `multiple_choice`: MCQ format
- `classification`: Categorization tasks

## Best Practices

1. **Corpus Size**: Provide at least 500 words for best results
2. **Question Count**: 10-50 questions per run is optimal
3. **LLM Model**: Use GPT-4 or Claude for best quality
4. **Difficulty**: Match difficulty to your target audience
5. **Review**: Always review generated questions before use

## Troubleshooting

### Low Quality Questions

- Increase `min_quality_score` in config
- Provide more detailed corpus text
- Use a better LLM model

### Lack of Diversity

- Ensure `enable_diversity` is True
- Generate more questions (diversity improves with volume)
- Check that corpus covers multiple topics

### Slow Generation

- Reduce `parallel_batch_size` if rate limited
- Use a faster LLM model
- Reduce `oversample_factor`

## Future Improvements

- [ ] Add caching for repeated concepts
- [ ] Implement semantic deduplication
- [ ] Add domain-specific templates
- [ ] Include confidence scores
- [ ] Support for multi-modal questions

## Migration from Legacy

To migrate from the legacy 5-agent system:

1. Replace `AgenticBenchmarkOrchestrator` with `StreamlinedOrchestrator`
2. Replace `generate_agentic_questions_*` with `generate_evaluation_questions`
3. Update imports to use streamlined agents
4. Adjust configuration for simpler options
5. Test with your existing corpora

## Support

For issues or questions:
- Check the logs for detailed error messages
- Verify LLM connectivity and API keys
- Review the fallback questions for clues
- Ensure corpus text is properly formatted
