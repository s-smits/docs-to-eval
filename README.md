# docs-to-eval - Automated LLM Evaluation System

🤖 **docs-to-eval** is a comprehensive Python system that automatically generates domain-specific benchmarks from text corpora and evaluates LLMs using appropriate methodologies. It intelligently determines whether to use deterministic (exact match) or non-deterministic (similarity-based) evaluation methods based on the content type.

## 🌟 Key Features

### 🧠 Intelligent Evaluation Type Classification
- **LLM-powered analysis** of text corpora to determine optimal evaluation methodology
- **10+ evaluation types** including mathematical, code generation, factual QA, creative writing, etc.
- **Automatic differentiation** between deterministic vs non-deterministic evaluation needs

### 🏗️ Advanced Benchmark Generation
- **Agentic question generation** using multiple strategies (conceptual, application, comparison, analytical, synthesis)
- **Quality scoring and improvement** of generated questions
- **Domain-specific question tailoring** based on corpus content analysis

### 🔍 Comprehensive Verification Systems
- **Deterministic verification**: exact match, numerical matching, code execution, multiple choice
- **Mathematical verification**: Enhanced math expression parsing with math-verify library
- **Non-deterministic verification**: token overlap, n-gram similarity, ROUGE-L, semantic similarity
- **LLM-judge evaluation** for subjective content

### 📊 Rich Reporting & Analytics
- **Executive summaries** with performance categorization
- **Detailed analysis** including score distributions and error patterns
- **Benchmark quality assessment** and improvement recommendations
- **Comparative analysis** against baseline performance

## 🏗️ System Architecture

```
📁 docs-to-eval/
├── core_evaluation.py          # Core framework and data structures
├── eval_classifier.py          # LLM-based evaluation type classification
├── benchmark_generators.py     # Benchmark generation for different types
├── agentic_generator.py        # Advanced agentic question generation
├── verification_systems.py     # Deterministic & non-deterministic verification
├── mock_llm_interface.py       # Mock LLM for testing pipeline
├── reporting_system.py         # Comprehensive reporting and visualization
├── main_interface.py           # Interactive user interface
└── demo.py                     # Quick demonstration script
```

## 🚀 Quick Start

### Basic Usage

```python
python main_interface.py
```

Follow the interactive prompts to:
1. **Input your corpus** (text entry, file upload, or sample)
2. **Configure parameters** (number of questions, evaluation type)
3. **Generate benchmark** (standard or advanced agentic generation)
4. **Run evaluation** (with mock LLM)
5. **View results** (comprehensive analysis and recommendations)

### Programmatic Usage

```python
from main_interface import docs-to-evalSystem

# Initialize system
system = docs-to-evalSystem()

# Or use individual components
from eval_classifier import classify_and_configure
from benchmark_generators import generate_domain_benchmark
from verification_systems import VerificationOrchestrator

# Classify corpus and generate config
corpus = "Your domain-specific text here..."
config = classify_and_configure(corpus, num_questions=50)

# Generate benchmark
benchmark = generate_domain_benchmark(corpus, config['eval_type'], 50)

# Verify responses (example)
verifier = VerificationOrchestrator()
result = verifier.verify("prediction", "ground_truth", config['eval_type'])
```

## 📋 Supported Evaluation Types

### 🎯 Deterministic Evaluations (Exact Answers)
- **Mathematical**: Enhanced mathematical expression verification using math-verify library
- **Math Expression**: Plain mathematical expressions (e.g., "1/2" ≡ "0.5")
- **LaTeX Math**: LaTeX mathematical expressions (e.g., "$\\sqrt{4}$" ≡ "$2$")
- **Code Generation**: Function correctness, syntax validation, execution testing
- **Factual QA**: Objective facts with exact match verification
- **Multiple Choice**: Standardized answer selection
- **Domain Knowledge**: Specific domain facts and concepts

### 🎨 Non-Deterministic Evaluations (Similarity-Based)
- **Summarization**: ROUGE-L, semantic similarity, content coverage
- **Translation**: BLEU scores, semantic preservation
- **Reading Comprehension**: Hybrid exact match + similarity
- **Creative Writing**: LLM-judge evaluation for creativity, coherence, fluency

## 🔧 Core Components

### EvaluationTypeClassifier
Uses LLM reasoning to analyze corpus content and determine the most appropriate evaluation methodology:

```python
classifier = EvaluationTypeClassifier()
result = classifier.classify_corpus(corpus_text)
# Returns: primary_type, secondary_types, confidence, reasoning
```

### AgenticQuestionGenerator
Advanced question generation using multiple strategies:

```python
generator = AgenticQuestionGenerator()
benchmark = generator.generate_comprehensive_benchmark(
    corpus_text, num_questions=50, eval_type='domain_knowledge'
)
# Includes quality scoring, category diversity, difficulty balancing
```

### VerificationOrchestrator
Intelligent verification system that chooses appropriate methods:

```python
verifier = VerificationOrchestrator()
result = verifier.verify(prediction, ground_truth, eval_type)
# Automatically selects: exact_match, similarity, code_execution, llm_judge, math_verify
```

### MathVerifyVerifier
Enhanced mathematical verification using the math-verify library:

```python
from docs_to_eval.core.verification import MathVerifyVerifier

# Initialize verifier
math_verifier = MathVerifyVerifier()

# Basic mathematical verification (supports LaTeX and expressions)
result = math_verifier.math_verify_match("${1,2,3,4}$", "${1,3} \\cup {2,4}$")
print(f"Score: {result.score}")  # 1.0 for mathematically equivalent

# LaTeX expression matching
result = math_verifier.latex_expression_match("$\\sqrt{4}$", "$2$")

# Plain expression matching  
result = math_verifier.expression_match("0.5", "1/2")

# Through orchestrator (recommended)
orchestrator = VerificationOrchestrator()
result = orchestrator.verify("0.5", "1/2", "math_expression")
result = orchestrator.verify("$\\sqrt{4}$", "$2$", "latex_math") 
result = orchestrator.verify("${1,2,3,4}$", "${1,3} \\cup {2,4}$", "mathematical")
```

## 📊 Evaluation Metrics

### Deterministic Metrics
- **Accuracy**: Exact match percentage
- **Pass Rate**: For code execution tasks
- **Normalized Accuracy**: Length-adjusted for multiple choice

### Non-Deterministic Metrics  
- **ROUGE-L**: Longest common subsequence similarity
- **BLEU**: N-gram overlap for translation-like tasks
- **Token Overlap F1**: Precision, recall, and F1 of token intersections
- **Semantic Similarity**: Contextual embedding similarity (mocked)

### Quality Metrics
- **Question Quality Score**: Clarity, relevance, difficulty appropriateness
- **Coverage Analysis**: Topic diversity and vocabulary richness
- **Benchmark Reliability**: Consistency across multiple runs

## 🎛️ Configuration Options

### Corpus Analysis
- **Automatic classification** with confidence scoring
- **Manual override** for evaluation type selection
- **Content analysis** (mathematical patterns, code structures, factual content)

### Generation Methods
- **Standard generation**: Pattern-based question creation
- **Agentic generation**: Multi-strategy approach with quality improvement
- **Question count**: 1-200 questions per benchmark

### Verification Settings
- **Temperature control**: 0.0 for deterministic, 0.7 for creative tasks
- **Similarity thresholds**: Configurable acceptance criteria
- **Multi-metric evaluation**: Combined scoring approaches

## 📈 Sample Output

```
🎯 Overall Performance:
   Mean Score: 0.756
   Median Score: 0.782
   Min Score: 0.445
   Max Score: 0.950
   Questions Evaluated: 50

📈 Detailed Metrics:
   exact_match_mean: 0.720
   token_overlap_f1_mean: 0.785
   rouge_l_f1_mean: 0.762

🔍 Sample Evaluations:
   Example 1:
   Question: What is machine learning in the context of artificial intelligence?
   Expected: A subset of AI that learns from data...
   Predicted: Machine learning is a field of AI that uses algorithms...
   Score: 0.850
   Method: token_overlap
```

## 🏭 Industrial Applications

### 🎓 Educational Technology
- **Automated quiz generation** from textbooks and course materials
- **Student assessment** with domain-appropriate metrics
- **Content quality evaluation** for educational resources

### 🏢 Enterprise AI
- **Model evaluation pipelines** for domain-specific deployments  
- **Benchmark creation** for proprietary knowledge bases
- **Performance monitoring** of production LLM systems

### 🔬 Research & Development
- **Reproducible evaluation** across different model architectures
- **Domain adaptation assessment** for specialized applications
- **Comparative studies** with standardized benchmarks

## 🛠️ Technical Requirements

- **Python 3.7+**
- **Functional programming approach** (no explicit type annotations)
- **Mock LLM interface** for testing (easily replaceable with real LLMs)
- **JSON-based configuration** and result storage
- **Modular architecture** for easy extension

## 🔄 Extensibility

The system is designed for easy extension:

### Adding New Evaluation Types
```python
# In benchmark_generators.py
class CustomBenchmarkGenerator(BenchmarkGenerator):
    def generate_benchmark(self, corpus_text, num_questions):
        # Custom generation logic
        pass

# Register in factory
BenchmarkGeneratorFactory.generators['custom_type'] = CustomBenchmarkGenerator
```

### Adding New Verification Methods
```python
# In verification_systems.py
def custom_verification(prediction, ground_truth):
    # Custom verification logic
    return VerificationResult(score, metrics, method)

# Add to orchestrator
VerificationOrchestrator.verify_methods['custom'] = custom_verification
```

## 📝 Future Enhancements

- **Real LLM integration** (OpenAI, Anthropic, Hugging Face APIs)
- **Semantic embedding models** for improved similarity metrics
- **Multi-language support** for international corpora
- **Batch processing** for large-scale evaluations
- **Web interface** for non-technical users
- **Integration plugins** for popular ML frameworks

## 🤝 Contributing

This system demonstrates a complete automated evaluation pipeline. Key areas for enhancement:

1. **Real LLM Integration**: Replace mock interfaces with actual model APIs
2. **Advanced Similarity Metrics**: Implement semantic embedding similarity
3. **Expanded Evaluation Types**: Add more domain-specific evaluation methods
4. **Performance Optimization**: Batch processing and caching improvements
5. **UI/UX Enhancement**: Web interface and visualization improvements

---

**docs-to-eval** provides a solid foundation for automated LLM evaluation that can be adapted to various domains and use cases. Its modular design and comprehensive feature set make it suitable for both research and production environments.