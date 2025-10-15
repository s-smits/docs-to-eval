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
├── run_server.py               # FastAPI server entry point
├── docs_to_eval/
│   ├── core/                   # Core evaluation framework
│   │   ├── classification.py   # LLM-based evaluation type classification
│   │   ├── evaluation.py       # Core evaluation framework
│   │   ├── verification.py     # Verification systems
│   │   ├── benchmarks.py       # Benchmark generation
│   │   └── agentic/            # Advanced agentic generation
│   │       ├── generator.py    # Agentic question generation
│   │       └── orchestrator.py # Generation orchestration
│   ├── llm/                    # LLM interfaces
│   │   ├── openrouter_interface.py # OpenRouter API integration
│   │   └── mock_interface.py   # Mock LLM for testing
│   ├── ui_api/                 # Web API and interface
│   │   ├── main.py            # FastAPI application
│   │   ├── routes.py          # API routes
│   │   └── websockets.py      # Real-time updates
│   └── utils/                  # Utilities and helpers
└── frontend/                   # Web interface assets
```

## 🚀 Quick Start

### Web Interface

```bash
python run_server.py
```

Then open your browser to: **http://localhost:8080**

The web interface allows you to:
1. **Upload your corpus** (text entry, file upload, or sample datasets)
2. **Configure parameters** (number of questions, evaluation type, model selection)
3. **Generate benchmark** (standard or advanced agentic generation)
4. **Run evaluation** (with multiple LLM options)
5. **View results** (comprehensive analysis and interactive reports)

### Programmatic Usage

```python
# Use individual core components
from docs_to_eval.core.classification import classify_evaluation_type
from docs_to_eval.core.agentic.generator import AgenticQuestionGenerator
from docs_to_eval.core.verification import VerificationOrchestrator

# Classify corpus and determine evaluation type
corpus = "Your domain-specific text here..."
classification = classify_evaluation_type(corpus)
eval_type = classification['primary_type']

# Generate benchmark using agentic approach
generator = AgenticQuestionGenerator()
benchmark = generator.generate_comprehensive_benchmark(corpus, eval_type, 50)

# Verify responses
verifier = VerificationOrchestrator()
result = verifier.verify("prediction", "ground_truth", eval_type)
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

## 🛠️ Installation & Requirements

### Prerequisites
- **Python 3.8+**
- **uv package manager** (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd docs-to-eval

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the web server
python run_server.py

# Open browser to http://localhost:8080
```

### Technical Features
- **FastAPI web interface** with real-time updates
- **Multiple LLM providers** (OpenRouter, local models, mock interface)
- **WebSocket support** for live progress updates
- **File upload and corpus management**
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

## 📝 Recent Enhancements

### ✅ Implemented
- **Web interface** with FastAPI and real-time updates
- **Real LLM integration** via OpenRouter (GPT, Claude, Gemini, etc.)
- **File upload and corpus management**
- **WebSocket support** for live progress tracking
- **Advanced agentic question generation**
- **Mathematical verification** with math-verify library

### 🚧 Future Enhancements
- **Semantic embedding models** for improved similarity metrics
- **Multi-language support** for international corpora
- **Batch processing** for large-scale evaluations
- **Advanced visualization dashboards**
- **API rate limiting and caching**
- **Integration plugins** for popular ML frameworks
- **Export to common benchmark formats** (JSON, CSV, HuggingFace datasets)

## 🤝 Contributing

This system provides a complete automated evaluation pipeline with web interface and real LLM integration. Key areas for enhancement:

1. **Advanced Similarity Metrics**: Implement semantic embedding similarity
2. **Expanded Evaluation Types**: Add more domain-specific evaluation methods  
3. **Performance Optimization**: Batch processing and caching improvements
4. **UI/UX Enhancement**: Advanced visualization and dashboard features
5. **Integration Ecosystem**: Plugins for popular ML frameworks and platforms

---

**docs-to-eval** provides a solid foundation for automated LLM evaluation that can be adapted to various domains and use cases. Its modular design and comprehensive feature set make it suitable for both research and production environments.