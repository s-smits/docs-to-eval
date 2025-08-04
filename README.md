# AutoEval - Production-Ready Automated LLM Evaluation System

🚀 **AutoEval** is a sophisticated, production-ready system for automated LLM evaluation that generates domain-specific benchmarks and evaluates language models using multiple methodologies. Built with enterprise-grade architecture, statistical rigor following lm-evaluation-harness standards, and comprehensive quality assurance.

## 🌟 Key Features

### 🧠 Intelligent Evaluation Pipeline
- **Smart content detection** with automatic template-content mismatch prevention
- **Intelligent verification routing** based on actual question content analysis
- **10+ evaluation types** with task-specific statistical baselines
- **Agentic benchmark generation** with multi-agent quality enhancement

### 🏗️ Production-Grade Architecture
- **Async-native pipeline** with concurrency safety and adaptive rate limiting
- **Comprehensive error handling** with explicit quality warnings
- **Statistical rigor** following lm-evaluation-harness gold standards
- **Bootstrap confidence intervals** with bias-corrected sampling
- **Thread-safe operations** with async locks and proper resource management

### 🔍 Advanced Verification Systems
- **Mathematical verification** with math-verify integration for LaTeX and expressions
- **Intelligent routing** based on content analysis (mathematical, factual, code, creative)
- **Multi-modal verification** including exact match, similarity, execution, and LLM-judge
- **Task-specific baselines** (0.0 for mathematical, 0.25 for multiple choice, etc.)

### 📊 Comprehensive Analytics & Quality Assurance
- **Statistical significance testing** with proper one-sample t-tests
- **Quality status tracking** with fallback detection and degradation warnings
- **Performance monitoring** with adaptive rate limiting based on LLM response times
- **Detailed reporting** following lm-evaluation-harness standards

## 🏗️ System Architecture

```
📁 AutoEval Production System/
├── 🎯 Core Engine
│   ├── pipeline.py              # Unified evaluation pipeline orchestrator
│   ├── evaluation.py            # Evaluation framework & type definitions
│   ├── classification.py        # Intelligent evaluation type classifier
│   ├── verification.py          # Multi-modal verification with smart routing
│   ├── benchmarks.py           # Benchmark generation factory
│   └── agentic/                # Advanced agentic benchmark generation
│       ├── orchestrator.py     # Multi-agent pipeline with adaptive rate limiting
│       ├── agents.py           # Specialized agents (ConceptMiner, QuestionWriter, etc.)
│       ├── models.py           # Enhanced data models with concept ownership
│       └── validation.py       # Quality validation and improvement
├── 🔧 Interfaces
│   ├── cli/                    # Command-line interface (Typer + Rich)
│   │   ├── main.py            # Full CLI with progress bars and statistics
│   │   └── interactive.py     # Step-by-step guided evaluation
│   └── ui_api/                # Web API (FastAPI + WebSockets)
│       ├── main.py            # CORS, static files, lifespan management
│       ├── routes.py          # API endpoints with background tasks
│       └── websockets.py      # Real-time progress updates
├── 🤖 LLM Integration
│   ├── base.py                # Abstract LLM interface with capability scoring
│   ├── mock_interface.py      # Comprehensive testing and simulation
│   └── openrouter_interface.py # Production LLM access via OpenRouter
├── 🛠️ Utilities
│   ├── config.py              # Pydantic configuration with validation
│   ├── statistical_analysis.py # Bootstrap CI, significance testing
│   ├── similarity.py          # Multi-method similarity calculations
│   ├── text_processing.py     # Answer normalization and extraction
│   └── logging.py            # Structured JSON logging with context
└── 🧪 Testing Framework
    ├── test_system_integration.py # End-to-end workflow testing
    ├── test_math_verify_integration.py # Mathematical verification
    ├── test_pipeline.py        # Core pipeline functionality
    └── test_benchmark_factory.py # Generator factory testing
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd docs-to-eval

# Install dependencies (using uv recommended)
uv sync --python 3.11
# Or using pip
pip install -e .

# Optional: Install math-verify for enhanced mathematical verification
pip install math-verify>=0.1.0
```

### Environment Setup

```bash
# Optional: Set environment variables
export DOCS_TO_EVAL_LOG_LEVEL=DEBUG
export OPENROUTER_API_KEY=your_key_here  # For production LLM evaluation
```

### Command Line Usage

```bash
# Full evaluation with progress tracking
uv run python -m docs_to_eval.cli.main evaluate corpus.txt --questions 20 --agentic

# Interactive guided evaluation
uv run python -m docs_to_eval.cli.main interactive

# Classification only
uv run python -m docs_to_eval.cli.main classify corpus.txt

# Start web API server
uv run python -m docs_to_eval.cli.main server --port 8000 --reload
```

### Programmatic Usage

```python
from docs_to_eval.core.pipeline import PipelineFactory
from docs_to_eval.utils.config import create_default_config

# Create pipeline with default configuration
config = create_default_config()
config.generation.num_questions = 10
config.llm.use_agentic = True

pipeline = PipelineFactory.create_pipeline(config)

# Run evaluation
corpus_text = "Your domain-specific text here..."
results = await pipeline.run_async(corpus_text)

print(f"Mean Score: {results['mean_score']:.3f}")
print(f"95% CI: [{results['confidence_interval_95'][0]:.3f}, {results['confidence_interval_95'][1]:.3f}]")
print(f"Statistically Significant: {results['statistically_significant']}")
```

### Web API Usage

```bash
# Start server
uv run python -m docs_to_eval.cli.main server

# API Examples
curl -X POST http://localhost:8000/api/v1/evaluation/start \
  -H "Content-Type: application/json" \
  -d '{"corpus_text": "Your text here", "num_questions": 10, "use_agentic": true}'

# WebSocket connection for real-time progress
# Connect to: ws://localhost:8000/ws/{run_id}
```

## 📋 Supported Evaluation Types

### 🎯 Deterministic Evaluations (Exact/Verifiable)
- **Mathematical**: Enhanced with math-verify library supporting LaTeX and expressions
  - Example: `"${1,2,3,4}$" ≡ "${1,3} \\cup {2,4}$"` → Score: 1.0
  - Baseline: 0.0 (mathematical answers are exact)
- **Code Generation**: Functional correctness with sandbox execution
  - Example: Function definition with unit tests → Pass/Fail verification
  - Baseline: 0.0 (code either works or doesn't)
- **Multiple Choice**: Normalized accuracy with length adjustment
  - Example: A/B/C/D selection with confidence scoring
  - Baseline: 0.25 (random chance for 4 options)
- **Factual QA**: Exact match with fallback to similarity
  - Example: "What is the capital of France?" → "Paris"
  - Baseline: 0.0 (factual answers require precision)

### 🎨 Non-Deterministic Evaluations (Similarity-Based)
- **Creative Writing**: LLM-judge evaluation with quality scoring
  - Baseline: 0.3 (subjective quality expectations)
- **Summarization**: ROUGE-L, semantic similarity, content coverage
  - Baseline: 0.2 (some overlap expected by chance)
- **Translation**: BLEU scores with semantic preservation
  - Baseline: 0.1 (minimal lexical overlap expected)
- **Domain Knowledge**: Partial credit for conceptual understanding
  - Baseline: 0.1 (some domain knowledge overlap possible)

## 🔧 Core Components

### Enhanced Pipeline Orchestrator
Unified workflow with intelligent routing and quality assurance:

```python
from docs_to_eval.core.pipeline import EvaluationPipeline

pipeline = EvaluationPipeline(config)
results = await pipeline.run_async(corpus_text)

# Automatic phases:
# 1. Classification → Benchmark Generation → LLM Evaluation → 
# 2. Verification → Results Aggregation → Statistical Analysis
```

### Agentic Benchmark Generation
Multi-agent system with concept ownership and feedback loops:

```python
from docs_to_eval.core.agentic import AgenticBenchmarkOrchestrator

orchestrator = AgenticBenchmarkOrchestrator(llm_pool, config)
benchmark = await orchestrator.generate(
    corpus_text, eval_type, num_questions=50, difficulty="hard"
)

# Agents: ConceptMiner → QuestionWriter → Adversary → Refiner → Validator
# Features: Adaptive retry with feedback, concept ownership, quality tracking
```

### Intelligent Verification System
Content-aware verification with automatic method selection:

```python
from docs_to_eval.core.verification import VerificationOrchestrator

verifier = VerificationOrchestrator()
result = verifier.verify(
    prediction="2", 
    ground_truth="$\\sqrt{4}$", 
    eval_type="mathematical",  # Will auto-detect and use LaTeX verification
    question="What is the square root of 4?"
)
# Automatically detects LaTeX and uses math-verify library
```

### Statistical Analysis with lm-eval-harness Standards
Comprehensive statistical analysis following gold-standard practices:

```python
from docs_to_eval.utils.statistical_analysis import EvaluationStatistics

# Calculate with task-specific baseline
stats = EvaluationStatistics.calculate_comprehensive_metrics(
    scores=[0.8, 0.9, 0.7, 0.85], 
    eval_type="mathematical"  # Uses 0.0 baseline
)

print(f"Mean: {stats.mean:.3f}")
print(f"95% CI: [{stats.confidence_interval_95[0]:.3f}, {stats.confidence_interval_95[1]:.3f}]")
print(f"Statistical significance (p-value): {stats.statistical_significance:.3f}")
print(f"Significant at α=0.05: {stats.statistical_significance < 0.05}")
```

## 📊 Quality Assurance Features

### Fallback Detection and Quality Warnings
The system now explicitly tracks and reports quality degradation:

```python
# API Response includes quality status
{
  "aggregate_metrics": { ... },
  "quality_status": {
    "has_fallback_questions": false,
    "quality_degraded": false,
    "warnings": [],
    "generation_modes": ["agentic"],
    "total_warnings": 0,
    "fallback_percentage": 0.0
  }
}
```

### Content-Type Mismatch Prevention
Automatic detection prevents nonsensical questions:

```python
# Before fix: "Solve for male." (random word + math template)
# After fix: Detects Etruscan mythology content → Uses factual_qa templates
# Result: "What is the significance of Tinia in Etruscan mythology?"
```

### Adaptive Performance Monitoring
Rate limiting adapts to LLM performance:

```python
# Pipeline automatically adjusts delays based on:
# - Average response times
# - Error rates  
# - Recent performance history
# - Exponential backoff for failures
```

## 📈 Sample Output

```
🎯 AutoEval Results - Production Grade
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Statistical Analysis (lm-evaluation-harness standard)
Mean Score: 0.847
95% Confidence Interval: [0.789, 0.905]
Statistical Significance: ✓ Significant (p=0.003)
Baseline Tested: 0.0 (mathematical)
Sample Size: 25 questions

🔬 Evaluation Quality
Generation Mode: agentic (100%)
Quality Status: ✓ No degradation warnings
Sufficient Samples: ✓ Yes (≥20)
Reliable CI: ✓ Yes (width < 0.2)

🎯 Verification Methods
mathematical (math-verify): 15 questions (avg: 0.893)
factual_qa (exact+similarity): 8 questions (avg: 0.782)
domain_knowledge (similarity): 2 questions (avg: 0.845)

⚡ Performance Stats
Total Processing Time: 45.2s
Average Question Generation: 1.8s
Rate Limiter: Adaptive (current delay: 0.3s)
Agent Success Rate: 96.8%
```

## 🏭 Production Applications

### 🎓 Educational Technology
- **Automated assessment generation** from textbooks with proper difficulty scaling
- **Student evaluation** with statistical significance testing
- **Content quality assurance** with fallback detection

### 🏢 Enterprise AI Evaluation
- **Model benchmarking** with task-specific baselines and confidence intervals
- **Performance monitoring** with quality degradation alerts
- **A/B testing** with proper statistical significance testing

### 🔬 Research & Development
- **Reproducible evaluation** following lm-evaluation-harness standards
- **Comparative studies** with bootstrap confidence intervals
- **Domain adaptation assessment** with intelligent content detection

## 🛠️ Technical Requirements

- **Python 3.8+** (Tested with 3.11)
- **Dependencies**: FastAPI, Typer, Rich, Pydantic, NumPy
- **Optional**: math-verify library for enhanced mathematical verification
- **Production**: OpenRouter API key for real LLM evaluation

## 🔄 Extension Points

### Adding New Evaluation Types
```python
# 1. Add to EvaluationType enum in evaluation.py
# 2. Create generator in benchmarks.py
# 3. Add verification logic in verification.py
# 4. Update task-specific baseline in statistical_analysis.py
```

### Adding New LLM Interfaces
```python
# Inherit from BaseLLMInterface
class CustomLLMInterface(BaseLLMInterface):
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        # Implementation here
        pass
```

### Adding New Verification Methods
```python
# Add verifier class in verification.py
# Update VerificationOrchestrator routing
# Add to VerificationMethod enum in config.py
```

## 🧪 Testing

```bash
# Run full test suite
uv run pytest

# Run specific test categories
uv run pytest tests/test_math_verify_integration.py  # Mathematical verification
uv run pytest tests/test_system_integration.py      # End-to-end workflow
uv run pytest tests/test_pipeline.py                # Core pipeline

# Test with different evaluation types
uv run python -m docs_to_eval.cli.main evaluate sample_corpus.txt --questions 5 --eval-type mathematical
```

## 📈 Performance & Scalability

### Optimization Features
- **Async-native pipeline** for concurrent processing
- **Adaptive rate limiting** based on LLM performance
- **Batch request processing** with configurable concurrency
- **Intelligent caching** for repeated evaluations
- **Background task processing** for web API

### Monitoring & Observability
- **Structured JSON logging** with evaluation context
- **Performance metrics tracking** (response times, success rates)
- **Quality status monitoring** (fallback usage, degradation alerts)
- **Statistical confidence reporting** with sample size recommendations

## 🔒 Production Deployment

### Security Considerations
- **API key management** via environment variables
- **Input validation** with Pydantic models
- **Rate limiting** for API endpoints
- **No secret logging** or storage

### Infrastructure Requirements
- **Python runtime** with async support
- **File system access** for logs and outputs
- **Network access** for LLM API calls (if using real LLMs)
- **Optional: Redis** for caching (future enhancement)

---

## 🎯 What Makes AutoEval Production-Ready

1. **Statistical Rigor**: Bootstrap confidence intervals, task-specific baselines, proper significance testing
2. **Quality Assurance**: Explicit fallback detection, content-mismatch prevention, degradation warnings
3. **Robustness**: Thread-safe operations, adaptive rate limiting, comprehensive error handling
4. **Observability**: Structured logging, performance monitoring, quality status tracking
5. **Standards Compliance**: Following lm-evaluation-harness principles and best practices

**AutoEval** bridges the gap between research-grade evaluation tools and production-ready systems, providing enterprise-scale reliability with the flexibility needed for diverse evaluation scenarios.