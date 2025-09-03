# LLM Provider Integration Guide

This guide explains how to integrate and use Groq and direct Gemini SDK with the docs-to-eval system.

## 🚀 Quick Start

### 1. Install Dependencies

**For Groq:**
```bash
pip install groq
```

**For Gemini SDK:**
```bash
pip install google-generativeai
```

### 2. Set API Keys

**Groq:**
```bash
export GROQ_API_KEY=your_groq_api_key_here
```

**Gemini:**
```bash
export GEMINI_API_KEY=your_gemini_api_key_here
# OR
export GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Test the Integration

```bash
python test_integrations.py
```

## 🔌 Provider Details

### Groq Integration

**Features:**
- ⚡ Ultra-fast inference (optimized hardware)
- 🦙 Multiple Llama 3 models (8B, 70B)
- 🔄 Batch processing support
- 📊 Built-in rate limiting
- 🎯 Evaluation-type aware prompting

**Available Models:**
- `llama3-8b-8192` - Llama 3 8B (very fast, good for most tasks)
- `llama3-70b-8192` - Llama 3 70B (slower but more capable)
- `mixtral-8x7b-32768` - Mixtral 8x7B (large context window)
- `gemma-7b-it` - Gemma 7B (efficient for basic tasks)

**Example Usage:**
```python
from docs_to_eval.llm import create_llm_interface

# Create Groq interface
groq_llm = create_llm_interface('groq', model='llama3-8b-8192')

# Generate response
response = await groq_llm.generate_response(
    prompt="What is machine learning?",
    eval_type="factual_qa"
)
print(response.text)
```

### Gemini SDK Integration

**Features:**
- 🧠 Google's latest Gemini models
- 🌍 Massive context windows (up to 2M tokens)
- 💡 Advanced reasoning capabilities
- 🔄 Batch processing support
- ⚙️ Configurable safety settings
- 🎯 Evaluation-type aware system instructions

**Available Models:**
- `gemini-1.5-flash-latest` - Fast, cost-effective (recommended)
- `gemini-1.5-pro-latest` - Most capable, 2M token context
- `gemini-1.0-pro` - Stable production model
- `gemini-pro` - General purpose model

**Example Usage:**
```python
from docs_to_eval.llm import create_llm_interface

# Create Gemini interface
gemini_llm = create_llm_interface('gemini', model='gemini-1.5-flash-latest')

# Generate response
response = await gemini_llm.generate_response(
    prompt="Explain quantum computing",
    eval_type="factual_qa"
)
print(response.text)
```

## 🔄 Batch Processing

Both providers support efficient batch processing:

### Groq Batch Processing

```python
from docs_to_eval.llm.groq_interface import GroqBatchInterface

batch_interface = GroqBatchInterface(max_concurrent=5)
results = await batch_interface.process_batch(
    prompts=[
        "What is AI?",
        "Explain neural networks",
        "What is deep learning?"
    ],
    eval_type="factual_qa"
)

# Get statistics
stats = batch_interface.get_batch_stats()
print(f"Success rate: {stats['success_rate']:.1f}%")
```

### Gemini Batch Processing

```python
from docs_to_eval.llm.gemini_sdk_interface import GeminiBatchInterface

batch_interface = GeminiBatchInterface(max_concurrent=3)
results = await batch_interface.process_batch(
    prompts=[
        "What is machine learning?",
        "How do transformers work?",
        "What is AGI?"
    ],
    eval_type="factual_qa"
)

stats = batch_interface.get_batch_stats()
print(f"Average response time: {stats['average_response_time']:.2f}s")
```

## 🎯 Evaluation Type Support

Both integrations support evaluation-type specific prompting:

- **mathematical** - Optimized for math problems with step-by-step solutions
- **code_generation** - Focused on clean, documented code generation
- **factual_qa** - Precise, factual responses with citations
- **creative_writing** - Engaging narrative content
- **domain_knowledge** - Specialized domain expertise
- **reasoning** - Step-by-step logical reasoning

## 🔧 Advanced Configuration

### Groq Configuration

```python
from docs_to_eval.llm.groq_interface import GroqConfig, GroqInterface

config = GroqConfig(
    model="llama3-70b-8192",
    temperature=0.3,
    max_tokens=1024,
    top_p=0.9
)

groq_interface = GroqInterface(config)
```

### Gemini Configuration

```python
from docs_to_eval.llm.gemini_sdk_interface import GeminiConfig, GeminiSDKInterface

config = GeminiConfig(
    model="gemini-1.5-pro-latest",
    temperature=0.7,
    max_output_tokens=2048,
    top_p=0.95,
    top_k=40,
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_LOW_AND_ABOVE"
    }
)

gemini_interface = GeminiSDKInterface(config)
```

## 🔗 Integration with docs-to-eval Pipeline

### Using with the Core Pipeline

```python
from docs_to_eval.core.pipeline import EvaluationPipeline
from docs_to_eval.utils.config import create_default_config
from docs_to_eval.llm import create_llm_interface

# Create configuration
config = create_default_config()
config.llm.provider = "groq"  # or "gemini"
config.llm.model = "llama3-8b-8192"  # or "gemini-1.5-flash-latest"

# Create LLM interface
llm_interface = create_llm_interface(
    provider=config.llm.provider,
    model=config.llm.model
)

# Create and run pipeline
pipeline = EvaluationPipeline(config)
# Note: You'll need to modify the pipeline to use your LLM interface
```

### Using with CLI

You can modify the CLI to use different providers by setting environment variables:

```bash
export LLM_PROVIDER=groq
export LLM_MODEL=llama3-8b-8192
python -m docs_to_eval.cli.main evaluate your_corpus.txt
```

### Using with Web Interface

The FastAPI interface can be configured to use different providers:

```python
# In your application startup
from docs_to_eval.llm import create_llm_interface

app_llm = create_llm_interface('groq', model='llama3-8b-8192')
# Use app_llm in your route handlers
```

## 📊 Performance Comparison

| Provider | Speed | Cost | Context | Best For |
|----------|-------|------|---------|----------|
| Groq (Llama 3 8B) | ⚡⚡⚡ | 💰 | 8K | Fast evaluation, testing |
| Groq (Llama 3 70B) | ⚡⚡ | 💰💰 | 8K | Better reasoning |
| Gemini Flash | ⚡⚡ | 💰 | 1M | Long documents |
| Gemini Pro | ⚡ | 💰💰💰 | 2M | Complex reasoning |

## 🚨 Error Handling

Both integrations include comprehensive error handling:

```python
try:
    response = await llm_interface.generate_response(
        prompt="Your prompt here",
        eval_type="factual_qa"
    )
    
    if response.text:
        print(f"Response: {response.text}")
        print(f"Confidence: {response.confidence}")
    else:
        print("No response generated")
        
except Exception as e:
    print(f"Error: {e}")
    # Check metadata for more details
    print(f"Error details: {response.metadata.get('error')}")
```

## 🔍 Monitoring and Analytics

### Performance Statistics

```python
# Get performance stats
stats = llm_interface.get_performance_stats()
print(f"Total calls: {stats['total_calls']}")
print(f"Average response time: {stats['average_response_time']:.2f}s")
print(f"Total tokens used: {stats['total_tokens_used']}")
```

### Rate Limiting

Both providers include built-in rate limiting:

```python
# Check rate limiter status
rate_stats = llm_interface.rate_limiter.get_stats()
print(f"Requests last minute: {rate_stats['requests_last_minute']}")
print(f"Current rate: {rate_stats['current_rate']:.2f}/min")
```

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure environment variables are set correctly
   - Check for typos in variable names
   - Restart your shell after setting variables

2. **Import Errors**
   - Install required packages: `pip install groq google-generativeai`
   - Check Python environment and package versions

3. **Rate Limiting**
   - Both providers have built-in rate limiting
   - Adjust `max_concurrent` in batch interfaces
   - Monitor rate limiter statistics

4. **Model Not Available**
   - Check model names against available models
   - Use `list_available_models()` to see options
   - Some models may not be available in all regions

### Getting Help

1. Check the test script: `python test_integrations.py`
2. Review logs for detailed error messages
3. Check API documentation:
   - [Groq API Documentation](https://console.groq.com/docs)
   - [Gemini API Documentation](https://ai.google.dev/docs)

## 📈 Next Steps

1. **Set up API keys** for the providers you want to use
2. **Run the test script** to verify everything works
3. **Integrate with your evaluation pipeline** using the examples above
4. **Monitor performance** using the built-in statistics
5. **Scale up** using batch processing for large evaluations

The integrations are designed to be drop-in replacements for existing LLM interfaces, making it easy to experiment with different providers and find the best fit for your specific use case.
