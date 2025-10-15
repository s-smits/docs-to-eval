"""
Pytest configuration and fixtures for docs-to-eval tests
Provides common test fixtures and configuration for all test modules
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import os

from docs_to_eval.utils.config import create_default_config
from docs_to_eval.llm.mock_interface import MockLLMInterface

"""
Ensure pytest-asyncio plugin is available so @pytest.mark.asyncio tests run without
"async not supported" errors in environments where auto-discovery may fail.
"""
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_corpus():
    """Sample corpus text for testing"""
    return """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    
    Machine learning is a subset of AI that focuses on algorithms that can 
    learn from and make predictions or decisions based on data. Deep learning 
    is a subset of machine learning that uses neural networks with multiple 
    layers to model and understand complex patterns in data.
    """


@pytest.fixture
def etruscan_corpus():
    """Sample Etruscan corpus for domain-specific testing"""
    return """
    The Etruscan civilization flourished in central Italy between the 8th and 3rd centuries BCE.
    The Etruscans were known for their art, architecture, and religious practices.
    They developed a sophisticated society with advanced metallurgy and urban planning.
    Etruscan religion included divination practices and elaborate burial customs.
    The civilization significantly influenced early Roman culture and society.
    """


@pytest.fixture
def mathematical_corpus():
    """Sample mathematical corpus for math evaluation testing"""
    return """
    Linear algebra is the branch of mathematics concerning vector spaces and linear mappings.
    A matrix is a rectangular array of numbers arranged in rows and columns.
    The determinant of a 2x2 matrix [[a,b],[c,d]] is calculated as ad - bc.
    Eigenvalues and eigenvectors are fundamental concepts in linear algebra.
    The equation 2x + 3 = 7 has solution x = 2.
    """


@pytest.fixture
def code_corpus():
    """Sample code corpus for code generation testing"""
    return """
    Python is a high-level programming language with dynamic typing.
    Functions in Python are defined using the 'def' keyword.
    Example: def hello_world(): return "Hello, World!"
    Lists are ordered collections that can contain different data types.
    Dictionary comprehensions provide a concise way to create dictionaries.
    """


@pytest.fixture
def test_config():
    """Create test configuration with safe defaults"""
    config = create_default_config()
    config.generation.num_questions = 5  # Smaller for testing
    config.generation.use_agentic = False  # Standard generation for tests
    config.system.log_level = "ERROR"  # Reduce log noise
    config.llm.temperature = 0.5  # Consistent for testing
    return config


@pytest.fixture
def mock_llm():
    """Create mock LLM interface for testing"""
    return MockLLMInterface(temperature=0.5)


@pytest.fixture
def api_test_client():
    """Create FastAPI test client"""
    from fastapi.testclient import TestClient
    from docs_to_eval.ui_api.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables and cleanup"""
    # Store original environment
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "ERROR"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def clear_evaluation_runs():
    """Clear evaluation runs before and after tests"""
    from docs_to_eval.ui_api.routes import evaluation_runs
    
    # Clear before test
    evaluation_runs._runs.clear()
    
    yield
    
    # Clear after test
    evaluation_runs._runs.clear()


@pytest.fixture
def sample_benchmark_items():
    """Sample benchmark items for testing"""
    return [
        {
            "question": "What is artificial intelligence?",
            "answer": "AI is intelligence demonstrated by machines",
            "eval_type": "domain_knowledge",
            "concept": "artificial intelligence",
            "difficulty": "basic"
        },
        {
            "question": "How does machine learning work?",
            "answer": "ML algorithms learn from data to make predictions",
            "eval_type": "domain_knowledge", 
            "concept": "machine learning",
            "difficulty": "intermediate"
        },
        {
            "question": "What is 2 + 2?",
            "answer": "4",
            "eval_type": "mathematical",
            "concept": "basic arithmetic",
            "difficulty": "basic"
        }
    ]


@pytest.fixture
def sample_llm_results():
    """Sample LLM evaluation results for testing"""
    return [
        {
            "question": "What is AI?",
            "ground_truth": "Artificial intelligence",
            "prediction": "AI is machine intelligence",
            "confidence": 0.8,
            "source": "test_llm"
        },
        {
            "question": "How does ML work?",
            "ground_truth": "Machine learning uses algorithms to learn from data",
            "prediction": "ML algorithms analyze data to find patterns",
            "confidence": 0.7,
            "source": "test_llm"
        }
    ]


@pytest.fixture
def sample_verification_results():
    """Sample verification results for testing"""
    return [
        {
            "question": "What is AI?",
            "prediction": "AI is machine intelligence", 
            "ground_truth": "Artificial intelligence",
            "score": 0.75,
            "method": "token_overlap",
            "details": {"overlap_ratio": 0.75, "precision": 0.8, "recall": 0.7}
        },
        {
            "question": "How does ML work?",
            "prediction": "ML algorithms analyze data to find patterns",
            "ground_truth": "Machine learning uses algorithms to learn from data",
            "score": 0.65,
            "method": "semantic_similarity",
            "details": {"similarity_score": 0.65, "method": "mock_embedding"}
        }
    ]


# Test markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "agentic: marks tests as agentic pipeline tests"
    )
    config.addinivalue_line(
        "markers", "websocket: marks tests as WebSocket tests"
    )
    config.addinivalue_line(
        "markers", "ui_workflow: marks tests as UI workflow tests"
    )


# Skip certain tests based on environment
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment"""
    import os
    
    # Skip agentic tests if no API key is configured
    skip_agentic = pytest.mark.skip(reason="No API key configured for agentic tests")
    
    # Skip slow tests in CI unless explicitly requested
    skip_slow = pytest.mark.skip(reason="Slow test skipped (use --runslow to run)")
    
    for item in items:
        # Skip agentic tests if no API key
        if "agentic" in item.keywords and not os.environ.get("OPENROUTER_API_KEY"):
            item.add_marker(skip_agentic)
        
        # Skip slow tests unless requested
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runintegration", action="store_true", default=False, help="run integration tests"
    )


# Test data constants
TEST_CORPORA = {
    "ai": """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to natural intelligence displayed by humans and animals.
    """,
    "math": """
    Mathematics is the study of numbers, shapes, and patterns.
    Basic arithmetic operations include addition, subtraction, multiplication, and division.
    """,
    "code": """
    Programming is the process of creating instructions for computers.
    Python is a popular programming language known for its simplicity.
    """
}

TEST_QUESTIONS = {
    "domain_knowledge": [
        {"question": "What is AI?", "answer": "Artificial intelligence"},
        {"question": "What is ML?", "answer": "Machine learning"}
    ],
    "mathematical": [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "What is 5*3?", "answer": "15"}
    ],
    "code_generation": [
        {"question": "Write a hello world function", "answer": "def hello(): return 'Hello World'"}
    ]
}


# Helper functions for tests
def create_test_file(temp_dir: Path, filename: str, content: str) -> Path:
    """Create a test file with given content"""
    file_path = temp_dir / filename
    file_path.write_text(content)
    return file_path


def assert_valid_evaluation_result(result_data: dict):
    """Assert that evaluation result has valid structure"""
    required_fields = ["run_id", "aggregate_metrics", "individual_results", "classification"]
    for field in required_fields:
        assert field in result_data, f"Missing required field: {field}"
    
    # Check aggregate metrics
    metrics = result_data["aggregate_metrics"]
    assert "mean_score" in metrics
    assert "num_samples" in metrics
    assert 0 <= metrics["mean_score"] <= 1
    assert metrics["num_samples"] > 0
    
    # Check individual results
    individual = result_data["individual_results"]
    assert isinstance(individual, list)
    assert len(individual) > 0
    
    for result in individual:
        assert "question" in result
        assert "score" in result
        assert 0 <= result["score"] <= 1


def assert_valid_websocket_message(message: dict):
    """Assert that WebSocket message has valid structure"""
    required_fields = ["type", "run_id", "timestamp"]
    for field in required_fields:
        assert field in message, f"Missing required field: {field}"
    
    valid_types = ["phase_start", "progress_update", "phase_complete", "evaluation_complete", "error", "log"]
    assert message["type"] in valid_types, f"Invalid message type: {message['type']}"