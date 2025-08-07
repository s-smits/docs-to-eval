"""
Integration test for LoRA fine-tuning with real data splits
Tests the complete pipeline from question generation to training data preparation
"""

import pytest
import tempfile
import json
from pathlib import Path
from docs_to_eval.core.evaluation import (
    create_finetune_test_set, 
    BenchmarkWithFinetuneSet,
    BenchmarkConfig
)
from docs_to_eval.core.finetune.lora_integration import (
    LoRAFinetuningOrchestrator,
    LoRAFinetuningConfig
)


@pytest.fixture
def sample_questions():
    """Sample questions for testing"""
    return [
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "id": "q1"
        },
        {
            "question": "What is deep learning?",
            "answer": "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
            "id": "q2"
        },
        {
            "question": "What is natural language processing?",
            "answer": "Natural language processing is a field of AI that deals with the interaction between computers and human language.",
            "id": "q3"
        },
        {
            "question": "What is computer vision?",
            "answer": "Computer vision is a field of AI that enables computers to interpret and understand visual information.",
            "id": "q4"
        },
        {
            "question": "What is reinforcement learning?",
            "answer": "Reinforcement learning is a type of machine learning where agents learn through interaction with an environment.",
            "id": "q5"
        }
    ]


@pytest.fixture
def lora_orchestrator():
    """LoRA orchestrator for testing"""
    return LoRAFinetuningOrchestrator()


def test_finetune_test_set_creation(sample_questions):
    """Test creation of finetune test set with proper splits"""
    
    # Create finetune test set with 20% test split
    finetune_set = create_finetune_test_set(
        questions=sample_questions,
        test_percentage=0.2,
        random_seed=42
    )
    
    # Verify splits
    assert finetune_set.test_set_size == 1  # 20% of 5 = 1
    assert finetune_set.train_set_size == 4  # 80% of 5 = 4
    assert len(finetune_set.test_questions) == 1
    assert len(finetune_set.train_questions) == 4
    
    # Verify no overlap between train and test
    test_ids = {q["id"] for q in finetune_set.test_questions}
    train_ids = {q["id"] for q in finetune_set.train_questions}
    assert test_ids.isdisjoint(train_ids)
    
    # Verify all questions are accounted for
    all_ids = test_ids | train_ids
    expected_ids = {q["id"] for q in sample_questions}
    assert all_ids == expected_ids


def test_lora_data_preparation(sample_questions, lora_orchestrator):
    """Test LoRA training data preparation"""
    
    # Create finetune test set
    finetune_set = create_finetune_test_set(
        questions=sample_questions,
        test_percentage=0.2,
        random_seed=42
    )
    
    # Create temporary work directory
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        
        # Prepare training data
        train_file, valid_file, test_file = lora_orchestrator.prepare_training_data(
            finetune_set, work_dir
        )
        
        # Verify files were created
        assert train_file.exists()
        assert valid_file.exists() 
        assert test_file.exists()
        
        # Read and verify training data format
        with open(train_file, 'r') as f:
            train_data = [json.loads(line) for line in f]
        
        with open(valid_file, 'r') as f:
            valid_data = [json.loads(line) for line in f]
            
        with open(test_file, 'r') as f:
            test_data = [json.loads(line) for line in f]
        
        # Verify data format
        for item in train_data + valid_data + test_data:
            assert "text" in item
            assert "Question:" in item["text"]
            assert "Answer:" in item["text"]
        
        # Verify training data count (80% of 4 train questions = 3 for training, 1 for validation)
        assert len(train_data) == 3
        assert len(valid_data) == 1
        assert len(test_data) == 1  # From original test split
        
        print(f"✅ Successfully prepared training data:")
        print(f"   📝 Training samples: {len(train_data)}")
        print(f"   🔍 Validation samples: {len(valid_data)}")
        print(f"   🧪 Test samples: {len(test_data)}")
        
        # Verify content format
        example = train_data[0]
        assert example["text"].startswith("Question:")
        assert "Answer:" in example["text"]


def test_small_dataset_handling(lora_orchestrator):
    """Test handling of very small datasets"""
    
    # Create minimal dataset (2 questions to ensure split works)
    small_questions = [
        {
            "question": "What is AI?",
            "answer": "AI stands for Artificial Intelligence.",
            "id": "q1"
        },
        {
            "question": "What is ML?",
            "answer": "ML stands for Machine Learning.",
            "id": "q2"
        }
    ]
    
    # Create finetune test set
    finetune_set = create_finetune_test_set(
        questions=small_questions,
        test_percentage=0.5,  # 50% split - 1 train, 1 test
        random_seed=42
    )
    
    # Should handle gracefully
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        
        train_file, valid_file, test_file = lora_orchestrator.prepare_training_data(
            finetune_set, work_dir
        )
        
        # Verify files exist and have content
        assert train_file.exists()
        assert valid_file.exists()
        assert test_file.exists()


def test_empty_dataset_error(lora_orchestrator):
    """Test error handling for empty datasets"""
    
    # Create empty finetune test set
    empty_finetune_set = create_finetune_test_set(
        questions=[],
        test_percentage=0.2,
        random_seed=42
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        
        # Should raise error for empty training data
        with pytest.raises(ValueError, match="No training questions available"):
            lora_orchestrator.prepare_training_data(empty_finetune_set, work_dir)


def test_lora_config_validation():
    """Test LoRA configuration validation"""
    
    # Test default config
    config = LoRAFinetuningConfig()
    assert config.model_path == "mlx_model"
    assert config.lora_layers == 16
    assert config.batch_size == 4
    assert config.learning_rate == 1e-5
    assert config.max_iters == 1000
    
    # Test custom config
    custom_config = LoRAFinetuningConfig(
        model_path="custom_model",
        lora_layers=8,
        max_iters=500
    )
    assert custom_config.model_path == "custom_model"
    assert custom_config.lora_layers == 8
    assert custom_config.max_iters == 500


def test_data_format_validation(sample_questions, lora_orchestrator):
    """Test validation of data format"""
    
    # Test with malformed data
    malformed_questions = [
        {"question": "", "answer": "Empty question"},  # Empty question
        {"question": "Valid question?", "answer": ""},  # Empty answer
        {"question": "Valid question?", "answer": "Valid answer"}  # Valid
    ]
    
    finetune_set = create_finetune_test_set(
        questions=malformed_questions,
        test_percentage=0.3,
        random_seed=42
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        
        # Should raise error for empty question/answer
        with pytest.raises(ValueError, match="Empty question or answer"):
            lora_orchestrator.prepare_training_data(finetune_set, work_dir)


if __name__ == "__main__":
    # Run a basic integration test
    questions = [
        {"question": f"What is topic {i}?", "answer": f"Topic {i} is important.", "id": f"q{i}"}
        for i in range(10)
    ]
    
    finetune_set = create_finetune_test_set(questions, test_percentage=0.2)
    orchestrator = LoRAFinetuningOrchestrator()
    
    print(f"Created finetune set:")
    print(f"  Train: {len(finetune_set.train_questions)}")
    print(f"  Test: {len(finetune_set.test_questions)}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            train_file, valid_file, test_file = orchestrator.prepare_training_data(
                finetune_set, Path(temp_dir)
            )
            print("✅ Integration test passed!")
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            raise