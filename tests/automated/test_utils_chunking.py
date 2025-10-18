"""
Utility-level tests for configuration validation and chunk creation helpers.
"""

import pytest

from docs_to_eval.utils.config import ChunkingConfig
from docs_to_eval.utils.text_processing import create_smart_chunks


def test_chunking_config_validation_adjusts_target():
    config = ChunkingConfig(enable_chonkie=False)
    assert config.max_chunk_size > config.min_chunk_size
    assert config.max_token_size > config.min_token_size

    with pytest.raises(ValueError):
        ChunkingConfig(min_chunk_size=1000, max_chunk_size=500, enable_chonkie=False)


def test_create_smart_chunks_without_chonkie(sample_corpus):
    config = ChunkingConfig(
        enable_chonkie=False,
        min_chunk_size=200,
        max_chunk_size=600,
        target_chunk_size=400,
        overlap_size=100,
    )

    chunks = create_smart_chunks(sample_corpus, chunking_config=config)

    assert chunks, "Expected at least one chunk to be produced"
    assert all("metadata" in chunk for chunk in chunks)
    assert any(chunk["metadata"].get("chunk_method") for chunk in chunks)
