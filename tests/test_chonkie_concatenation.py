import math
from typing import List, Dict, Any

import pytest

from docs_to_eval.utils.config import ChunkingConfig
from docs_to_eval.utils.text_processing import _concatenate_small_chunks, create_smart_chunks


def _get_token_counter():
    from tiktoken import get_encoding
    return get_encoding("cl100k_base")


def _generate_text_with_min_tokens(min_tokens: int) -> str:
    """Generate text with at least min_tokens using tiktoken for accuracy."""
    enc = _get_token_counter()
    # Use a base sentence that tokenizes to ~8-10 tokens to keep loops small
    base = "This is a test sentence with domain-specific terms and context. "
    text_parts: List[str] = []
    total = 0
    while total < min_tokens:
        text_parts.append(base)
        total = len(enc.encode("".join(text_parts)))
    return "".join(text_parts)


def _make_small_chunks(num_chunks: int, tokens_per_chunk: int) -> List[Dict[str, Any]]:
    enc = _get_token_counter()
    chunks: List[Dict[str, Any]] = []
    for i in range(num_chunks):
        text = _generate_text_with_min_tokens(tokens_per_chunk)
        token_count = len(enc.encode(text))
        chunks.append({
            'text': text,
            'index': i,
            'method': 'test_small',
            'size': len(text),
            'token_count': token_count,
            'token_status': 'exact',
            'semantic_score': 1.0,
            'word_count': len(text.split()),
            'is_smart_chunking': True,
        })
    return chunks


def test_concatenate_small_chunks_meets_min_tokens():
    """
    Ensure that small chunks are concatenated until they meet the min token threshold
    (user-level desired range 500-4000 tokens).
    """
    enc = _get_token_counter()
    config = ChunkingConfig(
        use_token_chunking=True,
        enable_chonkie=True,   # Path under test is the chonkie post-process util
        min_token_size=500,
        max_token_size=4000,
        target_token_size=3000,
        overlap_tokens=100,
    )

    # Create 8 small chunks ~100-120 tokens each → expect concatenation into >=500 tokens
    small_chunks = _make_small_chunks(num_chunks=8, tokens_per_chunk=120)
    token_counter = enc

    concatenated = _concatenate_small_chunks(small_chunks, config, token_counter)

    assert len(concatenated) >= 1
    # Each resulting chunk should be within configured range or explicitly flagged
    for c in concatenated:
        assert 'token_count' in c
        assert c['token_count'] > 0
        # Must meet min tokens unless it finalized due to approaching max (rare here)
        assert c['token_count'] >= config.min_token_size or c['token_count'] > int(config.max_token_size * 0.8)
        assert c['index'] >= 0


def test_create_smart_chunks_without_chonkie_respects_min_tokens():
    """
    With chonkie disabled, smart chunking should still produce chunks whose token counts
    are within the configured [min, max] range when token-aware chunking is on.
    """
    config = ChunkingConfig(
        use_token_chunking=True,
        enable_chonkie=False,  # explicitly test simple chunking path
        min_token_size=500,
        max_token_size=4000,
        target_token_size=2000,
        overlap_tokens=100,
    )

    # Build a corpus ~2500 tokens → expect 1 chunk around target or multiple valid ranges
    corpus_text = _generate_text_with_min_tokens(2500)
    chunks = create_smart_chunks(corpus_text, chunking_config=config)

    assert len(chunks) >= 1
    for ch in chunks:
        assert 'token_count' in ch
        # Allow estimated if tiktoken not available, but in our env it should be exact
        assert ch['token_count'] > 0
        # Chunks should be within [min, max] (simple chunker aims for this window)
        assert config.min_token_size <= ch['token_count'] <= config.max_token_size

