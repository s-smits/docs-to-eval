"""
Text processing utilities for evaluation framework
"""

import re
import time
from typing import List, Optional


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip())


def extract_numbers(text: str) -> List[str]:
    """Extract numbers from text"""
    return re.findall(r'-?\d+\.?\d*', text)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    if not answer:
        return ""
    
    answer = clean_text(answer.lower())
    # Remove punctuation except for periods in numbers
    answer = re.sub(r'[^\w\s\.]', '', answer)
    # Remove extra whitespace
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer


def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 50) -> List[str]:
    """Extract keywords from text"""
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + r',}\b', text.lower())
    
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
        'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our',
        'their', 'myself', 'yourself', 'himself', 'herself', 'itself',
        'ourselves', 'yourselves', 'themselves'
    }
    
    # Filter and count
    word_freq = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top keywords
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
    return [keyword[0] for keyword in keywords]


def truncate_text(text: str, max_length: int = 500, ellipsis: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    
    # Try to truncate at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can find a reasonable word boundary
        truncated = truncated[:last_space]
    
    return truncated + ellipsis


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    # Simple sentence splitting on periods, exclamation marks, and question marks
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def calculate_readability_score(text: str) -> float:
    """Calculate simple readability score (0-1, higher is more readable)"""
    if not text:
        return 0.0
    
    sentences = split_into_sentences(text)
    words = text.split()
    
    if not sentences or not words:
        return 0.0
    
    avg_sentence_length = len(words) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Simple scoring: shorter sentences and words = more readable
    sentence_score = max(0, 1 - (avg_sentence_length - 10) / 20)  # Optimal ~10 words/sentence
    word_score = max(0, 1 - (avg_word_length - 5) / 5)  # Optimal ~5 chars/word
    
    return (sentence_score + word_score) / 2


def extract_quoted_text(text: str) -> List[str]:
    """Extract quoted text from a string"""
    # Find text in quotes (both single and double)
    quotes = re.findall(r'"([^"]*)"|\'([^\']*)\'', text)
    # Flatten the tuples and filter empty strings
    return [quote for quote_tuple in quotes for quote in quote_tuple if quote]


def count_syllables(word: str) -> int:
    """Estimate syllable count for a word"""
    word = word.lower()
    if len(word) <= 3:
        return 1
    
    # Remove ending e
    if word.endswith('e'):
        word = word[:-1]
    
    # Count vowel groups
    vowels = 'aeiouy'
    syllable_count = 0
    prev_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel
    
    return max(1, syllable_count)


def extract_named_entities_simple(text: str) -> List[str]:
    """Simple named entity extraction using capitalization patterns"""
    # Find sequences of capitalized words
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    
    # Filter out common words that might be capitalized
    common_caps = {'The', 'This', 'That', 'These', 'Those', 'And', 'Or', 'But'}
    entities = [entity for entity in entities if entity not in common_caps]
    
    return list(set(entities))  # Remove duplicates


def contains_code_patterns(text: str) -> bool:
    """Check if text contains code-like patterns"""
    code_patterns = [
        r'def\s+\w+\s*\(',  # Python function definition
        r'function\s+\w+\s*\(',  # JavaScript function
        r'class\s+\w+\s*{',  # Class definition
        r'\w+\s*=\s*\w+\s*\(',  # Function call assignment
        r'import\s+\w+',  # Import statement
        r'from\s+\w+\s+import',  # From import
        r'if\s*\(.+\)\s*{',  # If statement
        r'for\s*\(.+\)\s*{',  # For loop
        r'while\s*\(.+\)\s*{',  # While loop
        r'console\.log\(',  # Console log
        r'print\s*\(',  # Print statement
    ]
    
    for pattern in code_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def contains_math_patterns(text: str) -> bool:
    """Check if text contains mathematical patterns"""
    math_patterns = [
        r'\d+\s*[\+\-\*\/]\s*\d+',  # Basic arithmetic
        r'=\s*\d+',  # Equations
        r'\b(equation|formula|calculate|solve)\b',  # Math keywords
        r'\b(sin|cos|tan|log|ln|sqrt)\b',  # Math functions
        r'\b\d+%',  # Percentages
        r'\$\d+',  # Money
        r'\b\d+\.\d+\b',  # Decimals
    ]
    
    for pattern in math_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def create_smart_chunks(text: str, target_chunk_size: int = 3000, overlap_percent: float = 5.0, 
                       chunking_config=None) -> List[dict]:
    """
    Create intelligent semantic chunks optimized for LLM processing using Chonkie library.
    Falls back to sentence-based chunking if Chonkie isn't available.
    
    Args:
        text: Input text to chunk
        target_chunk_size: Target size in characters (2k-4k recommended for LLMs)
        overlap_percent: Percentage overlap between chunks (5% recommended)
        chunking_config: Optional ChunkingConfig object with advanced settings
        
    Returns:
        List of chunk dictionaries with metadata
    """
    if not text or len(text) < 500:
        return [{"text": text, "start_pos": 0, "end_pos": len(text), "chunk_id": 0, "method": "single"}]
    
    # Use config values if provided
    if chunking_config:
        if chunking_config.adaptive_sizing:
            # Adaptive sizing based on text length
            text_length = len(text)
            if text_length > 20000:
                target_chunk_size = min(chunking_config.max_chunk_size, 4000)
            elif text_length < 5000:
                target_chunk_size = max(chunking_config.min_chunk_size, 2000)
            else:
                target_chunk_size = chunking_config.target_chunk_size
        else:
            target_chunk_size = chunking_config.target_chunk_size
        
        overlap_percent = chunking_config.overlap_percent
        
        # Force specific chunker if requested
        if chunking_config.force_chunker:
            return _create_forced_chunks(text, chunking_config.force_chunker, target_chunk_size, overlap_percent, chunking_config)
        
        # Skip Chonkie if disabled
        if not chunking_config.enable_chonkie:
            return _create_sentence_based_chunks(text, target_chunk_size, overlap_percent)
    
    # Try using Chonkie for semantic chunking first
    try:
        return _create_chonkie_chunks(text, target_chunk_size, overlap_percent, chunking_config)
    except ImportError:
        print("[INFO] Chonkie not available, using sentence-based chunking fallback")
        return _create_sentence_based_chunks(text, target_chunk_size, overlap_percent)
    except Exception as e:
        print(f"[WARNING] Chonkie chunking failed ({e}), falling back to sentence-based")
        return _create_sentence_based_chunks(text, target_chunk_size, overlap_percent)


def _create_chonkie_chunks(text: str, target_chunk_size: int, overlap_percent: float, chunking_config=None) -> List[dict]:
    """Create optimized semantic chunks using Chonkie library for 2k-4k context windows"""
    from chonkie import SemanticChunker, RecursiveChunker, SentenceChunker, LateChunker
    
    # Analyze text characteristics for optimal chunker selection  
    word_count = len(text.split())
    char_count = len(text)
    has_math = contains_math_patterns(text) if not chunking_config or chunking_config.preserve_math_expressions else False
    has_code = contains_code_patterns(text) if not chunking_config or chunking_config.preserve_code_blocks else False
    has_complex_content = has_math or has_code
    
    # Calculate optimal overlap in characters (Chonkie uses character-based overlap)
    overlap_chars = int(target_chunk_size * overlap_percent / 100)
    
    # Get semantic threshold from config
    semantic_threshold = chunking_config.semantic_threshold if chunking_config else 0.5
    
    # Advanced chunker selection based on web research best practices
    if char_count > 15000 and not has_complex_content:
        # Use LateChunker for very large documents - preserves global context
        try:
            chunker = LateChunker(
                chunk_size=target_chunk_size,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # Efficient model
            )
            chunker_type = "late"
        except Exception as e:
            # Fallback to SemanticChunker if LateChunker fails
            chunker = SemanticChunker(
                chunk_size=target_chunk_size,
                threshold=semantic_threshold  # Correct parameter name
            )
            chunker_type = "semantic"
    elif word_count > 2000 and not has_complex_content:
        # Use SemanticChunker for large, general text (recommended for 2k+ words)
        chunker = SemanticChunker(
            chunk_size=target_chunk_size,
            threshold=semantic_threshold  # Correct parameter name for semantic boundaries
        )
        chunker_type = "semantic"
    elif has_complex_content:
        # Use RecursiveChunker for technical content with optimized separators
        from chonkie import RecursiveRules, SplitRule
        
        # Create rules for recursive chunking
        separators = ["\n\n\n", "\n\n"]
        
        # Add code-specific separators if code detected and preservation enabled
        if has_code and (not chunking_config or chunking_config.preserve_code_blocks):
            separators.extend(["\n```", "```\n"])
        
        separators.extend(["\n#", ". ", "! ", "? ", "; ", "\n", " "])
        
        # Convert separators to SplitRules
        rules = RecursiveRules([SplitRule(sep) for sep in separators])
        
        chunker = RecursiveChunker(
            chunk_size=target_chunk_size,
            rules=rules
        )
        chunker_type = "recursive"
    else:
        # Use SentenceChunker for smaller or simpler content
        chunker = SentenceChunker(
            chunk_size=target_chunk_size,
            chunk_overlap=overlap_chars  # Correct parameter name
        )
        chunker_type = "sentence"
    
    # Create chunks with error handling
    try:
        chonkie_chunks = chunker(text)
    except Exception as e:
        print(f"[WARNING] Chonkie chunking failed with {chunker_type} chunker: {e}")
        # Fallback to basic SentenceChunker
        chunker = SentenceChunker(chunk_size=target_chunk_size, overlap=overlap_chars)
        chonkie_chunks = chunker(text)
        chunker_type = "sentence_fallback"
    
    # Convert to our format with enhanced metadata
    chunks = []
    current_pos = 0
    
    for i, chunk in enumerate(chonkie_chunks):
        chunk_text = chunk.text
        
        # Improved position tracking
        chunk_start = text.find(chunk_text, current_pos)
        if chunk_start == -1:
            # More sophisticated fallback positioning
            chunk_start = current_pos
            
        chunk_end = chunk_start + len(chunk_text)
        
        # Enhanced metadata based on Chonkie capabilities
        chunk_dict = {
            "text": chunk_text,
            "start_pos": chunk_start,
            "end_pos": chunk_end,
            "chunk_id": i,
            "method": f"chonkie_{chunker_type}",
            "token_count": getattr(chunk, 'token_count', len(chunk_text.split())),
            "char_count": len(chunk_text),
            "semantic_score": getattr(chunk, 'similarity_score', 1.0)
        }
        
        # Add chunker-specific metadata
        if chunker_type == "semantic":
            chunk_dict["semantic_coherence"] = getattr(chunk, 'coherence_score', 1.0)
        elif chunker_type == "recursive":
            chunk_dict["hierarchy_level"] = getattr(chunk, 'level', 0)
        elif chunker_type == "late":
            chunk_dict["global_context_preserved"] = True
            
        chunks.append(chunk_dict)
        current_pos = chunk_end
    
    return chunks


def _create_forced_chunks(text: str, chunker_type: str, target_chunk_size: int, overlap_percent: float, chunking_config) -> List[dict]:
    """Create chunks using a specific forced chunker type"""
    try:
        from chonkie import SemanticChunker, RecursiveChunker, SentenceChunker, LateChunker, TokenChunker
        
        overlap_chars = int(target_chunk_size * overlap_percent / 100)
        semantic_threshold = chunking_config.semantic_threshold if chunking_config else 0.5
        
        if chunker_type == "semantic":
            chunker = SemanticChunker(
                chunk_size=target_chunk_size,
                threshold=semantic_threshold
            )
        elif chunker_type == "recursive":
            from chonkie import RecursiveRules, SplitRule
            separators = ["\n\n\n", "\n\n", "\n```", "```\n", "\n#", ". ", "! ", "? ", "; ", "\n", " "]
            rules = RecursiveRules([SplitRule(sep) for sep in separators])
            chunker = RecursiveChunker(
                chunk_size=target_chunk_size,
                rules=rules
            )
        elif chunker_type == "sentence":
            chunker = SentenceChunker(
                chunk_size=target_chunk_size,
                chunk_overlap=overlap_chars
            )
        elif chunker_type == "late":
            chunker = LateChunker(
                chunk_size=target_chunk_size,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2"
            )
        elif chunker_type == "token":
            from chonkie import TokenChunker
            chunker = TokenChunker(
                chunk_size=target_chunk_size // 4  # Rough char to token conversion
            )
        else:
            # Default to sentence chunker
            chunker = SentenceChunker(
                chunk_size=target_chunk_size, 
                chunk_overlap=overlap_chars
            )
            chunker_type = "sentence_default"
        
        # Create chunks
        chonkie_chunks = chunker(text)
        
        # Convert to our format
        chunks = []
        current_pos = 0
        
        for i, chunk in enumerate(chonkie_chunks):
            chunk_text = chunk.text
            chunk_start = text.find(chunk_text, current_pos)
            
            if chunk_start == -1:
                chunk_start = current_pos
                
            chunk_end = chunk_start + len(chunk_text)
            
            chunks.append({
                "text": chunk_text,
                "start_pos": chunk_start,
                "end_pos": chunk_end,
                "chunk_id": i,
                "method": f"chonkie_{chunker_type}_forced",
                "token_count": getattr(chunk, 'token_count', len(chunk_text.split())),
                "char_count": len(chunk_text),
                "semantic_score": getattr(chunk, 'similarity_score', 1.0),
                "forced_chunker": True
            })
            
            current_pos = chunk_end
        
        return chunks
        
    except Exception as e:
        print(f"[WARNING] Forced chunker '{chunker_type}' failed: {e}, falling back to sentence-based")
        return _create_sentence_based_chunks(text, target_chunk_size, overlap_percent)


def _create_sentence_based_chunks(text: str, target_chunk_size: int, overlap_percent: float) -> List[dict]:
    """Fallback sentence-based chunking (original implementation)"""
    # Calculate overlap size
    overlap_size = int(target_chunk_size * (overlap_percent / 100))
    
    # Split text into sentences for better chunking boundaries
    sentences = split_into_sentences(text)
    if not sentences:
        # Fallback to word-based chunking
        return _create_word_based_chunks(text, target_chunk_size, overlap_size)
    
    chunks = []
    current_chunk = ""
    current_start = 0
    chunk_id = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        
        # Check if adding this sentence would exceed target size
        if current_chunk and len(current_chunk + " " + sentence) > target_chunk_size:
            # Finalize current chunk
            end_pos = current_start + len(current_chunk)
            chunks.append({
                "text": current_chunk.strip(),
                "start_pos": current_start,
                "end_pos": end_pos,
                "chunk_id": chunk_id,
                "method": "sentence_based",
                "sentence_count": len([s for s in current_chunk.split('.') if s.strip()])
            })
            
            # Start new chunk with overlap
            overlap_text = ""
            if overlap_size > 0 and current_chunk:
                # Find overlap by going backwards from current position
                overlap_chars = min(overlap_size, len(current_chunk))
                overlap_start = len(current_chunk) - overlap_chars
                overlap_text = current_chunk[overlap_start:].strip()
                
                # Try to start overlap at sentence boundary
                overlap_sentences = overlap_text.split('.')
                if len(overlap_sentences) > 1:
                    overlap_text = '.'.join(overlap_sentences[1:]).strip()
            
            current_chunk = overlap_text + (" " if overlap_text else "") + sentence
            current_start = end_pos - len(overlap_text)
            chunk_id += 1
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        
        i += 1
    
    # Add final chunk if any content remains
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "start_pos": current_start,
            "end_pos": current_start + len(current_chunk),
            "chunk_id": chunk_id,
            "method": "sentence_based",
            "sentence_count": len([s for s in current_chunk.split('.') if s.strip()])
        })
    
    return chunks


def _create_word_based_chunks(text: str, target_chunk_size: int, overlap_size: int) -> List[dict]:
    """Fallback word-based chunking when sentence splitting fails"""
    words = text.split()
    chunks = []
    
    # Estimate words per chunk (average 5 chars per word)
    words_per_chunk = target_chunk_size // 5
    overlap_words = overlap_size // 5
    
    chunk_id = 0
    for i in range(0, len(words), words_per_chunk - overlap_words):
        chunk_words = words[i:i + words_per_chunk]
        if len(chunk_words) >= 10:  # Minimum viable chunk
            chunk_text = ' '.join(chunk_words)
            start_pos = len(' '.join(words[:i]))
            
            chunks.append({
                "text": chunk_text,
                "start_pos": start_pos,
                "end_pos": start_pos + len(chunk_text),
                "chunk_id": chunk_id,
                "word_count": len(chunk_words)
            })
            chunk_id += 1
    
    return chunks


def predict_optimal_questions(text: str, eval_type: str = None) -> dict:
    """
    Predict optimal number of questions based on text characteristics.
    
    Args:
        text: Input corpus text
        eval_type: Optional evaluation type hint
        
    Returns:
        Dictionary with suggested question counts and reasoning
    """
    if not text:
        return {"suggested": 10, "min": 5, "max": 20, "reasoning": "Default for empty text"}
    
    # Basic text metrics
    word_count = len(text.split())
    sentence_count = len(split_into_sentences(text))
    char_count = len(text)
    
    # Content complexity indicators
    has_math = contains_math_patterns(text)
    has_code = contains_code_patterns(text)
    entities = extract_named_entities_simple(text)
    keywords = extract_keywords(text, max_keywords=100)
    
    # Base calculation: roughly 1 question per 200-300 words
    base_questions = max(5, min(200, word_count // 250))
    
    # Adjust based on content type
    if has_math:
        # Mathematical content: fewer questions but more focused
        base_questions = int(base_questions * 0.7)
        content_factor = "mathematical content (focused questions)"
    elif has_code:
        # Code content: moderate number, function/concept based
        base_questions = int(base_questions * 0.8)
        content_factor = "code content (concept-based questions)"
    elif len(entities) > 20:
        # Entity-rich content: more factual questions possible
        base_questions = int(base_questions * 1.2)
        content_factor = "entity-rich content (factual questions)"
    elif sentence_count > word_count // 10:
        # Short sentences: likely technical/factual content
        base_questions = int(base_questions * 1.1)
        content_factor = "technical/factual content"
    else:
        content_factor = "general content"
    
    # Complexity adjustment
    unique_words = len(set(text.lower().split()))
    vocabulary_richness = unique_words / max(1, word_count)
    
    if vocabulary_richness > 0.6:
        # High vocabulary diversity suggests complex content
        base_questions = int(base_questions * 1.15)
        complexity_note = "high vocabulary diversity"
    elif vocabulary_richness < 0.3:
        # Low diversity might be repetitive or simple
        base_questions = int(base_questions * 0.9)
        complexity_note = "repetitive content"
    else:
        complexity_note = "balanced vocabulary"
    
    # Final bounds
    suggested = max(5, min(200, base_questions))
    min_recommended = max(3, suggested // 2)
    max_recommended = min(300, suggested * 2)
    
    # Reasoning explanation
    reasoning_parts = [
        f"Based on {word_count:,} words ({sentence_count} sentences)",
        f"Content type: {content_factor}",
        f"Complexity: {complexity_note}",
        f"Unique concepts: ~{len(keywords)}"
    ]
    
    if has_math or has_code:
        reasoning_parts.append("Adjusted for technical content")
    
    return {
        "suggested": suggested,
        "min": min_recommended,
        "max": max_recommended,
        "reasoning": "; ".join(reasoning_parts),
        "text_stats": {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "char_count": char_count,
            "unique_words": unique_words,
            "vocabulary_richness": round(vocabulary_richness, 3),
            "has_math": has_math,
            "has_code": has_code,
            "entity_count": len(entities),
            "keyword_count": len(keywords)
        }
    }
    
    # Add performance metrics if chunking was requested
    # Note: This is a prediction function, not actual chunking, so no performance metrics here


def _calculate_chunking_performance_metrics(original_text: str, chunks: List[dict]) -> dict:
    """Calculate performance metrics for chunking quality assessment"""
    if not chunks:
        return {}
    
    total_chars = len(original_text)
    chunk_sizes = [len(chunk["text"]) for chunk in chunks]
    
    # Size distribution metrics
    size_stats = {
        "mean_size": sum(chunk_sizes) / len(chunk_sizes),
        "size_std": (sum((size - sum(chunk_sizes)/len(chunk_sizes))**2 for size in chunk_sizes) / len(chunk_sizes))**0.5,
        "size_coefficient_of_variation": 0
    }
    
    if size_stats["mean_size"] > 0:
        size_stats["size_coefficient_of_variation"] = size_stats["size_std"] / size_stats["mean_size"]
    
    # Coverage and efficiency
    coverage_metrics = {
        "total_coverage_chars": sum(chunk_sizes),
        "coverage_efficiency": sum(chunk_sizes) / total_chars if total_chars > 0 else 0,
        "chunk_density": len(chunks) / (total_chars / 1000) if total_chars > 0 else 0,  # chunks per 1k chars
    }
    
    # Quality indicators
    quality_metrics = {
        "optimal_size_ratio": sum(1 for size in chunk_sizes if 2000 <= size <= 4000) / len(chunk_sizes),
        "undersized_chunks": sum(1 for size in chunk_sizes if size < 1000),
        "oversized_chunks": sum(1 for size in chunk_sizes if size > 5000),
        "avg_semantic_score": sum(chunk.get("semantic_score", 1.0) for chunk in chunks) / len(chunks)
    }
    
    # Method performance
    methods = [chunk.get("method", "unknown") for chunk in chunks]
    method_stats = {
        "primary_method": max(set(methods), key=methods.count) if methods else "unknown",
        "method_consistency": methods.count(max(set(methods), key=methods.count)) / len(methods) if methods else 0,
        "uses_advanced_chunking": any(method.startswith("chonkie_") for method in methods)
    }
    
    return {
        "size_distribution": size_stats,
        "coverage": coverage_metrics,
        "quality": quality_metrics,
        "method_performance": method_stats,
        "total_chunks": len(chunks),
        "timestamp": time.time()
    }