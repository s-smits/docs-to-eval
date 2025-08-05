"""
Text processing utilities for evaluation framework
"""

import os
# Force CPU usage to avoid MPS compatibility issues on Apple Silicon
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['SENTENCE_TRANSFORMERS_DEVICE'] = 'cpu'

import re
from typing import List, Optional, Dict, Any
from .config import ChunkingConfig


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip())


def extract_numbers(text: str) -> List[str]:
    """Extract numbers from text"""
    return re.findall(r'-?\d+\.?\d*', text)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (lm-eval-harness style)"""
    if not answer:
        return ""
    
    answer = clean_text(answer.lower())
    # Remove punctuation except for periods in numbers
    answer = re.sub(r'[^\w\s\.]', '', answer)
    # Remove extra whitespace
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer


class AnswerExtractor:
    """Answer extraction filters following lm-eval-harness patterns"""
    
    @staticmethod
    def extract_multiple_choice(text: str) -> str:
        """Extract multiple choice answer (A, B, C, D)"""
        # Look for patterns like "The answer is A" or "A)" or "(A)"
        patterns = [
            r'(?:answer is|answer:|^)\s*([ABCD])\b',
            r'\b([ABCD])\)',  # A)
            r'\(([ABCD])\)',  # (A)
            r'^([ABCD])\.',   # A.
            r'\b([ABCD])\b(?=\s|$)',  # Standalone A, B, C, D
        ]
        
        text_clean = text.strip().upper()
        for pattern in patterns:
            match = re.search(pattern, text_clean, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        
        # Fallback: look for first A/B/C/D in text
        first_letter = re.search(r'\b([ABCD])\b', text_clean)
        return first_letter.group(1) if first_letter else text[:1].upper()
    
    @staticmethod
    def extract_numerical_answer(text: str) -> str:
        """Extract numerical answer from text"""
        # Common patterns for numerical answers
        patterns = [
            r'(?:answer is|answer:|equals?|=)\s*([+-]?(?:\d+\.?\d*|\.\d+))',
            r'(?:^|\s)([+-]?(?:\d+\.?\d*|\.\d+))(?:\s|$|\.)',
            r'\$([+-]?(?:\d+\.?\d*|\.\d+))',  # Currency
            r'([+-]?(?:\d+\.?\d*|\.\d+))%',   # Percentage
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Fallback: extract first number
        numbers = extract_numbers(text)
        return numbers[0] if numbers else ""
    
    @staticmethod
    def extract_yes_no(text: str) -> str:
        """Extract yes/no answer"""
        text_clean = text.lower().strip()
        
        # Look for clear yes/no indicators
        if re.search(r'\b(yes|true|correct|right)\b', text_clean):
            return "yes"
        elif re.search(r'\b(no|false|incorrect|wrong)\b', text_clean):
            return "no"
        
        # Look for positive/negative sentiment
        positive_words = ['agree', 'support', 'accept', 'approve']
        negative_words = ['disagree', 'reject', 'deny', 'oppose']
        
        for word in positive_words:
            if word in text_clean:
                return "yes"
        for word in negative_words:
            if word in text_clean:
                return "no"
        
        return text_clean[:3]  # Return first few chars as fallback
    
    @staticmethod
    def extract_short_answer(text: str, max_words: int = 5) -> str:
        """Extract short answer (for reading comprehension tasks)"""
        # Remove common prefixes
        prefixes_to_remove = [
            r'^(?:the answer is|answer:|it is|this is|that is)\s*',
            r'^(?:based on|according to).*?[,:]?\s*',
        ]
        
        cleaned = text.strip()
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)
        
        # Extract first sentence or up to max_words
        words = cleaned.split()
        if len(words) <= max_words:
            return cleaned
        
        # Try to get first complete phrase/sentence
        first_sentence = re.split(r'[.!?]', cleaned)[0]
        if len(first_sentence.split()) <= max_words:
            return first_sentence.strip()
        
        # Return first max_words
        return ' '.join(words[:max_words])


class FilterPipeline:
    """Configurable filter pipeline for answer processing following lm-eval-harness"""
    
    def __init__(self):
        self.filters = []
        
    def add_filter(self, filter_func, **kwargs):
        """Add a filter function to the pipeline"""
        self.filters.append((filter_func, kwargs))
        
    def process(self, text: str) -> str:
        """Process text through all filters in sequence"""
        result = text
        for filter_func, kwargs in self.filters:
            result = filter_func(result, **kwargs)
        return result
    
    @classmethod
    def create_multiple_choice_pipeline(cls):
        """Create pipeline for multiple choice tasks"""
        pipeline = cls()
        pipeline.add_filter(normalize_answer)
        pipeline.add_filter(AnswerExtractor.extract_multiple_choice)
        return pipeline
    
    @classmethod
    def create_numerical_pipeline(cls):
        """Create pipeline for numerical tasks"""
        pipeline = cls()
        pipeline.add_filter(normalize_answer)
        pipeline.add_filter(AnswerExtractor.extract_numerical_answer)
        return pipeline
    
    @classmethod
    def create_short_answer_pipeline(cls, max_words: int = 5):
        """Create pipeline for short answer tasks"""
        pipeline = cls()
        pipeline.add_filter(AnswerExtractor.extract_short_answer, max_words=max_words)
        pipeline.add_filter(normalize_answer)
        return pipeline
    
    @classmethod
    def create_exact_match_pipeline(cls):
        """Create pipeline for exact match tasks"""
        pipeline = cls()
        pipeline.add_filter(normalize_answer)
        return pipeline


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
    
    # Generic/vague words to filter out for better domain focus
    generic_words = {
        'such', 'find', 'sources', 'article', 'citations', 'information',
        'content', 'text', 'data', 'example', 'case', 'way', 'time', 'place',
        'thing', 'part', 'people', 'work', 'life', 'world', 'year', 'day',
        'man', 'woman', 'child', 'person', 'group', 'number', 'part', 'way',
        'back', 'see', 'get', 'make', 'come', 'give', 'take', 'use', 'look',
        'know', 'want', 'say', 'tell', 'ask', 'feel', 'try', 'leave', 'put'
    }
    
    # Filter and count
    word_freq = {}
    for word in words:
        if word not in stop_words and word not in generic_words:
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


def create_smart_chunks_from_files(file_contents: List[Dict[str, str]], chunking_config: Optional[ChunkingConfig] = None) -> List[Dict[str, Any]]:
    """
    Create smart chunks from multiple files, concatenating them to reach optimal chunk sizes
    
    Args:
        file_contents: List of dicts with 'filename' and 'content' keys
        chunking_config: Configuration for chunking behavior
        
    Returns:
        List of chunk dictionaries with text, metadata, and semantic information
    """
    if not file_contents:
        return []
    
    # Use default config if none provided
    if chunking_config is None:
        chunking_config = ChunkingConfig()
    
    # Concatenate files intelligently to reach target token size
    concatenated_chunks = _concatenate_files_to_target_size(file_contents, chunking_config)
    
    # Now apply smart chunking to each concatenated segment
    all_chunks = []
    for i, concat_chunk in enumerate(concatenated_chunks):
        chunks = create_smart_chunks(concat_chunk['text'], chunking_config)
        
        # Update metadata to include file source information
        for chunk in chunks:
            chunk['file_sources'] = concat_chunk['sources']
            chunk['concatenated_chunk_index'] = i
            chunk['metadata']['file_sources'] = concat_chunk['sources']
            chunk['metadata']['is_multi_file'] = len(concat_chunk['sources']) > 1
        
        all_chunks.extend(chunks)
    
    return all_chunks


def create_smart_chunks(text: str, chunking_config: Optional[ChunkingConfig] = None) -> List[Dict[str, Any]]:
    """
    Create smart chunks from text using various strategies including chonkie integration
    
    Args:
        text: Input text to chunk
        chunking_config: Configuration for chunking behavior
        
    Returns:
        List of chunk dictionaries with text, metadata, and semantic information
    """
    if not text or not text.strip():
        return []
    
    # Use default config if none provided
    if chunking_config is None:
        chunking_config = ChunkingConfig()
    
    # Try chonkie integration first if enabled
    if chunking_config.enable_chonkie:
        try:
            return _create_chonkie_chunks(text, chunking_config)
        except ImportError:
            print("⚠️ Chonkie not available, falling back to simple chunking")
            print("💡 Install chonkie for advanced semantic chunking: pip install chonkie")
        except Exception as e:
            error_msg = str(e)
            if "aten::_embedding_bag" in error_msg and "MPS device" in error_msg:
                print(f"⚠️ MPS device compatibility issue detected: {error_msg}")
                print("💡 This is an Apple Silicon compatibility issue with PyTorch embeddings")
                print("✅ The system has automatically set PYTORCH_ENABLE_MPS_FALLBACK=1 to fix this")
                print("🔧 Falling back to simple chunking for now")
            else:
                print(f"⚠️ Chonkie chunking failed: {e}, falling back to simple chunking")
    
    # Fallback to simple chunking
    return _create_simple_chunks(text, chunking_config)


def _concatenate_files_to_target_size(file_contents: List[Dict[str, str]], config: ChunkingConfig) -> List[Dict[str, Any]]:
    """
    Concatenate files together to reach target token size (~3k tokens per chunk)
    
    Args:
        file_contents: List of dicts with 'filename' and 'content' keys
        config: Chunking configuration
        
    Returns:
        List of concatenated chunks with metadata about source files
    """
    concatenated_chunks = []
    current_chunk = ""
    current_sources = []
    
    # Estimate target size in characters (3k tokens ≈ ~11k characters)
    target_size = config.target_token_size * 3.7 if config.use_token_chunking else config.target_chunk_size
    max_size = config.max_token_size * 3.7 if config.use_token_chunking else config.max_chunk_size
    
    # Try to load tiktoken for accurate token counting
    token_counter = None
    if config.use_token_chunking:
        try:
            from tiktoken import get_encoding
            token_counter = get_encoding("cl100k_base")
        except ImportError:
            print("⚠️ tiktoken not available, using character approximation")
    
    def get_size(text: str) -> int:
        """Get actual token count or character count"""
        if token_counter and config.use_token_chunking:
            try:
                return len(token_counter.encode(text))
            except (AttributeError, TypeError) as e:
                # Token encoding failed, use character approximation
                return len(text) // 4  # Fallback approximation
        return len(text)
    
    def should_add_file(current_text: str, new_content: str) -> bool:
        """Check if adding this file would exceed max size"""
        combined_size = get_size(current_text + "\n\n" + new_content)
        return combined_size <= max_size
    
    for file_info in file_contents:
        filename = file_info.get('filename', 'unknown')
        content = file_info.get('content', '').strip()
        
        if not content:
            continue
        
        # If current chunk is empty, start with this file
        if not current_chunk:
            current_chunk = content
            current_sources = [filename]
            continue
        
        # Check if we can add this file to current chunk
        if should_add_file(current_chunk, content):
            # Add file to current chunk
            current_chunk += f"\n\n============================================================\nFILE: {filename}\n{content}"
            current_sources.append(filename)
        else:
            # Current chunk is big enough, finalize it
            if current_chunk:
                chunk_size = get_size(current_chunk)
                concatenated_chunks.append({
                    'text': current_chunk,
                    'sources': current_sources.copy(),
                    'size': chunk_size,
                    'is_target_size': chunk_size >= (target_size * 0.8),  # Within 80% of target
                    'file_count': len(current_sources)
                })
                
                print(f"📁 Created concatenated chunk: {len(current_sources)} files, "
                      f"{chunk_size} {'tokens' if config.use_token_chunking else 'chars'}")
            
            # Start new chunk with current file
            current_chunk = content
            current_sources = [filename]
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_size = get_size(current_chunk)
        concatenated_chunks.append({
            'text': current_chunk,
            'sources': current_sources,
            'size': chunk_size,
            'is_target_size': chunk_size >= (target_size * 0.8),
            'file_count': len(current_sources)
        })
        
        print(f"📁 Created final concatenated chunk: {len(current_sources)} files, "
              f"{chunk_size} {'tokens' if config.use_token_chunking else 'chars'}")
    
    print(f"✅ File concatenation complete: {len(concatenated_chunks)} chunks from {len(file_contents)} files")
    return concatenated_chunks


def _detect_device_compatibility() -> Dict[str, Any]:
    """Detect device compatibility and provide recommendations"""
    import platform
    import os
    
    info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "is_apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64",
        "mps_fallback_set": os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1',
        "recommendations": []
    }
    
    if info["is_apple_silicon"]:
        info["recommendations"].append("Apple Silicon detected - MPS fallback enabled for embeddings")
        if not info["mps_fallback_set"]:
            info["recommendations"].append("Consider setting PYTORCH_ENABLE_MPS_FALLBACK=1 for better compatibility")
    
    return info


def _create_chonkie_chunks(text: str, config: ChunkingConfig) -> List[Dict[str, Any]]:
    """Create chunks using chonkie library for semantic chunking"""
    try:
        # Set PyTorch MPS fallback to CPU for embedding operations on Apple Silicon
        import os
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        
        # Detect device compatibility
        device_info = _detect_device_compatibility()
        if device_info["is_apple_silicon"]:
            print(f"🍎 Apple Silicon detected - MPS fallback enabled for PyTorch embeddings")
        
        from chonkie import SemanticChunker, SentenceChunker, RecursiveChunker, TokenChunker
        
        # Use token-aware SEMANTIC chunking (smart chunking within 2k-4k token range)
        if config.use_token_chunking:
            try:
                # Try to use semantic chunking with token awareness
                # Approximate character count for target token range (1 token ≈ 3.5-4 chars average)
                target_chars = int(config.target_token_size * 3.7)  # ~3k tokens = ~11k chars
                max_chars = int(config.max_token_size * 3.7)        # ~4k tokens = ~15k chars  
                min_chars = int(config.min_token_size * 3.7)        # ~2k tokens = ~7.5k chars
                overlap_chars = int(config.overlap_tokens * 3.7)    # ~300 tokens = ~1.1k chars
                
                # Use SEMANTIC chunking with token-approximated sizes
                if config.force_chunker == "semantic" or config.chunking_strategy == "semantic":
                    chunker = SemanticChunker(
                        chunk_size=target_chars,  # Smart semantic chunking at ~3k tokens
                        # Note: SemanticChunker doesn't support overlap parameter
                    )
                    method = "chonkie_semantic_token_aware"
                elif config.force_chunker == "recursive":
                    chunker = RecursiveChunker(
                        chunk_size=target_chars,  # Smart recursive chunking at ~3k tokens
                        overlap=overlap_chars
                    )
                    method = "chonkie_recursive_token_aware"  
                else:
                    # Try advanced SDPM (Semantic Double-Pass Merge) for smartest chunking
                    try:
                        from chonkie import SDPMChunker
                        chunker = SDPMChunker(
                            chunk_size=target_chars,
                            # Note: Need to check SDPM parameters
                        )
                        method = "chonkie_sdpm_token_aware"
                    except ImportError:
                        # Fallback to semantic
                        chunker = SemanticChunker(
                            chunk_size=target_chars,
                            # Note: SemanticChunker doesn't support overlap parameter
                        )
                        method = "chonkie_semantic_token_aware"
                
            except ImportError:
                print("⚠️ Advanced chonkie chunkers not available, falling back to character-based chunking")
                # Fallback to character-based with larger sizes to approximate tokens
                char_size = config.target_token_size * 4
                char_overlap = config.overlap_tokens * 4
                
                if config.force_chunker == "semantic" or config.chunking_strategy == "semantic":
                    chunker = SemanticChunker(
                        chunk_size=char_size,
                        # Note: SemanticChunker doesn't support overlap parameter
                    )
                    method = "chonkie_semantic_fallback"
                else:
                    chunker = RecursiveChunker(
                        chunk_size=char_size,
                        overlap=char_overlap
                    )
                    method = "chonkie_recursive_fallback"
        else:
            # Traditional character-based chunking
            if config.force_chunker == "semantic" or config.chunking_strategy == "semantic":
                chunker = SemanticChunker(
                    chunk_size=config.target_chunk_size,
                    # Note: SemanticChunker doesn't support overlap parameter
                )
                method = "chonkie_semantic"
            elif config.force_chunker == "recursive":
                chunker = RecursiveChunker(
                    chunk_size=config.target_chunk_size,
                    overlap=config.overlap_size
                )
                method = "chonkie_recursive"
            elif config.force_chunker == "sentence":
                chunker = SentenceChunker(
                    chunk_size=config.target_chunk_size,
                    overlap=config.overlap_size
                )
                method = "chonkie_sentence"
            else:
                # Default to semantic for best results
                chunker = SemanticChunker(
                    chunk_size=config.target_chunk_size,
                    # Note: SemanticChunker doesn't support overlap parameter
                )
                method = "chonkie_semantic"
        
        # Perform chunking
        raw_chunks = chunker.chunk(text)
        
        # Convert to our format with metadata and token validation
        chunks = []
        
        # Try to load tiktoken for accurate token counting
        token_counter = None
        if config.use_token_chunking:
            try:
                from tiktoken import get_encoding
                token_counter = get_encoding("cl100k_base")  # Use GPT-4 tokenizer for validation
            except ImportError:
                print("⚠️ tiktoken not available for token validation")
        
        for i, chunk_text in enumerate(raw_chunks):
            # Calculate actual token count if available
            actual_tokens = None
            token_status = "estimated"
            
            if token_counter:
                try:
                    actual_tokens = len(token_counter.encode(chunk_text))
                    token_status = "exact"
                    
                    # Validate token range (2k-4k tokens)
                    if config.use_token_chunking:
                        if actual_tokens < config.min_token_size:
                            print(f"⚠️ Chunk {i} below min tokens: {actual_tokens} < {config.min_token_size}")
                        elif actual_tokens > config.max_token_size:
                            print(f"⚠️ Chunk {i} above max tokens: {actual_tokens} > {config.max_token_size}")
                        else:
                            print(f"✅ Chunk {i} within token range: {actual_tokens} tokens ({len(chunk_text)} chars)")
                            
                except Exception as e:
                    print(f"⚠️ Token counting failed for chunk {i}: {e}")
                    actual_tokens = len(chunk_text) // 4  # Rough approximation
            else:
                # Estimate tokens from characters (rough approximation)
                actual_tokens = len(chunk_text) // 4
            
            # Calculate semantic coherence score (simplified)
            semantic_score = _calculate_semantic_score(chunk_text, text)
            
            chunk_dict = {
                'text': chunk_text,
                'index': i,
                'method': method,
                'size': len(chunk_text),
                'token_count': actual_tokens,
                'token_status': token_status,
                'semantic_score': semantic_score,
                'word_count': len(chunk_text.split()),
                'is_smart_chunking': True,  # This is semantic/smart chunking, not fixed-size
                'token_range_valid': (
                    config.min_token_size <= actual_tokens <= config.max_token_size 
                    if config.use_token_chunking and actual_tokens else None
                ),
                'metadata': {
                    'chunking_strategy': config.chunking_strategy,
                    'chonkie_enabled': True,
                    'chunk_method': method,
                    'target_token_size': config.target_token_size if config.use_token_chunking else None,
                    'overlap_tokens': config.overlap_tokens if config.use_token_chunking else config.overlap_size
                }
            }
            chunks.append(chunk_dict)
        
        return chunks
        
    except ImportError as e:
        raise ImportError(f"Chonkie not installed: {e}")
    except Exception as e:
        raise Exception(f"Chonkie chunking failed: {e}")


def _create_simple_chunks(text: str, config: ChunkingConfig) -> List[Dict[str, Any]]:
    """Create chunks using simple text splitting strategies"""
    
    # Split by paragraphs first for better boundaries
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    chunk_index = 0
    
    for paragraph in paragraphs:
        # Check if adding this paragraph would exceed max size
        if current_chunk and len(current_chunk) + len(paragraph) > config.max_chunk_size:
            # Finalize current chunk if it meets minimum size
            if len(current_chunk) >= config.min_chunk_size:
                chunk_dict = {
                    'text': current_chunk.strip(),
                    'index': chunk_index,
                    'method': 'simple_paragraph',
                    'size': len(current_chunk),
                    'semantic_score': 1.0,  # Default score for simple chunks
                    'metadata': {
                        'chunking_strategy': config.chunking_strategy,
                        'chonkie_enabled': False,
                        'chunk_method': 'simple_paragraph'
                    }
                }
                chunks.append(chunk_dict)
                chunk_index += 1
                
                # Start new chunk with overlap if configured
                if config.overlap_size > 0:
                    overlap_text = current_chunk[-config.overlap_size:]
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Current chunk too small, add paragraph
                current_chunk += "\n\n" + paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add final chunk if it exists and meets minimum size
    if current_chunk and len(current_chunk) >= config.min_chunk_size:
        chunk_dict = {
            'text': current_chunk.strip(),
            'index': chunk_index,
            'method': 'simple_paragraph',
            'size': len(current_chunk),
            'semantic_score': 1.0,
            'metadata': {
                'chunking_strategy': config.chunking_strategy,
                'chonkie_enabled': False,
                'chunk_method': 'simple_paragraph'
            }
        }
        chunks.append(chunk_dict)
    
    # If no chunks created (text too small), create single chunk
    if not chunks and text.strip():
        chunk_dict = {
            'text': text.strip(),
            'index': 0,
            'method': 'simple_single',
            'size': len(text),
            'semantic_score': 1.0,
            'metadata': {
                'chunking_strategy': config.chunking_strategy,
                'chonkie_enabled': False,
                'chunk_method': 'simple_single'
            }
        }
        chunks.append(chunk_dict)
    
    return chunks


def _calculate_semantic_score(chunk_text: str, full_text: str) -> float:
    """Calculate a simple semantic coherence score for a chunk"""
    if not chunk_text or not full_text:
        return 1.0
    
    # Simple coherence based on keyword consistency
    chunk_keywords = set(extract_keywords(chunk_text, max_keywords=20))
    full_keywords = set(extract_keywords(full_text, max_keywords=100))
    
    if not chunk_keywords or not full_keywords:
        return 1.0
    
    # Calculate overlap ratio
    overlap = len(chunk_keywords.intersection(full_keywords))
    coherence_score = min(1.0, overlap / len(chunk_keywords))
    
    # Boost score for chunks with good sentence structure
    sentences = split_into_sentences(chunk_text)
    if len(sentences) >= 2:  # Multi-sentence chunks are generally more coherent
        coherence_score = min(1.0, coherence_score * 1.2)
    
    return round(coherence_score, 3)