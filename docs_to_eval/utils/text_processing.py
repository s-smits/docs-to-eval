"""
Text processing utilities for evaluation framework
"""

import re
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