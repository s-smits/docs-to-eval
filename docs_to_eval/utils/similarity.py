"""
Similarity calculation utilities for evaluation framework
"""

import re
from typing import List, Tuple, Dict, Optional
from collections import Counter
import math

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def token_overlap_similarity(text1: str, text2: str) -> float:
    """Calculate token overlap similarity between two texts"""
    if not text1 or not text2:
        return 0.0
    
    # Tokenize and normalize
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    return intersection / union if union > 0 else 0.0


def character_overlap_similarity(text1: str, text2: str) -> float:
    """Calculate character-level overlap similarity"""
    if not text1 or not text2:
        return 0.0
    
    chars1 = Counter(text1.lower())
    chars2 = Counter(text2.lower())
    
    # Calculate intersection
    intersection = sum((chars1 & chars2).values())
    union = sum((chars1 | chars2).values())
    
    return intersection / union if union > 0 else 0.0


def levenshtein_similarity(text1: str, text2: str) -> float:
    """Calculate Levenshtein distance-based similarity"""
    if RAPIDFUZZ_AVAILABLE:
        return fuzz.ratio(text1, text2) / 100.0
    else:
        # Fallback implementation
        return _levenshtein_similarity_fallback(text1, text2)


def _levenshtein_similarity_fallback(text1: str, text2: str) -> float:
    """Fallback Levenshtein similarity implementation"""
    if not text1 or not text2:
        return 0.0
    
    if text1 == text2:
        return 1.0
    
    # Simple character-based similarity as fallback
    return character_overlap_similarity(text1, text2)


def ngram_similarity(text1: str, text2: str, n: int = 2) -> float:
    """Calculate n-gram similarity between two texts"""
    def get_ngrams(text: str, n: int) -> List[str]:
        text = text.lower()
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    if not text1 or not text2:
        return 0.0
    
    ngrams1 = set(get_ngrams(text1, n))
    ngrams2 = set(get_ngrams(text2, n))
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    
    return intersection / union if union > 0 else 0.0


def rouge_l_similarity(text1: str, text2: str) -> float:
    """Calculate ROUGE-L similarity (Longest Common Subsequence)"""
    def lcs_length(x: str, y: str) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    if not text1 or not text2:
        return 0.0
    
    # Tokenize
    tokens1 = text1.lower().split()
    tokens2 = text2.lower().split()
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # Calculate LCS on token level
    lcs_len = lcs_length(tokens1, tokens2)
    
    # ROUGE-L formula
    if len(tokens1) + len(tokens2) == 0:
        return 0.0
    
    recall = lcs_len / len(tokens2) if len(tokens2) > 0 else 0.0
    precision = lcs_len / len(tokens1) if len(tokens1) > 0 else 0.0
    
    if recall + precision == 0:
        return 0.0
    
    f1 = 2 * (recall * precision) / (recall + precision)
    return f1


def bleu_similarity_simple(reference: str, candidate: str, n: int = 4) -> float:
    """Simple BLEU score approximation"""
    if not reference or not candidate:
        return 0.0
    
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if not ref_tokens or not cand_tokens:
        return 0.0
    
    # Calculate n-gram precisions
    precisions = []
    
    for i in range(1, min(n + 1, len(cand_tokens) + 1)):
        ref_ngrams = Counter([' '.join(ref_tokens[j:j+i]) for j in range(len(ref_tokens) - i + 1)])
        cand_ngrams = Counter([' '.join(cand_tokens[j:j+i]) for j in range(len(cand_tokens) - i + 1)])
        
        overlap = sum((ref_ngrams & cand_ngrams).values())
        total = sum(cand_ngrams.values())
        
        if total > 0:
            precisions.append(overlap / total)
        else:
            precisions.append(0.0)
    
    if not precisions:
        return 0.0
    
    # Geometric mean of precisions
    geometric_mean = 1.0
    for p in precisions:
        geometric_mean *= max(p, 1e-10)  # Avoid zero
    
    geometric_mean = geometric_mean ** (1.0 / len(precisions))
    
    # Brevity penalty
    bp = min(1.0, len(cand_tokens) / len(ref_tokens)) if len(ref_tokens) > 0 else 0.0
    
    return bp * geometric_mean


def semantic_similarity_mock(text1: str, text2: str) -> float:
    """Mock semantic similarity (placeholder for real embeddings)"""
    # This is a placeholder - in production, you'd use sentence transformers
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        return semantic_similarity_real(text1, text2)
    else:
        # Fallback to combined token and n-gram similarity
        token_sim = token_overlap_similarity(text1, text2)
        ngram_sim = ngram_similarity(text1, text2)
        return (token_sim + ngram_sim) / 2


def semantic_similarity_real(text1: str, text2: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """Real semantic similarity using sentence transformers"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return semantic_similarity_mock(text1, text2)
    
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode([text1, text2])
        
        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        # Convert to 0-1 range
        return (similarity + 1) / 2
        
    except Exception:
        # Fallback to mock if model loading fails
        return semantic_similarity_mock(text1, text2)


def calculate_similarity(text1: str, text2: str, method: str = "token_overlap") -> float:
    """Calculate similarity using specified method"""
    methods = {
        "token_overlap": token_overlap_similarity,
        "character_overlap": character_overlap_similarity,
        "levenshtein": levenshtein_similarity,
        "ngram": ngram_similarity,
        "rouge_l": rouge_l_similarity,
        "bleu": bleu_similarity_simple,
        "semantic": semantic_similarity_mock,
    }
    
    if method not in methods:
        raise ValueError(f"Unknown similarity method: {method}")
    
    return methods[method](text1, text2)


def calculate_multi_similarity(text1: str, text2: str) -> Dict[str, float]:
    """Calculate multiple similarity metrics"""
    methods = ["token_overlap", "character_overlap", "ngram", "rouge_l", "bleu"]
    
    if RAPIDFUZZ_AVAILABLE:
        methods.append("levenshtein")
    
    results = {}
    for method in methods:
        try:
            results[method] = calculate_similarity(text1, text2, method)
        except Exception as e:
            results[method] = 0.0
    
    # Add semantic similarity
    try:
        results["semantic"] = semantic_similarity_mock(text1, text2)
    except Exception:
        results["semantic"] = 0.0
    
    return results


def find_best_similarity_method(text_pairs: List[Tuple[str, str]], ground_truth_scores: List[float]) -> str:
    """Find the best similarity method for given text pairs and ground truth scores"""
    if len(text_pairs) != len(ground_truth_scores):
        raise ValueError("Number of text pairs must match number of ground truth scores")
    
    methods = ["token_overlap", "character_overlap", "ngram", "rouge_l", "bleu", "semantic"]
    if RAPIDFUZZ_AVAILABLE:
        methods.append("levenshtein")
    
    best_method = "token_overlap"
    best_correlation = -1
    
    for method in methods:
        try:
            predicted_scores = [calculate_similarity(pair[0], pair[1], method) for pair in text_pairs]
            correlation = pearson_correlation(predicted_scores, ground_truth_scores)
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_method = method
        except Exception:
            continue
    
    return best_method


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient"""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum(xi * xi for xi in x)
    sum_yy = sum(yi * yi for yi in y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def get_similarity_confidence(score: float, method: str = "token_overlap") -> float:
    """Get confidence level for similarity score based on method"""
    # Different methods have different reliability characteristics
    confidence_factors = {
        "token_overlap": 0.8,
        "character_overlap": 0.6,
        "levenshtein": 0.9,
        "ngram": 0.7,
        "rouge_l": 0.85,
        "bleu": 0.75,
        "semantic": 0.95,
    }
    
    base_confidence = confidence_factors.get(method, 0.7)
    
    # Adjust confidence based on score
    if score > 0.8:
        return min(1.0, base_confidence + 0.1)
    elif score < 0.2:
        return max(0.1, base_confidence - 0.2)
    else:
        return base_confidence