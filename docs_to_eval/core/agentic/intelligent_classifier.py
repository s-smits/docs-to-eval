"""
Intelligent benchmark type classifier with multi-signal analysis
Provides sophisticated automatic determination of optimal benchmark types
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter

from ..evaluation import EvaluationType


@dataclass
class ContentSignals:
    """Multi-dimensional content analysis signals"""
    
    # Density metrics (0-1 scale)
    math_density: float
    code_density: float
    technical_density: float
    factual_density: float
    narrative_density: float
    
    # Complexity metrics
    vocabulary_diversity: float
    avg_sentence_length: float
    domain_specificity: float
    
    # Structural metrics
    has_code_blocks: bool
    has_formulas: bool
    has_lists: bool
    has_diagrams: bool
    question_count: int
    
    # Composite scores
    complexity_score: float  # 0-10
    abstractness_score: float  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'math_density': self.math_density,
            'code_density': self.code_density,
            'technical_density': self.technical_density,
            'factual_density': self.factual_density,
            'narrative_density': self.narrative_density,
            'vocabulary_diversity': self.vocabulary_diversity,
            'avg_sentence_length': self.avg_sentence_length,
            'domain_specificity': self.domain_specificity,
            'complexity_score': self.complexity_score,
            'abstractness_score': self.abstractness_score,
        }


@dataclass
class BenchmarkRecommendation:
    """Recommended benchmark configuration"""
    
    primary_type: EvaluationType
    secondary_types: List[EvaluationType]
    confidence: float
    reasoning: str
    
    # Configuration suggestions
    suggested_difficulty: str  # "basic", "intermediate", "hard", "expert"
    suggested_question_count: int
    use_agentic: bool
    
    # Quality thresholds
    min_validation_score: float
    answer_type_distribution: Dict[str, float]  # Expected distribution of answer types
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'primary_type': self.primary_type.value if hasattr(self.primary_type, 'value') else str(self.primary_type),
            'secondary_types': [t.value if hasattr(t, 'value') else str(t) for t in self.secondary_types],
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'suggested_difficulty': self.suggested_difficulty,
            'suggested_question_count': self.suggested_question_count,
            'use_agentic': self.use_agentic,
            'min_validation_score': self.min_validation_score,
            'answer_type_distribution': self.answer_type_distribution
        }


class IntelligentBenchmarkClassifier:
    """
    Advanced classifier that analyzes corpus to recommend optimal benchmark types
    
    Uses multi-signal analysis including:
    - Content patterns (mathematical, code, technical terminology)
    - Structural characteristics (lists, code blocks, formulas)
    - Linguistic complexity (vocabulary, sentence structure)
    - Domain specificity
    """
    
    def __init__(self):
        self.signal_extractors = {
            'math': self._extract_math_signals,
            'code': self._extract_code_signals,
            'technical': self._extract_technical_signals,
            'factual': self._extract_factual_signals,
            'narrative': self._extract_narrative_signals,
        }
    
    def analyze_corpus(self, corpus_text: str) -> ContentSignals:
        """Perform comprehensive multi-signal analysis"""
        
        # Extract all signals
        math_signals = self._extract_math_signals(corpus_text)
        code_signals = self._extract_code_signals(corpus_text)
        technical_signals = self._extract_technical_signals(corpus_text)
        factual_signals = self._extract_factual_signals(corpus_text)
        narrative_signals = self._extract_narrative_signals(corpus_text)
        
        # Calculate complexity metrics
        vocabulary_div = self._calculate_vocabulary_diversity(corpus_text)
        avg_sent_length = self._calculate_avg_sentence_length(corpus_text)
        domain_spec = self._calculate_domain_specificity(corpus_text)
        
        # Structural analysis
        has_code = self._detect_code_blocks(corpus_text)
        has_formulas = self._detect_formulas(corpus_text)
        has_lists = self._detect_lists(corpus_text)
        has_diagrams = self._detect_diagrams(corpus_text)
        question_count = len(re.findall(r'\?', corpus_text))
        
        # Composite scores
        complexity = self._calculate_complexity_score(
            vocabulary_div, avg_sent_length, technical_signals
        )
        abstractness = self._calculate_abstractness(corpus_text)
        
        return ContentSignals(
            math_density=math_signals,
            code_density=code_signals,
            technical_density=technical_signals,
            factual_density=factual_signals,
            narrative_density=narrative_signals,
            vocabulary_diversity=vocabulary_div,
            avg_sentence_length=avg_sent_length,
            domain_specificity=domain_spec,
            has_code_blocks=has_code,
            has_formulas=has_formulas,
            has_lists=has_lists,
            has_diagrams=has_diagrams,
            question_count=question_count,
            complexity_score=complexity,
            abstractness_score=abstractness
        )
    
    def recommend_benchmark_type(self, corpus_text: str) -> BenchmarkRecommendation:
        """Recommend optimal benchmark type with full configuration"""
        
        # Analyze corpus
        signals = self.analyze_corpus(corpus_text)
        
        # Score each evaluation type
        type_scores = self._score_evaluation_types(signals)
        
        # Get primary and secondary types
        sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
        primary_type, primary_score = sorted_types[0]
        secondary_types = [eval_type for eval_type, score in sorted_types[1:4] if score > 0.3]
        
        # Calculate confidence
        confidence = self._calculate_confidence(primary_score, sorted_types)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(primary_type, signals, type_scores)
        
        # Suggest configuration
        difficulty = self._suggest_difficulty(signals)
        question_count = self._suggest_question_count(signals, primary_type)
        use_agentic = self._should_use_agentic(primary_type, signals)
        min_validation = self._suggest_validation_threshold(primary_type)
        answer_dist = self._suggest_answer_distribution(primary_type, signals)
        
        return BenchmarkRecommendation(
            primary_type=primary_type,
            secondary_types=secondary_types,
            confidence=confidence,
            reasoning=reasoning,
            suggested_difficulty=difficulty,
            suggested_question_count=question_count,
            use_agentic=use_agentic,
            min_validation_score=min_validation,
            answer_type_distribution=answer_dist
        )
    
    def _extract_math_signals(self, text: str) -> float:
        """Extract mathematical content density"""
        
        math_patterns = [
            r'\b(equation|formula|theorem|proof|lemma|corollary)\b',
            r'\b(calculate|compute|solve|derive|integrate|differentiate)\b',
            r'\b(function|variable|constant|parameter|coefficient)\b',
            r'\b(matrix|vector|scalar|tensor|determinant)\b',
            r'[=≈≠><≤≥±∫∑∏√]',
            r'\b\d+\s*[+\-*/^]\s*\d+\b',
            r'\b(sin|cos|tan|log|exp|sqrt)\s*\(',
        ]
        
        total_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                           for pattern in math_patterns)
        word_count = len(text.split())
        
        return min(1.0, total_matches / max(word_count * 0.1, 1))
    
    def _extract_code_signals(self, text: str) -> float:
        """Extract programming/code content density"""
        
        code_patterns = [
            r'\b(def|function|class|interface|struct)\s+\w+',
            r'\b(import|require|include|using|from)\s+\w+',
            r'\b(if|else|elif|for|while|switch|case)\s*[\(\{]',
            r'\b(return|yield|break|continue|pass)\b',
            r'[a-zA-Z_]\w*\s*\([^)]*\)\s*[{:]',
            r'\b(int|str|bool|float|char|void|var|let|const)\b',
            r'[=!<>]=|[+\-*/]=|\+\+|--',
        ]
        
        # Also check for code blocks
        code_block_count = len(re.findall(r'```|`[^`]+`|^\s{4,}\w+', text, re.MULTILINE))
        
        total_matches = sum(len(re.findall(pattern, text)) for pattern in code_patterns)
        total_matches += code_block_count * 5  # Code blocks are strong signals
        
        word_count = len(text.split())
        return min(1.0, total_matches / max(word_count * 0.08, 1))
    
    def _extract_technical_signals(self, text: str) -> float:
        """Extract technical/domain-specific terminology density"""
        
        # Detect technical patterns
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b',  # CamelCase terms
            r'\b\w+[-_]\w+\b',  # Hyphenated/underscored technical terms
            r'\b(system|process|method|technique|algorithm|protocol)\b',
            r'\b(architecture|framework|infrastructure|implementation)\b',
        ]
        
        total_matches = sum(len(re.findall(pattern, text)) for pattern in technical_patterns)
        word_count = len(text.split())
        
        return min(1.0, total_matches / max(word_count * 0.15, 1))
    
    def _extract_factual_signals(self, text: str) -> float:
        """Extract factual/encyclopedic content density"""
        
        factual_patterns = [
            r'\b(is|was|are|were|has|have|had)\s+(a|an|the)\b',
            r'\b(located|founded|established|created|discovered|invented)\b',
            r'\b(in\s+\d{4}|during|between\s+\d{4})',  # Date references
            r'\b(consists of|composed of|made up of|includes)\b',
            r'\b(known as|called|named|referred to as)\b',
        ]
        
        total_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                           for pattern in factual_patterns)
        word_count = len(text.split())
        
        return min(1.0, total_matches / max(word_count * 0.1, 1))
    
    def _extract_narrative_signals(self, text: str) -> float:
        """Extract narrative/descriptive content density"""
        
        narrative_patterns = [
            r'\b(story|narrative|tale|account|describes)\b',
            r'\b(character|protagonist|antagonist|setting)\b',
            r'\b(begins|continues|concludes|ultimately)\b',
            r'\b(however|although|despite|nevertheless)\b',
        ]
        
        total_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                           for pattern in narrative_patterns)
        word_count = len(text.split())
        
        return min(1.0, total_matches / max(word_count * 0.1, 1))
    
    def _calculate_vocabulary_diversity(self, text: str) -> float:
        """Calculate vocabulary diversity (type-token ratio)"""
        words = text.lower().split()
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        return unique_words / total_words
    
    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if not sentences:
            return 0.0
        
        total_words = sum(len(s.split()) for s in sentences)
        return total_words / len(sentences)
    
    def _calculate_domain_specificity(self, text: str) -> float:
        """Calculate how domain-specific the content is"""
        
        # Look for domain-specific indicators
        words = text.lower().split()
        word_freq = Counter(words)
        
        # Find words that appear multiple times (likely domain terms)
        repeated_specialized = sum(1 for word, count in word_freq.items() 
                                  if count > 2 and len(word) > 6)
        
        # Look for capitalized multi-word terms
        proper_terms = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text))
        
        total_indicator = repeated_specialized + proper_terms
        word_count = len(words)
        
        return min(1.0, total_indicator / max(word_count * 0.05, 1))
    
    def _detect_code_blocks(self, text: str) -> bool:
        """Detect presence of code blocks"""
        return bool(re.search(r'```|^\s{4,}\w+\(|^\t\w+', text, re.MULTILINE))
    
    def _detect_formulas(self, text: str) -> bool:
        """Detect mathematical formulas"""
        return bool(re.search(r'[∫∑∏√±≈≠≤≥]|\$.*\$|\\[a-z]+\{', text))
    
    def _detect_lists(self, text: str) -> bool:
        """Detect lists or enumerations"""
        return bool(re.search(r'^\s*[\d•\-\*]\s*\w+', text, re.MULTILINE))
    
    def _detect_diagrams(self, text: str) -> bool:
        """Detect diagram/figure references"""
        return bool(re.search(r'\b(figure|diagram|chart|graph|illustration)\s+\d+', text, re.IGNORECASE))
    
    def _calculate_complexity_score(self, vocab_div: float, avg_sent: float, tech_density: float) -> float:
        """Calculate overall content complexity (0-10 scale)"""
        
        # Normalize components
        vocab_score = vocab_div * 3  # 0-3
        sent_score = min(4, avg_sent / 5)  # 0-4
        tech_score = tech_density * 3  # 0-3
        
        return vocab_score + sent_score + tech_score
    
    def _calculate_abstractness(self, text: str) -> float:
        """Calculate how abstract vs concrete the content is"""
        
        concrete_patterns = r'\b(object|thing|person|place|time|number)\b'
        abstract_patterns = r'\b(concept|idea|theory|principle|notion|abstraction)\b'
        
        concrete_count = len(re.findall(concrete_patterns, text, re.IGNORECASE))
        abstract_count = len(re.findall(abstract_patterns, text, re.IGNORECASE))
        
        total = concrete_count + abstract_count
        if total == 0:
            return 0.5
        
        return abstract_count / total
    
    def _score_evaluation_types(self, signals: ContentSignals) -> Dict[EvaluationType, float]:
        """Score each evaluation type based on signals"""
        
        scores = {}
        
        # Mathematical
        scores[EvaluationType.MATHEMATICAL] = (
            signals.math_density * 0.5 +
            (1.0 if signals.has_formulas else 0.0) * 0.3 +
            (signals.complexity_score / 10) * 0.2
        )
        
        # Code Generation
        scores[EvaluationType.CODE_GENERATION] = (
            signals.code_density * 0.6 +
            (1.0 if signals.has_code_blocks else 0.0) * 0.3 +
            signals.technical_density * 0.1
        )
        
        # Factual QA
        scores[EvaluationType.FACTUAL_QA] = (
            signals.factual_density * 0.5 +
            (1.0 if signals.has_lists else 0.0) * 0.2 +
            (1.0 - signals.abstractness_score) * 0.3
        )
        
        # Domain Knowledge
        scores[EvaluationType.DOMAIN_KNOWLEDGE] = (
            signals.domain_specificity * 0.4 +
            signals.technical_density * 0.3 +
            (signals.complexity_score / 10) * 0.3
        )
        
        # Reading Comprehension
        scores[EvaluationType.READING_COMPREHENSION] = (
            signals.narrative_density * 0.4 +
            (signals.avg_sentence_length / 30) * 0.3 +
            (signals.complexity_score / 10) * 0.3
        )
        
        # Multiple Choice (derived from other factors)
        scores[EvaluationType.MULTIPLE_CHOICE] = (
            max(scores[EvaluationType.FACTUAL_QA], 
                scores[EvaluationType.DOMAIN_KNOWLEDGE]) * 0.7
        )
        
        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def _calculate_confidence(self, primary_score: float, sorted_types: List[Tuple]) -> float:
        """Calculate confidence in the classification"""
        
        if len(sorted_types) < 2:
            return primary_score
        
        second_score = sorted_types[1][1]
        gap = primary_score - second_score
        
        # High confidence if large gap and high primary score
        confidence = primary_score * (0.5 + gap * 0.5)
        
        return min(1.0, confidence)
    
    def _generate_reasoning(self, primary_type: EvaluationType, 
                          signals: ContentSignals,
                          type_scores: Dict[EvaluationType, float]) -> str:
        """Generate human-readable reasoning for the classification"""
        
        reasons = []
        
        if primary_type == EvaluationType.MATHEMATICAL:
            reasons.append(f"High mathematical content density ({signals.math_density:.1%})")
            if signals.has_formulas:
                reasons.append("Contains mathematical formulas")
        
        elif primary_type == EvaluationType.CODE_GENERATION:
            reasons.append(f"High code density ({signals.code_density:.1%})")
            if signals.has_code_blocks:
                reasons.append("Contains code blocks")
        
        elif primary_type == EvaluationType.DOMAIN_KNOWLEDGE:
            reasons.append(f"High domain specificity ({signals.domain_specificity:.1%})")
            reasons.append(f"Complex vocabulary (diversity: {signals.vocabulary_diversity:.1%})")
        
        elif primary_type == EvaluationType.FACTUAL_QA:
            reasons.append(f"High factual content density ({signals.factual_density:.1%})")
            reasons.append("Concrete, verifiable information")
        
        elif primary_type == EvaluationType.READING_COMPREHENSION:
            reasons.append(f"Narrative structure ({signals.narrative_density:.1%})")
            reasons.append(f"Complex sentences (avg: {signals.avg_sentence_length:.1f} words)")
        
        return "; ".join(reasons)
    
    def _suggest_difficulty(self, signals: ContentSignals) -> str:
        """Suggest appropriate difficulty level"""
        
        if signals.complexity_score < 3:
            return "basic"
        elif signals.complexity_score < 6:
            return "intermediate"
        elif signals.complexity_score < 8:
            return "hard"
        else:
            return "expert"
    
    def _suggest_question_count(self, signals: ContentSignals, eval_type: EvaluationType) -> int:
        """Suggest appropriate number of questions"""
        
        # Base counts by type
        base_counts = {
            EvaluationType.MATHEMATICAL: 30,
            EvaluationType.CODE_GENERATION: 25,
            EvaluationType.FACTUAL_QA: 50,
            EvaluationType.DOMAIN_KNOWLEDGE: 40,
            EvaluationType.READING_COMPREHENSION: 35,
            EvaluationType.MULTIPLE_CHOICE: 50,
        }
        
        base = base_counts.get(eval_type, 40)
        
        # Adjust based on complexity
        if signals.complexity_score > 7:
            return int(base * 0.7)  # Fewer for complex
        elif signals.complexity_score < 3:
            return int(base * 1.3)  # More for simple
        
        return base
    
    def _should_use_agentic(self, eval_type: EvaluationType, signals: ContentSignals) -> bool:
        """Determine if agentic generation should be used"""
        
        # Use agentic for complex, domain-specific content
        if signals.complexity_score > 5:
            return True
        
        if signals.domain_specificity > 0.3:
            return True
        
        # Always use for certain types
        if eval_type in [EvaluationType.DOMAIN_KNOWLEDGE, 
                        EvaluationType.CODE_GENERATION,
                        EvaluationType.MATHEMATICAL]:
            return True
        
        return False
    
    def _suggest_validation_threshold(self, eval_type: EvaluationType) -> float:
        """Suggest minimum validation score threshold"""
        
        thresholds = {
            EvaluationType.MATHEMATICAL: 0.8,  # High precision needed
            EvaluationType.CODE_GENERATION: 0.7,
            EvaluationType.FACTUAL_QA: 0.75,
            EvaluationType.DOMAIN_KNOWLEDGE: 0.6,
            EvaluationType.READING_COMPREHENSION: 0.55,
            EvaluationType.MULTIPLE_CHOICE: 0.7,
        }
        
        return thresholds.get(eval_type, 0.65)
    
    def _suggest_answer_distribution(self, eval_type: EvaluationType, 
                                    signals: ContentSignals) -> Dict[str, float]:
        """Suggest expected distribution of answer types"""
        
        if eval_type == EvaluationType.MATHEMATICAL:
            return {
                'numeric_exact': 0.7,
                'string_exact': 0.2,
                'free_text': 0.1
            }
        
        elif eval_type == EvaluationType.CODE_GENERATION:
            return {
                'code': 0.9,
                'free_text': 0.1
            }
        
        elif eval_type == EvaluationType.FACTUAL_QA:
            return {
                'string_exact': 0.5,
                'free_text': 0.4,
                'numeric_exact': 0.1
            }
        
        elif eval_type == EvaluationType.DOMAIN_KNOWLEDGE:
            return {
                'free_text': 0.8,
                'string_exact': 0.2
            }
        
        elif eval_type == EvaluationType.MULTIPLE_CHOICE:
            return {
                'multiple_choice': 1.0
            }
        
        else:
            return {
                'free_text': 0.7,
                'string_exact': 0.3
            }

