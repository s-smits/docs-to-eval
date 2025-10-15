"""
Agentic question generation system
Uses multiple strategies to generate high-quality domain-specific questions from corpus
"""

import re
import random
from collections import Counter
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from docs_to_eval.core.evaluation import extract_key_concepts, sample_corpus_segments, EvaluationType


@dataclass
class QuestionItem:
    """Structured question item with quality metrics"""
    question: str
    answer: str
    context: Optional[str] = None
    category: str = "general"
    difficulty: str = "intermediate"
    quality_score: float = 0.5
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AdvancedMockLLM:
    """Advanced mock LLM for sophisticated question generation"""
    
    def __init__(self):
        self.model_name = "AdvancedMockLLM-v2"
        self.generation_patterns = {
            'conceptual': [
                "What is the fundamental concept behind {}?",
                "How would you define {} in this context?",
                "What are the core principles of {}?",
                "Explain the significance of {} in the domain."
            ],
            'application': [
                "How would you apply {} in the following scenario: {}?",
                "What would happen if {} was combined with {}?",
                "Describe a practical use case for {}.",
                "How does {} solve the problem of {}?"
            ],
            'comparison': [
                "Compare and contrast {} with {}.",
                "What are the similarities and differences between {} and {}?",
                "How does {} relate to {}?",
                "What advantages does {} have over {}?"
            ],
            'analytical': [
                "Analyze the impact of {} on {}.",
                "What are the implications of {} for {}?",
                "Evaluate the effectiveness of {} in {}.",
                "What factors influence {} in the context of {}?"
            ],
            'synthesis': [
                "How do {}, {}, and {} work together?",
                "What would be the result of combining {} with {}?",
                "Create a framework that incorporates {}.",
                "Design a system that utilizes {}."
            ]
        }
    
    def generate_question(self, concept: str, category: str, context: str = "") -> Dict[str, str]:
        """Generate a single question based on concept and category"""
        templates = self.generation_patterns.get(category, self.generation_patterns['conceptual'])
        template = random.choice(templates)
        
        # Simple template filling with some randomization
        if '{}' in template:
            if template.count('{}') == 1:
                question = template.format(concept)
            elif template.count('{}') == 2:
                # Use concept and a related term
                related_terms = ['technology', 'process', 'system', 'approach', 'method']
                second_concept = random.choice(related_terms)
                question = template.format(concept, second_concept)
            else:
                # Fill with concept and variations
                question = template.format(concept, concept + " system", concept + " approach")
        else:
            question = template + " " + concept
        
        # Generate mock answer without placeholders
        answer_templates = {
            'conceptual': f"{concept} is a fundamental concept that involves structured processes and systematic approaches",
            'application': f"When applying {concept}, one would typically use data-driven methods and established frameworks",
            'comparison': f"Comparing {concept} with other approaches shows distinct advantages in efficiency and scalability", 
            'analytical': f"The analysis of {concept} reveals complex relationships between components and underlying mechanisms",
            'synthesis': f"Combining {concept} with other elements creates enhanced capabilities and improved performance"
        }
        
        answer = answer_templates.get(category, f"The answer involves understanding {concept} through systematic analysis and practical application")
        
        return {
            'question': question,
            'answer': answer,
            'concept': concept,
            'category': category
        }


class AgenticQuestionGenerator:
    """Advanced question generation using multiple strategies and self-improvement"""
    
    def __init__(self, llm=None):
        self.llm = llm or AdvancedMockLLM()
        self.question_strategies = [
            self._generate_conceptual_questions,
            self._generate_application_questions,
            self._generate_comparison_questions, 
            self._generate_analytical_questions,
            self._generate_synthesis_questions
        ]
        self.quality_metrics = ['clarity', 'difficulty', 'relevance', 'uniqueness']
    
    def generate_comprehensive_benchmark(self, corpus_text: str, num_questions: int = 50, 
                                       eval_type: EvaluationType = EvaluationType.DOMAIN_KNOWLEDGE) -> Dict[str, Any]:
        """Generate a comprehensive benchmark using multiple question generation strategies with quality filtering"""
        
        # Analyze corpus structure
        corpus_analysis = self._analyze_corpus_structure(corpus_text)
        
        # Use oversampling to ensure we have enough quality questions after filtering
        oversample_factor = 2.0  # Generate 2.5x more questions than needed
        target_generation = int(num_questions * oversample_factor)
        
        # Generate questions with oversampling
        all_questions, strategy_distribution = self._generate_with_oversampling(corpus_text, corpus_analysis, target_generation, eval_type)
        
        # Apply quality filtering and improvement to get exactly num_questions
        filtered_questions = self._filter_and_ensure_count(all_questions, corpus_analysis, num_questions, corpus_text, eval_type)
        
        # Generate quality statistics
        quality_stats = self._compute_quality_statistics(filtered_questions)
        
        # Convert to dictionary format for compatibility
        question_dicts = []
        for q in filtered_questions:
            question_dicts.append({
                'question': q.question,
                'answer': q.answer,
                'context': q.context,
                'category': q.category,
                'difficulty': q.difficulty,
                'quality_score': q.quality_score,
                'metadata': q.metadata
            })
        
        return {
            'questions': question_dicts,
            'strategy_distribution': strategy_distribution,
            'quality_stats': quality_stats,
            'corpus_analysis': corpus_analysis,
            'total_generated': len(question_dicts),
            'eval_type': eval_type
        }
    
    def _analyze_corpus_structure(self, corpus_text: str) -> Dict[str, Any]:
        """Analyze corpus structure for informed question generation"""
        
        # Extract various elements
        key_concepts = extract_key_concepts(corpus_text, max_concepts=30)
        segments = sample_corpus_segments(corpus_text, num_segments=10, segment_length=300)
        
        # Analyze text complexity
        sentences = re.split(r'[.!?]+', corpus_text)
        words = corpus_text.split()
        
        # Find domain-specific patterns
        domain_indicators = {
            'technical_terms': len(re.findall(r'\b[A-Z][a-z]*[A-Z][a-z]*\b', corpus_text)),  # CamelCase
            'acronyms': len(re.findall(r'\b[A-Z]{2,}\b', corpus_text)),
            'numbers': len(re.findall(r'\d+', corpus_text)),
            'parenthetical_explanations': len(re.findall(r'\([^)]*\)', corpus_text))
        }
        
        # Topic clustering (simplified)
        word_freq = Counter(word.lower() for word in words if len(word) > 3)
        top_topics = [word for word, freq in word_freq.most_common(15)]
        
        return {
            'key_concepts': key_concepts,
            'segments': segments,
            'text_stats': {
                'total_words': len(words),
                'total_sentences': len([s for s in sentences if s.strip()]),
                'avg_sentence_length': len(words) / max(len(sentences), 1),
                'unique_words': len(set(words))
            },
            'domain_indicators': domain_indicators,
            'top_topics': top_topics,
            'complexity_score': self._calculate_complexity_score(corpus_text)
        }
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate text complexity score (0-1)"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        if not words or not sentences:
            return 0.5
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_sentence_length = len(words) / len(sentences)
        unique_word_ratio = len(set(words)) / len(words)
        
        # Normalize and combine factors
        complexity = (
            min(avg_word_length / 8, 1) * 0.3 +  # Word complexity
            min(avg_sentence_length / 20, 1) * 0.4 +  # Sentence complexity  
            unique_word_ratio * 0.3  # Vocabulary diversity
        )
        
        return min(max(complexity, 0), 1)
    
    def _generate_conceptual_questions(self, corpus_text: str, analysis: Dict[str, Any], 
                                     num_questions: int, eval_type: str) -> List[QuestionItem]:
        """Generate conceptual understanding questions"""
        questions = []
        concepts = analysis['key_concepts'][:num_questions * 2]  # Get more concepts than needed
        
        for i in range(min(num_questions, len(concepts))):
            concept = concepts[i]
            generated = self.llm.generate_question(concept, 'conceptual', corpus_text)
            
            quality_score = self._assess_question_quality(generated['question'], analysis)
            difficulty = self._determine_difficulty(generated['question'], analysis)
            
            questions.append(QuestionItem(
                question=generated['question'],
                answer=generated['answer'],
                context=corpus_text,
                category='conceptual',
                difficulty=difficulty,
                quality_score=quality_score,
                metadata={'concept': concept, 'generation_method': 'conceptual'}
            ))
        
        return questions
    
    def _generate_application_questions(self, corpus_text: str, analysis: Dict[str, Any], 
                                      num_questions: int, eval_type: str) -> List[QuestionItem]:
        """Generate application-based questions"""
        questions = []
        concepts = analysis['key_concepts']
        
        for i in range(num_questions):
            concept = concepts[i % len(concepts)] if concepts else f"concept_{i}"
            generated = self.llm.generate_question(concept, 'application', corpus_text)
            
            quality_score = self._assess_question_quality(generated['question'], analysis) + 0.1  # Slight boost for application
            difficulty = 'advanced'  # Application questions are typically more difficult
            
            questions.append(QuestionItem(
                question=generated['question'],
                answer=generated['answer'],
                context=random.choice(analysis['segments']) if analysis['segments'] else corpus_text,
                category='application',
                difficulty=difficulty,
                quality_score=min(quality_score, 1.0),
                metadata={'concept': concept, 'generation_method': 'application'}
            ))
        
        return questions
    
    def _generate_comparison_questions(self, corpus_text: str, analysis: Dict[str, Any], 
                                     num_questions: int, eval_type: str) -> List[QuestionItem]:
        """Generate comparison questions"""
        questions = []
        concepts = analysis['key_concepts']
        
        for i in range(num_questions):
            if len(concepts) >= 2:
                concept = random.choice(concepts)
                generated = self.llm.generate_question(concept, 'comparison', corpus_text)
            else:
                concept = f"topic_{i}"
                generated = self.llm.generate_question(concept, 'comparison', corpus_text)
            
            quality_score = self._assess_question_quality(generated['question'], analysis)
            
            questions.append(QuestionItem(
                question=generated['question'],
                answer=generated['answer'],
                context=corpus_text,
                category='comparison',
                difficulty='intermediate',
                quality_score=quality_score,
                metadata={'concept': concept, 'generation_method': 'comparison'}
            ))
        
        return questions
    
    def _generate_analytical_questions(self, corpus_text: str, analysis: Dict[str, Any], 
                                     num_questions: int, eval_type: str) -> List[QuestionItem]:
        """Generate analytical questions"""
        questions = []
        concepts = analysis['key_concepts']
        
        for i in range(num_questions):
            concept = concepts[i % len(concepts)] if concepts else f"element_{i}"
            generated = self.llm.generate_question(concept, 'analytical', corpus_text)
            
            quality_score = self._assess_question_quality(generated['question'], analysis) + 0.15  # Boost for analytical
            
            questions.append(QuestionItem(
                question=generated['question'],
                answer=generated['answer'],
                context=random.choice(analysis['segments']) if analysis['segments'] else corpus_text,
                category='analytical',
                difficulty='advanced',
                quality_score=min(quality_score, 1.0),
                metadata={'concept': concept, 'generation_method': 'analytical'}
            ))
        
        return questions
    
    def _generate_synthesis_questions(self, corpus_text: str, analysis: Dict[str, Any], 
                                    num_questions: int, eval_type: str) -> List[QuestionItem]:
        """Generate synthesis questions"""
        questions = []
        concepts = analysis['key_concepts']
        
        for i in range(num_questions):
            if len(concepts) >= 3:
                concept = random.choice(concepts[:10])  # Use top concepts
                generated = self.llm.generate_question(concept, 'synthesis', corpus_text)
            else:
                concept = f"system_{i}"
                generated = self.llm.generate_question(concept, 'synthesis', corpus_text)
            
            quality_score = self._assess_question_quality(generated['question'], analysis) + 0.2  # Highest boost for synthesis
            
            questions.append(QuestionItem(
                question=generated['question'],
                answer=generated['answer'],
                context=corpus_text,
                category='synthesis',
                difficulty='expert',
                quality_score=min(quality_score, 1.0),
                metadata={'concept': concept, 'generation_method': 'synthesis'}
            ))
        
        return questions
    
    def _assess_question_quality(self, question: str, analysis: Dict[str, Any]) -> float:
        """Assess question quality based on various factors"""
        
        # Base quality factors
        length_score = min(len(question.split()) / 15, 1)  # Optimal around 10-15 words
        
        # Complexity alignment
        question_complexity = len([w for w in question.split() if len(w) > 6]) / max(len(question.split()), 1)
        complexity_alignment = 1 - abs(question_complexity - analysis['complexity_score'])
        
        # Concept relevance
        concepts_in_question = sum(1 for concept in analysis['key_concepts'][:10] 
                                 if concept.lower() in question.lower())
        relevance_score = min(concepts_in_question / 2, 1)
        
        # Question type diversity (bonus for question words)
        question_words = ['what', 'how', 'why', 'when', 'where', 'which']
        has_question_word = any(word in question.lower() for word in question_words)
        question_type_score = 1.0 if has_question_word else 0.7
        
        # Combine scores
        quality = (
            length_score * 0.2 +
            complexity_alignment * 0.3 +
            relevance_score * 0.3 +
            question_type_score * 0.2
        )
        
        return min(max(quality, 0.1), 1.0)
    
    def _determine_difficulty(self, question: str, analysis: Dict[str, Any]) -> str:
        """Determine question difficulty level"""
        
        # Analyze question complexity
        words = question.split()
        complex_words = [w for w in words if len(w) > 7]
        technical_terms = sum(1 for concept in analysis['key_concepts'][:5] 
                            if concept.lower() in question.lower())
        
        complexity_indicators = [
            'analyze', 'evaluate', 'synthesize', 'compare', 'contrast',
            'implications', 'significance', 'relationship', 'framework'
        ]
        
        advanced_indicators = sum(1 for indicator in complexity_indicators 
                                if indicator in question.lower())
        
        # Scoring
        difficulty_score = (
            len(complex_words) / max(len(words), 1) * 3 +
            technical_terms * 2 +
            advanced_indicators * 2
        )
        
        if difficulty_score >= 4:
            return 'expert'
        elif difficulty_score >= 2:
            return 'advanced'
        elif difficulty_score >= 1:
            return 'intermediate'
        else:
            return 'basic'
    
    def _generate_with_oversampling(self, corpus_text: str, corpus_analysis: Dict[str, Any], 
                                  target_count: int, eval_type: EvaluationType):
        """Generate questions with oversampling across strategies"""
        
        # Distribute questions across strategies
        questions_per_strategy = max(1, target_count // len(self.question_strategies))
        remaining_questions = target_count % len(self.question_strategies)
        
        all_questions = []
        strategy_distribution = []
        
        for i, strategy in enumerate(self.question_strategies):
            strategy_questions = questions_per_strategy
            if i < remaining_questions:
                strategy_questions += 1
            
            try:
                questions = strategy(corpus_text, corpus_analysis, strategy_questions, str(eval_type))
                all_questions.extend(questions)
                
                strategy_distribution.append({
                    'strategy': strategy.__name__,
                    'questions_generated': len(questions),
                    'avg_quality': sum(q.quality_score for q in questions) / len(questions) if questions else 0
                })
            except Exception as e:
                print(f"Warning: Strategy {strategy.__name__} failed: {e}")
                strategy_distribution.append({
                    'strategy': strategy.__name__,
                    'questions_generated': 0,
                    'avg_quality': 0,
                    'error': str(e)
                })
        
        return all_questions, strategy_distribution
    
    def _filter_and_ensure_count(self, questions: List[QuestionItem], analysis: Dict[str, Any], 
                               target_count: int, corpus_text: str, eval_type: EvaluationType) -> List[QuestionItem]:
        """Filter questions with quality control and ensure exact count through regeneration"""
        
        min_quality_threshold = 0.4  # Minimum acceptable quality
        max_attempts = 3  # Maximum regeneration attempts
        
        for attempt in range(max_attempts):
            # Apply quality filtering
            quality_filtered = self._apply_quality_filter(questions, min_quality_threshold)
            
            # Remove duplicates
            unique_filtered = self._remove_duplicates(quality_filtered)
            
            # Sort by quality score
            unique_filtered.sort(key=lambda q: q.quality_score, reverse=True)
            
            if len(unique_filtered) >= target_count:
                # We have enough high-quality questions
                return unique_filtered[:target_count]
            
            # Need to regenerate more questions
            shortage = target_count - len(unique_filtered)
            print(f"Regenerating {shortage} questions (attempt {attempt + 1}/{max_attempts})")
            
            # Generate additional questions to fill the gap
            additional_questions = self._generate_additional_questions(
                corpus_text, analysis, shortage * 2, eval_type  # Generate 2x shortage for safety
            )
            questions.extend(additional_questions)
        
        # If we still don't have enough after max attempts, return what we have
        # and lower quality threshold progressively
        for lower_threshold in [0.3, 0.2, 0.1]:
            quality_filtered = self._apply_quality_filter(questions, lower_threshold)
            unique_filtered = self._remove_duplicates(quality_filtered)
            unique_filtered.sort(key=lambda q: q.quality_score, reverse=True)
            
            if len(unique_filtered) >= target_count:
                return unique_filtered[:target_count]
        
        # Final fallback: return best available questions, even if below target count
        print(f"Warning: Could only generate {len(unique_filtered)} questions out of {target_count} requested")
        return unique_filtered
    
    def _apply_quality_filter(self, questions: List[QuestionItem], min_threshold: float) -> List[QuestionItem]:
        """Filter out questions below quality threshold"""
        return [q for q in questions if q.quality_score >= min_threshold]
    
    def _remove_duplicates(self, questions: List[QuestionItem]) -> List[QuestionItem]:
        """Remove duplicate or very similar questions"""
        filtered_questions = []
        seen_questions = set()
        
        for question in questions:
            # Simple similarity check
            question_words = set(question.question.lower().split())
            
            is_duplicate = False
            for seen in seen_questions:
                seen_words = set(seen.lower().split())
                if len(question_words.intersection(seen_words)) / len(question_words.union(seen_words)) > 0.7:
                    is_duplicate = True
                    break
            
            # Additional checks for too simple questions
            if not is_duplicate and not self._is_too_simple(question):
                filtered_questions.append(question)
                seen_questions.add(question.question)
        
        return filtered_questions
    
    def _is_too_simple(self, question: QuestionItem) -> bool:
        """Check if a question is too simple or low-quality"""
        q_text = question.question.lower()
        a_text = question.answer.lower()
        
        # Check for extremely short content (very restrictive)
        if len(question.question.split()) < 3:
            return True
        
        if len(question.answer.split()) < 1:
            return True
        
        # Check for placeholder content
        placeholders = ['...', 'todo', 'placeholder', 'insert here', 'fill in']
        if any(placeholder in q_text or placeholder in a_text for placeholder in placeholders):
            return True
        
        # Check for extremely vague questions (very restrictive)
        extremely_simple_patterns = [
            'what?', 'how?', 'why?', 'yes.', 'no.', 'maybe.'
        ]
        
        # Only flag questions that are extremely simple
        if q_text.strip() in extremely_simple_patterns or a_text.strip() in extremely_simple_patterns:
            return True
        
        # Check for very short questions with very generic patterns (more restrictive)
        if len(question.question.split()) <= 4:
            very_generic = ['what is it', 'how does it work', 'why does it']
            if any(pattern in q_text for pattern in very_generic):
                return True
        
        return False
    
    def _generate_additional_questions(self, corpus_text: str, analysis: Dict[str, Any], 
                                     count: int, eval_type: EvaluationType) -> List[QuestionItem]:
        """Generate additional questions when we need more to reach target count"""
        # Use conceptual strategy for additional questions as it tends to be most reliable
        return self._generate_conceptual_questions(corpus_text, analysis, count, str(eval_type))
    
    def _compute_quality_statistics(self, questions: List[QuestionItem]) -> Dict[str, Any]:
        """Compute quality statistics for generated questions"""
        
        if not questions:
            return {'avg_quality': 0, 'quality_distribution': {}}
        
        quality_scores = [q.quality_score for q in questions]
        difficulties = [q.difficulty for q in questions]
        categories = [q.category for q in questions]
        
        difficulty_dist = Counter(difficulties)
        category_dist = Counter(categories)
        
        return {
            'avg_quality': sum(quality_scores) / len(quality_scores),
            'min_quality': min(quality_scores),
            'max_quality': max(quality_scores),
            'quality_std': self._calculate_std(quality_scores),
            'difficulty_distribution': dict(difficulty_dist),
            'category_distribution': dict(category_dist),
            'total_questions': len(questions)
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


if __name__ == "__main__":
    # Test the agentic generator
    generator = AgenticQuestionGenerator()
    
    sample_corpus = """
    Artificial neural networks are computing systems inspired by biological neural networks.
    They consist of interconnected nodes (neurons) that process information using mathematical models.
    Deep learning uses multiple layers of neural networks to learn hierarchical representations of data.
    Common applications include image recognition, natural language processing, and autonomous systems.
    Training involves adjusting connection weights through backpropagation algorithms.
    """
    
    print("Testing Agentic Question Generator:")
    print("=" * 50)
    
    benchmark = generator.generate_comprehensive_benchmark(
        sample_corpus, 
        num_questions=50, 
        eval_type=EvaluationType.DOMAIN_KNOWLEDGE
    )
    
    print(f"Generated {benchmark['total_generated']} questions")
    print(f"Quality Statistics: {benchmark['quality_stats']}")
    print(f"Strategy Distribution: {benchmark['strategy_distribution']}")
    
    print("\nSample Questions:")
    for i, question in enumerate(benchmark['questions'][:5], 1):
        print(f"\n{i}. [{question['category']}] {question['question']}")
        print(f"   Answer: {question['answer'][:80]}...")
        print(f"   Difficulty: {question['difficulty']} | Quality: {question['quality_score']:.2f}")