"""
Streamlined agent classes for production-ready agentic benchmark generation
Simplified from 5 agents to 3 essential agents with clear responsibilities
"""

import asyncio
import json
import re
import random
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import time
import logging

from .models import (
    BenchmarkCandidate,
    ConceptExtractionResult,
    ValidationResult,
    DifficultyLevel,
    AnswerType
)
from ..verification import VerificationOrchestrator
from ...llm.base import BaseLLMInterface


logger = logging.getLogger(__name__)


# Question type templates for diversity
QUESTION_TYPES = {
    "factual": {
        "templates": [
            "What is {concept}?",
            "Define {concept}.",
            "What are the key characteristics of {concept}?",
            "Describe {concept} in the context provided."
        ],
        "keywords": ["what", "define", "describe", "identify"]
    },
    "analytical": {
        "templates": [
            "Why is {concept} important in this context?",
            "How does {concept} relate to {related_concept}?",
            "Analyze the role of {concept} in the described process.",
            "What factors influence {concept}?"
        ],
        "keywords": ["why", "how", "analyze", "explain", "factors"]
    },
    "comparative": {
        "templates": [
            "Compare {concept} with {related_concept}.",
            "What distinguishes {concept} from {related_concept}?",
            "How does {concept} differ from {related_concept} in this context?",
            "What are the key differences between {concept} and {related_concept}?"
        ],
        "keywords": ["compare", "contrast", "distinguish", "versus", "different"]
    },
    "application": {
        "templates": [
            "How would you apply {concept} to solve a similar problem?",
            "Give an example of {concept} in practice.",
            "In what scenarios would {concept} be most effective?",
            "What are the practical implications of {concept}?"
        ],
        "keywords": ["apply", "example", "practice", "implement", "use"]
    },
    "evaluative": {
        "templates": [
            "Evaluate the effectiveness of {concept} in this scenario.",
            "What are the strengths and weaknesses of {concept}?",
            "Is {concept} the best approach for this situation? Why?",
            "Assess the impact of {concept} on the outcome."
        ],
        "keywords": ["evaluate", "assess", "judge", "critique", "strengths"]
    },
    "synthesis": {
        "templates": [
            "How do {concept} and {related_concept} work together?",
            "Integrate {concept} with the other elements described.",
            "What would result from combining {concept} with {related_concept}?",
            "Create a unified understanding of {concept} and its connections."
        ],
        "keywords": ["integrate", "combine", "synthesize", "together", "unified"]
    }
}


class BaseAgent(ABC):
    """Base class for all streamlined agents"""
    
    def __init__(self, llm_interface: Optional[BaseLLMInterface] = None):
        self.llm = llm_interface
        self.agent_name = self.__class__.__name__
        self.call_count = 0
        self.total_processing_time = 0.0
    
    @abstractmethod
    async def produce(self, *args, **kwargs) -> Any:
        """Main production method for the agent"""
        pass
    
    async def _call_llm(self, prompt: str, temperature: float = 0.7, max_retries: int = 2) -> str:
        """Call LLM with simple retry logic"""
        if not self.llm:
            raise ValueError(f"{self.agent_name} requires LLM interface")
        
        for attempt in range(max_retries + 1):
            try:
                response = await self.llm.generate_response(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=1000
                )
                return response.text.strip()
            except Exception as e:
                if attempt == max_retries:
                    raise e
                await asyncio.sleep(0.5 * (attempt + 1))
        
        raise RuntimeError(f"Failed to get LLM response after {max_retries} retries")


class ConceptExtractor(BaseAgent):
    """
    Extracts key concepts from corpus - simplified version
    Focuses on finding the most important topics to generate questions about
    """
    
    async def produce(self, corpus_text: str, num_concepts: int = 20) -> ConceptExtractionResult:
        """Extract key concepts from corpus"""
        start_time = time.time()
        
        try:
            # Simple chunking for concept extraction
            chunks = self._create_chunks(corpus_text, chunk_size=500)
            
            # Extract concepts
            if self.llm:
                concepts = await self._extract_with_llm(chunks, num_concepts)
            else:
                concepts = self._extract_with_keywords(corpus_text, num_concepts)
            
            # Create supporting snippets
            snippets = {}
            for concept in concepts[:num_concepts]:
                snippet = self._find_best_snippet(concept, corpus_text)
                snippets[concept] = snippet
            
            self.total_processing_time += time.time() - start_time
            self.call_count += 1
            
            return ConceptExtractionResult(
                key_concepts=concepts[:num_concepts],
                supporting_snippets=snippets,
                concept_importance_scores={c: 0.8 for c in concepts[:num_concepts]},
                chunk_ids=[f"chunk_{i}" for i in range(min(5, len(chunks)))]
            )
            
        except Exception as e:
            logger.error(f"ConceptExtractor failed: {str(e)}")
            # Return simple fallback concepts
            fallback_concepts = self._extract_with_keywords(corpus_text, num_concepts)
            return ConceptExtractionResult(
                key_concepts=fallback_concepts,
                supporting_snippets={c: corpus_text[:200] for c in fallback_concepts},
                concept_importance_scores={c: 0.5 for c in fallback_concepts},
                chunk_ids=["chunk_0"]
            )
    
    def _create_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Create simple text chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk) > 50:
                chunks.append(chunk)
        return chunks
    
    async def _extract_with_llm(self, chunks: List[str], num_concepts: int) -> List[str]:
        """Extract concepts using LLM"""
        # Process first few chunks to get concepts
        sample_text = ' '.join(chunks[:3])[:2000]
        
        prompt = f"""Extract {num_concepts} key concepts from this text. 
Focus on the most important topics, entities, and ideas.

Text: {sample_text}

Return a simple JSON list of concepts:
{{"concepts": ["concept1", "concept2", ...]}}"""
        
        try:
            response = await self._call_llm(prompt, temperature=0.3)
            data = json.loads(re.search(r'\{.*\}', response, re.DOTALL).group())
            return data.get('concepts', [])
        except Exception:
            return self._extract_with_keywords(sample_text, num_concepts)
    
    def _extract_with_keywords(self, text: str, num_concepts: int) -> List[str]:
        """Simple keyword extraction as fallback"""
        # Remove common words and extract important terms
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)  # Keep original case for proper noun detection
        
        # Comprehensive stopwords list including generic placeholder terms
        stopwords = {
            # Common stop words
            'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 
            'what', 'when', 'where', 'which', 'while', 'these', 'those', 'through', 
            'about', 'after', 'before', 'under', 'over', 'their', 'there', 'would',
            'could', 'should', 'your', 'into', 'very', 'just', 'only', 'also',
            
            # Generic/vague terms that make poor concepts
            'such', 'some', 'many', 'most', 'much', 'more', 'less', 'other', 'another',
            'different', 'various', 'several', 'certain', 'particular', 'specific',
            'general', 'common', 'similar', 'related', 'important', 'significant',
            'major', 'minor', 'large', 'small', 'great', 'good', 'best', 'better',
            'first', 'last', 'next', 'previous', 'following', 'above', 'below',
            'same', 'each', 'every', 'both', 'either', 'neither', 'former', 'latter',
            
            # Generic question/document words
            'question', 'answer', 'example', 'instance', 'case', 'situation',
            'context', 'content', 'text', 'document', 'information', 'data',
            'thing', 'things', 'element', 'elements', 'item', 'items', 'object',
            
            # Time/sequence words (unless specific dates)
            'time', 'times', 'year', 'years', 'period', 'phase', 'moment',
            'today', 'tomorrow', 'yesterday', 'recent', 'current', 'ancient',
            
            # Generic descriptors
            'type', 'types', 'kind', 'kinds', 'form', 'forms', 'part', 'parts',
            'aspect', 'aspects', 'feature', 'features', 'basis', 'nature',
            
            # Location generics (unless specific place names)
            'place', 'places', 'area', 'areas', 'region', 'regions', 'location',
            
            # People generics (unless specific names/roles)
            'people', 'person', 'women', 'woman', 'men', 'man', 'someone', 'anyone'
        }
        
        word_freq = {}
        for word in words:
            word_lower = word.lower()
            # Skip stop words and very short words
            if word_lower not in stopwords and len(word) >= 5:
                # Prefer proper nouns (capitalized words not at sentence start)
                if word[0].isupper() and not word.isupper():  # Capitalized but not acronym
                    word_freq[word] = word_freq.get(word, 0) + 2  # Boost proper nouns
                else:
                    word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # Also extract potential technical terms
        # Look for hyphenated words, CamelCase, acronyms
        tech_patterns = [
            (r'\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b', 3),  # CamelCase (high weight)
            (r'\b[a-z]+(?:-[a-z]+)+\b', 2),         # hyphenated-terms (medium weight)
            (r'\b[A-Z]{2,}\b', 2),                  # ACRONYMS (medium weight)
        ]
        
        for pattern, weight in tech_patterns:
            matches = re.findall(pattern, text)
            for match in matches[:10]:  # Limit technical terms
                if match.lower() not in stopwords:
                    word_freq[match] = word_freq.get(match, 0) + weight
        
        # Get top concepts, ensuring quality
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out any remaining low-quality concepts
        quality_concepts = []
        for word, freq in sorted_words:
            if len(quality_concepts) >= num_concepts:
                break
            # Additional quality checks
            if (len(word) >= 5 or word[0].isupper() or '-' in word or 
                freq >= 3 or re.match(r'^[A-Z]{2,}$', word)):
                quality_concepts.append(word)
        
        # If we still don't have enough concepts, try to extract from noun phrases
        if len(quality_concepts) < num_concepts // 2:
            # Extract potential noun phrases (simple pattern)
            noun_phrases = re.findall(r'\b(?:[A-Z][a-z]+\s+){1,3}[A-Z][a-z]+\b', text)
            for phrase in noun_phrases[:5]:
                if phrase not in quality_concepts:
                    quality_concepts.append(phrase)
        
        return quality_concepts[:num_concepts]
    
    def _find_best_snippet(self, concept: str, text: str, max_length: int = 300) -> str:
        """Find the best snippet containing the concept"""
        sentences = text.split('.')
        best_snippet = ""
        
        for sentence in sentences:
            if concept.lower() in sentence.lower():
                if len(sentence) > 30:  # Meaningful sentence
                    best_snippet = sentence.strip()
                    break
        
        if not best_snippet:
            # Return first part of text as fallback
            best_snippet = text[:max_length]
        
        return best_snippet[:max_length]


class QuestionGenerator(BaseAgent):
    """
    Generates diverse, high-quality questions
    Combines functionality of QuestionWriter, Adversary, and Refiner
    """
    
    def __init__(self, llm_interface: Optional[BaseLLMInterface] = None):
        super().__init__(llm_interface)
        self.question_type_rotation = list(QUESTION_TYPES.keys())
        self.current_type_index = 0
    
    async def produce(self, concept: str, context: str, eval_type: str, 
                     difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE) -> BenchmarkCandidate:
        """Generate a high-quality question"""
        start_time = time.time()
        
        try:
            # Select question type for diversity
            question_type = self._select_question_type()
            
            # Generate question based on type
            if self.llm:
                question_data = await self._generate_with_llm(
                    concept, context, eval_type, question_type, difficulty
                )
            else:
                question_data = self._generate_with_template(
                    concept, context, eval_type, question_type, difficulty
                )
            
            # Ensure quality and formatting
            question = self._format_question(question_data['question'])
            answer = self._format_answer(question_data['answer'], eval_type)
            
            # Determine answer type
            answer_type = self._determine_answer_type(answer, eval_type)
            
            # Add multiple choice options if needed
            options = None
            if answer_type == AnswerType.MULTIPLE_CHOICE or eval_type == "multiple_choice":
                options = await self._generate_options(question, answer, concept)
            
            self.total_processing_time += time.time() - start_time
            self.call_count += 1
            
            return BenchmarkCandidate(
                question=question,
                answer=answer,
                context=context[:500],
                options=options,
                concept=concept,
                expected_answer_type=answer_type,
                difficulty=difficulty,
                reasoning_chain=[f"Generated {question_type} question"],
                adversarial_techniques=[],
                variables={"question_type": question_type}
            )
            
        except Exception as e:
            logger.error(f"QuestionGenerator failed: {str(e)}")
            return self._create_fallback_question(concept, context, eval_type, difficulty)
    
    def _select_question_type(self) -> str:
        """Rotate through question types for diversity"""
        question_type = self.question_type_rotation[self.current_type_index]
        self.current_type_index = (self.current_type_index + 1) % len(self.question_type_rotation)
        return question_type
    
    async def _generate_with_llm(self, concept: str, context: str, eval_type: str, 
                                 question_type: str, difficulty: DifficultyLevel) -> Dict[str, Any]:
        """Generate question using LLM"""
        
        # Get question type info
        type_info = QUESTION_TYPES[question_type]
        keywords = ', '.join(type_info['keywords'])
        
        # Adjust prompt based on difficulty
        difficulty_hints = {
            DifficultyLevel.BASIC: "simple and straightforward",
            DifficultyLevel.INTERMEDIATE: "requiring some thought and understanding",
            DifficultyLevel.HARD: "challenging and requiring deep understanding",
            DifficultyLevel.EXPERT: "complex, requiring expert-level knowledge"
        }
        
        prompt = f"""Generate a {question_type} question about "{concept}".

Context: {context[:500]}

Requirements:
1. Question type: {question_type} (use keywords like: {keywords})
2. Difficulty: {difficulty} ({difficulty_hints.get(difficulty, 'moderate')})
3. The question must be answerable from the context
4. The answer must be clear and definitive
5. Evaluation type: {eval_type}

Return JSON only:
{{"question": "...", "answer": "..."}}"""
        
        try:
            response = await self._call_llm(prompt, temperature=0.7)
            data = json.loads(re.search(r'\{.*\}', response, re.DOTALL).group())
            return data
        except Exception:
            return self._generate_with_template(concept, context, eval_type, question_type, difficulty)
    
    def _generate_with_template(self, concept: str, context: str, eval_type: str,
                                question_type: str, difficulty: DifficultyLevel) -> Dict[str, Any]:
        """Generate question using templates"""
        templates = QUESTION_TYPES[question_type]["templates"]
        template = random.choice(templates)
        
        # Find a related concept for comparative/synthesis questions
        # Extract meaningful words from context, excluding common words
        words = re.findall(r'\b[a-zA-Z]{5,}\b', context)
        stopwords = {'these', 'those', 'their', 'there', 'where', 'which', 'while', 'other', 
                     'some', 'such', 'many', 'much', 'more', 'less', 'elements', 'things'}
        
        # Find words that are frequent and meaningful (not stopwords, not the main concept)
        word_freq = {}
        for word in words:
            word_lower = word.lower()
            if word_lower != concept.lower() and word_lower not in stopwords:
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # Get related concepts sorted by frequency
        related_candidates = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        related_concepts = [word for word, freq in related_candidates if freq > 1][:3]
        
        # Fallback: if no good related concepts, use a variation of the main concept
        if not related_concepts:
            # Create meaningful variations instead of generic terms
            if eval_type == "mathematical":
                related_concept = f"{concept} calculations"
            elif eval_type == "code_generation":
                related_concept = f"{concept} implementation"  
            elif eval_type == "factual_qa":
                related_concept = f"{concept} details"
            else:
                # Use domain-appropriate fallback
                related_concept = f"the {concept} approach"
        else:
            related_concept = related_concepts[0]
        
        # Handle template formatting - some templates need multiple concepts
        if template.count('{') > 2:  # Templates with multiple placeholders
            # For templates needing 3 concepts
            extra_concept = related_concepts[1] if len(related_concepts) > 1 else f"{concept} methods"
            question = template.format(concept, related_concept, extra_concept)
        elif '{concept}' in template and '{related_concept}' in template:
            question = template.format(concept=concept, related_concept=related_concept)
        elif '{concept}' in template:
            question = template.format(concept=concept)
        else:
            # Fallback for templates with positional arguments
            question = template.format(concept)
        
        # Generate appropriate answer based on question type
        if question_type == "factual":
            answer = self._extract_concise_answer(context, concept)
        elif question_type == "analytical":
            answer = f"The importance of {concept} lies in..."
        elif question_type == "comparative":
            answer = f"{concept} differs from {related_concept} in that..."
        elif question_type == "application":
            answer = f"To apply {concept}, one would..."
        elif question_type == "evaluative":
            answer = f"The effectiveness of {concept} can be evaluated as..."
        else:  # synthesis
            answer = f"{concept} integrates with other elements by..."
        
        return {"question": question, "answer": answer}
    
    def _format_question(self, question: str) -> str:
        """Format and clean the question"""
        question = question.strip()
        
        # Ensure proper length
        if len(question) > 150:
            # Truncate at sentence boundary
            sentences = question.split('.')
            question = sentences[0] + '?' if sentences else question[:147] + '...'
        
        # Ensure proper punctuation
        if not question.endswith('?') and not question.endswith('.'):
            if any(word in question.lower() for word in ['what', 'how', 'why', 'when', 'where', 'which', 'who']):
                question += '?'
            else:
                question += '.'
        
        # Capitalize first letter
        if question:
            question = question[0].upper() + question[1:]
        
        return question
    
    def _format_answer(self, answer: str, eval_type: str) -> str:
        """Format the answer appropriately"""
        answer = answer.strip()
        
        # Keep answers concise for deterministic types
        if eval_type in ["factual_qa", "mathematical", "code_generation"]:
            # Limit to essential information
            sentences = answer.split('.')
            if len(sentences) > 2:
                answer = '. '.join(sentences[:2]) + '.'
        
        return answer
    
    def _determine_answer_type(self, answer: str, eval_type: str) -> AnswerType:
        """Determine the answer type"""
        # Check for specific patterns
        if re.search(r'^\d+\.?\d*$', answer.strip()):
            return AnswerType.NUMERIC_EXACT
        elif answer.lower().strip() in ['true', 'false', 'yes', 'no']:
            return AnswerType.BOOLEAN
        elif eval_type == "multiple_choice":
            return AnswerType.MULTIPLE_CHOICE
        elif eval_type == "code_generation" or 'def ' in answer or 'function' in answer:
            return AnswerType.CODE
        elif len(answer.split()) <= 3:
            return AnswerType.STRING_EXACT
        else:
            return AnswerType.FREE_TEXT
    
    async def _generate_options(self, question: str, correct_answer: str, concept: str) -> List[str]:
        """Generate multiple choice options"""
        options = [correct_answer]
        
        if self.llm:
            try:
                prompt = f"""Generate 3 plausible but incorrect options for this question:
Question: {question}
Correct Answer: {correct_answer}

Return JSON: {{"options": ["wrong1", "wrong2", "wrong3"]}}"""
                
                response = await self._call_llm(prompt, temperature=0.8)
                data = json.loads(re.search(r'\{.*\}', response, re.DOTALL).group())
                options.extend(data.get('options', []))[:4]
            except Exception:
                pass
        
        # Fallback options if needed
        while len(options) < 4:
            options.append(f"Incorrect option {len(options)}")
        
        random.shuffle(options)
        return options[:4]
    
    def _create_fallback_question(self, concept: str, context: str, eval_type: str, 
                                 difficulty: DifficultyLevel) -> BenchmarkCandidate:
        """Create a simple fallback question"""
        question = f"What is {concept}?"
        answer = self._extract_concise_answer(context, concept)
        
        return BenchmarkCandidate(
            question=question,
            answer=answer,
            context=context[:500],
            concept=concept,
            expected_answer_type=AnswerType.FREE_TEXT,
            difficulty=difficulty,
            reasoning_chain=["Fallback generation"],
            adversarial_techniques=[]
        )

    def _extract_concise_answer(self, context: str, concept: str, max_chars: int = 220) -> str:
        """Extract concise grounded answer from context using definitional cues."""
        if not context:
            return f"{concept} is ..."
        sentences = [s.strip() for s in re.split(r'[\.!?]\s+', context) if s.strip()]
        concept_lower = concept.lower()
        cues = [' is ', ' are ', ' refers to ', ' defined as ', ' known as ', ' consists of ', ' comprises ']
        for s in sentences:
            s_lower = s.lower()
            if concept_lower in s_lower and any(cue in s_lower for cue in cues):
                return s[:max_chars].rstrip(',;: ') + ('...' if len(s) > max_chars else '')
        for s in sentences:
            if concept_lower in s.lower():
                return s[:max_chars].rstrip(',;: ') + ('...' if len(s) > max_chars else '')
        return (context[:max_chars].rstrip(',;: ') + ('...' if len(context) > max_chars else ''))


class QualityValidator(BaseAgent):
    """
    Validates question quality and ensures standards
    Simplified version focusing on essential quality checks
    """
    
    def __init__(self, llm_interface: Optional[BaseLLMInterface] = None):
        super().__init__(llm_interface)
        self.verifier = VerificationOrchestrator()
    
    async def produce(self, candidate: BenchmarkCandidate, corpus_text: str, 
                     min_score: float = 0.5) -> ValidationResult:
        """Validate candidate quality"""
        start_time = time.time()
        
        try:
            issues = []
            recommendations = []
            
            # Basic quality checks
            if not candidate.question or len(candidate.question.strip()) < 10:
                issues.append("Question too short or empty")
            
            if not candidate.answer or len(candidate.answer.strip()) < 2:
                issues.append("Answer too short or empty")
            
            if len(candidate.question) > 150:
                issues.append("Question exceeds length limit")
            
            # Check answer type consistency
            if candidate.expected_answer_type == AnswerType.MULTIPLE_CHOICE:
                if not candidate.options or len(candidate.options) < 2:
                    issues.append("Multiple choice lacks options")
            
            # Calculate quality score
            quality_score = await self._calculate_quality_score(candidate, corpus_text)
            
            # Determine if accepted
            accepted = quality_score >= min_score and len(issues) == 0
            
            if not accepted:
                recommendations.append("Improve question clarity and ensure answer is derivable from context")
            
            self.total_processing_time += time.time() - start_time
            self.call_count += 1
            
            return ValidationResult(
                accepted=accepted,
                score=quality_score,
                issues=issues,
                recommendations=recommendations,
                deterministic_check_passed=True,
                verification_method_used="streamlined_validation"
            )
            
        except Exception as e:
            logger.error(f"QualityValidator failed: {str(e)}")
            return ValidationResult(
                accepted=False,
                score=0.0,
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Review question for issues"],
                deterministic_check_passed=False,
                verification_method_used="error"
            )
    
    async def _calculate_quality_score(self, candidate: BenchmarkCandidate, corpus_text: str) -> float:
        """Calculate overall quality score"""
        scores = []
        
        # Question clarity (simple heuristics)
        question_score = 1.0
        if '?' not in candidate.question and 'what' not in candidate.question.lower():
            question_score -= 0.2
        if len(candidate.question.split()) < 5:
            question_score -= 0.3
        scores.append(max(0, question_score))
        
        # Answer completeness
        answer_score = 1.0
        if len(candidate.answer.split()) < 3 and candidate.expected_answer_type == AnswerType.FREE_TEXT:
            answer_score -= 0.3
        scores.append(max(0, answer_score))
        
        # Context relevance (check if concept appears in context)
        relevance_score = 0.5
        if candidate.concept.lower() in (candidate.context or corpus_text[:500]).lower():
            relevance_score = 1.0
        scores.append(relevance_score)
        
        # LLM validation if available
        if self.llm:
            llm_score = await self._validate_with_llm(candidate, corpus_text)
            scores.append(llm_score)
        
        return sum(scores) / len(scores)
    
    async def _validate_with_llm(self, candidate: BenchmarkCandidate, corpus_text: str) -> float:
        """Use LLM to validate question quality"""
        try:
            prompt = f"""Rate this question-answer pair quality (0-1):

Question: {candidate.question}
Answer: {candidate.answer}
Context: {candidate.context or corpus_text[:300]}

Consider:
1. Is the question clear and answerable?
2. Is the answer correct based on context?
3. Is this challenging for modern LLMs?

Return JSON: {{"score": 0.7, "reason": "..."}}"""
            
            response = await self._call_llm(prompt, temperature=0.1)
            data = json.loads(re.search(r'\{.*\}', response, re.DOTALL).group())
            return float(data.get('score', 0.5))
        except Exception:
            return 0.5  # Neutral score on error
