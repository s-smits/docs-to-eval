"""
Enhanced agent implementations with improved prompts and logic
Provides more sophisticated question generation and quality control
"""

import asyncio
import json
import re
import random
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import time

from .models import (
    BenchmarkDraft,
    BenchmarkCandidate,
    EnhancedBenchmarkItem,
    ConceptExtractionResult,
    ValidationResult,
    DifficultyLevel,
    AnswerType,
)
from ..evaluation import EvaluationType, is_deterministic
from ..verification import VerificationOrchestrator
from ...llm.base import BaseLLMInterface
from ...utils.text_processing import extract_named_entities_simple, create_smart_chunks
from ...utils.config import ChunkingConfig

# Import base agents to extend
from .agents import BaseAgent, ConceptMiner, Adversary, Refiner, Validator


class EnhancedQuestionWriter(BaseAgent):
    """
    Improved QuestionWriter with better prompts and domain-specific logic
    
    Key improvements:
    - More specific prompts for each evaluation type
    - Better grounding in corpus snippets
    - Improved answer type determination
    - Multi-shot reasoning for complex questions
    """
    
    IMPROVED_SYSTEM_PROMPTS = {
        EvaluationType.DOMAIN_KNOWLEDGE: """You are an expert educational content creator specializing in domain-specific knowledge assessment.

Your task is to create questions that:
1. Test DEEP understanding of domain concepts, not just memorization
2. Are STRICTLY grounded in the provided snippet - no outside knowledge
3. Use precise terminology from the domain
4. Require synthesizing information, not just recalling facts
5. Have answers that demonstrate mastery (2-4 sentences explaining the concept)

Quality standards:
- Question must use specific terms and entities from the snippet
- Answer must explain relationships, significance, or implications
- Avoid yes/no or single-word answers for complex domains
- Focus on "how" and "why" rather than just "what"
""",
        
        EvaluationType.FACTUAL_QA: """You are an expert fact-checker and knowledge assessor.

Your task is to create precise, verifiable questions that:
1. Test specific facts explicitly stated in the snippet
2. Have unambiguous, deterministic answers
3. Focus on key entities, relationships, dates, or attributes
4. Can be answered in 1-3 sentences with clear facts

Quality standards:
- Question must have ONE clear correct answer
- Answer must be directly verifiable from the snippet
- Use "What", "When", "Where", "Who" for clarity
- Avoid ambiguity or interpretation
""",
        
        EvaluationType.MATHEMATICAL: """You are an expert mathematics educator.

Your task is to create mathematical problems that:
1. Use numbers, formulas, or relationships from the provided text
2. Test mathematical reasoning, not just calculation
3. Have precise numerical or algebraic answers
4. Include step-by-step reasoning when appropriate

Quality standards:
- Answers must be exact numbers or expressions
- Show clear derivation path in reasoning
- Use proper mathematical notation
- Verify answer is computable from given information
""",
        
        EvaluationType.CODE_GENERATION: """You are an expert programming instructor.

Your task is to create coding challenges that:
1. Relate to concepts or algorithms in the text
2. Have clear, testable specifications
3. Can be solved with clean, working code
4. Include example inputs/outputs if applicable

Quality standards:
- Specification must be unambiguous
- Answer should be functional code
- Include necessary imports/setup
- Focus on core logic, not boilerplate
""",
        
        EvaluationType.READING_COMPREHENSION: """You are an expert reading comprehension assessor.

Your task is to create questions that:
1. Test understanding of main ideas and details
2. Require inference or synthesis from the passage
3. Have answers well-supported by the text
4. Assess both surface and deep understanding

Quality standards:
- Answer must be supportable from the passage
- Avoid trivial or obvious questions
- Test comprehension, not memorization
- Include reasoning about relationships or implications
"""
    }
    
    async def produce(self, concept: str, corpus_text: str, eval_type: EvaluationType, 
                     context_snippet: Optional[str] = None) -> BenchmarkDraft:
        """Generate improved question draft with evaluation-specific logic"""
        start_time = time.time()
        
        try:
            # Select appropriate snippet if not provided
            if not context_snippet:
                context_snippet = self._find_relevant_snippet(concept, corpus_text, 
                                                             max_length=600)
            
            # Use improved prompting strategy
            if self.llm:
                question_data = await self._generate_with_improved_prompt(
                    concept, context_snippet, eval_type
                )
            else:
                question_data = self._generate_improved_template(
                    concept, context_snippet, eval_type
                )
            
            # Determine answer type with enhanced logic
            answer_type = self._determine_answer_type_improved(
                question_data['answer'], question_data['question'], eval_type
            )
            
            processing_time = time.time() - start_time
            self.call_count += 1
            self.total_processing_time += processing_time
            
            return BenchmarkDraft(
                question=question_data['question'][:200],
                answer=question_data['answer'],
                concept=concept,
                context_snippet=context_snippet[:800],
                expected_answer_type=answer_type,
                reasoning_chain=question_data.get('reasoning_chain', []),
                difficulty_estimate=self._estimate_difficulty(question_data, eval_type)
            )
            
        except Exception as e:
            # Improved fallback
            return self._create_improved_fallback(concept, context_snippet or corpus_text[:500], eval_type)
    
    async def _generate_with_improved_prompt(self, concept: str, snippet: str, 
                                            eval_type: EvaluationType) -> Dict[str, Any]:
        """Generate question using evaluation-type-specific improved prompts"""
        
        system_prompt = self.IMPROVED_SYSTEM_PROMPTS.get(
            eval_type,
            self.IMPROVED_SYSTEM_PROMPTS[EvaluationType.DOMAIN_KNOWLEDGE]
        )
        
        task_prompt = f"""
Given concept to assess: "{concept}"

Source snippet:
---
{snippet}
---

Create ONE high-quality question following the guidelines above.

Step 1: Identify the most important aspect of "{concept}" in this snippet
Step 2: Formulate a question that tests understanding of that aspect
Step 3: Write a complete, well-grounded answer
Step 4: List your reasoning steps

Return ONLY valid JSON in this exact format:
{{
    "question": "Your question here",
    "answer": "Your complete answer here",
    "reasoning_chain": ["step 1", "step 2", "step 3"]
}}
"""
        
        response_text = await self._call_llm_with_retry(
            prompt=f"{system_prompt}\n\n{task_prompt}",
            eval_type=str(eval_type)
        )
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*"question"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                # Validate and clean
                if data.get('question') and data.get('answer'):
                    return data
        except:
            pass
        
        # Improved fallback parsing
        return self._parse_improved_fallback(response_text, concept, snippet, eval_type)
    
    def _determine_answer_type_improved(self, answer: str, question: str, 
                                       eval_type: EvaluationType) -> AnswerType:
        """Enhanced answer type determination with contextual awareness"""
        
        # Check evaluation type first for strong signals
        if eval_type == EvaluationType.MATHEMATICAL:
            if re.search(r'\d+\.?\d*|\w+\s*=\s*\w+', answer):
                return AnswerType.NUMERIC_EXACT
        
        if eval_type == EvaluationType.CODE_GENERATION:
            if any(kw in answer.lower() for kw in ['def ', 'function', 'class ', 'import ', 'return ']):
                return AnswerType.CODE
        
        if eval_type == EvaluationType.MULTIPLE_CHOICE:
            return AnswerType.MULTIPLE_CHOICE
        
        # Content-based determination
        answer_clean = answer.strip()
        
        # Boolean
        if answer_clean.lower() in ['true', 'false', 'yes', 'no']:
            return AnswerType.BOOLEAN
        
        # Numeric (strict)
        if re.match(r'^-?\d+\.?\d*$', answer_clean):
            return AnswerType.NUMERIC_EXACT
        
        # Short factual (for factual QA only)
        if eval_type == EvaluationType.FACTUAL_QA:
            word_count = len(answer_clean.split())
            if word_count <= 5:
                # Check if it's a proper noun or specific term
                if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', answer_clean):
                    return AnswerType.STRING_EXACT
        
        # Default to free text for domain knowledge and complex answers
        return AnswerType.FREE_TEXT
    
    def _estimate_difficulty(self, question_data: Dict[str, Any], 
                            eval_type: EvaluationType) -> DifficultyLevel:
        """Estimate difficulty based on question characteristics"""
        
        question = question_data.get('question', '')
        answer = question_data.get('answer', '')
        reasoning = question_data.get('reasoning_chain', [])
        
        complexity_score = 0
        
        # Question complexity
        if len(question.split()) > 20:
            complexity_score += 1
        if any(word in question.lower() for word in ['analyze', 'compare', 'evaluate', 'synthesize', 'contrast']):
            complexity_score += 2
        
        # Answer complexity
        if len(answer.split()) > 30:
            complexity_score += 1
        if len(reasoning) > 2:
            complexity_score += 1
        
        # Evaluation type base difficulty
        type_base = {
            EvaluationType.MATHEMATICAL: 1,
            EvaluationType.CODE_GENERATION: 2,
            EvaluationType.DOMAIN_KNOWLEDGE: 1,
            EvaluationType.FACTUAL_QA: 0,
            EvaluationType.READING_COMPREHENSION: 1
        }
        complexity_score += type_base.get(eval_type, 1)
        
        # Map to difficulty levels
        if complexity_score <= 2:
            return DifficultyLevel.BASIC
        elif complexity_score <= 4:
            return DifficultyLevel.INTERMEDIATE
        elif complexity_score <= 6:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.EXPERT
    
    def _generate_improved_template(self, concept: str, snippet: str, 
                                   eval_type: EvaluationType) -> Dict[str, Any]:
        """Generate question using improved templates"""
        
        templates = {
            EvaluationType.DOMAIN_KNOWLEDGE: [
                "How does {concept} function in the context described in the passage?",
                "Explain the significance of {concept} based on the given information.",
                "What role does {concept} play in the described system or domain?",
                "Analyze the relationship between {concept} and the main topic discussed."
            ],
            EvaluationType.FACTUAL_QA: [
                "What specific characteristics of {concept} are described?",
                "According to the text, what is {concept}?",
                "What are the key attributes of {concept} mentioned in the passage?"
            ],
            EvaluationType.MATHEMATICAL: [
                "Based on the relationship involving {concept}, what would be the result?",
                "Calculate the value associated with {concept} from the given information.",
            ]
        }
        
        template_set = templates.get(eval_type, templates[EvaluationType.DOMAIN_KNOWLEDGE])
        question = random.choice(template_set).format(concept=concept)
        
        # Generate contextual answer
        answer = self._generate_contextual_answer(concept, snippet, eval_type)
        
        return {
            'question': question,
            'answer': answer,
            'reasoning_chain': [
                f"Identify {concept} in context",
                "Extract relevant information from snippet",
                "Formulate grounded response"
            ]
        }
    
    def _generate_contextual_answer(self, concept: str, snippet: str, 
                                   eval_type: EvaluationType) -> str:
        """Generate a contextually appropriate answer"""
        
        # Extract sentences containing the concept
        sentences = [s.strip() for s in snippet.split('.') if concept.lower() in s.lower()]
        
        if sentences:
            # For factual QA, keep it concise
            if eval_type == EvaluationType.FACTUAL_QA:
                return sentences[0] if sentences else f"{concept} is a key element in the described context."
            
            # For domain knowledge, provide explanation
            else:
                if len(sentences) >= 2:
                    return f"{sentences[0]}. Additionally, {sentences[1].lower()}"
                else:
                    return f"{concept} is described in the context as a significant element with specific characteristics and relationships to the overall topic discussed."
        
        return f"Based on the passage, {concept} is presented with specific attributes and contextual significance."
    
    def _parse_improved_fallback(self, response_text: str, concept: str, 
                                snippet: str, eval_type: EvaluationType) -> Dict[str, Any]:
        """Improved fallback parsing when JSON extraction fails"""
        
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        question = None
        answer = None
        reasoning = []
        
        for line in lines:
            if line.lower().startswith('question:') or line.startswith('Q:'):
                question = re.sub(r'^(question:|Q:)\s*', '', line, flags=re.IGNORECASE)
            elif line.lower().startswith('answer:') or line.startswith('A:'):
                answer = re.sub(r'^(answer:|A:)\s*', '', line, flags=re.IGNORECASE)
            elif line.startswith('-') or line.startswith('•'):
                reasoning.append(line.lstrip('-•').strip())
        
        if not question or not answer:
            # Use template fallback
            return self._generate_improved_template(concept, snippet, eval_type)
        
        return {
            'question': question,
            'answer': answer,
            'reasoning_chain': reasoning if reasoning else ["Generated from snippet analysis"]
        }
    
    def _create_improved_fallback(self, concept: str, snippet: str, 
                                 eval_type: EvaluationType) -> BenchmarkDraft:
        """Create improved fallback draft"""
        
        question_data = self._generate_improved_template(concept, snippet, eval_type)
        answer_type = self._determine_answer_type_improved(
            question_data['answer'], question_data['question'], eval_type
        )
        
        return BenchmarkDraft(
            question=question_data['question'],
            answer=question_data['answer'],
            concept=concept,
            context_snippet=snippet,
            expected_answer_type=answer_type,
            reasoning_chain=question_data['reasoning_chain'],
            difficulty_estimate=DifficultyLevel.INTERMEDIATE
        )
    
    def _find_relevant_snippet(self, concept: str, corpus_text: str, max_length: int = 600) -> str:
        """Find most relevant snippet with improved relevance scoring"""
        sentences = corpus_text.split('.')
        
        # Score sentences by relevance
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_lower = sentence.lower()
            concept_lower = concept.lower()
            
            # Exact concept match
            if concept_lower in sentence_lower:
                score += 10
            
            # Partial matches
            concept_words = concept_lower.split()
            for word in concept_words:
                if len(word) > 3 and word in sentence_lower:
                    score += 2
            
            # Proximity bonus for adjacent sentences
            if i > 0 and concept_lower in sentences[i-1].lower():
                score += 3
            if i < len(sentences) - 1 and concept_lower in sentences[i+1].lower():
                score += 3
            
            if score > 0:
                scored_sentences.append((score, i, sentence.strip()))
        
        if scored_sentences:
            # Sort by score
            scored_sentences.sort(reverse=True, key=lambda x: x[0])
            
            # Build snippet from best sentences
            snippet = ""
            for _, idx, sentence in scored_sentences:
                if len(snippet + sentence) <= max_length:
                    snippet += sentence + ". "
                else:
                    break
            
            return snippet.strip()
        
        # Fallback: return first part of corpus
        return corpus_text[:max_length]
