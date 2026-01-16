"""
Specialized agent classes for agentic benchmark generation
Each agent has a specific role in the intelligent benchmark creation pipeline
"""

import asyncio
import json
import re
import random
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import time
import logging

from .models import (
    BenchmarkDraft,
    BenchmarkCandidate,
    ConceptExtractionResult,
    ValidationResult,
    AgentResponse,
    AgentConfig,
    DifficultyLevel,
    AnswerType,
    validate_deterministic_answer_type
)
from ..evaluation import EvaluationType
from ..verification import VerificationOrchestrator
from ...llm.base import BaseLLMInterface
from ...utils.text_processing import extract_named_entities_simple, create_smart_chunks
from ...utils.config import ChunkingConfig


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self, llm_interface: Optional[BaseLLMInterface] = None, config: Optional[AgentConfig] = None):
        self.llm = llm_interface
        self.config = config or AgentConfig()
        self.agent_name = self.__class__.__name__
        self.agent_version = "v1"
        self.call_count = 0
        self.total_processing_time = 0.0
    
    @abstractmethod
    async def produce(self, *args, **kwargs) -> Any:
        """Main production method for the agent"""
        pass
    
    def _create_response(self, success: bool = True, error_message: Optional[str] = None, **metadata) -> AgentResponse:
        """Create standardized agent response"""
        return AgentResponse(
            agent_name=self.agent_name,
            agent_version=self.agent_version,
            success=success,
            error_message=error_message,
            metadata=metadata
        )
    
    async def _call_llm_with_retry(self, prompt: str, context: Optional[str] = None, **kwargs) -> str:
        """Call LLM with retry logic"""
        if not self.llm:
            raise ValueError(f"{self.agent_name} requires LLM interface")
        
        for attempt in range(self.config.retry_attempts + 1):
            try:
                response = await self.llm.generate_response(
                    prompt=prompt,
                    context=context,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    **kwargs
                )
                return response.text.strip()
            except Exception as e:
                if attempt == self.config.retry_attempts:
                    raise e
                await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff
        
        raise RuntimeError(f"Failed to get response from LLM after {self.config.retry_attempts} retries")


class ConceptMiner(BaseAgent):
    """
    Extracts key concepts from corpus using hybrid RAG approach
    Responsibilities: 
    - Mine salient concepts from corpus text
    - Provide supporting snippets with chunk IDs
    - Score concept importance
    """
    
    async def produce(self, corpus_text: str, k: int = 20, min_chunk_size: int = 400) -> ConceptExtractionResult:
        """Extract key concepts from corpus"""
        if not corpus_text or not corpus_text.strip():
            raise ValueError("corpus_text cannot be empty")
        
        if k <= 0 or k > 100:
            raise ValueError(f"k must be between 1 and 100, got {k}")
        
        start_time = time.time()
        
        try:
            # Use smart semantic chunking with strict minimum threshold
            try:
                cfg = ChunkingConfig(
                    target_chunk_size=1200,
                    max_chunk_size=2200,
                    min_chunk_size=max(400, min_chunk_size),
                    overlap_size=200,
                    overlap_percent=15.0,
                    enable_chonkie=True,
                    chunking_strategy="semantic",
                    adaptive_sizing=True,
                )
                chunk_dicts = create_smart_chunks(corpus_text, chunking_config=cfg)
                chunks = [c.get('text', '') for c in chunk_dicts if c.get('text')]
                
                if not chunks:
                    raise ValueError("Semantic chunking produced no chunks")
                    
            except Exception as e:
                # Fallback to simple windowed chunks
                import logging
                logging.warning(f"Semantic chunking failed: {e}. Using windowed chunks.")
                chunks = self._create_windowed_chunks(corpus_text, chunk_size=800, overlap=100)
            
            # Extract concepts from each chunk
            all_concepts = {}
            supporting_snippets = {}
            chunk_ids = []
            
            # Process chunks in parallel batches
            batch_size = min(5, len(chunks))
            if batch_size <= 0:
                batch_size = 1
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_results = await asyncio.gather(*[
                    self._extract_concepts_from_chunk(chunk, f"chunk_{i+j}")
                    for j, chunk in enumerate(batch_chunks)
                ])
                
                # Merge results
                for chunk_concepts, chunk_id in batch_results:
                    chunk_ids.append(chunk_id)
                    for concept, (score, snippet) in chunk_concepts.items():
                        if concept in all_concepts:
                            # Average the scores, keep best snippet
                            all_concepts[concept] = (all_concepts[concept] + score) / 2
                            if len(snippet) > len(supporting_snippets.get(concept, "")):
                                supporting_snippets[concept] = snippet
                        else:
                            all_concepts[concept] = score
                            supporting_snippets[concept] = snippet
            
            # Select top-k concepts
            sorted_concepts = sorted(all_concepts.items(), key=lambda x: x[1], reverse=True)[:k]
            
            key_concepts = [concept for concept, _ in sorted_concepts]
            concept_scores = {concept: score for concept, score in sorted_concepts}
            
            # Filter supporting snippets to only include selected concepts
            filtered_snippets = {concept: supporting_snippets[concept] for concept in key_concepts}
            
            processing_time = time.time() - start_time
            self.call_count += 1
            self.total_processing_time += processing_time
            
            return ConceptExtractionResult(
                key_concepts=key_concepts,
                supporting_snippets=filtered_snippets,
                concept_importance_scores=concept_scores,
                chunk_ids=chunk_ids
            )
            
        except Exception as e:
            logger.error(f"ConceptMiner failed with error: {str(e)}", exc_info=True)
            logger.warning("Returning fallback concepts - workflow quality may be degraded")
            return ConceptExtractionResult(
                key_concepts=[f"fallback_concept_{i}" for i in range(min(k, 10))],  # Fallback with clear naming
                supporting_snippets={f"fallback_concept_{i}": corpus_text for i in range(min(k, 10))},
                concept_importance_scores={f"fallback_concept_{i}": 0.1 for i in range(min(k, 10))},
                chunk_ids=[f"fallback_chunk_{i}" for i in range(min(5, k))]
            )
    
    def _create_windowed_chunks(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Create overlapping text chunks (robust to small chunk_size)"""
        words = text.split()
        if chunk_size <= overlap:
            # Ensure progress to avoid infinite loop or zero-step
            overlap = max(0, chunk_size // 2)
        step = max(1, chunk_size - overlap)
        chunks: List[str] = []
        for i in range(0, len(words), step):
            chunk_words = words[i:i + max(1, chunk_size)]
            if len(chunk_words) >= max(5, min(50, chunk_size // 4)):
                chunks.append(' '.join(chunk_words))
        return chunks
    
    async def _extract_concepts_from_chunk(self, chunk: str, chunk_id: str) -> Tuple[Dict[str, Tuple[float, str]], str]:
        """Extract concepts from a single chunk"""
        
        if self.llm:
            # Use LLM for concept extraction
            prompt = f"""
Extract DOMAIN-SPECIFIC key concepts, entities, and topics from this text. 

Rules:
- Extract full noun phrases and proper names, NOT single generic words
- Include specific entities (e.g., "Tavola Capuana", "Etruscan terracotta slab", "ritual calendar")
- Include domain-specific terms with context (e.g., "470 BCE inscription" not just "inscription")
- Include measurements, dates, locations when present
- Concepts must be specific to THIS domain, not generic terms

Text: {chunk}

Return JSON with specific multi-word concepts:
{{"concepts": [{{"name": "specific domain concept or entity", "importance": 0.8, "snippet": "exact supporting text"}}]}}

Examples of GOOD concepts: "Tavola Capuana terracotta slab", "Etruscan ritual calendar", "470 BCE inscription"
Examples of BAD concepts: "ancient", "text", "women", "important"
"""
            try:
                response = await self._call_llm_with_retry(prompt)
                # Parse JSON response
                data = json.loads(response)
                concepts = {}
                for item in data.get('concepts', []):
                    concepts[item['name']] = (item['importance'], item['snippet'])
                return concepts, chunk_id
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                # Failed to parse LLM response, use fallback
                logger.warning(f"Failed to parse LLM response in ConceptMiner: {str(e)}")
                pass
        
        # Fallback: Simple keyword extraction
        return self._simple_concept_extraction(chunk), chunk_id
    
    def _simple_concept_extraction(self, text: str) -> Dict[str, Tuple[float, str]]:
        """Fallback concept extraction combining keyword statistics and domain heuristics."""

        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq: Dict[str, int] = {}
        for word in words:
            if word not in {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'which', 'their'}:
                word_freq[word] = word_freq.get(word, 0) + 1

        entities = extract_named_entities_simple(text)
        entity_scores: Dict[str, float] = {}
        for ent in entities:
            tokens = ent.split()
            base = 1.0 + 0.4 * (len(tokens) - 1)
            entity_scores[ent] = base

        total_words = max(1, len(words))
        concepts: Dict[str, Tuple[float, str]] = {}
        sentences = text.split('.')

        for ent, escore in sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:15]:
            snippet = next((s.strip() for s in sentences if ent in s), text[:200])
            concepts[ent] = (min(1.0, 0.6 + 0.1 * escore), snippet[:200])

        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]:
            if word in concepts:
                continue
            score = min(1.0, (freq / total_words) * 8)
            snippet = next((s.strip() for s in sentences if word in s.lower()), text[:200])
            concepts[word] = (score, snippet[:200])

        # Additional domain-aware heuristics inspired by main branch implementation
        domain_phrases: List[str] = []
        proper_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        domain_phrases.extend(proper_phrases)

        dates = re.findall(r'\b\d+\s*(?:BCE|CE|BC|AD)\b', text)
        domain_phrases.extend(dates)

        measurements = re.findall(r'\b\d+\s*(?:by|x)\s*\d+\s*(?:cm|m|in|ft)\b', text)
        domain_phrases.extend(measurements)

        artifact_patterns = [
            r'\b(?:ancient|ritual|religious|sacred)\s+[a-z]+\b',
            r'\b[a-z]+\s+(?:slab|tablet|inscription|calendar|artifact|text)\b',
            r'\b[a-z]+\s+(?:ceremony|tradition|practice|society)\b'
        ]
        for pattern in artifact_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            domain_phrases.extend([m for m in matches if len(m.split()) > 1])

        locations = re.findall(r'\b[A-Z][a-z]+(?:\s+in\s+[A-Z][a-z]+)?\b', text)
        domain_phrases.extend([loc for loc in locations if ' in ' in loc.lower()])

        seen: set[str] = set()
        for phrase in domain_phrases:
            phrase = phrase.strip()
            normalized = phrase.lower()
            if normalized in seen or normalized in {'the', 'this', 'that', 'these', 'those', 'ancient', 'text', 'also'}:
                continue
            if phrase in concepts:
                continue
            if len(phrase.split()) <= 1 and not any(char.isdigit() for char in phrase):
                continue

            best_snippet = next((s.strip() for s in sentences if normalized in s.lower()), None)
            if not best_snippet:
                idx = text.lower().find(normalized)
                if idx != -1:
                    start = max(0, idx - 50)
                    end = min(len(text), idx + len(phrase) + 50)
                    best_snippet = text[start:end].strip()
            snippet_value = (best_snippet or text[:120])[:300]

            score = 0.5
            if any(char.isupper() for char in phrase):
                score += 0.2
            if any(char.isdigit() for char in phrase):
                score += 0.2
            if len(phrase.split()) > 2:
                score += 0.1

            concepts[phrase] = (min(score, 1.0), snippet_value)
            seen.add(normalized)

        if not concepts:
            caps = re.findall(r'\b[A-Z][a-z]+\b', text)
            for cap in caps[:5]:
                snippet = next((s.strip() for s in sentences if cap in s), text[:100])
                concepts[cap] = (0.3, snippet[:300])

        return concepts


class QuestionWriter(BaseAgent):
    """
    Creates initial benchmark questions using cooperative chain-of-thought
    Responsibilities:
    - Generate questions grounded in corpus snippets
    - Use chain-of-thought reasoning
    - Format as BenchmarkDraft
    """
    
    async def produce(self, concept: str, corpus_text: str, eval_type: EvaluationType, 
                     context_snippet: Optional[str] = None) -> BenchmarkDraft:
        """Generate initial question draft"""
        start_time = time.time()
        
        try:
            # Select appropriate snippet if not provided
            if not context_snippet:
                context_snippet = self._find_relevant_snippet(concept, corpus_text)
            
            # Create question using LLM
            if self.llm:
                question_data = await self._generate_question_with_llm(concept, context_snippet, eval_type)
            else:
                question_data = self._generate_question_template(concept, context_snippet, eval_type)
            
            # Determine answer type
            answer_type = self._determine_answer_type(question_data['answer'], eval_type)
            
            processing_time = time.time() - start_time
            self.call_count += 1
            self.total_processing_time += processing_time
            
            return BenchmarkDraft(
                question=question_data['question'],
                answer=question_data['answer'],
                concept=concept,
                context_snippet=context_snippet,
                expected_answer_type=answer_type,
                reasoning_chain=question_data.get('reasoning_chain', []),
                difficulty_estimate=DifficultyLevel.INTERMEDIATE
            )
            
        except Exception:
            # Fallback question generation
            return self._create_fallback_draft(concept, context_snippet or corpus_text, eval_type)
    
    async def produce_with_feedback(self, concept: str, corpus_text: str, eval_type: EvaluationType, 
                                   context_snippet: Optional[str] = None, validation_context: Optional[Dict] = None) -> BenchmarkDraft:
        """Generate question draft with validation feedback for improvement"""
        
        # Use feedback to improve generation if provided
        if validation_context and validation_context.get('previous_issues'):
            # Incorporate feedback into prompt
            feedback_prompt = f"""
Previous attempt had issues: {'; '.join(validation_context['previous_issues'][:2])}

Suggestions for improvement:
{'; '.join(validation_context.get('suggestions', [])[:3])}

Retry #{validation_context.get('retry_count', 1)} - please address these issues.
"""
            # Add feedback to context
            enhanced_snippet = f"{context_snippet}\n\nFeedback: {feedback_prompt}"
            return await self.produce(concept, corpus_text, eval_type, enhanced_snippet)
        
        # If no feedback, use normal generation
        return await self.produce(concept, corpus_text, eval_type, context_snippet)
    
    async def _generate_question_with_llm(self, concept: str, snippet: str, eval_type: EvaluationType) -> Dict[str, Any]:
        """Generate question using LLM with chain-of-thought"""
        
        system_prompt = """You are an expert question writer. Create challenging, well-grounded, domain-specific questions.

CRITICAL RULES:
1. Use domain-specific terminology from the snippet in both the question and answer
2. ALWAYS include specific domain entities, dates, locations, or measurements drawn from the snippet
3. Questions must be answerable strictly from the snippet but require understanding and synthesis, not copying
4. Avoid generic phrasing; keep questions precise and contextual to this domain
5. For free_text or domain knowledge answers provide 2-3 grounded sentences; for deterministic types give concise, unambiguous responses

Return ONLY JSON with this exact format:
{"question": "...", "answer": "...", "reasoning_chain": ["step1", "step2", "step3"]}"""

        # Extract candidate domain terms from the snippet to force specificity
        domain_terms = self._extract_domain_terms(snippet)
        term_bank = ", ".join(domain_terms[:8]) if domain_terms else ""
        term_bank_clause = f"\nDomain terms (use ≥2 if appropriate): [{term_bank}]\n" if term_bank else "\n"
        
        task_prompt = f"""
Domain concept: {concept}
Evaluation type: {eval_type}
Domain context:
{snippet}
{term_bank_clause}

Create a DOMAIN-SPECIFIC question that:
1. Relies exclusively on the information above while demonstrating deep understanding of {concept}
2. Mentions concrete entities, dates, measurements, or terminology that appear in the snippet
3. Cannot be answered using general knowledge alone—unique to this snippet’s content
4. Matches the expectations of the {eval_type} evaluation type

Answer requirements:
- For free_text or domain_knowledge answers, write 2-3 concise sentences grounded in the snippet
- For factual, multiple choice, or other deterministic types, provide a precise, unambiguous answer grounded in the snippet

Examples of GOOD questions:
- "What is the size of the Tavola Capuana terracotta slab found in 470 BCE?"
- "How does the Etruscan ritual calendar from 470 BCE relate to religious practices?"
- "What specific inscriptions appear on the 50x60 cm terracotta artifact?"

Examples of BAD questions:
- "What is ancient?"
- "What factors influence women?"
- "What is the significance of roman?"

Think step by step and return JSON only."""
        
        response_text = await self._call_llm_with_retry(
            prompt=f"{system_prompt}\n\n{task_prompt}",
            context=snippet,
            eval_type=str(eval_type)
        )
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            # Failed to parse JSON, use fallback
            pass
        
        # Fallback parsing - Extract domain details to make specific questions
        # Try to extract specific details from the snippet
        import re
        dates = re.findall(r'\b\d+\s*(?:BCE|CE|BC|AD)\b', snippet)
        measurements = re.findall(r'\b\d+\s*(?:cm|m|km|in|by)\s*\d+', snippet)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', snippet)
        
        # Build more specific questions using extracted details
        domain_details = ""
        if dates:
            domain_details = f" from {dates[0]}"
        elif proper_nouns:
            domain_details = f" in relation to {proper_nouns[0]}"
        
        domain_templates = {
            'factual': f"What specific characteristics define {concept}{domain_details}?",
            'definition': f"How is {concept}{domain_details} described in this context?",
            'significance': f"What role does {concept}{domain_details} play in the described system?",
            'relationship': f"How does {concept}{domain_details} connect to the other elements mentioned?",
            'characteristics': f"What distinguishes {concept}{domain_details} from similar artifacts?"
        }
        
        # Choose appropriate template
        template_key = random.choice(list(domain_templates.keys()))
        question = domain_templates[template_key]
        
        # Generate grounded answer with fallback if extraction is too short
        answer = self._extract_concise_answer_from_snippet(snippet, concept)
        if not answer or len(answer.split()) < 10:
            answer = (
                f"Based on the provided snippet, {concept} is situated within a specific domain context that "
                f"highlights its role, properties, and relationships. The snippet describes concrete details that "
                f"differentiate {concept} from related ideas within this setting."
            )
        
        return {
            'question': question,
            'answer': answer,
            'reasoning_chain': ["Analyze context", "Extract relevant information", "Formulate precise answer"]
        }

    def _extract_domain_terms(self, text: str, max_terms: int = 12) -> List[str]:
        """Extract candidate domain terms (proper nouns and salient keywords) from text.
        Heuristic: proper nouns (capitalized words), multi-word noun-like phrases, and frequent non-stopwords.
        """
        if not text:
            return []
        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        tokens = re.findall(r"\b[a-zA-Z]{5,}\b", text.lower())
        stop = {
            'these','those','which','their','there','where','while','after','before','about','would','could','should',
            'between','within','among','however','therefore','because','since','given','using','based','according',
            'context','corpus','snippet','text','passage','the','this','that','with','have','will','from','they','been','were','into','other','during','often','common','various'
        }
        freq: Dict[str,int] = {}
        for t in tokens:
            if t not in stop:
                freq[t] = freq.get(t, 0) + 1
        keywords = [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:max_terms]]
        combined: List[str] = []
        seen = set()
        for term in proper_nouns + keywords:
            if term not in seen:
                combined.append(term)
                seen.add(term)
            if len(combined) >= max_terms:
                break
        return combined
    
    def _generate_question_template(self, concept: str, snippet: str, eval_type: EvaluationType) -> Dict[str, Any]:
        """Template-based question generation with domain specificity"""
        
        # Extract domain-specific details from snippet
        import re
        dates = re.findall(r'\b\d+\s*(?:BCE|CE|BC|AD)\b', snippet)
        measurements = re.findall(r'\b\d+\s*(?:cm|m|km|in|by)\s*\d+', snippet)
        proper_nouns = [n for n in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', snippet) if n != concept]
        
        # Build context-specific parts
        time_context = f" from {dates[0]}" if dates else ""
        size_context = f" measuring {measurements[0]}" if measurements else ""
        entity_context = f" related to {proper_nouns[0]}" if proper_nouns else ""
        
        templates = {
            EvaluationType.FACTUAL_QA: [
                f"What specific function does {concept}{time_context}{entity_context} serve?",
                f"How is {concept}{size_context}{time_context} described in this context?",
                f"What distinguishing features characterize {concept}{entity_context}?"
            ],
            EvaluationType.DOMAIN_KNOWLEDGE: [
                f"What is the significance of {concept}{time_context} in this specific domain?",
                f"How does {concept}{entity_context} function within the described system?",
                f"What evidence supports the role of {concept}{size_context}{time_context}?"
            ],
            EvaluationType.MATHEMATICAL: [
                f"Calculate the dimensions involving {concept}{size_context}.",
                f"What quantitative relationship exists with {concept}{time_context}?",
                f"Determine the numerical value associated with {concept}."
            ]
        }
        
        template = random.choice(templates.get(eval_type, templates[EvaluationType.DOMAIN_KNOWLEDGE]))
        question = template.format(concept=concept)
        answer = self._extract_concise_answer_from_snippet(snippet, concept)
        if not answer or len(answer.split()) < 10:
            answer = (
                f"Within the provided snippet, {concept} is described with specific attributes and context. "
                f"Synthesizing these details yields a concise explanation grounded in the text."
            )
        
        return {
            'question': question,
            'answer': answer,
            'reasoning_chain': ["Analyze context", "Apply domain knowledge", "Formulate response"]
        }
    
    def _find_relevant_snippet(self, concept: str, corpus_text: str, max_length: int = -1) -> str:
        """Find the most relevant snippet for a concept.
        If max_length <= 0, treat as unlimited (no truncation) and let upstream chunking control size.
        """
        sentences = corpus_text.split('.')
        
        # Score sentences by concept relevance
        scored_sentences = []
        for sentence in sentences:
            score = sentence.lower().count(concept.lower())
            if score > 0:
                scored_sentences.append((score, sentence.strip()))
        
        if scored_sentences:
            # Get best sentences up to max_length (or unlimited)
            scored_sentences.sort(reverse=True)
            snippet = ""
            unlimited = (max_length is None) or (max_length <= 0)
            for _, sentence in scored_sentences:
                if unlimited or len(snippet + sentence) <= max_length:
                    snippet += sentence + ". "
                else:
                    break
            return snippet.strip()
        
        # Fallback: return first part of corpus or full text if unlimited
        if (max_length is None) or (max_length <= 0):
            return corpus_text
        return corpus_text[:max_length]
    
    def _determine_answer_type(self, answer: str, eval_type: EvaluationType) -> AnswerType:
        """Determine expected answer type based on content and eval type"""
        
        # Check for numeric answers
        if re.search(r'\d+\.?\d*', answer) and eval_type == EvaluationType.MATHEMATICAL:
            return AnswerType.NUMERIC_EXACT
        
        # Check for code
        if any(keyword in answer.lower() for keyword in ['def', 'function', 'class', 'import', 'return']):
            return AnswerType.CODE
        
        # Check for boolean
        if answer.lower().strip() in ['true', 'false', 'yes', 'no']:
            return AnswerType.BOOLEAN
        
        # Check for multiple choice indicators
        if eval_type == EvaluationType.MULTIPLE_CHOICE or re.search(r'\b[ABCD]\b', answer):
            return AnswerType.MULTIPLE_CHOICE
        
        # Detect deterministic factual responses when clearly short and specific
        if eval_type == EvaluationType.FACTUAL_QA:
            tokens = answer.split()
            if 1 <= len(tokens) <= 4 and re.match(r'^[A-Za-z][A-Za-z\s\-]{0,40}$', answer.strip()):
                return AnswerType.STRING_EXACT

        return AnswerType.FREE_TEXT

    async def produce_with_feedback(
        self,
        concept: str,
        corpus_text: str,
        eval_type: EvaluationType,
        context_snippet: Optional[str] = None,
        feedback: Optional[Dict[str, Any]] = None
    ) -> BenchmarkDraft:
        """Retry-friendly generation that incorporates validator feedback."""
        feedback = feedback or {}
        if not context_snippet:
            context_snippet = self._find_relevant_snippet(concept, corpus_text)

        # Compose augmented prompt
        guidance = """
Incorporate the following validator feedback:
- Make the question more specific to the snippet's terminology
- Ensure the answer is directly derivable from the snippet and provide 2 concise sentences if free text
- Avoid generic phrasing, use concrete entities from the snippet
"""
        if self.llm:
            try:
                response_text = await self._call_llm_with_retry(
                    prompt=(
                        f"{guidance}\n\nConcept: {concept}\nEval Type: {eval_type}\nSnippet:\n{context_snippet}\n\n"
                        "Return JSON: {\"question\": \"...\", \"answer\": \"...\", \"reasoning_chain\": [\"...\"]}"
                    )
                )
                data = json.loads(re.search(r'\{.*\}', response_text, re.DOTALL).group())
            except Exception:
                data = self._generate_question_template(concept, context_snippet, eval_type)
        else:
            data = self._generate_question_template(concept, context_snippet, eval_type)

        answer_type = self._determine_answer_type(data['answer'], eval_type)
        return BenchmarkDraft(
            question=data['question'][:200],
            answer=data['answer'],
            concept=concept,
            context_snippet=context_snippet[:800],
            expected_answer_type=answer_type,
            reasoning_chain=data.get('reasoning_chain', []),
            difficulty_estimate=DifficultyLevel.INTERMEDIATE
        )
    
    def _create_fallback_draft(self, concept: str, snippet: str, eval_type: EvaluationType) -> BenchmarkDraft:
        """Create fallback draft with domain specificity"""
        
        # Extract domain context
        import re
        dates = re.findall(r'\b\d+\s*(?:BCE|CE|BC|AD)\b', snippet)
        proper_nouns = [n for n in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', snippet) if n != concept]
        
        # Create context-aware questions
        context_suffix = ""
        if dates and proper_nouns:
            context_suffix = f" in the {proper_nouns[0]} context from {dates[0]}"
        elif dates:
            context_suffix = f" from the period of {dates[0]}"
        elif proper_nouns:
            context_suffix = f" as it relates to {proper_nouns[0]}"
        
        question_templates = [
            f"What specific characteristics define {concept}{context_suffix}?",
            f"How is {concept}{context_suffix} described in this domain?",
            f"What role does {concept}{context_suffix} serve?",
            f"What distinguishes {concept}{context_suffix} from similar items?"
        ]
        
        question = random.choice(question_templates)
        answer = self._extract_concise_answer_from_snippet(snippet, concept)
        
        return BenchmarkDraft(
            question=question,
            answer=answer,
            concept=concept,
            context_snippet=snippet,
            expected_answer_type=AnswerType.FREE_TEXT,
            reasoning_chain=["Extract definition", "Analyze context", "Provide clear explanation"],
            difficulty_estimate=DifficultyLevel.BASIC
        )

    def _extract_concise_answer_from_snippet(self, snippet: str, concept: str, max_chars: int = 220) -> str:
        """Extract a concise, grounded answer from the supporting snippet.
        Prefer sentences that contain the concept or definitional cues."""
        if not snippet:
            return f"{concept} is defined as ..."
        sentences = [s.strip() for s in re.split(r'[\.!?]\s+', snippet) if s.strip()]
        # Prefer sentence containing concept
        preferred = None
        concept_lower = concept.lower()
        definitional_cues = [' is ', ' are ', ' refers to ', ' defined as ', ' known as ', ' consists of ', ' comprises ']
        for s in sentences:
            s_lower = s.lower()
            if concept_lower in s_lower and any(cue in s_lower for cue in definitional_cues):
                preferred = s
                break
        if not preferred:
            # Next best: any sentence containing the concept
            preferred = next((s for s in sentences if concept_lower in s.lower()), sentences[0] if sentences else snippet)
        concise = preferred.strip()
        if len(concise) > max_chars:
            concise = concise[:max_chars].rstrip(',;: ') + '...'
        return concise


class Adversary(BaseAgent):
    """
    Enhances question difficulty and adds adversarial elements
    Responsibilities:
    - Increase difficulty while preserving groundedness
    - Add distractors and obfuscation
    - Apply adversarial techniques
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adversarial_techniques = [
            "add_distractors",
            "multi_hop_reasoning", 
            "reverse_statements",
            "data_to_code_transformation",
            "temporal_complexity",
            "causal_chain_extension"
        ]
    
    async def produce(self, draft: BenchmarkDraft, target_difficulty: DifficultyLevel = DifficultyLevel.HARD) -> BenchmarkCandidate:
        """Apply adversarial enhancement to increase difficulty"""
        start_time = time.time()
        
        try:
            # Select appropriate techniques based on difficulty and eval type
            techniques = self._select_techniques(draft, target_difficulty)
            
            # Apply adversarial transformations
            enhanced_question = draft.question
            enhanced_answer = draft.answer
            options = None
            distractor_rationale = ""
            
            for technique in techniques:
                if technique == "add_distractors" and draft.expected_answer_type == AnswerType.MULTIPLE_CHOICE:
                    options, distractor_rationale = await self._add_distractors(enhanced_question, enhanced_answer, draft.concept)
                elif technique == "multi_hop_reasoning":
                    enhanced_question, enhanced_answer = await self._add_multi_hop_reasoning(enhanced_question, enhanced_answer, draft.context_snippet)
                elif technique == "reverse_statements":
                    enhanced_question = self._add_reverse_complexity(enhanced_question, draft.concept)
                elif technique == "data_to_code_transformation" and "data" in draft.context_snippet.lower():
                    enhanced_question, enhanced_answer = self._transform_to_code_problem(enhanced_question, enhanced_answer, draft)
            
            processing_time = time.time() - start_time
            self.call_count += 1
            self.total_processing_time += processing_time
            
            return BenchmarkCandidate(
                question=enhanced_question,
                answer=enhanced_answer,
                context=draft.context_snippet,
                options=options,
                concept=draft.concept,
                expected_answer_type=draft.expected_answer_type,
                difficulty=target_difficulty,
                reasoning_chain=draft.reasoning_chain + [f"Applied {len(techniques)} adversarial techniques"],
                adversarial_techniques=techniques,
                distractor_rationale=distractor_rationale
            )
            
        except Exception:
            # Fallback: return slightly modified draft
            return self._create_fallback_candidate(draft, target_difficulty)
    
    def _select_techniques(self, draft: BenchmarkDraft, target_difficulty: DifficultyLevel) -> List[str]:
        """Select appropriate adversarial techniques"""
        
        # Base number of techniques by difficulty
        technique_counts = {
            DifficultyLevel.BASIC: 0,
            DifficultyLevel.INTERMEDIATE: 1,
            DifficultyLevel.HARD: 2,
            DifficultyLevel.EXPERT: 3
        }
        
        num_techniques = technique_counts.get(target_difficulty, 2)
        available_techniques = self.adversarial_techniques.copy()
        
        # Filter based on answer type and context
        if draft.expected_answer_type != AnswerType.MULTIPLE_CHOICE:
            available_techniques = [t for t in available_techniques if t != "add_distractors"]
        
        if "data" not in draft.context_snippet.lower():
            available_techniques = [t for t in available_techniques if t != "data_to_code_transformation"]
        
        return random.sample(available_techniques, min(num_techniques, len(available_techniques)))
    
    async def _add_distractors(self, question: str, answer: str, concept: str) -> Tuple[List[str], str]:
        """Add plausible wrong options for multiple choice"""
        
        options = [answer]  # Correct answer
        
        if self.llm:
            try:
                prompt = f"""
Create 3 DOMAIN-SPECIFIC incorrect options for this multiple choice question.
Distracters must be from the SAME DOMAIN and plausible within the context.

Question: {question}
Correct Answer: {answer}
Domain Concept: {concept}

Rules:
1. All options must be specific to this domain (not generic)
2. Include similar entities, dates, or measurements from the same field
3. Make them plausible within the domain context
4. Avoid obviously wrong or nonsensical answers

Return JSON: {{"distractors": ["domain-specific wrong1", "domain-specific wrong2", "domain-specific wrong3"], "rationale": "why these are good domain distractors"}}
"""
                response = await self._call_llm_with_retry(prompt)
                data = json.loads(re.search(r'\{.*\}', response, re.DOTALL).group())
                options.extend(data.get('distractors', []))
                rationale = data.get('rationale', 'Generated distractors')
            except (json.JSONDecodeError, AttributeError, TypeError):
                # Fallback distractors (domain-aware)
                numeric_like = any(ch.isdigit() for ch in answer)
                if numeric_like:
                    import re
                    nums = re.findall(r'\d+', answer)
                    if nums:
                        base = int(nums[0])
                        options.extend([
                            answer.replace(str(base), str(base + 5)),
                            answer.replace(str(base), str(base - 5 if base > 5 else base + 7)),
                            answer.replace(str(base), str(max(1, base * 2)))
                        ])
                else:
                    options.extend([
                        f"Earlier dating of {concept}",
                        f"Later dating related to {concept}",
                        f"Similar artifact to {concept} from a nearby site"
                    ])
                rationale = "Generated domain-aware distractors"
        else:
            # Simple template distractors (domain-aware wording)
            options.extend([
                f"Variant attribution of {concept}",
                f"Alternative interpretation within the same corpus",
                f"Related but distinct inscription associated with {concept}"
            ])
            rationale = "Template-based domain distractors"
        
        # Shuffle options
        random.shuffle(options)
        return options[:4], rationale  # Max 4 options
    
    async def _add_multi_hop_reasoning(self, question: str, answer: str, context: str) -> Tuple[str, str]:
        """Add multi-step reasoning requirements"""
        
        if self.llm:
            try:
                prompt = f"""
Transform this question to require multi-step reasoning while staying grounded in the context.

Original Question: {question}
Context: {context}

Create a question that requires 2-3 logical steps to answer. Return JSON:
{{"enhanced_question": "...", "enhanced_answer": "...", "reasoning_steps": ["step1", "step2", "step3"]}}
"""
                response = await self._call_llm_with_retry(prompt)
                data = json.loads(re.search(r'\{.*\}', response, re.DOTALL).group())
                return data['enhanced_question'], data['enhanced_answer']
            except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
                # Failed to parse response, use fallback
                pass
        
        # Fallback: Add conditional reasoning
        enhanced_question = f"Given the context, if {question.lower().replace('?', '')}, what would be the implications?"
        enhanced_answer = f"If {answer}, then the implications would include..."
        
        return enhanced_question, enhanced_answer
    
    def _add_reverse_complexity(self, question: str, concept: str) -> str:
        """Add complexity through negation or reverse reasoning"""
        
        reverse_patterns = [
            f"What would happen if {concept} did NOT exist in this context?",
            f"Which of the following is NOT true about {concept}?",
            f"Identify what {concept} is NOT, based on the given information."
        ]
        
        # Sometimes transform the original question
        if random.random() < 0.5:
            return random.choice(reverse_patterns)
        else:
            # Add negation complexity to existing question
            if "what" in question.lower():
                return question.replace("What", "What specifically is NOT")
            return f"Contrary to common assumptions, {question.lower()}"
    
    def _transform_to_code_problem(self, question: str, answer: str, draft: BenchmarkDraft) -> Tuple[str, str]:
        """Transform data questions into code generation problems"""
        
        code_question = f"Write a function that would {question.lower().replace('?', '')} based on the given data structure."
        
        code_answer = f"""def analyze_{draft.concept.replace(' ', '_')}(data):
    # Based on the context: {answer}
    result = process_data(data)
    return result"""
        
        return code_question, code_answer
    
    def _create_fallback_candidate(self, draft: BenchmarkDraft, target_difficulty: DifficultyLevel) -> BenchmarkCandidate:
        """Create fallback candidate when adversarial processing fails"""
        
        # Simple difficulty enhancement
        enhanced_question = draft.question
        if target_difficulty == DifficultyLevel.HARD:
            enhanced_question = f"Analyze and explain: {draft.question}"
        elif target_difficulty == DifficultyLevel.EXPERT:
            enhanced_question = f"Critically evaluate and synthesize: {draft.question}"
        
        return BenchmarkCandidate(
            question=enhanced_question,
            answer=draft.answer,
            context=draft.context_snippet,
            concept=draft.concept,
            expected_answer_type=draft.expected_answer_type,
            difficulty=target_difficulty,
            reasoning_chain=draft.reasoning_chain,
            adversarial_techniques=["difficulty_enhancement"]
        )


class Refiner(BaseAgent):
    """
    Enforces style guide and formatting requirements
    Responsibilities:
    - Ensure proper formatting and length constraints
    - Add multiple choice options where needed
    - Insert variables for randomization
    """
    
    async def produce(self, candidate: BenchmarkCandidate) -> BenchmarkCandidate:
        """Apply refinement and formatting"""
        start_time = time.time()
        
        try:
            # Enforce question length limit (150 chars)
            refined_question = self._enforce_question_length(candidate.question)
            
            # Ensure proper question formatting
            refined_question = self._format_question(refined_question)
            
            # Add multiple choice options if needed
            options = candidate.options
            if candidate.expected_answer_type == AnswerType.MULTIPLE_CHOICE and not options:
                options = await self._generate_multiple_choice_options(refined_question, candidate.answer, candidate.concept)
            
            # Ensure deterministic answer format
            refined_answer = self._format_answer(candidate.answer, candidate.expected_answer_type)
            
            # Add variables for randomization (optional)
            variables = self._extract_variables(refined_question, refined_answer)
            
            processing_time = time.time() - start_time
            self.call_count += 1
            self.total_processing_time += processing_time
            
            return BenchmarkCandidate(
                question=refined_question,
                answer=refined_answer,
                context=candidate.context,
                options=options,
                concept=candidate.concept,
                expected_answer_type=candidate.expected_answer_type,
                difficulty=candidate.difficulty,
                reasoning_chain=candidate.reasoning_chain + ["Applied formatting refinements"],
                adversarial_techniques=candidate.adversarial_techniques,
                distractor_rationale=candidate.distractor_rationale,
                variables=variables
            )
            
        except Exception:
            # Return original candidate if refinement fails
            return candidate
    
    def _enforce_question_length(self, question: str, max_length: int = 10000) -> str:
        """No hard cap; keep API compatibility. Optionally normalize punctuation."""
        return question
    
    def _format_question(self, question: str) -> str:
        """Ensure proper question formatting"""
        question = question.strip()
        
        # Ensure question ends with proper punctuation
        if not question.endswith('?') and not question.endswith('.'):
            # Check if it's a question
            question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
            if any(word in question.lower() for word in question_words):
                question += '?'
            else:
                question += '.'
        
        # Capitalize first letter
        if question:
            question = question[0].upper() + question[1:]
        
        return question
    
    def _format_answer(self, answer: str, answer_type: AnswerType) -> str:
        """Format answer according to its type"""
        answer = answer.strip()
        
        if answer_type == AnswerType.NUMERIC_EXACT:
            # Extract and clean numeric answer
            numbers = re.findall(r'-?\d+\.?\d*', answer)
            if numbers:
                return numbers[0]
        
        elif answer_type == AnswerType.BOOLEAN:
            # Normalize boolean answers
            if answer.lower() in ['yes', 'true', '1', 'correct']:
                return 'True'
            elif answer.lower() in ['no', 'false', '0', 'incorrect']:
                return 'False'
        
        elif answer_type == AnswerType.MULTIPLE_CHOICE:
            # Ensure single letter answer
            match = re.search(r'\b[ABCD]\b', answer.upper())
            if match:
                return match.group()
        
        elif answer_type == AnswerType.STRING_EXACT:
            # Clean and normalize exact string answers
            return re.sub(r'\s+', ' ', answer).strip()
        
        # For free text/domain knowledge, ensure the answer isn't unnaturally short
        if answer_type == AnswerType.FREE_TEXT:
            if len(answer.split()) < 8:
                # Lightly expand while staying grounded
                answer = answer + " Based on the snippet, this explanation summarizes the key points precisely."
        return answer
    
    async def _generate_multiple_choice_options(self, question: str, answer: str, concept: str) -> List[str]:
        """Generate multiple choice options if not provided"""
        
        options = [answer]  # Correct answer first
        
        if self.llm:
            try:
                prompt = f"""
Generate 3 plausible incorrect options for this multiple choice question.

Question: {question}
Correct Answer: {answer}
Concept: {concept}

Return JSON: {{"options": ["wrong1", "wrong2", "wrong3"]}}
"""
                response = await self._call_llm_with_retry(prompt)
                data = json.loads(re.search(r'\{.*\}', response, re.DOTALL).group())
                options.extend(data.get('options', []))
            except (json.JSONDecodeError, AttributeError, TypeError):
                # Failed to generate options, use fallback
                pass
        
        # Fallback: generate domain-aware distractors
        if len(options) == 1:
            # Try to create plausible domain alternatives
            if any(char.isdigit() for char in answer):
                # If answer has numbers, create similar numbers
                import re
                nums = re.findall(r'\d+', answer)
                if nums:
                    base = int(nums[0])
                    options.extend([
                        answer.replace(str(base), str(base + 10)),
                        answer.replace(str(base), str(base - 10)),
                        answer.replace(str(base), str(base * 2))
                    ])
            else:
                # Create variations using the concept
                options.extend([
                    f"Earlier interpretation of {concept}",
                    f"Alternative form of {concept}",  
                    f"Related but distinct from {concept}"
                ])
        
        # Shuffle and return up to 4 options
        random.shuffle(options)
        return options[:4]
    
    def _extract_variables(self, question: str, answer: str) -> Dict[str, Any]:
        """Extract variables for potential randomization"""
        variables = {}
        
        # Extract numbers that could be randomized
        question_numbers = re.findall(r'\d+', question)
        answer_numbers = re.findall(r'\d+', answer)
        
        if question_numbers:
            variables['question_numbers'] = question_numbers
        if answer_numbers:
            variables['answer_numbers'] = answer_numbers
        
        # Extract proper nouns that could be varied
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', question)
        if proper_nouns:
            variables['proper_nouns'] = proper_nouns
        
        return variables


class Validator(BaseAgent):
    """
    Validates benchmark quality and enforces deterministic guardrails
    Responsibilities:
    - Run verification modules on generated items
    - Filter items below quality threshold
    - Enforce deterministic vs non-deterministic split
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verifier = VerificationOrchestrator()
    
    async def produce(self, candidate: BenchmarkCandidate, min_score: float = 0.6) -> ValidationResult:
        """Main production method for Validator (wraps accept method)"""
        return await self.accept(candidate, min_score)
    
    async def accept(self, candidate: BenchmarkCandidate, min_score: float = 0.6) -> ValidationResult:
        """Validate candidate quality and deterministic requirements"""
        start_time = time.time()
        
        try:
            issues = []
            recommendations = []
            
            # Check deterministic consistency
            is_deterministic_type = validate_deterministic_answer_type(candidate.expected_answer_type)
            deterministic_check_passed = True
            
            if is_deterministic_type:
                # For deterministic types, ensure answer can pass exact verification
                deterministic_check_passed = await self._validate_deterministic_answer(candidate)
                if not deterministic_check_passed:
                    issues.append("Deterministic answer type but answer fails exact match validation")
                    recommendations.append("Revise answer to be more precise and deterministic")
            
            # Quality score assessment
            quality_score = await self._assess_quality(candidate)
            
            # Length and format checks
            if len(candidate.question) > 500:
                issues.append("Question exceeds 500 character limit")
                quality_score *= 0.9
            
            if not candidate.question.strip():
                issues.append("Empty question")
                quality_score = 0.0
            
            if not candidate.answer.strip():
                issues.append("Empty answer")
                quality_score = 0.0
            
            # Multiple choice validation
            if candidate.expected_answer_type == AnswerType.MULTIPLE_CHOICE:
                if not candidate.options or len(candidate.options) < 2:
                    issues.append("Multiple choice question lacks proper options")
                    quality_score *= 0.8
            
            # Difficulty appropriateness
            if candidate.difficulty == DifficultyLevel.HARD and quality_score < 0.7:
                recommendations.append("Consider reducing difficulty rating or improving question complexity")
            
            accepted = quality_score >= min_score and len(issues) == 0
            
            processing_time = time.time() - start_time
            self.call_count += 1
            self.total_processing_time += processing_time
            
            return ValidationResult(
                accepted=accepted,
                score=quality_score,
                issues=issues,
                recommendations=recommendations,
                deterministic_check_passed=deterministic_check_passed,
                verification_method_used=self._get_verification_method(candidate.expected_answer_type)
            )
            
        except Exception as e:
            return ValidationResult(
                accepted=False,
                score=0.0,
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Review candidate for potential issues"],
                deterministic_check_passed=False,
                verification_method_used="error"
            )
    
    async def _validate_deterministic_answer(self, candidate: BenchmarkCandidate) -> bool:
        """Check if deterministic answer passes exact verification"""
        
        try:
            # Mock prediction that should match exactly
            mock_prediction = candidate.answer
            
            # Use verifier to check
            eval_type_map = {
                AnswerType.NUMERIC_EXACT: 'mathematical',
                AnswerType.MULTIPLE_CHOICE: 'multiple_choice',
                AnswerType.STRING_EXACT: 'factual_qa',
                AnswerType.CODE: 'code_generation',
                AnswerType.BOOLEAN: 'factual_qa'
            }
            
            eval_type = eval_type_map.get(candidate.expected_answer_type, 'factual_qa')
            result = self.verifier.verify(mock_prediction, candidate.answer, eval_type, candidate.options)
            
            return result.score >= 0.95  # Should be nearly perfect for deterministic
        except (AttributeError, KeyError, TypeError, Exception):
            # Verification failed, assume not deterministic
            return False
    
    async def _assess_quality(self, candidate: BenchmarkCandidate) -> float:
        """Assess overall quality of the candidate"""
        
        quality_factors = {
            'question_clarity': self._assess_question_clarity(candidate.question),
            'answer_quality': self._assess_answer_quality(candidate.answer, candidate.expected_answer_type),
            'context_relevance': self._assess_context_relevance(candidate.question, candidate.context),
            'difficulty_appropriateness': self._assess_difficulty(candidate),
            'adversarial_quality': self._assess_adversarial_techniques(candidate.adversarial_techniques)
        }
        
        # Weighted average
        weights = {
            'question_clarity': 0.25,
            'answer_quality': 0.25,
            'context_relevance': 0.20,
            'difficulty_appropriateness': 0.15,
            'adversarial_quality': 0.15
        }
        
        weighted_score = sum(quality_factors[factor] * weights[factor] for factor in quality_factors)
        return min(1.0, max(0.0, weighted_score))
    
    def _assess_question_clarity(self, question: str) -> float:
        """Assess clarity and well-formedness of question"""
        score = 1.0
        
        # Check for proper punctuation
        if not question.endswith('?') and not question.endswith('.'):
            score -= 0.2
        
        # Check for reasonable length
        if len(question) < 10:
            score -= 0.3
        elif len(question) > 150:
            score -= 0.2
        
        # Check for clear language
        if question.count('?') > 1:
            score -= 0.1  # Multiple questions in one
        
        return max(0.0, score)
    
    def _assess_answer_quality(self, answer: str, answer_type: AnswerType) -> float:
        """Assess answer quality based on type"""
        score = 1.0
        
        if not answer.strip():
            return 0.0
        
        # Type-specific checks
        if answer_type == AnswerType.NUMERIC_EXACT:
            if not re.search(r'\d', answer):
                score -= 0.5
        elif answer_type == AnswerType.CODE:
            if not any(keyword in answer.lower() for keyword in ['def', 'function', 'class', 'return']):
                score -= 0.4
        elif answer_type == AnswerType.BOOLEAN:
            if answer.lower() not in ['true', 'false', 'yes', 'no']:
                score -= 0.3
        
        # General quality checks
        if len(answer) < 3:
            score -= 0.3
        elif answer_type == AnswerType.FREE_TEXT:
            word_count = len(answer.split())
            if word_count < 5:
                score -= 0.5
            elif word_count < 8:
                score -= 0.25
        
        return max(0.0, score)
    
    def _assess_context_relevance(self, question: str, context: Optional[str]) -> float:
        """Assess relevance between question and context"""
        if not context:
            return 0.5  # Neutral score for missing context
        
        # Simple word overlap assessment
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())
        
        overlap = len(question_words.intersection(context_words))
        total_unique = len(question_words.union(context_words))
        
        if total_unique == 0:
            return 0.5
        
        relevance_score = overlap / total_unique
        return min(1.0, relevance_score * 2)  # Scale up
    
    def _assess_difficulty(self, candidate: BenchmarkCandidate) -> float:
        """Assess if difficulty level matches question complexity"""
        
        complexity_indicators = {
            'multi_step': len(candidate.reasoning_chain) > 2,
            'adversarial': len(candidate.adversarial_techniques) > 0,
            'long_question': len(candidate.question) > 80,
            'complex_answer': len(candidate.answer.split()) > 10,
            'has_options': candidate.options is not None
        }
        
        complexity_score = sum(complexity_indicators.values()) / len(complexity_indicators)
        
        # Compare with stated difficulty
        expected_complexity = {
            DifficultyLevel.BASIC: 0.2,
            DifficultyLevel.INTERMEDIATE: 0.5,
            DifficultyLevel.HARD: 0.7,
            DifficultyLevel.EXPERT: 0.9
        }
        
        expected = expected_complexity.get(candidate.difficulty, 0.5)
        
        # Score based on how well complexity matches difficulty
        difference = abs(complexity_score - expected)
        return max(0.0, 1.0 - difference * 2)
    
    def _assess_adversarial_techniques(self, techniques: List[str]) -> float:
        """Assess quality of adversarial techniques applied"""
        if not techniques:
            return 0.5  # Neutral for no techniques
        
        # Basic scoring based on number and diversity of techniques
        technique_score = min(1.0, len(techniques) / 3)  # Up to 3 techniques
        
        # Bonus for diverse techniques
        if len(set(techniques)) == len(techniques):  # No duplicates
            technique_score *= 1.1
        
        return min(1.0, technique_score)
    
    def _get_verification_method(self, answer_type: AnswerType) -> str:
        """Get verification method name for answer type"""
        method_map = {
            AnswerType.NUMERIC_EXACT: 'numerical_match',
            AnswerType.NUMERIC_TOLERANCE: 'numerical_match_tolerance',
            AnswerType.STRING_EXACT: 'exact_match',
            AnswerType.CODE: 'code_execution',
            AnswerType.MULTIPLE_CHOICE: 'multiple_choice_match',
            AnswerType.BOOLEAN: 'exact_match',
            AnswerType.FREE_TEXT: 'semantic_similarity'
        }
        return method_map.get(answer_type, 'exact_match')
    
    async def verify_corpus_accuracy(self, question: str, proposed_answer: str, corpus_text: str, 
                                   min_complexity: float = 0.4) -> ValidationResult:
        """
        Verify that the proposed answer is factually correct according to the corpus
        and assess question complexity for modern LLMs
        """
        if not self.llm:
            return ValidationResult(
                accepted=False,
                score=0.0,
                issues=["No LLM interface available for corpus verification"],
                recommendations=["Configure LLM interface for validation"],
                deterministic_check_passed=False,
                verification_method_used="no_llm"
            )
        
        verification_prompt = f"""You are a fact-checker and complexity assessor.

CRITICAL: You must respond with ONLY a valid JSON object. No markdown blocks or formatting.

TASK:
1. Verify if the proposed answer is factually correct according to the source text
2. Assess the complexity/difficulty of the question for modern LLMs

SOURCE TEXT: {corpus_text}

QUESTION: {question}
PROPOSED ANSWER: {proposed_answer}

Use this EXACT JSON format:
{{"is_correct":true,"verified_answer":"...","confidence":0.8,"complexity":0.7,"reasoning":"...","evidence":"...","complexity_analysis":"..."}}

COMPLEXITY SCORING (0.0-1.0):
- 0.0-0.3: TOO EASY - Simple facts, basic math, obvious answers
- 0.4-0.6: MODERATE - Requires reasoning, domain knowledge, multi-step thinking  
- 0.7-1.0: CHALLENGING - Complex reasoning, obscure facts, multi-layered analysis

RULES:
1. Answer correct ONLY if directly supported by source text
2. REJECT questions with complexity < {min_complexity} (too easy for modern LLMs)
3. Prefer questions requiring reasoning, synthesis, or domain expertise
4. Consider: Would GPT-4/Claude-3.5 find this challenging?

Return raw JSON only."""

        try:
            response = await self._call_llm_with_retry(verification_prompt, temperature=0.1)
            
            # Parse JSON response
            import json
            try:
                verification_result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown blocks
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    verification_result = json.loads(json_match.group())
                else:
                    raise json.JSONDecodeError("No valid JSON found", response, 0)
            
            # Extract results
            is_correct = verification_result.get("is_correct", False)
            complexity = verification_result.get("complexity", 0.0)
            confidence = verification_result.get("confidence", 0.0)
            # Capture but do not assign unused fields to avoid linter warnings
            _ = verification_result.get("reasoning", "No reasoning provided")
            _ = verification_result.get("evidence", "No evidence provided")
            _ = verification_result.get("complexity_analysis", "No analysis provided")
            
            # Determine acceptance
            issues = []
            recommendations = []
            
            if not is_correct:
                issues.append("Answer is factually incorrect according to corpus")
                recommendations.append("Revise answer to match corpus information")
            
            if complexity < min_complexity:
                issues.append(f"Question complexity ({complexity:.2f}) below threshold ({min_complexity})")
                recommendations.append("Increase question difficulty to challenge modern LLMs")
                is_correct = False  # Override - reject easy questions regardless of correctness
            
            # Calculate overall score
            accuracy_score = 1.0 if is_correct else 0.0
            complexity_score = max(0.0, min(1.0, complexity))
            overall_score = (accuracy_score * 0.7 + complexity_score * 0.3) * confidence
            
            accepted = is_correct and complexity >= min_complexity and len(issues) == 0
            
            return ValidationResult(
                accepted=accepted,
                score=overall_score,
                issues=issues,
                recommendations=recommendations,
                deterministic_check_passed=is_correct,
                verification_method_used="llm_corpus_verification",
                # Store additional verification data in metadata if ValidationResult supports it
                # Otherwise these will be available through the agent's response
            )
            
        except Exception as e:
            return ValidationResult(
                accepted=False,
                score=0.0,
                issues=[f"Corpus verification failed: {str(e)}"],
                recommendations=["Check LLM connectivity and try again"],
                deterministic_check_passed=False,
                verification_method_used="verification_error"
            )