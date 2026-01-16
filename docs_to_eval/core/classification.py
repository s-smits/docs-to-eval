"""
LLM-based evaluation type classifier
Determines the best evaluation methodology for a given corpus using an LLM
"""

import json
import re
from typing import Dict, List, Any
from docs_to_eval.utils.config import EvaluationType, EVAL_TYPES, analyze_corpus_content
from ..utils.logging import get_logger


class MockLLM:
    """Mock LLM interface for classification"""
    
    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate mock LLM response based on prompt analysis"""
        if "mathematical" in prompt.lower() or "math" in prompt.lower():
            return self._generate_math_response()
        elif "code" in prompt.lower() or "programming" in prompt.lower():
            return self._generate_code_response()
        elif "factual" in prompt.lower() or "knowledge" in prompt.lower():
            return self._generate_factual_response()
        elif "creative" in prompt.lower() or "writing" in prompt.lower():
            return self._generate_creative_response()
        else:
            return self._generate_default_response(prompt)
    
    def _generate_math_response(self) -> str:
        return """Analysis: This corpus contains mathematical content with equations, calculations, and numerical reasoning.

Primary Evaluation Type: mathematical
Secondary Types: factual_qa
Confidence: 0.85

Reasoning: The presence of mathematical expressions, formulas, and numerical problem-solving indicates that mathematical evaluation with exact answer verification is most appropriate."""
    
    def _generate_code_response(self) -> str:
        return """Analysis: This corpus contains programming concepts, code examples, and technical implementation details.

Primary Evaluation Type: code_generation
Secondary Types: domain_knowledge
Confidence: 0.90

Reasoning: Code-related content requires execution-based verification to determine correctness."""
    
    def _generate_factual_response(self) -> str:
        return """Analysis: This corpus contains factual information and knowledge-based content.

Primary Evaluation Type: factual_qa
Secondary Types: domain_knowledge, multiple_choice
Confidence: 0.80

Reasoning: Factual content is best evaluated through exact match verification for specific facts."""
    
    def _generate_creative_response(self) -> str:
        return """Analysis: This corpus contains creative or subjective content requiring qualitative assessment.

Primary Evaluation Type: creative_writing
Secondary Types: summarization
Confidence: 0.75

Reasoning: Creative content requires similarity-based or LLM-judge evaluation due to subjective nature."""
    
    def _generate_default_response(self, prompt: str) -> str:
        # Analyze prompt content to determine response
        content_scores = analyze_corpus_content(prompt)
        primary_type = max(content_scores.items(), key=lambda x: x[1])[0] if content_scores else EvaluationType.DOMAIN_KNOWLEDGE
        
        return f"""Analysis: This corpus contains domain-specific content that requires specialized evaluation.

Primary Evaluation Type: {primary_type}
Secondary Types: factual_qa, reading_comprehension
Confidence: 0.70

Reasoning: Based on content analysis, this appears to be specialized domain knowledge that would benefit from {primary_type} evaluation methodology."""


class ClassificationResult:
    """Structured result from corpus classification"""
    
    def __init__(self, primary_type: EvaluationType, secondary_types: List[EvaluationType],
                 confidence: float, analysis: str, reasoning: str):
        self.primary_type = primary_type
        self.secondary_types = secondary_types
        self.confidence = confidence
        self.analysis = analysis
        self.reasoning = reasoning
        self.pipeline = self._create_pipeline_config(primary_type)
        self.fallback_analysis = {}
    
    def _create_pipeline_config(self, eval_type: EvaluationType) -> Dict[str, Any]:
        """Create pipeline configuration for evaluation type"""
        if eval_type not in EVAL_TYPES:
            eval_type = EvaluationType.DOMAIN_KNOWLEDGE
        
        config = EVAL_TYPES[eval_type].model_dump()
        config['eval_type'] = eval_type
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'primary_type': self.primary_type,
            'secondary_types': self.secondary_types,
            'confidence': self.confidence,
            'analysis': self.analysis,
            'reasoning': self.reasoning,
            'pipeline': self.pipeline,
            'fallback_analysis': self.fallback_analysis
        }


class EvaluationTypeClassifier:
    """Classifies the best evaluation type for a given corpus using LLM reasoning"""
    
    def __init__(self, llm=None):
        self.llm = llm or MockLLM(temperature=0.0)
        self.classification_prompt_template = self._get_classification_prompt()
        self.logger = get_logger(self.__class__.__name__)
    
    def _get_classification_prompt(self) -> str:
        """Get the classification prompt template"""
        available_types = json.dumps({k.value: v.model_dump() for k, v in EVAL_TYPES.items()}, indent=2)
        
        return f"""
You are an expert in LLM evaluation methodologies. Your task is to analyze a text corpus and determine 
the most appropriate evaluation type(s) for benchmarking LLMs on this content.

Available Evaluation Types:
{available_types}

For the given corpus, please:
1. Analyze the content type, domain, and structure
2. Identify patterns that suggest specific evaluation approaches
3. Consider whether answers will be deterministic (exact match) or non-deterministic (similarity-based)
4. Recommend the primary evaluation type and 1-2 secondary types
5. Provide confidence scores and reasoning

Corpus to analyze:
{{corpus_text}}

Please respond in this exact format:
Analysis: [Your analysis of the corpus content and characteristics]

Primary Evaluation Type: [single type from the available list]
Secondary Types: [1-2 additional types, comma-separated]
Confidence: [0.0-1.0 confidence score]

Reasoning: [Detailed explanation for your choice, focusing on why this evaluation type 
is most appropriate for measuring LLM performance on this content type]
"""
    
    def classify_corpus(self, corpus_text: str, max_length: int = 128000) -> ClassificationResult:
        """Classify corpus and return structured result"""
        # Truncate corpus if too long for LLM context
        if len(corpus_text) > max_length:
            corpus_text = corpus_text[:max_length] + "..."
        
        self.logger.info("Starting corpus classification", corpus_length=len(corpus_text))
        
        # Use statistical analysis as primary method for reliability
        content_scores = analyze_corpus_content(corpus_text)
        
        # Get the most confident statistical match
        if content_scores and max(content_scores.values()) > 0:
            statistical_primary = max(content_scores.items(), key=lambda x: x[1])[0]
            max_score = max(content_scores.values())
            confidence = min(0.9, content_scores[statistical_primary] / max_score * 0.8)
        else:
            statistical_primary = EvaluationType.DOMAIN_KNOWLEDGE
            confidence = 0.6
        
        # Ensure we have a valid evaluation type
        if statistical_primary not in EVAL_TYPES:
            statistical_primary = EvaluationType.DOMAIN_KNOWLEDGE
        
        # Get secondary types (next best matches)
        secondary_types = []
        if content_scores:
            sorted_scores = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
            for eval_type, score in sorted_scores[1:3]:  # Get 2nd and 3rd best
                if eval_type in EVAL_TYPES and eval_type != statistical_primary:
                    secondary_types.append(eval_type)
        
        # If we don't have enough secondary types, add defaults
        if len(secondary_types) == 0:
            secondary_types = [EvaluationType.FACTUAL_QA] if statistical_primary != EvaluationType.FACTUAL_QA else [EvaluationType.READING_COMPREHENSION]
        
        analysis = f"Corpus analysis identified {statistical_primary} as the primary evaluation type based on content patterns"
        reasoning = f"Statistical analysis shows strong indicators for {statistical_primary} evaluation methodology"
        
        result = ClassificationResult(
            primary_type=statistical_primary,
            secondary_types=secondary_types,
            confidence=confidence,
            analysis=analysis,
            reasoning=reasoning
        )
        
        # Add fallback analysis
        result.fallback_analysis = {
            'statistical_primary_type': statistical_primary,
            'content_scores': content_scores if content_scores else {},
            'corpus_stats': {
                'word_count': len(corpus_text.split()),
                'char_count': len(corpus_text),
                'sentences': len(re.findall(r'[.!?]+', corpus_text))
            }
        }
        
        self.logger.info("Classification completed", 
                        primary_type=statistical_primary,
                        confidence=confidence,
                        secondary_types=secondary_types)
        
        return result
    
    def _parse_classification_response(self, response: str, original_corpus: str) -> ClassificationResult:
        """Parse LLM response into structured format"""
        
        # Default values
        primary_type = EvaluationType.DOMAIN_KNOWLEDGE
        secondary_types = []
        confidence = 0.5
        analysis = ""
        reasoning = ""
        
        try:
            # Parse Analysis
            analysis_match = re.search(r'Analysis:\s*(.+?)(?=Primary Evaluation Type:|$)', response, re.DOTALL | re.IGNORECASE)
            if analysis_match:
                analysis = analysis_match.group(1).strip()
            
            # Parse Primary Type
            primary_match = re.search(r'Primary Evaluation Type:\s*([a-zA-Z_]+)', response, re.IGNORECASE)
            if primary_match:
                primary_type_str = primary_match.group(1).strip()
                try:
                    primary_type = EvaluationType(primary_type_str)
                except ValueError:
                    # Try to match against available types
                    for eval_type in EvaluationType:
                        if eval_type.value == primary_type_str:
                            primary_type = eval_type
                            break
            
            # Parse Secondary Types
            secondary_match = re.search(r'Secondary Types:\s*([a-zA-Z_,\s]+)', response, re.IGNORECASE)
            if secondary_match:
                secondary_types_str = [t.strip() for t in secondary_match.group(1).split(',')]
                for type_str in secondary_types_str:
                    try:
                        secondary_types.append(EvaluationType(type_str))
                    except ValueError:
                        continue
            
            # Parse Confidence
            confidence_match = re.search(r'Confidence:\s*([0-9.]+)', response, re.IGNORECASE)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                confidence = min(1.0, max(0.0, confidence))
            
            # Parse Reasoning
            reasoning_match = re.search(r'Reasoning:\s*(.+)', response, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                
        except Exception as e:
            self.logger.error("Error parsing LLM response", error=str(e))
        
        return ClassificationResult(primary_type, secondary_types, confidence, analysis, reasoning)
    
    def classify_with_examples(self, corpus_text: str, num_examples: int = 3) -> Dict[str, Any]:
        """Generate example questions for the recommended evaluation type"""
        classification = self.classify_corpus(corpus_text)
        
        # Generate sample questions based on the classification
        examples = self._generate_sample_questions(
            corpus_text, 
            classification.primary_type, 
            num_examples
        )
        
        result = classification.to_dict()
        result['sample_questions'] = examples
        return result
    
    def _generate_sample_questions(self, corpus_text: str, eval_type: EvaluationType, num_examples: int) -> List[Dict[str, Any]]:
        """Mock question generation based on evaluation type"""
        examples = []
        
        if eval_type == EvaluationType.MATHEMATICAL:
            examples = [
                {"question": "Solve: 2x + 5 = 15", "answer": "5", "type": "algebra"},
                {"question": "What is 25% of 80?", "answer": "20", "type": "percentage"},
                {"question": "Calculate the area of a circle with radius 3", "answer": "28.27", "type": "geometry"}
            ]
        elif eval_type == EvaluationType.CODE_GENERATION:
            examples = [
                {"question": "Write a function to reverse a string", "answer": "def reverse_string(s): return s[::-1]", "type": "string_manipulation"},
                {"question": "Implement binary search", "answer": "def binary_search(arr, target): ...", "type": "algorithms"},
                {"question": "Create a class for a linked list node", "answer": "class ListNode: def __init__(self, val=0): ...", "type": "data_structures"}
            ]
        elif eval_type == EvaluationType.FACTUAL_QA:
            examples = [
                {"question": "What is the capital of France?", "answer": "Paris", "type": "geography"},
                {"question": "Who invented the telephone?", "answer": "Alexander Graham Bell", "type": "history"},
                {"question": "What is the chemical symbol for gold?", "answer": "Au", "type": "science"}
            ]
        else:
            # Generate domain-specific examples from corpus
            from ..utils.text_processing import extract_keywords
            key_concepts = extract_keywords(corpus_text, max_keywords=5)
            
            for i, concept in enumerate(key_concepts[:num_examples]):
                examples.append({
                    "question": f"What is {concept}?",
                    "answer": f"[Answer about {concept} from corpus]",
                    "type": "domain_knowledge"
                })
        
        return examples[:num_examples]


def classify_and_configure(corpus_text: str, num_questions: int = 50) -> Dict[str, Any]:
    """Main function to classify corpus and generate evaluation configuration"""
    
    classifier = EvaluationTypeClassifier()
    classification = classifier.classify_with_examples(corpus_text, num_examples=5)
    
    # Create complete evaluation configuration
    config = {
        'classification': classification,
        'benchmark_config': {
            'num_questions': num_questions,
            'eval_type': classification['primary_type'],
            'pipeline': classification['pipeline'],
            'secondary_types': classification['secondary_types'],
            'temperature': 0.0 if classification['pipeline']['deterministic'] else 0.7,
            'verification_method': classification['pipeline']['verification'],
            'metrics': classification['pipeline']['metrics']
        },
        'corpus_analysis': classification['fallback_analysis']
    }
    
    return config


if __name__ == "__main__":
    # Test the classifier
    test_corpus = """
    Linear algebra is a branch of mathematics dealing with vector spaces and linear transformations.
    Key concepts include matrices, eigenvalues, and eigenvectors. A matrix A is invertible if there 
    exists a matrix B such that AB = BA = I, where I is the identity matrix. The determinant of a 
    2x2 matrix [[a,b],[c,d]] is calculated as ad - bc.
    """
    
    config = classify_and_configure(test_corpus, num_questions=20)
    print("Classification and Configuration:")
    print(json.dumps(config, indent=2, default=str))