from typing import List, Dict
from .agentic import AgenticBenchmarkGenerator
from ..utils.config import EvaluationType
from .verification import VerificationOrchestrator
from ..llm.mock_interface import MockLLMInterface


class LocalQwenEvaluator:
    """Local evaluation system using mock LLM to simulate Qwen behavior"""
    
    def __init__(self):
        self.mock_llm = MockLLMInterface()
        self.verification_orchestrator = VerificationOrchestrator()

    async def create_fictional_benchmark(self, corpus_text: str, num_questions: int = 5):
        """Create benchmark from fictional content using agentic generation"""
        # Use your agentic benchmark generator
        generator = AgenticBenchmarkGenerator(EvaluationType.DOMAIN_KNOWLEDGE)
        
        benchmark_items = await generator.generate_benchmark_async(
            corpus_text=corpus_text,
            num_questions=num_questions
        )
        
        if not benchmark_items:
            return self._fallback_question_generation(corpus_text, num_questions)
        
        # Convert to standard format for compatibility (dict)
        standard_items = []
        for item in benchmark_items:
            if hasattr(item, 'to_standard_benchmark_item'):
                standard_items.append(item.to_standard_benchmark_item())
            else:
                standard_items.append(item)
                
        return standard_items

    def _fallback_question_generation(self, corpus_text: str, num_questions: int):
        """Fallback question generation if agentic system fails"""
        # Extract key facts and numbers from the fictional corpus
        import re
        
        # Find numbers for mathematical questions
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', corpus_text)
        
        # Find key entities/concepts
        words = corpus_text.split()
        concepts = [w for w in words if len(w) > 5 and w[0].isupper()][:10]
        
        # Create simple questions manually
        fallback_questions = []
        
        if numbers:
            fallback_questions.append({
                'question': 'What is the first number mentioned in the text?',
                'answer': numbers[0],
                'context': corpus_text[:200] + '...',
                'concept': 'numerical_fact',
                'type': 'factual'
            })
        
        if concepts:
            fallback_questions.append({
                'question': f'What is {concepts[0]}?',
                'answer': f'{concepts[0]} is a key entity mentioned in the fictional world.',
                'context': corpus_text[:200] + '...',
                'concept': concepts[0].lower(),
                'type': 'factual'
            })
        
        # Add more generic questions
        fallback_questions.extend([
            {
                'question': 'What is the main setting described in this text?',
                'answer': 'A fictional fantasy world with magical elements.',
                'context': corpus_text[:200] + '...',
                'concept': 'setting',
                'type': 'comprehension'
            },
            {
                'question': 'What kind of world is being described?',
                'answer': 'A fantasy world with fictional characters and magical elements.',
                'context': corpus_text[:200] + '...',
                'concept': 'genre',
                'type': 'analysis'
            }
        ])
        
        return fallback_questions[:num_questions]

    async def simulate_qwen_responses(self, questions: List[Dict]):
        """Simulate Qwen responses using mock LLM"""
        responses = []
        for q in questions:
            # Handle both dicts and Pydantic models
            if isinstance(q, dict):
                question = q['question']
                context = q.get('context', '')
                expected_answer = q['answer']
            else:
                question = getattr(q, 'question', '')
                context = getattr(q, 'context', '')
                expected_answer = getattr(q, 'answer', '')

            prompt = f"Question: {question}\nContext: {context}"
            mock_resp = await self.mock_llm.generate_response(prompt)
            responses.append({
                'question': question,
                'qwen_response': mock_resp.text,
                'expected_answer': expected_answer,
                'context': context
            })
        return responses

    def evaluate_responses(self, qwen_responses: List[Dict]):
        """Evaluate responses using verification system"""
        evaluation_results = []
        scores = []
        
        for response_data in qwen_responses:
            qwen_response = response_data['qwen_response']
            expected_answer = response_data['expected_answer']
            question = response_data['question']
            
            verification_result = self.verification_orchestrator.verify(
                prediction=qwen_response,
                ground_truth=expected_answer,
                eval_type="domain_factual",
                question=question
            )
            
            scores.append(verification_result.score)
            evaluation_results.append({
                'question': question,
                'qwen_response': qwen_response,
                'expected_answer': expected_answer,
                'score': verification_result.score,
                'method': verification_result.method,
                'details': verification_result.details
            })
        
        return evaluation_results, scores

    def generate_report(self, evaluation_results, scores, corpus_info):
        """Generate summary report"""
        mean_score = sum(scores) / len(scores) if scores else 0
        return {
            'aggregate_metrics': {
                'mean_score': mean_score,
                'num_questions': len(evaluation_results)
            },
            'corpus_info': corpus_info,
            'results': evaluation_results,
            'evaluation_type': 'qwen_local',
            'model': 'Qwen3-0.6B (Local)',
            'system_capabilities': {
                'local_inference': True,
                'gpu_acceleration': False,
                'no_api_required': True
            }
        }

class RealQwenEvaluator:
    """🚀 Real Local Qwen Evaluation using HuggingFace transformers"""
    
    def __init__(self, model_key: str = "qwen3-0.6b"):
        self.model_key = model_key
        self.qwen_interface = None
        self.verification_orchestrator = VerificationOrchestrator()
        self.model_loaded = False
        
    async def initialize_model(self):
        """Initialize the Qwen model"""
        if self.model_loaded:
            return True
            
        try:
            from docs_to_eval.llm.qwen_local_interface import QwenModelFactory
            self.qwen_interface = QwenModelFactory.create_interface(
                self.model_key,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True
            )
            
            success = self.qwen_interface.load_model()
            if success:
                self.model_loaded = True
                return True
            return False
                
        except Exception:
            return False
    
    async def create_fictional_benchmark(self, corpus_text: str, num_questions: int = 5):
        """Create benchmark from fictional content using agentic generation"""
        generator = AgenticBenchmarkGenerator(EvaluationType.DOMAIN_KNOWLEDGE)
        benchmark_items = await generator.generate_benchmark_async(
            corpus_text=corpus_text,
            num_questions=num_questions
        )
        if not benchmark_items:
            # Re-use logic from LocalQwenEvaluator
            return LocalQwenEvaluator()._fallback_question_generation(corpus_text, num_questions)
            
        # Convert to standard format for compatibility (dict)
        standard_items = []
        for item in benchmark_items:
            if hasattr(item, 'to_standard_benchmark_item'):
                standard_items.append(item.to_standard_benchmark_item())
            else:
                standard_items.append(item)
                
        return standard_items
    
    async def evaluate_with_real_qwen(self, questions):
        """🚀 Evaluate questions using real Qwen model"""
        if not self.model_loaded:
            success = await self.initialize_model()
            if not success:
                raise RuntimeError("Failed to initialize Qwen model")
        
        qwen_responses = []
        for item in questions:
            if hasattr(item, 'question'):
                question, context, answer = item.question, item.context, item.answer
            else:
                question, context, answer = item.get('question', ''), item.get('context', ''), item.get('answer', '')
            
            prompt = f"Context: {context}\nQuestion: {question}" if context else f"Question: {question}"
            
            try:
                response = await self.qwen_interface.generate_response(prompt)
                qwen_responses.append({
                    'question': question,
                    'qwen_response': response.text if response.success else f"Error: {response.error}",
                    'expected_answer': answer,
                    'context': context,
                    'tokens_generated': response.tokens_generated if response.success else 0,
                    'generation_time': response.generation_time
                })
            except Exception as e:
                qwen_responses.append({
                    'question': question,
                    'qwen_response': f"Error: {str(e)}",
                    'expected_answer': answer,
                    'context': context,
                    'tokens_generated': 0,
                    'generation_time': 0.0
                })
        
        return qwen_responses
    
    def evaluate_responses(self, qwen_responses):
        """Evaluate Qwen responses using verification system"""
        return LocalQwenEvaluator().evaluate_responses(qwen_responses)
    
    def cleanup(self):
        """Clean up model resources"""
        if self.qwen_interface:
            self.qwen_interface.unload_model()
            self.qwen_interface = None
        self.model_loaded = False
