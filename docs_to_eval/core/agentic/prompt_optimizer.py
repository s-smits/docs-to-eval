"""
Prompt Optimization System using Qwen3 30B
Analyzes and improves agent prompts for better performance
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

from ...llm.openrouter_interface import OpenRouterInterface, OpenRouterConfig


logger = logging.getLogger(__name__)


@dataclass
class PromptAnalysis:
    """Analysis results for a prompt"""
    original_prompt: str
    agent_name: str
    prompt_type: str
    
    # Analysis results
    clarity_score: float  # 0-1
    specificity_score: float  # 0-1
    instruction_quality_score: float  # 0-1
    overall_score: float  # 0-1
    
    # Identified issues
    issues: List[str]
    strengths: List[str]
    
    # Improvement suggestions
    suggestions: List[str]
    improved_prompt: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'agent_name': self.agent_name,
            'prompt_type': self.prompt_type,
            'scores': {
                'clarity': self.clarity_score,
                'specificity': self.specificity_score,
                'instruction_quality': self.instruction_quality_score,
                'overall': self.overall_score
            },
            'issues': self.issues,
            'strengths': self.strengths,
            'suggestions': self.suggestions,
            'has_improvement': self.improved_prompt is not None
        }


class PromptOptimizer:
    """
    Uses Qwen3 30B to analyze and improve agent prompts
    """
    
    def __init__(self, qwen_interface: Optional[OpenRouterInterface] = None):
        """
        Initialize prompt optimizer
        
        Args:
            qwen_interface: Qwen interface (creates default if None)
        """
        if qwen_interface is None:
            try:
                config = OpenRouterConfig()
                self.qwen = OpenRouterInterface(config)
            except Exception as e:
                logger.warning(f"Failed to initialize Qwen interface: {e}")
                self.qwen = None
        else:
            self.qwen = qwen_interface
        
        # Prompt analysis criteria
        self.analysis_criteria = {
            'clarity': "How clear and unambiguous are the instructions?",
            'specificity': "How specific and detailed are the requirements?",
            'instruction_quality': "How well-structured and actionable are the instructions?",
            'output_format': "How clearly is the expected output format specified?",
            'examples': "Are there sufficient examples or guidance provided?"
        }
    
    async def analyze_prompt(
        self, 
        prompt: str, 
        agent_name: str, 
        prompt_type: str,
        context: Optional[str] = None
    ) -> PromptAnalysis:
        """
        Analyze a single prompt for quality and effectiveness
        
        Args:
            prompt: The prompt to analyze
            agent_name: Name of the agent using this prompt
            prompt_type: Type of prompt (e.g., 'system', 'instruction', 'template')
            context: Optional context about the prompt's purpose
            
        Returns:
            PromptAnalysis with detailed analysis results
        """
        
        if not self.qwen:
            # Return default analysis if Qwen is not available
            return PromptAnalysis(
                original_prompt=prompt,
                agent_name=agent_name,
                prompt_type=prompt_type,
                clarity_score=0.5,
                specificity_score=0.5,
                instruction_quality_score=0.5,
                overall_score=0.5,
                issues=["Unable to analyze - Qwen interface not available"],
                strengths=[],
                suggestions=["Set up OpenRouter API key to enable analysis"]
            )
        
        analysis_prompt = f"""You are an expert in LLM prompt engineering. Analyze the following prompt for quality and effectiveness.

Agent: {agent_name}
Prompt Type: {prompt_type}
Context: {context or 'Not provided'}

PROMPT TO ANALYZE:
```
{prompt}
```

Please provide a comprehensive analysis in the following JSON format:

{{
    "clarity_score": 0.85,
    "specificity_score": 0.75,
    "instruction_quality_score": 0.90,
    "overall_score": 0.83,
    "issues": [
        "List specific issues found",
        "Be concrete and actionable"
    ],
    "strengths": [
        "List what works well",
        "Highlight effective elements"
    ],
    "suggestions": [
        "Specific improvement recommendations",
        "Actionable next steps"
    ]
}}

Focus on:
1. **Clarity**: Are instructions clear and unambiguous?
2. **Specificity**: Are requirements detailed enough?
3. **Instruction Quality**: Are instructions well-structured and actionable?
4. **Output Format**: Is the expected output format clearly specified?
5. **Examples**: Are there sufficient examples or guidance?

Provide scores from 0.0 to 1.0 where:
- 0.9-1.0: Excellent
- 0.7-0.89: Good  
- 0.5-0.69: Fair
- 0.3-0.49: Poor
- 0.0-0.29: Very Poor

Return only the JSON response:"""

        try:
            response = await self.qwen.generate_response(
                analysis_prompt,
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=1000
            )
            
            # Parse JSON response
            analysis_data = self._parse_analysis_response(response.text)
            
            return PromptAnalysis(
                original_prompt=prompt,
                agent_name=agent_name,
                prompt_type=prompt_type,
                clarity_score=analysis_data.get('clarity_score', 0.5),
                specificity_score=analysis_data.get('specificity_score', 0.5),
                instruction_quality_score=analysis_data.get('instruction_quality_score', 0.5),
                overall_score=analysis_data.get('overall_score', 0.5),
                issues=analysis_data.get('issues', []),
                strengths=analysis_data.get('strengths', []),
                suggestions=analysis_data.get('suggestions', [])
            )
        
        except Exception as e:
            logger.error(f"Failed to analyze prompt for {agent_name}: {e}")
            return PromptAnalysis(
                original_prompt=prompt,
                agent_name=agent_name,
                prompt_type=prompt_type,
                clarity_score=0.5,
                specificity_score=0.5,
                instruction_quality_score=0.5,
                overall_score=0.5,
                issues=[f"Analysis failed: {str(e)}"],
                strengths=[],
                suggestions=["Fix analysis system and retry"]
            )
    
    async def improve_prompt(
        self, 
        analysis: PromptAnalysis,
        improvement_focus: Optional[List[str]] = None
    ) -> str:
        """
        Generate an improved version of the prompt based on analysis
        
        Args:
            analysis: PromptAnalysis from analyze_prompt
            improvement_focus: Specific areas to focus on (optional)
            
        Returns:
            Improved prompt text
        """
        
        if not self.qwen:
            return analysis.original_prompt  # Return original if no Qwen
        
        focus_areas = improvement_focus or ['clarity', 'specificity', 'instruction_quality']
        
        improvement_prompt = f"""You are an expert prompt engineer. Based on the analysis below, create an improved version of the prompt.

ORIGINAL PROMPT:
```
{analysis.original_prompt}
```

ANALYSIS RESULTS:
- Agent: {analysis.agent_name}
- Type: {analysis.prompt_type}
- Overall Score: {analysis.overall_score:.2f}
- Clarity Score: {analysis.clarity_score:.2f}
- Specificity Score: {analysis.specificity_score:.2f}
- Instruction Quality Score: {analysis.instruction_quality_score:.2f}

IDENTIFIED ISSUES:
{chr(10).join(f"- {issue}" for issue in analysis.issues)}

IMPROVEMENT SUGGESTIONS:
{chr(10).join(f"- {suggestion}" for suggestion in analysis.suggestions)}

FOCUS AREAS: {', '.join(focus_areas)}

Please create an improved version that addresses the issues and implements the suggestions. The improved prompt should:

1. **Maintain the original intent** - Don't change what the prompt is trying to achieve
2. **Improve clarity** - Make instructions clearer and less ambiguous
3. **Add specificity** - Provide more detailed requirements and constraints
4. **Better structure** - Organize instructions logically and clearly
5. **Include examples** - Add examples where helpful
6. **Specify output format** - Clearly define expected response format

Return ONLY the improved prompt text, without additional commentary:"""

        try:
            response = await self.qwen.generate_response(
                improvement_prompt,
                temperature=0.4,  # Slightly higher temperature for creativity
                max_tokens=1500
            )
            
            improved_prompt = response.text.strip()
            
            # Clean up the response (remove any meta-commentary)
            improved_prompt = self._clean_improved_prompt(improved_prompt)
            
            return improved_prompt
        
        except Exception as e:
            logger.error(f"Failed to improve prompt for {analysis.agent_name}: {e}")
            return analysis.original_prompt
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON analysis response from Qwen"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\\{.*\\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Try parsing the whole response as JSON
                return json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON analysis response")
            return {
                'clarity_score': 0.5,
                'specificity_score': 0.5,
                'instruction_quality_score': 0.5,
                'overall_score': 0.5,
                'issues': ['Failed to parse analysis response'],
                'strengths': [],
                'suggestions': ['Improve response parsing']
            }
    
    def _clean_improved_prompt(self, prompt: str) -> str:
        """Clean up improved prompt by removing meta-commentary"""
        
        # Remove common prefixes that might be added
        prefixes_to_remove = [
            "Here's the improved prompt:",
            "Improved prompt:",
            "Here is the improved version:",
            "The improved prompt is:",
            "IMPROVED PROMPT:"
        ]
        
        for prefix in prefixes_to_remove:
            if prompt.lower().startswith(prefix.lower()):
                prompt = prompt[len(prefix):].strip()
        
        # Remove markdown code blocks if present
        if prompt.startswith('```') and prompt.endswith('```'):
            lines = prompt.split('\\n')
            if len(lines) > 2:
                prompt = '\\n'.join(lines[1:-1])
        
        return prompt.strip()


class AgentPromptReviewer:
    """
    Reviews and improves all prompts used by agentic system agents
    """
    
    def __init__(self, qwen_interface: Optional[OpenRouterInterface] = None):
        """Initialize with Qwen interface"""
        self.optimizer = PromptOptimizer(qwen_interface)
        self.extracted_prompts = {}
        self.analysis_results = {}
    
    def extract_agent_prompts(self) -> Dict[str, Dict[str, str]]:
        """
        Extract all prompts used by agents in the system
        
        Returns:
            Dictionary mapping agent_name -> {prompt_type: prompt_text}
        """
        
        # Import agents to extract their prompts
        
        extracted = {}
        
        # ConceptMiner prompts
        extracted['ConceptMiner'] = {
            'concept_extraction': """
Extract key concepts from this text. Return JSON only.

Text: {chunk}

Return format:
{"concepts": [{"name": "concept", "importance": 0.8, "snippet": "supporting text"}]}
""".strip()
        }
        
        # QuestionWriter prompts
        extracted['QuestionWriter'] = {
            'system_prompt': """You are an expert question writer. Create challenging, well-grounded questions.
Always return JSON only with this exact format:
{"question": "...", "answer": "...", "reasoning_chain": ["step1", "step2", "step3"]}""",
            
            'task_prompt_template': """
Given concept: {concept}
Evaluation type: {eval_type}
Supporting snippet: {snippet}

Create a question that:
1. Is answerable using the snippet
2. Tests deep understanding of {concept}
3. Is non-trivial (requires reasoning, not just lookup)
4. Fits the {eval_type} evaluation type

Think step by step, then return JSON only.
"""
        }
        
        # Adversary prompts
        extracted['Adversary'] = {
            'distractor_generation': """
Create 3 plausible but incorrect options for this multiple choice question.
Make them believable distractors that test common misconceptions.

Question: {question}
Correct Answer: {answer}
Concept: {concept}

Return JSON: {"distractors": ["wrong1", "wrong2", "wrong3"], "rationale": "why these are good distractors"}
""",
            
            'multi_hop_reasoning': """
Transform this question to require multi-step reasoning while staying grounded in the context.

Original Question: {question}
Context: {context}

Create a question that requires 2-3 logical steps to answer. Return JSON:
{"enhanced_question": "...", "enhanced_answer": "...", "reasoning_steps": ["step1", "step2", "step3"]}
"""
        }
        
        # Refiner prompts
        extracted['Refiner'] = {
            'multiple_choice_generation': """
Generate 3 plausible incorrect options for this multiple choice question.

Question: {question}
Correct Answer: {answer}
Concept: {concept}

Return JSON: {"options": ["wrong1", "wrong2", "wrong3"]}
"""
        }
        
        # Additional system prompts used throughout
        extracted['System'] = {
            'json_instruction': "Always return JSON only with the specified format.",
            'grounding_instruction': "Ensure all questions are answerable using the provided context.",
            'difficulty_instruction': "Create questions that require reasoning, not just lookup."
        }
        
        self.extracted_prompts = extracted
        return extracted
    
    async def analyze_all_prompts(self) -> Dict[str, Dict[str, PromptAnalysis]]:
        """
        Analyze all extracted prompts
        
        Returns:
            Dictionary mapping agent_name -> {prompt_type: PromptAnalysis}
        """
        
        if not self.extracted_prompts:
            self.extract_agent_prompts()
        
        analysis_results = {}
        
        for agent_name, prompts in self.extracted_prompts.items():
            agent_analyses = {}
            
            for prompt_type, prompt_text in prompts.items():
                logger.info(f"Analyzing {agent_name}.{prompt_type}")
                
                analysis = await self.optimizer.analyze_prompt(
                    prompt_text,
                    agent_name,
                    prompt_type,
                    context=f"Used in {agent_name} agent for {prompt_type}"
                )
                
                agent_analyses[prompt_type] = analysis
            
            analysis_results[agent_name] = agent_analyses
        
        self.analysis_results = analysis_results
        return analysis_results
    
    async def generate_improvements(self, min_score_threshold: float = 0.7) -> Dict[str, Dict[str, str]]:
        """
        Generate improved prompts for those below threshold
        
        Args:
            min_score_threshold: Only improve prompts below this score
            
        Returns:
            Dictionary mapping agent_name -> {prompt_type: improved_prompt}
        """
        
        if not self.analysis_results:
            await self.analyze_all_prompts()
        
        improvements = {}
        
        for agent_name, agent_analyses in self.analysis_results.items():
            agent_improvements = {}
            
            for prompt_type, analysis in agent_analyses.items():
                if analysis.overall_score < min_score_threshold:
                    logger.info(f"Improving {agent_name}.{prompt_type} (score: {analysis.overall_score:.2f})")
                    
                    improved_prompt = await self.optimizer.improve_prompt(analysis)
                    agent_improvements[prompt_type] = improved_prompt
                else:
                    logger.info(f"Skipping {agent_name}.{prompt_type} (score: {analysis.overall_score:.2f} >= {min_score_threshold})")
            
            if agent_improvements:
                improvements[agent_name] = agent_improvements
        
        return improvements
    
    def generate_review_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive review report
        
        Returns:
            Detailed report with analysis and recommendations
        """
        
        if not self.analysis_results:
            return {"error": "No analysis results available. Run analyze_all_prompts() first."}
        
        # Calculate aggregate statistics
        all_scores = []
        agent_summaries = {}
        issues_by_category = {}
        
        for agent_name, agent_analyses in self.analysis_results.items():
            agent_scores = []
            agent_issues = []
            
            for prompt_type, analysis in agent_analyses.items():
                all_scores.append(analysis.overall_score)
                agent_scores.append(analysis.overall_score)
                agent_issues.extend(analysis.issues)
                
                # Categorize issues
                for issue in analysis.issues:
                    category = self._categorize_issue(issue)
                    if category not in issues_by_category:
                        issues_by_category[category] = []
                    issues_by_category[category].append(f"{agent_name}.{prompt_type}: {issue}")
            
            agent_summaries[agent_name] = {
                'avg_score': sum(agent_scores) / len(agent_scores) if agent_scores else 0,
                'min_score': min(agent_scores) if agent_scores else 0,
                'max_score': max(agent_scores) if agent_scores else 0,
                'prompt_count': len(agent_scores),
                'issues_count': len(agent_issues)
            }
        
        # Generate report
        report = {
            'summary': {
                'total_prompts': len(all_scores),
                'avg_overall_score': sum(all_scores) / len(all_scores) if all_scores else 0,
                'min_score': min(all_scores) if all_scores else 0,
                'max_score': max(all_scores) if all_scores else 0,
                'prompts_needing_improvement': len([s for s in all_scores if s < 0.7])
            },
            'agent_summaries': agent_summaries,
            'issues_by_category': issues_by_category,
            'detailed_analyses': {
                agent_name: {
                    prompt_type: analysis.to_dict()
                    for prompt_type, analysis in agent_analyses.items()
                }
                for agent_name, agent_analyses in self.analysis_results.items()
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _categorize_issue(self, issue: str) -> str:
        """Categorize an issue for reporting"""
        issue_lower = issue.lower()
        
        if any(keyword in issue_lower for keyword in ['unclear', 'ambiguous', 'confusing']):
            return 'Clarity'
        elif any(keyword in issue_lower for keyword in ['specific', 'detail', 'vague']):
            return 'Specificity'
        elif any(keyword in issue_lower for keyword in ['format', 'output', 'structure']):
            return 'Output Format'
        elif any(keyword in issue_lower for keyword in ['example', 'guidance', 'instruction']):
            return 'Instructions'
        else:
            return 'Other'
    
    def _generate_recommendations(self) -> List[str]:
        """Generate high-level recommendations based on analysis"""
        recommendations = [
            "Review prompts with scores below 0.7 for improvement opportunities",
            "Add specific output format requirements to prompts lacking them",
            "Include examples in prompts where helpful for clarity",
            "Ensure all prompts have clear, actionable instructions",
            "Test improved prompts with actual agent execution",
            "Consider prompt versioning for systematic improvements"
        ]
        
        return recommendations


# Convenience functions

async def quick_prompt_review(api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick review of all agent prompts
    
    Args:
        api_key: OpenRouter API key (uses environment if None)
        
    Returns:
        Review report
    """
    
    try:
        config = OpenRouterConfig(api_key=api_key)
        qwen = OpenRouterInterface(config) if api_key or os.getenv('OPENROUTER_API_KEY') else None
        reviewer = AgentPromptReviewer(qwen)
        
        # Extract and analyze
        await reviewer.analyze_all_prompts()
        
        # Generate report
        report = reviewer.generate_review_report()
        
        return report
    
    except Exception as e:
        return {
            'error': f"Quick review failed: {e}",
            'summary': {'total_prompts': 0}
        }


async def improve_low_scoring_prompts(
    api_key: Optional[str] = None,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Analyze and improve prompts scoring below threshold
    
    Args:
        api_key: OpenRouter API key
        threshold: Score threshold for improvement
        
    Returns:
        Dictionary with improvements and report
    """
    
    try:
        config = OpenRouterConfig(api_key=api_key)
        qwen = OpenRouterInterface(config) if api_key or os.getenv('OPENROUTER_API_KEY') else None
        reviewer = AgentPromptReviewer(qwen)
        
        # Analyze all prompts
        await reviewer.analyze_all_prompts()
        
        # Generate improvements
        improvements = await reviewer.generate_improvements(threshold)
        
        # Generate report
        report = reviewer.generate_review_report()
        
        return {
            'improvements': improvements,
            'report': report,
            'improved_count': sum(len(agent_improvements) for agent_improvements in improvements.values())
        }
    
    except Exception as e:
        return {
            'error': f"Improvement process failed: {e}",
            'improvements': {},
            'improved_count': 0
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    import os
    
    async def main():
        """Test the prompt optimization system"""
        
        print("Testing Prompt Optimization System...")
        
        # Quick review
        report = await quick_prompt_review()
        
        if 'error' not in report:
            print("\\nPrompt Review Summary:")
            print(f"Total prompts: {report['summary']['total_prompts']}")
            print(f"Average score: {report['summary']['avg_overall_score']:.2f}")
            print(f"Prompts needing improvement: {report['summary']['prompts_needing_improvement']}")
            
            # Show agent summaries
            print("\\nAgent Summaries:")
            for agent_name, summary in report['agent_summaries'].items():
                print(f"  {agent_name}: {summary['avg_score']:.2f} avg ({summary['prompt_count']} prompts)")
        else:
            print(f"Review failed: {report['error']}")
        
        # Try improvements if API key is available
        if os.getenv('OPENROUTER_API_KEY'):
            print("\\nGenerating improvements...")
            improvement_results = await improve_low_scoring_prompts()
            
            if improvement_results['improved_count'] > 0:
                print(f"Generated {improvement_results['improved_count']} improvements")
            else:
                print("No improvements needed or failed to generate")
        else:
            print("\\nSet OPENROUTER_API_KEY to test improvements")
    
    # Run test
    asyncio.run(main())