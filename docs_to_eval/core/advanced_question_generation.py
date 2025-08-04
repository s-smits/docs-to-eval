"""
Advanced Question Generation System for Stress-Testing LLMs
Creates niche, synthesis-based questions that go beyond surface-level recall
"""

import re
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AdvancedQuestionTemplate:
    """Template for generating sophisticated evaluation questions"""
    base_concept: str
    complexity_layer: str  # 'synthesis', 'inference', 'ambiguity', 'extrapolation'
    question_template: str
    context_requirements: List[str]
    expected_reasoning_elements: List[str]
    difficulty_level: str  # 'advanced', 'expert', 'research'


class AdvancedQuestionGenerator:
    """
    Generates sophisticated questions that stress-test LLM capabilities beyond basic recall
    """
    
    def __init__(self, corpus_text: str, domain: str = "historical"):
        self.corpus_text = corpus_text
        self.domain = domain
        self.complexity_templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, List[AdvancedQuestionTemplate]]:
        """Initialize sophisticated question templates by category"""
        
        templates = {
            "comparative_analysis": [
                AdvancedQuestionTemplate(
                    base_concept="publication_patterns",
                    complexity_layer="synthesis",
                    question_template="Given that {entity1} published over {timespan1} and {entity2} over {timespan2}, and considering that {historical_context}, compare their average publication rates per decade and infer {reasoning_requirement}.",
                    context_requirements=["historical_context", "publication_data", "external_factors"],
                    expected_reasoning_elements=["calculation", "historical_analysis", "inference"],
                    difficulty_level="advanced"
                ),
                AdvancedQuestionTemplate(
                    base_concept="technological_evolution",
                    complexity_layer="inference",
                    question_template="Analyzing the progression from {early_tech} to {later_tech} over {timespan}, and knowing that {constraint}, estimate {calculation_target} and explain how {external_factor} might have influenced this development.",
                    context_requirements=["technological_progression", "constraints", "external_factors"],
                    expected_reasoning_elements=["technical_analysis", "historical_reasoning", "estimation"],
                    difficulty_level="expert"
                )
            ],
            
            "artifact_analysis": [
                AdvancedQuestionTemplate(
                    base_concept="construction_techniques",
                    complexity_layer="ambiguity",
                    question_template="Given {artifact} with dimensions {measurements}, and considering that ancient {technique} typically resulted in {characteristic}, calculate {derived_measurement} assuming {assumption}, then discuss why {design_choice} might have been {purpose}.",
                    context_requirements=["artifact_details", "construction_methods", "purpose_analysis"],
                    expected_reasoning_elements=["technical_calculation", "material_science", "functional_analysis"],
                    difficulty_level="expert"
                ),
                AdvancedQuestionTemplate(
                    base_concept="material_composition",
                    complexity_layer="extrapolation",
                    question_template="The {artifact} shows {property} suggesting {composition}. Calculate its {derived_property}, compare this to {reference_material}, and infer what this reveals about {historical_aspect} during the {time_period}.",
                    context_requirements=["material_properties", "reference_data", "historical_context"],
                    expected_reasoning_elements=["material_analysis", "comparison", "historical_inference"],
                    difficulty_level="expert"
                )
            ],
            
            "cultural_synthesis": [
                AdvancedQuestionTemplate(
                    base_concept="cultural_influence",
                    complexity_layer="synthesis",
                    question_template="If {cultural_influence} began around {start_date} and {conditions} were met, recalculate the {measurement} under {alternative_scenario}. What does this suggest about {cultural_aspect} and how might {external_factor} have contributed?",
                    context_requirements=["cultural_timeline", "conditions", "external_factors"],
                    expected_reasoning_elements=["scenario_analysis", "cultural_reasoning", "causation"],
                    difficulty_level="advanced"
                ),
                AdvancedQuestionTemplate(
                    base_concept="survival_analysis",
                    complexity_layer="inference",
                    question_template="Considering that {artifact} survived due to {preservation_factor}, estimate {time_calculation} and analyze how {environmental_factor} influenced the survival of {comparison_category} versus {reference_category} materials.",
                    context_requirements=["preservation_conditions", "environmental_factors", "comparative_data"],
                    expected_reasoning_elements=["survival_analysis", "environmental_science", "comparison"],
                    difficulty_level="advanced"
                )
            ],
            
            "methodological_complexity": [
                AdvancedQuestionTemplate(
                    base_concept="research_methodology",
                    complexity_layer="ambiguity",
                    question_template="Recent {discovery_type} suggests {finding}. If this changes our understanding of {established_fact}, recalculate {measurement} and discuss what methodological challenges this presents for {research_area}.",
                    context_requirements=["recent_research", "established_knowledge", "methodological_issues"],
                    expected_reasoning_elements=["research_analysis", "methodology_critique", "recalculation"],
                    difficulty_level="research"
                ),
                AdvancedQuestionTemplate(
                    base_concept="interdisciplinary_analysis",
                    complexity_layer="synthesis",
                    question_template="Combining evidence from {field1} and {field2}, analyze how {phenomenon} would affect {calculation_target}. Factor in {constraint} and explain why {interdisciplinary_connection} is crucial for understanding {broader_implication}.",
                    context_requirements=["interdisciplinary_data", "constraints", "broader_context"],
                    expected_reasoning_elements=["interdisciplinary_thinking", "complex_calculation", "systems_analysis"],
                    difficulty_level="research"
                )
            ]
        }
        
        return templates
    
    def extract_corpus_entities(self) -> Dict[str, List[str]]:
        """Extract key entities from corpus for use in advanced questions"""
        
        entities = {
            "artifacts": [],
            "publications": [],
            "dates": [],
            "measurements": [],
            "people": [],
            "places": [],
            "concepts": []
        }
        
        # Extract artifacts (capitalized terms often referring to objects)
        artifact_patterns = [
            r"\b([A-Z][a-z]+\s+of\s+[A-Z][a-z]+)\b",  # "Liver of Piacenza"
            r"\b(Corpus\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",  # "Corpus Speculorum"
            r"\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"  # Other proper nouns
        ]
        
        for pattern in artifact_patterns:
            matches = re.findall(pattern, self.corpus_text)
            entities["artifacts"].extend(matches)
        
        # Extract dates
        date_patterns = [
            r"\b(\d{1,4}\s*(?:BC|AD|CE|BCE))\b",
            r"\b(\d{1,4})\s*[-–]\s*(\d{1,4})\b"
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, self.corpus_text)
            entities["dates"].extend([m if isinstance(m, str) else ' - '.join(m) for m in matches])
        
        # Extract measurements
        measurement_patterns = [
            r"\b(\d+\.?\d*\s*(?:mm|cm|m|kg|g|years?|centuries?|decades?))\b",
            r"\b(\d+\s*[×x]\s*\d+(?:\s*[×x]\s*\d+)?)\b"
        ]
        
        for pattern in measurement_patterns:
            matches = re.findall(pattern, self.corpus_text, re.IGNORECASE)
            entities["measurements"].extend(matches)
        
        # Extract key concepts (important terms that appear multiple times)
        concept_candidates = re.findall(r'\b[A-Z][a-z]+(?:an|ic|ous|al)?\b', self.corpus_text)
        concept_frequency = {}
        for concept in concept_candidates:
            if len(concept) > 3:  # Filter short words
                concept_frequency[concept] = concept_frequency.get(concept, 0) + 1
        
        # Keep concepts that appear multiple times
        entities["concepts"] = [concept for concept, freq in concept_frequency.items() if freq >= 2]
        
        return entities
    
    def generate_advanced_question(self, template: AdvancedQuestionTemplate, 
                                 entities: Dict[str, List[str]]) -> Optional[Dict[str, Any]]:
        """Generate a single advanced question from template and entities"""
        
        try:
            # Select appropriate entities for this template
            template_vars = {}
            
            if "entity1" in template.question_template:
                if entities["publications"]:
                    template_vars["entity1"] = random.choice(entities["publications"])
                else:
                    template_vars["entity1"] = "First scholarly work"
            
            if "artifact" in template.question_template:
                if entities["artifacts"]:
                    template_vars["artifact"] = random.choice(entities["artifacts"])
                else:
                    template_vars["artifact"] = "the archaeological artifact"
            
            if "measurements" in template.question_template:
                if entities["measurements"]:
                    template_vars["measurements"] = random.choice(entities["measurements"])
                else:
                    template_vars["measurements"] = "standard dimensions"
            
            # Add domain-specific context based on complexity layer
            context_additions = self._generate_context_additions(template.complexity_layer, entities)
            template_vars.update(context_additions)
            
            # Generate the question
            question = template.question_template.format(**template_vars)
            
            # Generate expected reasoning elements
            expected_answer = self._generate_expected_answer(template, template_vars)
            
            return {
                "question": question,
                "answer": expected_answer,
                "concept": template.base_concept,
                "difficulty": template.difficulty_level,
                "complexity_layer": template.complexity_layer,
                "reasoning_elements": template.expected_reasoning_elements,
                "source": "advanced_generation"
            }
            
        except (KeyError, IndexError) as e:
            # Template couldn't be filled with available entities
            return None
    
    def _generate_context_additions(self, complexity_layer: str, 
                                  entities: Dict[str, List[str]]) -> Dict[str, str]:
        """Generate contextual additions based on complexity layer"""
        
        additions = {}
        
        if complexity_layer == "synthesis":
            additions.update({
                "historical_context": "scholarly publication standards evolved significantly during this period",
                "external_factor": "changing archaeological discovery methods",
                "reasoning_requirement": "potential factors affecting publication efficiency"
            })
        
        elif complexity_layer == "inference":
            additions.update({
                "constraint": "available technology and materials of the era",
                "external_factor": "environmental and political factors",
                "characteristic": "specific structural properties"
            })
        
        elif complexity_layer == "ambiguity":
            additions.update({
                "assumption": "accounting for typical construction variations",
                "design_choice": "this particular structural approach",
                "purpose": "optimal for its intended ritual function"
            })
        
        elif complexity_layer == "extrapolation":
            additions.update({
                "reference_material": "contemporary materials of known composition",
                "historical_aspect": "metallurgical practices",
                "time_period": "the relevant historical era"
            })
        
        return additions
    
    def _generate_expected_answer(self, template: AdvancedQuestionTemplate, 
                                template_vars: Dict[str, str]) -> str:
        """Generate expected answer structure for advanced questions"""
        
        # For advanced questions, we expect multi-part answers
        answer_components = []
        
        if "calculation" in template.expected_reasoning_elements:
            answer_components.append("Quantitative analysis with calculations")
        
        if "historical_analysis" in template.expected_reasoning_elements:
            answer_components.append("Historical context and analysis")
        
        if "inference" in template.expected_reasoning_elements:
            answer_components.append("Reasoned inference based on evidence")
        
        if "comparison" in template.expected_reasoning_elements:
            answer_components.append("Comparative analysis with reference materials")
        
        # Create a structured expected answer
        expected_answer = f"Multi-part response addressing: {', '.join(answer_components)}"
        
        return expected_answer
    
    def generate_advanced_question_set(self, num_questions: int = 10) -> List[Dict[str, Any]]:
        """Generate a set of advanced questions for stress-testing LLMs"""
        
        entities = self.extract_corpus_entities()
        generated_questions = []
        
        # Ensure variety across complexity layers
        complexity_distribution = {
            "synthesis": num_questions // 4,
            "inference": num_questions // 4,
            "ambiguity": num_questions // 4,
            "extrapolation": num_questions - (3 * (num_questions // 4))
        }
        
        for category, templates in self.complexity_templates.items():
            for complexity_layer, target_count in complexity_distribution.items():
                relevant_templates = [t for t in templates if t.complexity_layer == complexity_layer]
                
                if relevant_templates:
                    for _ in range(min(target_count, len(relevant_templates))):
                        template = random.choice(relevant_templates)
                        question = self.generate_advanced_question(template, entities)
                        
                        if question:
                            generated_questions.append(question)
        
        # Fill remaining slots with any available templates
        while len(generated_questions) < num_questions:
            all_templates = [t for templates in self.complexity_templates.values() for t in templates]
            template = random.choice(all_templates)
            question = self.generate_advanced_question(template, entities)
            
            if question:
                generated_questions.append(question)
            else:
                break  # Can't generate more questions
        
        return generated_questions[:num_questions]


def create_advanced_question_prompt(corpus_text: str, num_questions: int, eval_type: str) -> str:
    """
    Create an advanced prompt for generating sophisticated, stress-testing questions
    """
    
    advanced_instructions = {
        "mathematical": """Generate questions that require multi-step mathematical reasoning combined with domain expertise. Include:
- Calculations with realistic assumptions and constraints
- Comparisons requiring external knowledge inference
- Analysis of ratios, rates, and proportional relationships in historical context
- Integration of quantitative analysis with qualitative interpretation""",
        
        "factual_qa": """Generate questions that require synthesis across multiple sources and time periods. Include:
- Comparative analysis requiring inference beyond stated facts
- Questions with plausible ambiguity that test reasoning under uncertainty
- Integration of archaeological, historical, and material science knowledge
- Analysis of cause-and-effect relationships across cultural and temporal boundaries""",
        
        "domain_knowledge": """Generate questions that stress-test deep domain understanding. Include:
- Interdisciplinary connections requiring specialized knowledge
- Methodological challenges and research implications
- Analysis of conflicting evidence or alternative interpretations
- Extrapolation from specific cases to broader principles"""
    }
    
    instruction = advanced_instructions.get(eval_type, advanced_instructions["domain_knowledge"])
    
    return f"""You are an expert creating ADVANCED evaluation questions designed to stress-test LLMs beyond surface-level knowledge.

OBJECTIVE: Generate questions that require:
1. SYNTHESIS across multiple concepts and time periods
2. INFERENCE beyond directly stated facts  
3. REASONING under ambiguity and uncertainty
4. INTEGRATION of domain expertise with analytical thinking

{instruction}

QUESTION COMPLEXITY REQUIREMENTS:
- Force the model to make connections between disparate information
- Include plausible assumptions or "what-if" scenarios
- Require both calculation AND interpretation
- Test reasoning about methodology, causation, or broader implications
- Include context that requires external domain knowledge

FORMAT: Return JSON array with:
[
  {{
    "question": "Complex, multi-layered question requiring synthesis and inference",
    "answer": "Expected reasoning approach and key conclusions",
    "concept": "Primary concept being stress-tested",
    "difficulty": "advanced|expert|research",
    "complexity_layer": "synthesis|inference|ambiguity|extrapolation"
  }}
]

EXAMPLES OF COMPLEXITY:
- "Given X and considering that Y typically results in Z, calculate A under assumption B, then analyze what this suggests about C during period D."
- "If recent findings suggest X, recalculate Y and discuss methodological implications for understanding Z."
- "Comparing the progression from A to B over timespan C, and knowing constraint D, estimate E and explain how factor F influenced this development."

Generate {num_questions} questions that will truly challenge advanced reasoning capabilities."""


if __name__ == "__main__":
    # Test the advanced question generator
    sample_corpus = """
    The Liver of Piacenza measures 126 × 76 × 60 mm and is an important Etruscan bronze artifact.
    Eduard Gerhard's Etruskische Spiegel was published from 1843 to 1897.
    The Corpus Speculorum Etruscorum project began in 1981 and has produced 36 fascicles.
    """
    
    generator = AdvancedQuestionGenerator(sample_corpus)
    questions = generator.generate_advanced_question_set(5)
    
    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i}:")
        print(f"Complexity: {q['complexity_layer']}")
        print(f"Question: {q['question']}")
        print(f"Expected: {q['answer']}")