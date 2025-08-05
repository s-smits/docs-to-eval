# Enhanced Agentic Workflow with Ground Truth Verification

## 🎯 NEW ENHANCED 3-PHASE AGENTIC EVALUATION

The system now includes a **critical ground truth verification step** to ensure question-answer pairs are actually correct according to the corpus.

### Phase 1: Question & Answer Generation
**LLM Role**: Content creator with corpus access
- Analyzes corpus text thoroughly
- Generates questions testing key concepts
- Proposes answers based on corpus content
- Designed for maximum coverage of important topics

### Phase 2: Ground Truth Verification ⭐ **NEW**
**LLM Role**: Independent fact-checker with corpus access
- **Critically important**: Verifies proposed answers against corpus
- Rejects questions with incorrect answers
- Corrects partially incorrect answers
- Provides evidence citations from corpus
- Ensures only corpus-supported facts become ground truth

### Phase 3: Blind Evaluation
**LLM Role**: Test subject WITHOUT corpus access
- Receives only the verified question
- No access to ground truth answer or corpus
- Generates response based on knowledge only
- This is the "prediction" to be evaluated

### Phase 4: Performance Verification
**System Role**: Objective scorer
- Compares blind prediction vs verified ground truth
- Applies appropriate verification method
- Generates performance metrics

## 🔧 IMPLEMENTATION DETAILS

### Phase 2 Implementation (Ground Truth Verification):

```python
async def verify_ground_truth_against_corpus(question, proposed_answer, corpus_text, llm_config):
    verification_prompt = f"""You are a fact-checker. Verify if this answer is correct according to the source text.

SOURCE TEXT: {corpus_text}
QUESTION: {question}  
PROPOSED ANSWER: {proposed_answer}

Return JSON:
{{
  "is_correct": true/false,
  "verified_answer": "corrected answer if needed",
  "confidence": 0.0-1.0,
  "reasoning": "explanation",
  "evidence": "specific quote from source"
}}

Rules:
1. Answer correct ONLY if directly supported by source
2. Provide corrected version if partially wrong
3. Require clear textual evidence
4. Be strict - mark uncertain answers as incorrect
"""
```

### Quality Control Mechanisms:

1. **Strict Evidence Requirements**: 
   - Answers must be directly quotable from corpus
   - Vague or inferential answers are rejected
   - Citations required for verification

2. **Confidence Scoring**:
   - High confidence: Direct quote match
   - Medium confidence: Supported by implication  
   - Low confidence: Uncertain, rejected

3. **Automatic Rejection & Regeneration**:
   - Questions with incorrect answers are discarded
   - If >20% rejection rate, generates additional questions
   - Ensures sufficient high-quality questions

4. **Answer Correction**:
   - Partially correct answers are fixed
   - Uses verified answer as final ground truth
   - Maintains question quality while fixing errors

## 📊 VERIFICATION STATISTICS

The system tracks and reports:
- Total questions generated
- Questions verified as correct
- Questions rejected (with reasons)
- Questions corrected
- Overall verification success rate

Example output:
```
Ground truth verification: 8/10 questions verified (20.0% rejected)
✅ Verified Q1: What century did Etruscan civilization begin?
✅ Verified Q2: What metal were Etruscan artifacts made from?
❌ Rejected Q3: Answer not supported by corpus evidence
```

## 🎯 WHY THIS IS CRITICAL

### Without Ground Truth Verification:
❌ **Question**: "When did the Etruscan empire end?"
❌ **Generated Answer**: "500 BCE" (WRONG - not in corpus)
❌ **Result**: LLM evaluated against incorrect ground truth

### With Ground Truth Verification:
✅ **Question**: "When did the Etruscan empire end?"
✅ **Generated Answer**: "500 BCE"
✅ **Verification**: REJECTED - "No evidence in corpus supports 500 BCE"
✅ **Corrected Answer**: "3rd century BCE" (from corpus)
✅ **Result**: LLM evaluated against verified ground truth

## 🔄 COMPLETE ENHANCED WORKFLOW

```
CORPUS: "Etruscan civilization flourished from 8th century BCE to 3rd century BCE"

1. GENERATION:
   Q: "When did Etruscan civilization end?"
   A: "3rd century BCE"

2. VERIFICATION:
   ✅ Verified: "3rd century BCE" found in corpus
   Evidence: "flourished from 8th century BCE to 3rd century BCE"
   Confidence: 0.95

3. BLIND EVALUATION:
   LLM Response: "The Etruscan civilization ended around 264 BCE"

4. SCORING:
   Ground Truth: "3rd century BCE"
   Prediction: "The Etruscan civilization ended around 264 BCE"  
   Score: 1.0 (264 BCE is in 3rd century BCE)
   Method: "temporal_match"
```

## ✅ ENHANCED SYSTEM BENEFITS

1. **Guaranteed Accuracy**: Only corpus-supported facts become ground truth
2. **Quality Assurance**: Bad questions are automatically filtered out  
3. **Evidence-Based**: All answers have textual citations
4. **Self-Correcting**: System improves answer quality automatically
5. **Transparent**: Full verification reasoning is tracked
6. **Robust**: Handles edge cases and partial matches

This enhanced workflow ensures that your agentic evaluation system produces **reliable, corpus-verified benchmarks** for accurate LLM assessment!