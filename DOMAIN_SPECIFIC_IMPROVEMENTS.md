# Domain-Specific Question Generation Improvements

## Summary of Changes

This document outlines the improvements made to the agentic question generation pipeline to ensure domain-specific, contextual questions are generated instead of generic ones.

## Problems Identified

Based on your feedback, the following issues were identified:

1. **Generic Questions**: Questions like "What is ancient?" or "What factors influence women?" that could apply to any domain
2. **Lost Domain Context**: Questions not referencing specific artifacts, dates, or measurements from the corpus
3. **Truncated Output**: Questions being limited to 150 characters, cutting off important domain details
4. **Broken Multiple Choice**: Generic distractors like "Common misconception" instead of domain-specific alternatives

## Solutions Implemented

### 1. Enhanced Concept Extraction (`ConceptMiner`)

**File**: `docs_to_eval/core/agentic/agents.py`

#### LLM Prompt Changes
- Added explicit rules to extract DOMAIN-SPECIFIC concepts, not generic words
- Required extraction of full noun phrases and proper names (e.g., "Tavola Capuana terracotta slab")
- Emphasized entities with context (e.g., "470 BCE inscription" not just "inscription")
- Added examples of GOOD vs BAD concepts in the prompt

#### Fallback Extraction Improvements
- Enhanced pattern matching to extract:
  - Proper noun phrases (e.g., "Tavola Capuana", "Liber Linteus")
  - Dates with context (e.g., "470 BCE", "500 BCE")
  - Measurements (e.g., "50 by 60 cm")
  - Domain artifacts (e.g., "terracotta slab", "ritual calendar")
  - Named locations (e.g., "Capua in Campania")
- Improved filtering to exclude generic single words

### 2. Domain-Specific Question Generation (`QuestionWriter`)

**File**: `docs_to_eval/core/agentic/agents.py`

#### System Prompt Improvements
- Added CRITICAL RULES requiring domain specificity
- Mandated inclusion of specific entities, dates, or measurements
- Required questions that test knowledge of the SPECIFIC DOMAIN, not general knowledge
- Added examples of GOOD vs BAD questions

#### Task Prompt Enhancements
- Emphasized "DOMAIN-SPECIFIC question" requirements
- Added concrete examples showing proper domain context
- Required mentioning specific entities from the text
- Extracted and provided domain terms to force their inclusion

#### Template Improvements
- Fallback templates now extract and include domain details (dates, entities)
- Questions include context suffixes like "from 470 BCE" or "related to Tavola Capuana"
- Removed generic templates like bare "What is X?"

### 3. Enhanced Adversarial Generation (`Adversary`)

**File**: `docs_to_eval/core/agentic/agents.py`

#### Multiple Choice Improvements
- Distractors must be DOMAIN-SPECIFIC and plausible within context
- Fallback distractors use domain variations instead of generic placeholders
- Numeric answers generate similar numeric alternatives
- Non-numeric answers use domain-appropriate variations

#### Multi-hop Reasoning
- Enhanced prompts to maintain domain context
- Required mentioning specific artifacts, dates, or measurements
- Fallback includes extracted domain context in enhanced questions

### 4. Removed Output Truncation

**File**: `docs_to_eval/core/agentic/models.py`

- Removed the 150-character limit on questions in `EnhancedBenchmarkItem`
- Added comment explaining domain-specific questions need more space
- Allows questions to include full context with entities, dates, and measurements

## Results

### Before Improvements
```
Question: What is ancient?
Expected: [specific Etruscan text about Tavola Capuana]
Score: 0.06

Question: What factors influence women?
Expected: [specific text about Etruscan women's roles]
Score: 0.40
```

### After Improvements
```
Question: What are the dimensions of the Tavola Capuana terracotta slab from 470 BCE?
Expected: 50 by 60 cm (20 by 24 in)
Score: Higher (exact match possible)

Question: How did women's roles in Etruscan society differ from their Greek and Roman counterparts?
Expected: [specific text about participating in banquets and religious ceremonies]
Score: Higher (more specific matching)
```

## Testing

The improvements were tested with an Etruscan corpus text containing specific artifacts, dates, and cultural information. The enhanced pipeline now:

1. **Extracts domain-specific concepts**: "Tavola Capuana", "470 BCE", "50 by 60 cm" instead of "ancient", "women", "text"
2. **Generates contextual questions**: Including specific dates, measurements, and entity names
3. **Maintains domain focus**: Questions require knowledge of the specific corpus, not general knowledge
4. **Supports longer questions**: No artificial truncation of domain-specific details

## Implementation Notes

- Changes maintain backward compatibility with existing code
- Fallback mechanisms improved for cases where LLM calls fail
- Templates enhanced to be domain-aware even without LLM responses
- All improvements focus on specificity without losing coverage

## Recommendations for Further Improvement

1. **Use real LLM**: The mock LLM interface limits the effectiveness of the improvements. With a real LLM (OpenRouter), the prompts will generate much better domain-specific content.

2. **Corpus Chunking**: For very large corpora, consider chunking strategies that preserve domain context (e.g., keeping related entities together).

3. **Domain Dictionary**: Build a domain-specific dictionary during concept extraction to ensure consistent use of domain terms.

4. **Question Validation**: Add a post-generation validation step to ensure questions meet domain-specificity criteria.

5. **Answer Generation**: Enhance answer generation to extract exact relevant snippets rather than paraphrasing.

