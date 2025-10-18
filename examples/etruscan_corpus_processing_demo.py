#!/usr/bin/env python3
"""
Process Domain-Specific Etruscan Corpus using Existing Agentic Abstractions
Uses chonkie for semantic chunking and existing agentic pipeline
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import existing abstractions - NO NEW ABSTRACTIONS!
from docs_to_eval.core.agentic import AgenticBenchmarkGenerator
from docs_to_eval.utils.config import EvaluationType
from docs_to_eval.utils.text_processing import create_smart_chunks
from docs_to_eval.utils.config import ChunkingConfig


def load_etruscan_corpus() -> Dict[str, str]:
    """Load all Etruscan texts from data"""
    corpus_dir = Path("data/etruscan_texts")
    texts = {}
    
    for txt_file in corpus_dir.glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content and len(content) > 50:  # Only non-empty meaningful texts
                    texts[txt_file.stem] = content
        except Exception as e:
            print(f"⚠️ Could not load {txt_file}: {e}")
    
    return texts


def combine_corpus_texts(texts: Dict[str, str], max_length: int = 8000) -> str:
    """Combine corpus texts into a manageable corpus"""
    combined = ""
    for name, content in texts.items():
        if len(combined) + len(content) < max_length:
            combined += f"\n\n# {name.replace('_', ' ').title()}\n\n{content}"
        else:
            break
    return combined.strip()


def process_with_chonkie(corpus_text: str) -> List[Dict[str, Any]]:
    """Use existing chonkie integration to chunk corpus"""
    print("🧠 Processing with Chonkie semantic chunking...")
    
    # Use existing chunking config with chonkie enabled
    config = ChunkingConfig(
        target_chunk_size=1500,
        max_chunk_size=2500,
        min_chunk_size=800,
        overlap_size=200,
        chunking_strategy="semantic",
        enable_chonkie=True  # Use existing chonkie integration
    )
    
    # Use existing create_smart_chunks function
    chunks = create_smart_chunks(corpus_text, chunking_config=config)
    
    print(f"✅ Created {len(chunks)} semantic chunks")
    if chunks:
        sizes = [len(chunk['text']) for chunk in chunks]
        methods = [chunk.get('method', 'unknown') for chunk in chunks]
        print(f"   📏 Chunk sizes: {min(sizes)}-{max(sizes)} chars (avg: {sum(sizes)//len(sizes)})")
        print(f"   🔧 Methods: {set(methods)}")
        print(f"   🧠 Advanced chunking: {any(m.startswith('chonkie_') for m in methods)}")
    
    return chunks


async def generate_agentic_benchmarks(corpus_text: str, eval_type: EvaluationType, num_questions: int = 10) -> List[Any]:
    """Use existing agentic system to generate benchmarks"""
    print(f"🤖 Generating {num_questions} {eval_type.value} questions with agentic pipeline...")
    
    # Use existing AgenticBenchmarkGenerator
    generator = AgenticBenchmarkGenerator(eval_type)
    
    # Use existing async generation method
    items = await generator.generate_benchmark_async(
        corpus_text=corpus_text, 
        num_questions=num_questions
    )
    
    print(f"✅ Generated {len(items)} benchmark items")
    return items


def display_benchmark_sample(items: List[Any], max_display: int = 3):
    """Display sample generated benchmarks"""
    print(f"\n📋 Sample Generated Benchmarks (showing {min(len(items), max_display)}):")
    print("-" * 60)
    
    for i, item in enumerate(items[:max_display]):
        print(f"\n🔸 Question {i+1}:")
        
        # Handle both dict and object formats
        if hasattr(item, 'question'):
            print(f"   Q: {item.question}")
            print(f"   A: {item.answer}")
            if hasattr(item, 'metadata'):
                print(f"   Concept: {getattr(item.metadata, 'source_concept', 'N/A')}")
                print(f"   Difficulty: {getattr(item.metadata, 'difficulty_level', 'N/A')}")
        elif isinstance(item, dict):
            print(f"   Q: {item.get('question', 'N/A')}")
            print(f"   A: {item.get('answer', 'N/A')}")
            print(f"   Type: {item.get('eval_type', 'N/A')}")


def analyze_corpus_semantics(chunks: List[Dict[str, Any]]):
    """Analyze semantic structure of chunked corpus"""
    print("\n🔍 Corpus Semantic Analysis:")
    print("-" * 40)
    
    # Semantic scores analysis
    semantic_scores = []
    for chunk in chunks:
        score = chunk.get('semantic_score', 1.0)
        semantic_scores.append(score)
    
    if semantic_scores:
        avg_score = sum(semantic_scores) / len(semantic_scores)
        print(f"   🎯 Average semantic coherence: {avg_score:.3f}")
        print(f"   📊 Score range: {min(semantic_scores):.3f} - {max(semantic_scores):.3f}")
    
    # Content preview
    print("\n   📖 Semantic Chunks Preview:")
    for i, chunk in enumerate(chunks[:2]):
        text_preview = chunk['text'][:100].replace('\n', ' ').strip()
        method = chunk.get('method', 'unknown')
        score = chunk.get('semantic_score', 1.0)
        print(f"     {i+1}. [{method}, score={score:.3f}] {text_preview}...")


async def main():
    """Main processing function using existing abstractions"""
    print("🏺 ETRUSCAN CORPUS PROCESSING with Existing Agentic Abstractions")
    print("🧠 Using: chonkie + AgenticBenchmarkGenerator + existing utilities")
    print("=" * 80)
    
    start_time = time.time()
    
    # Step 1: Load domain corpus using built-in structure
    print("📚 Loading domain-specific Etruscan corpus...")
    etruscan_texts = load_etruscan_corpus()
    print(f"✅ Loaded {len(etruscan_texts)} Etruscan texts")
    
    if not etruscan_texts:
        print("❌ No texts found. Check data/etruscan_texts/")
        return
    
    # Display corpus sample
    text_names = list(etruscan_texts.keys())[:5]
    print(f"   📖 Sample texts: {', '.join(text_names)}")
    
    # Step 2: Combine and prepare corpus
    corpus_text = combine_corpus_texts(etruscan_texts)
    print(f"📝 Combined corpus: {len(corpus_text)} characters")
    
    # Step 3: Process with chonkie (existing integration)
    chunks = process_with_chonkie(corpus_text)
    
    # Step 4: Analyze semantic structure
    analyze_corpus_semantics(chunks)
    
    # Step 5: Generate agentic benchmarks for different evaluation types
    evaluation_scenarios = [
        (EvaluationType.DOMAIN_KNOWLEDGE, 8),
        (EvaluationType.FACTUAL_QA, 5),
        (EvaluationType.MULTIPLE_CHOICE, 4)
    ]
    
    all_results = {}
    
    for eval_type, num_questions in evaluation_scenarios:
        print(f"\n🎯 Processing {eval_type.value.upper()} evaluation...")
        try:
            items = await generate_agentic_benchmarks(corpus_text, eval_type, num_questions)
            all_results[eval_type.value] = items
            
            # Display sample for this type
            display_benchmark_sample(items, max_display=2)
            
        except Exception as e:
            print(f"⚠️ Error with {eval_type.value}: {e}")
            all_results[eval_type.value] = []
    
    # Final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n🎉 PROCESSING COMPLETE!")
    print("=" * 50)
    print(f"⏱️  Total processing time: {total_time:.2f}s")
    print(f"📚 Corpus processed: {len(etruscan_texts)} texts")
    print(f"🧠 Semantic chunks: {len(chunks)}")
    
    total_items = sum(len(items) for items in all_results.values())
    print(f"🤖 Benchmark items generated: {total_items}")
    
    print("\n📊 Results by evaluation type:")
    for eval_type, items in all_results.items():
        print(f"   • {eval_type}: {len(items)} items")
    
    print("\n✨ Key Features Demonstrated:")
    print("   ✓ Used existing AgenticBenchmarkGenerator")
    print("   ✓ Leveraged chonkie semantic chunking") 
    print("   ✓ Processed domain-specific Etruscan corpus")
    print("   ✓ Generated multiple evaluation types")
    print("   ✓ Zero new abstractions - all existing utilities!")
    
    return all_results


if __name__ == "__main__":
    # Run the processing pipeline
    results = asyncio.run(main())
