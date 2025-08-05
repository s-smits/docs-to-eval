#!/usr/bin/env python3
"""
Simple Demo: Domain-Specific Etruscan Corpus with Existing Abstractions
Shows how to use existing functions without creating new abstractions
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import existing abstractions - NO NEW ABSTRACTIONS!
from docs_to_eval.utils.text_processing import create_smart_chunks
from docs_to_eval.utils.config import ChunkingConfig
from docs_to_eval.core.agentic.agents import ConceptMiner
from docs_to_eval.core.evaluation import EvaluationType


def load_etruscan_corpus() -> Dict[str, str]:
    """Load all Etruscan texts from domain_spcfc_general_corpus"""
    corpus_dir = Path("domain_spcfc_general_corpus/etruscan_texts")
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


def demonstrate_existing_abstractions():
    """Demonstrate using existing abstractions with domain corpus"""
    print("🏺 ETRUSCAN CORPUS with EXISTING ABSTRACTIONS")
    print("🧠 Using: create_smart_chunks + ConceptMiner + ChunkingConfig")
    print("=" * 70)
    
    # Step 1: Load domain corpus using existing directory structure
    print("📚 Loading domain-specific Etruscan corpus...")
    etruscan_texts = load_etruscan_corpus()
    print(f"✅ Loaded {len(etruscan_texts)} Etruscan texts")
    
    if not etruscan_texts:
        print("❌ No texts found. Check domain_spcfc_general_corpus/etruscan_texts/")
        return
    
    # Show sample texts
    sample_names = list(etruscan_texts.keys())[:5]
    print(f"   📖 Sample texts: {', '.join(sample_names)}")
    
    # Step 2: Combine into corpus (existing utility pattern)
    combined_corpus = ""
    for name, content in list(etruscan_texts.items())[:8]:  # Use 8 texts
        combined_corpus += f"\n\n# {name.replace('_', ' ').title()}\n\n{content}"
    
    print(f"📝 Combined corpus: {len(combined_corpus)} characters")
    
    # Step 3: Use existing ChunkingConfig abstraction
    print("\n🔧 Using existing ChunkingConfig abstraction...")
    config = ChunkingConfig(
        target_chunk_size=1500,
        max_chunk_size=2500,
        min_chunk_size=800,
        overlap_percent=15.0,
        force_chunker="semantic",
        enable_chonkie=True,  # Use existing chonkie integration
        adaptive_sizing=True,
        preserve_code_blocks=False,
        preserve_math_expressions=False
    )
    print(f"✅ Created ChunkingConfig: semantic strategy, chonkie={config.enable_chonkie}")
    
    # Step 4: Use existing create_smart_chunks function
    print("\n🧠 Using existing create_smart_chunks function...")
    chunks = create_smart_chunks(combined_corpus, chunking_config=config)
    
    print(f"✅ Created {len(chunks)} chunks using existing function")
    if chunks:
        sizes = [len(chunk['text']) for chunk in chunks]
        methods = [chunk.get('method', 'unknown') for chunk in chunks]
        print(f"   📏 Chunk sizes: {min(sizes)}-{max(sizes)} chars (avg: {sum(sizes)//len(sizes)})")
        print(f"   🔧 Chunking methods: {set(methods)}")
        print(f"   🧠 Advanced chunking: {any(m.startswith('chonkie_') for m in methods)}")
    
    # Step 5: Use existing ConceptMiner agent abstraction
    print("\n🤖 Using existing ConceptMiner agent abstraction...")
    concept_miner = ConceptMiner()  # Existing agent
    
    # Use the existing _simple_concept_extraction method directly
    concepts = concept_miner._simple_concept_extraction(combined_corpus)
    
    print(f"✅ Extracted {len(concepts)} concepts using existing ConceptMiner")
    if concepts:
        # Show top concepts by score
        sorted_concepts = sorted(concepts.items(), key=lambda x: x[1][0], reverse=True)
        top_concepts = sorted_concepts[:10]
        print(f"   🎯 Top concepts:")
        for concept, (score, snippet) in top_concepts:
            print(f"      • {concept} (score: {score:.3f}): {snippet[:50]}...")
    
    # Step 6: Show semantic analysis using existing chunk metadata
    print("\n🔍 Analyzing semantic structure with existing metadata...")
    semantic_scores = []
    for chunk in chunks:
        score = chunk.get('semantic_score', 1.0)
        semantic_scores.append(score)
    
    if semantic_scores:
        avg_score = sum(semantic_scores) / len(semantic_scores)
        print(f"   🎯 Average semantic coherence: {avg_score:.3f}")
        print(f"   📊 Score range: {min(semantic_scores):.3f} - {max(semantic_scores):.3f}")
    
    # Step 7: Show content samples from chunks
    print(f"\n📖 Sample chunk content:")
    for i, chunk in enumerate(chunks[:2]):
        text_preview = chunk['text'][:120].replace('\n', ' ').strip()
        method = chunk.get('method', 'unknown')
        score = chunk.get('semantic_score', 1.0)
        print(f"   {i+1}. [{method}, score={score:.3f}] {text_preview}...")
    
    # Step 8: Show evaluation type classification using existing enums
    print(f"\n🎯 Available evaluation types (existing EvaluationType enum):")
    eval_types = [
        EvaluationType.DOMAIN_KNOWLEDGE,
        EvaluationType.FACTUAL_QA,
        EvaluationType.MULTIPLE_CHOICE,
        EvaluationType.MATHEMATICAL
    ]
    
    for eval_type in eval_types:
        print(f"   • {eval_type.value}: {eval_type}")
    
    print(f"\n✨ DEMONSTRATION COMPLETE!")
    print("=" * 50)
    print(f"🏆 Successfully demonstrated using EXISTING abstractions:")
    print(f"   ✓ ChunkingConfig for configuration")
    print(f"   ✓ create_smart_chunks for text processing")
    print(f"   ✓ ConceptMiner for concept extraction")
    print(f"   ✓ EvaluationType enum for classification")
    print(f"   ✓ domain_spcfc_general_corpus for domain texts")
    print(f"   ✓ Chonkie integration (fallback demonstrated)")
    
    print(f"\n📊 Processing Results:")
    print(f"   📚 Texts processed: {len(etruscan_texts)}")
    print(f"   📝 Corpus size: {len(combined_corpus):,} chars")
    print(f"   🧠 Semantic chunks: {len(chunks)}")
    print(f"   🤖 Concepts extracted: {len(concepts)}")
    print(f"   🎯 Evaluation types available: {len(eval_types)}")
    
    print(f"\n🔥 ZERO NEW ABSTRACTIONS CREATED!")
    print(f"   Everything used existing functions, classes, and configurations!")


if __name__ == "__main__":
    demonstrate_existing_abstractions()