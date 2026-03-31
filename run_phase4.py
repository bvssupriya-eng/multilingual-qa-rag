"""
Phase 4: Full RAG Pipeline (without XAI)
Simple question-answering with retrieval and generation
"""

import sys
from retrieval.search import Retriever
from generation.qa_generator import QAGenerator
from evaluation.faithfulness import compute_faithfulness
from config import REGENERATE_ON_LOW_FAITHFULNESS, FAITHFULNESS_RETRY_THRESHOLD

ROLES = ["beginner", "student", "teacher"]
MAX_CONTEXT_CHARS = 1200


def build_context(results):
    if results[0]["source"] == "external":
        text = results[0]["text"][:MAX_CONTEXT_CHARS]
        return text, [text]

    context_parts = []
    source_texts = []
    used_chars = 0

    for idx, r in enumerate(results[:5], 1):
        source_label = f"S{idx}"
        raw_text = r["text"].strip()
        remaining = MAX_CONTEXT_CHARS - used_chars
        if remaining <= 0:
            break

        snippet = raw_text[:remaining]
        part = f"[{source_label}] Title: {r.get('title') or 'Untitled'}\n{snippet}"
        context_parts.append(part)
        source_texts.append(snippet)
        used_chars += len(snippet)

    return "\n\n".join(context_parts), source_texts


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    retriever = Retriever()
    generator = QAGenerator()

    query = input("\nEnter your query: ")
    lang = input("Language code (en/hi/bn/ar): ").strip().lower()

    # Validate language
    if lang not in ["en", "hi", "bn", "ar"]:
        print(f"Error: Invalid language code '{lang}'. Must be one of: en, hi, bn, ar")
        sys.exit(1)
    
    # Validate query length
    if len(query.strip()) < 3:
        print("Error: Query too short. Please enter at least 3 characters.")
        sys.exit(1)
    
    if len(query) > 500:
        print("Warning: Query is very long. Truncating to 500 characters.")
        query = query[:500]

    # Retrieval
    print(f"\n[RETRIEVAL: {lang.upper()}]")
    results = retriever.search(query, language=lang)
    
    if results[0].get("code_mixed"):
        print("  → Code-mixed query detected")
    
    print(f"  → Method: {results[0]['retrieval_stage']} | Source: {results[0]['source']}")

    print("\n" + "="*50)
    print("Source :", results[0]["source"])
    print("Title  :", results[0].get("title"))
    print("="*50 + "\n")

    # Build context
    context, source_texts = build_context(results)

    print("\n--- Retrieved Context (First 400 chars) ---")
    print(context[:400])
    print("-------------------------------------------\n")

    # Translation if needed
    if lang != "en":
        query_en = retriever.translate(query, lang, "en")
        context_en = retriever.translate(context[:2000], lang, "en")
    else:
        query_en = query
        context_en = context

    # Generation
    print(f"\n[GENERATION: {lang.upper()} → EN → {lang.upper()}]" if lang != "en" else "\n[GENERATION: EN]")
    
    for role in ROLES:
        print(f"\n[{role.upper()}]")

        answer = generator.generate_answer(
            question=query_en,
            context=context_en,
            role=role,
            language="en"
        )
        
        answer_en = answer

        # Translate back
        if lang != "en":
            answer = retriever.translate(answer, "en", lang)

        support = compute_faithfulness(answer, source_texts)

        # Regenerate if low faithfulness
        if (
            REGENERATE_ON_LOW_FAITHFULNESS
            and support["faithfulness_score"] < FAITHFULNESS_RETRY_THRESHOLD
            and results[0]["source"] != "external"
        ):
            print(f"  ⚠ Low faithfulness, regenerating...")
            retry_instruction = (
                "Regenerate using only explicitly supported facts from the provided sources. "
                "Keep the answer short, cite source labels, and if support is weak say Not found."
            )
                
            answer = generator.generate_answer(
                question=query_en,
                context=context_en,
                role="eval",
                language="en",
                extra_instruction=retry_instruction
            )
            
            answer_en = answer
            
            if lang != "en":
                answer = retriever.translate(answer, "en", lang)
                
            support = compute_faithfulness(answer, source_texts)

        print(f"\n{answer}")
        print(f"\nConfidence: {support['confidence']} ({support['faithfulness_score']:.2f})")
        
        if results[0]["source"] != "external":
            print("Sources used:")
            for idx, r in enumerate(results[:5], 1):
                print(f"  [S{idx}] {r.get('title')} | score={r.get('hybrid_score'):.3f}")
