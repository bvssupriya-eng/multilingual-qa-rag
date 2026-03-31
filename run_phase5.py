import sys
import logging
import time
import mlflow
import json
from langdetect import detect, LangDetectException

from retrieval.search import Retriever
from generation.qa_generator import QAGenerator
from evaluation.faithfulness import compute_faithfulness
from evaluation.metrics import RAGMetrics
from config import REGENERATE_ON_LOW_FAITHFULNESS, FAITHFULNESS_RETRY_THRESHOLD
from explainability.shap_explainer import RAGShapExplainer
from explainability.counterfactual_explainer import CounterfactualExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

ROLES = ["beginner", "student", "teacher"]
MAX_CONTEXT_CHARS = 1200

# Language mapping for detection
LANG_MAP = {
    'en': 'en',
    'hi': 'hi',
    'bn': 'bn',
    'ar': 'ar',
    'ur': 'ar',
    'pa': 'hi',
    'ne': 'hi',
}

LANG_NAMES = {
    'en': 'English',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'ar': 'Arabic'
}

def detect_query_language(query):
    """Detect language of the query"""
    try:
        detected = detect(query)
        if detected in LANG_MAP:
            return LANG_MAP[detected]
        else:
            return 'en'
    except LangDetectException:
        return 'en'

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
    evaluator = RAGMetrics()
    
    # Initialize MLflow experiment
    try:
        mlflow.set_experiment("multilingual_qa_system")
    except Exception as e:
        logging.warning(f"MLflow initialization failed: {e}")

    print("\n" + "="*60)
    print("MULTILINGUAL RAG QA SYSTEM")
    print("="*60)

    # Step 1: Get query and detect language
    query = input("\nEnter your query (in any language): ").strip()
    
    if not query:
        print("Error: Query cannot be empty.")
        sys.exit(1)
    
    if len(query) < 3:
        print("Error: Query too short. Please enter at least 3 characters.")
        sys.exit(1)
    
    if len(query) > 500:
        print("Warning: Query is very long. Truncating to 500 characters.")
        query = query[:500]

    # Auto-detect query language
    query_lang = detect_query_language(query)
    print(f"\n✓ Detected query language: {LANG_NAMES[query_lang]} ({query_lang})")

    # Step 2: Choose answer language
    print("\n=== Answer Language Selection ===")
    print("1. English (en)")
    print("2. Hindi (hi)")
    print("3. Bengali (bn)")
    print("4. Arabic (ar)")
    
    answer_lang_choice = input("Choose answer language (1/2/3/4): ").strip()
    answer_lang_map = {'1': 'en', '2': 'hi', '3': 'bn', '4': 'ar'}
    answer_lang = answer_lang_map.get(answer_lang_choice, 'en')
    print(f"✓ Answer will be in: {LANG_NAMES[answer_lang]}")

    # Step 3: Choose role
    print("\n=== Role Selection ===")
    print("1. Beginner (simple, easy language)")
    print("2. Student (balanced, moderate detail)")
    print("3. Teacher (advanced, comprehensive)")
    
    role_choice = input("Choose role (1/2/3): ").strip()
    role_map = {'1': 'beginner', '2': 'student', '3': 'teacher'}
    selected_role = role_map.get(role_choice, 'student')
    print(f"✓ Selected role: {selected_role.upper()}")

    # Step 4: Choose XAI method
    print("\n=== XAI Method Selection ===")
    print("1. SHAP (rigorous, ~10-20s)")
    print("2. Counterfactual (intuitive, ~3-5s)")
    print("3. No XAI (skip explanations)")
    
    xai_choice = input("Choose XAI method (1/2/3): ").strip()
    
    if xai_choice == "1":
        explainer = RAGShapExplainer(generator, retriever)
        use_shap = True
        use_counterfactual = False
        use_xai = True
        xai_method = "shap"
    elif xai_choice == "2":
        explainer = CounterfactualExplainer(generator, retriever)
        use_shap = False
        use_counterfactual = True
        use_xai = True
        xai_method = "counterfactual"
    else:
        use_xai = False
        use_shap = False
        use_counterfactual = False
        explainer = None
        xai_method = "none"

    # Start MLflow run with tracing
    try:
        mlflow.start_run()
        
        # A. Log input parameters
        mlflow.log_param("query", query)
        mlflow.log_param("detected_language", query_lang)
        mlflow.log_param("answer_language", answer_lang)
        mlflow.log_param("role", selected_role)
        mlflow.log_param("xai_method", xai_method)
        
        # B. Log system config
        from config import RETRIEVAL_THRESHOLD, FINAL_TOP_K
        mlflow.log_param("retrieval_threshold", RETRIEVAL_THRESHOLD)
        mlflow.log_param("top_k", FINAL_TOP_K)
        
    except Exception as e:
        logging.warning(f"MLflow logging failed: {e}")

    # ---------------------------
    # RETRIEVAL WITH TRACING
    # ---------------------------
    print("\n" + "="*60)
    print(f"RETRIEVAL (Query Language: {query_lang.upper()})")
    print("="*60)
    
    retrieval_start = time.time()
    
    # Start retrieval trace span
    with mlflow.start_span(name="retrieval") as retrieval_span:
        retrieval_span.set_inputs({"query": query, "language": query_lang})
        
        # Dense search span
        with mlflow.start_span(name="dense_search") as dense_span:
            dense_start = time.time()
            results = retriever.search(query, language=query_lang)
            dense_time = time.time() - dense_start
            dense_span.set_attribute("duration_ms", dense_time * 1000)
        
        retrieval_span.set_outputs({"num_results": len(results), "source": results[0]['source']})
        retrieval_span.set_attribute("retrieval_time_ms", (time.time() - retrieval_start) * 1000)
    
    retrieval_time = time.time() - retrieval_start
    
    # C. Log retrieval behavior
    try:
        top_score = results[0].get('hybrid_score', 0) if results[0].get('hybrid_score') is not None else 0
        fallback_triggered = 1 if results[0]['source'] == 'external' else 0
        num_results = len(results)
        
        mlflow.log_metric("top_score", top_score)
        mlflow.log_metric("fallback_triggered", fallback_triggered)
        mlflow.log_metric("num_results", num_results)
        mlflow.log_metric("retrieval_time", retrieval_time)
    except Exception as e:
        logging.warning(f"MLflow retrieval logging failed: {e}")
    
    # Retrieval Transparency
    print(f"\n📊 Retrieval Transparency:")
    print(f"  Source: {results[0]['source'].upper()}")
    print(f"  Method: {results[0]['retrieval_stage']}")
    
    if results[0]['source'] == 'external':
        print(f"  ✓ Wikipedia Fallback Triggered")
        print(f"  Article: {results[0].get('title')}")
    else:
        print(f"  ✓ Local Corpus Used")
        print(f"\n  Top Retrieved Documents:")
        for idx, r in enumerate(results[:5], 1):
            score = r.get('hybrid_score', 0)
            print(f"    [{idx}] {r.get('title', 'Untitled')[:50]:50s} | Score: {score:.3f}")
    
    if results[0].get("code_mixed"):
        print("  ⚠ Code-mixed query detected")

    context, source_texts = build_context(results)

    print(f"\n--- Retrieved Context (First 400 chars) ---")
    print(context[:400])
    print("-------------------------------------------")

    # ---------------------------
    # XAI EXPLANATIONS WITH TRACING
    # ---------------------------
    if use_xai:
        print("\n" + "="*60)
        if results[0]['source'] == 'local':
            print("XAI EXPLANATIONS (Local Corpus)")
        else:
            print("XAI EXPLANATIONS (Wikipedia Fallback)")
        print("="*60)
        
        query_for_xai = query if query_lang == "en" else retriever.translate(query, query_lang, "en")
        context_for_xai = context[:1500] if query_lang == "en" else retriever.translate(context[:1500], query_lang, "en")
        
        # Start XAI trace span
        with mlflow.start_span(name="xai_explanation") as xai_span:
            xai_span.set_inputs({"method": xai_method, "query": query_for_xai})
            xai_start = time.time()
            
            if use_shap:
                print("  → Running SHAP analysis (this may take 10-20 seconds)...")
                query_exp = explainer.explain_query_importance(query_for_xai, language="en", num_samples=20)
                context_exp = explainer.explain_context_importance(
                    question=query_for_xai,
                    context=context_for_xai,
                    num_samples=30
                )
                print(explainer.get_summary(query_exp, context_exp))
                
                if results[0]['source'] == 'external':
                    print("\n  ℹ Note: For Wikipedia fallback:")
                    print("    - Query word importance shows which words triggered the search")
                    print("    - Context importance shows key terms in the Wikipedia article")
            
            elif use_counterfactual:
                print("  → Running counterfactual analysis (this may take 3-5 seconds)...")
                query_cf = explainer.explain_query_words(query_for_xai, language="en", top_k=5)
                sample_answer = generator.generate_answer(
                    question=query_for_xai,
                    context=context_for_xai,
                    role="student",
                    language="en"
                )
                context_cf = explainer.explain_context_usage(
                    question=query_for_xai,
                    context=context_for_xai,
                    answer=sample_answer
                )
                print(explainer.format_output(query_cf, context_cf))
                
                if results[0]['source'] == 'external':
                    print("\n  ℹ Note: For Wikipedia fallback:")
                    print("    - Shows impact of removing each word from the query")
                    print("    - Helps understand which words were critical for finding the article")
            
            xai_time = time.time() - xai_start
            xai_span.set_outputs({"completed": True})
            xai_span.set_attribute("xai_time_ms", xai_time * 1000)

    # ---------------------------
    # GENERATION WITH TRACING
    # ---------------------------
    print("\n" + "="*60)
    print(f"GENERATION (Role: {selected_role.upper()}, Language: {LANG_NAMES[answer_lang]})")
    print("="*60)
    
    query_en = query if query_lang == "en" else retriever.translate(query, query_lang, "en")
    context_en = context if query_lang == "en" else retriever.translate(context[:2000], query_lang, "en")

    generation_start = time.time()
    
    # Start generation trace span
    with mlflow.start_span(name="generation") as gen_span:
        gen_span.set_inputs({"query": query_en, "role": selected_role, "language": answer_lang})
        
        answer = generator.generate_answer(
            question=query_en,
            context=context_en,
            role=selected_role,
            language="en"
        )
        
        gen_span.set_outputs({"answer_length": len(answer)})
        gen_span.set_attribute("generation_time_ms", (time.time() - generation_start) * 1000)
    
    generation_time = time.time() - generation_start
    
    answer_en = answer

    if answer_lang != "en":
        with mlflow.start_span(name="translation") as trans_span:
            trans_span.set_inputs({"text_length": len(answer), "src": "en", "tgt": answer_lang})
            answer = retriever.translate(answer, "en", answer_lang)
            trans_span.set_outputs({"translated_length": len(answer)})

    # Start evaluation trace span
    with mlflow.start_span(name="evaluation") as eval_span:
        eval_start = time.time()
        
        metrics = evaluator.compute_all_metrics(
            question=query_en,
            answer=answer_en,
            context=context_en,
            retrieved_docs=results
        )
        
        support = compute_faithfulness(
            answer, 
            source_texts,
            retrieval_score=metrics['retrieval_score'],
            generation_score=metrics['generation_score']
        )
        
        eval_span.set_outputs({"overall_score": metrics['overall_score'], "confidence": support['faithfulness_score']})
        eval_span.set_attribute("evaluation_time_ms", (time.time() - eval_start) * 1000)

    if (
        REGENERATE_ON_LOW_FAITHFULNESS
        and support["faithfulness_score"] < FAITHFULNESS_RETRY_THRESHOLD
        and results[0]["source"] != "external"
    ):
        print(f"\n⚠ Low faithfulness detected, regenerating...")
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
        
        if answer_lang != "en":
            answer = retriever.translate(answer, "en", answer_lang)
        
        metrics = evaluator.compute_all_metrics(
            question=query_en,
            answer=answer_en,
            context=context_en,
            retrieved_docs=results
        )
        
        support = compute_faithfulness(
            answer, 
            source_texts,
            retrieval_score=metrics['retrieval_score'],
            generation_score=metrics['generation_score']
        )

    print(f"\n{answer}")
    print(f"\nConfidence: {support['confidence']} ({support['faithfulness_score']:.2f})")
    
    if results[0]["source"] != "external":
        print("\nSources used:")
        for idx, r in enumerate(results[:5], 1):
            print(f"  [S{idx}] {r.get('title')} | score={r.get('hybrid_score'):.3f}")
    
    print(evaluator.format_metrics(metrics))
    
    # D. Log evaluation metrics
    try:
        mlflow.log_metric("precision_at_5", metrics.get('precision_at_5', 0))
        mlflow.log_metric("mrr", metrics.get('mrr', 0))
        mlflow.log_metric("retrieval_quality", metrics.get('retrieval_quality', 0))
        mlflow.log_metric("retrieval_score", metrics.get('retrieval_score', 0))
        mlflow.log_metric("completeness", metrics.get('completeness', 0))
        mlflow.log_metric("relevance", metrics.get('relevance', 0))
        mlflow.log_metric("context_use", metrics.get('context_use', 0))
        mlflow.log_metric("fluency", metrics.get('fluency', 0))
        mlflow.log_metric("generation_score", metrics.get('generation_score', 0))
        mlflow.log_metric("overall_score", metrics.get('overall_score', 0))
        mlflow.log_metric("confidence", support['faithfulness_score'])
        
        # E. Log performance metrics
        mlflow.log_metric("generation_time", generation_time)
        
        # F. Log artifacts
        mlflow.log_text(answer, "answer.txt")
        
        # Sanitize results for JSON serialization
        results_serializable = []
        for r in results:
            r_clean = {
                'title': r.get('title', ''),
                'text': r.get('text', '')[:500],  # Truncate for size
                'source': r.get('source', ''),
                'language': r.get('language', ''),
                'score': float(r.get('score', 0)) if r.get('score') is not None else 0,
                'hybrid_score': float(r.get('hybrid_score', 0)) if r.get('hybrid_score') is not None else 0,
                'retrieval_stage': r.get('retrieval_stage', '')
            }
            results_serializable.append(r_clean)
        
        mlflow.log_dict(results_serializable, "retrieval.json")
        
    except Exception as e:
        logging.warning(f"MLflow metrics/artifacts logging failed: {e}")
    
    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass
    
    print("\n" + "="*60)
    print("SESSION COMPLETE")
    print("="*60)
