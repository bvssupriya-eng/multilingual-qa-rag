# Experimental Methodology: Multilingual RAG-based Question Answering System

## 📋 Project Overview

This document outlines the comprehensive experimental methodology for evaluating a Retrieval-Augmented Generation (RAG) system with explainable AI (XAI) capabilities across multiple languages (English, Hindi, Bengali, Arabic).

---

## 🛠️ Tools & Technologies

### Core Framework
- **Python 3.8+** - Primary programming language
- **Sentence Transformers** - Multilingual semantic embeddings
- **FAISS** - Vector similarity search
- **Hugging Face Transformers** - Language model integration
- **BM25** - Sparse retrieval baseline

### Explainable AI (XAI)
- **SHAP (SHapley Additive exPlanations)** - Feature importance analysis
- **Counterfactual Explanations** - What-if scenario analysis

### Evaluation & Analysis
- **NumPy & Pandas** - Data manipulation and analysis
- **Scikit-learn** - Evaluation metrics
- **Matplotlib/Seaborn** - Visualization (optional)

### Data Sources
- **Wikipedia API** - Multilingual knowledge base
- **Local Document Store** - Custom knowledge corpus
- **Hybrid Retrieval** - Combined semantic + keyword search

---

## 📂 Dataset & Knowledge Base

### Supported Languages
| Language | Code | Script | Sample Query |
|----------|------|--------|--------------|
| English | en | Latin | "What is Machine Learning?" |
| Hindi | hi | Devanagari | "मशीन लर्निंग क्या है?" |
| Bengali | bn | Bengali | "মেশিন লার্নিং কি?" |
| Arabic | ar | Arabic | "ما هو تعلم الآلة؟" |

### Knowledge Sources
1. **Local Documents** - Pre-indexed domain-specific documents
2. **Wikipedia Fallback** - Real-time retrieval for missing content
3. **Hybrid Index** - Combined semantic + BM25 scoring

### Protected/Controlled Attributes
- **Language** - Ensures fair performance across languages
- **Query Complexity** - Simple vs. complex questions
- **Domain** - Technical, general knowledge, educational

---

## 🔬 Experimental Methodology

### Phase 1: System Initialization

```python
# Import core libraries
from retrieval.search import Retriever
from generation.qa_generator import QAGenerator
from explainability.shap_explainer import RAGShapExplainer
from explainability.counterfactual_explainer import CounterfactualExplainer
from evaluation.metrics import RAGMetrics

# Initialize components
retriever = Retriever(
    index_path="data/faiss_index",
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

generator = QAGenerator(
    model_name="google/flan-t5-base",
    device="cpu"
)

# Initialize XAI modules
shap_explainer = RAGShapExplainer(generator, retriever)
cf_explainer = CounterfactualExplainer(generator, retriever)

# Initialize evaluation metrics
metrics = RAGMetrics()
```

---

### Phase 2: Query Processing & Retrieval

```python
# User query input
query = "What is Machine Learning?"
language = "en"  # Language code

# Retrieval with hybrid scoring
results = retriever.search(
    query=query,
    language=language,
    top_k=5,
    use_hybrid=True
)

# Extract top result
top_result = results[0]
context = top_result['content']
retrieval_score = top_result['hybrid_score']

print(f"Retrieval Score: {retrieval_score:.3f}")
print(f"Source: {top_result['source']}")
print(f"Method: {top_result['method']}")
```

**Retrieval Metrics:**
- Hybrid Score (0-1): Combination of semantic similarity + BM25
- Precision@K: Relevance of top-K results
- Mean Reciprocal Rank (MRR): Position of first relevant result

---

### Phase 3: Explainable AI Analysis

#### Option A: SHAP Analysis (~10-20 seconds)

```python
# SHAP-based feature importance
query_importance = shap_explainer.explain_query_importance(
    query=query,
    language=language,
    num_samples=100
)

context_importance = shap_explainer.explain_context_importance(
    question=query,
    context=context,
    num_samples=100
)

# Display SHAP summary
shap_summary = shap_explainer.get_summary(
    query_importance,
    context_importance
)
print(shap_summary)
```

**SHAP Metrics:**
- Shapley Values: Contribution of each word to retrieval/generation
- Feature Importance Ranking: Top-K most influential words
- Baseline vs. Perturbed Performance: Impact quantification

#### Option B: Counterfactual Analysis (~3-5 seconds)

```python
# Counterfactual "what-if" scenarios
query_cf = cf_explainer.explain_query_words(
    query=query,
    language=language,
    top_k=5
)

context_cf = cf_explainer.explain_context_usage(
    question=query,
    context=context,
    answer=answer  # Generated answer
)

# Display counterfactual results
cf_output = cf_explainer.format_output(query_cf, context_cf)
print(cf_output)
```

**Counterfactual Metrics:**
- Impact Score: Change in retrieval score when word removed
- Interpretation Categories: CRITICAL, IMPORTANT, HELPFUL, NEUTRAL, NOISE
- Context Sentence Contribution: Effect of removing each sentence

---

### Phase 4: Answer Generation (Role-Based)

```python
# Generate answers for different user roles
roles = ["beginner", "student", "teacher"]

for role in roles:
    print(f"\n{'='*50}")
    print(f"ROLE: {role.upper()}")
    print('='*50)
    
    # Generate answer
    answer = generator.generate_answer(
        question=query,
        context=context,
        role=role,
        language=language
    )
    
    print(f"\nAnswer:\n{answer}")
```

**Generation Parameters:**
- **Beginner**: Simple language, basic concepts, short answers
- **Student**: Moderate detail, educational tone, examples
- **Teacher**: Comprehensive, technical depth, structured format

---

### Phase 5: Comprehensive Evaluation

```python
# Evaluate retrieval quality
retrieval_metrics = metrics.evaluate_retrieval(
    query=query,
    retrieved_docs=results,
    top_k=5
)

# Evaluate generation quality
generation_metrics = metrics.evaluate_generation(
    question=query,
    context=context,
    answer=answer,
    language=language
)

# Combined evaluation
overall_score = metrics.get_overall_score(
    retrieval_metrics,
    generation_metrics
)

# Display results
print("\n" + "="*50)
print("EVALUATION METRICS")
print("="*50)
print(metrics.format_results(retrieval_metrics, generation_metrics))
```

---

## ⚖️ Evaluation Metrics

### 1️⃣ Retrieval Metrics

#### Precision@K
Measures relevance of top-K retrieved documents.

```python
def precision_at_k(retrieved_docs, k=5):
    """
    Precision@K = (Relevant docs in top-K) / K
    """
    relevant_count = sum(1 for doc in retrieved_docs[:k] 
                        if doc['hybrid_score'] > 0.5)
    return relevant_count / k
```

**Interpretation:**
- **> 0.8**: Excellent retrieval
- **0.6-0.8**: Good retrieval
- **< 0.6**: Poor retrieval, needs improvement

#### Mean Reciprocal Rank (MRR)
Measures position of first relevant result.

```python
def mean_reciprocal_rank(retrieved_docs, threshold=0.5):
    """
    MRR = 1 / (rank of first relevant document)
    """
    for i, doc in enumerate(retrieved_docs, 1):
        if doc['hybrid_score'] > threshold:
            return 1.0 / i
    return 0.0
```

**Interpretation:**
- **MRR = 1.0**: Best result is rank 1
- **MRR = 0.5**: Best result is rank 2
- **MRR < 0.3**: Relevant results ranked too low

#### Retrieval Quality Score
Weighted combination of similarity scores.

```python
def retrieval_quality_score(retrieved_docs):
    """
    Weighted average of top-K scores with decay
    """
    if not retrieved_docs:
        return 0.0
    
    weights = [1.0 / (i + 1) for i in range(len(retrieved_docs))]
    scores = [doc['hybrid_score'] for doc in retrieved_docs]
    
    weighted_score = sum(w * s for w, s in zip(weights, scores))
    return weighted_score / sum(weights)
```

---

### 2️⃣ Generation Metrics

#### Answer Completeness
Measures coverage of key concepts from context.

```python
def answer_completeness(context, answer):
    """
    Completeness = (Key terms in answer) / (Key terms in context)
    """
    context_words = set(extract_keywords(context))
    answer_words = set(extract_keywords(answer))
    
    if not context_words:
        return 0.0
    
    overlap = len(context_words & answer_words)
    return overlap / len(context_words)
```

**Interpretation:**
- **> 0.7**: Comprehensive answer
- **0.5-0.7**: Adequate coverage
- **< 0.5**: Incomplete answer

#### Answer Relevance
Measures semantic similarity between question and answer.

```python
def answer_relevance(question, answer, embedder):
    """
    Relevance = cosine_similarity(question_embedding, answer_embedding)
    """
    q_emb = embedder.encode(question)
    a_emb = embedder.encode(answer)
    
    similarity = np.dot(q_emb, a_emb) / (
        np.linalg.norm(q_emb) * np.linalg.norm(a_emb)
    )
    return float(similarity)
```

#### Context Utilization
Measures how well the answer uses retrieved context.

```python
def context_utilization(context, answer):
    """
    Utilization = (Context words in answer) / (Total answer words)
    """
    context_words = set(extract_keywords(context))
    answer_words = extract_keywords(answer)
    
    if not answer_words:
        return 0.0
    
    utilized = sum(1 for word in answer_words if word in context_words)
    return utilized / len(answer_words)
```

**Interpretation:**
- **> 0.6**: Strong context grounding
- **0.4-0.6**: Moderate context usage
- **< 0.4**: Weak context grounding (potential hallucination)

#### Fluency Score
Measures linguistic quality of generated answer.

```python
def fluency_score(answer):
    """
    Fluency based on:
    - Sentence structure
    - Length appropriateness
    - Punctuation usage
    """
    sentences = answer.split('.')
    
    # Penalize very short or very long answers
    length_score = min(len(answer) / 200, 1.0)
    
    # Reward proper sentence structure
    structure_score = min(len(sentences) / 3, 1.0)
    
    # Check for basic punctuation
    punct_score = 1.0 if any(p in answer for p in '.!?') else 0.5
    
    return (length_score + structure_score + punct_score) / 3
```

---

### 3️⃣ Combined Evaluation

```python
def get_overall_score(retrieval_metrics, generation_metrics):
    """
    Overall Score = 0.4 * Retrieval + 0.6 * Generation
    """
    retrieval_score = (
        retrieval_metrics['precision@5'] * 0.3 +
        retrieval_metrics['mrr'] * 0.3 +
        retrieval_metrics['retrieval_quality'] * 0.4
    )
    
    generation_score = (
        generation_metrics['completeness'] * 0.25 +
        generation_metrics['relevance'] * 0.25 +
        generation_metrics['context_utilization'] * 0.25 +
        generation_metrics['fluency'] * 0.25
    )
    
    overall = 0.4 * retrieval_score + 0.6 * generation_score
    
    return {
        'retrieval_score': retrieval_score,
        'generation_score': generation_score,
        'overall_score': overall
    }
```

---

## 📊 Result Analysis Framework

### Quantitative Analysis

Students/researchers must analyze:

1. **Retrieval Performance**
   - Does hybrid search outperform pure semantic or BM25?
   - Which language shows best/worst retrieval scores?
   - How does query complexity affect retrieval quality?

2. **Generation Quality**
   - Are answers grounded in retrieved context?
   - Do different roles produce appropriate answer styles?
   - Is there evidence of hallucination (low context utilization)?

3. **XAI Insights**
   - Which query words are most critical for retrieval?
   - Do SHAP values align with human intuition?
   - What counterfactual scenarios reveal system weaknesses?

4. **Cross-Lingual Fairness**
   - Are evaluation scores consistent across languages?
   - Does the system favor certain languages?
   - Are non-English queries handled fairly?

---

### Qualitative Analysis

#### Answer Quality Assessment

| Criterion | Poor (0-0.4) | Fair (0.4-0.6) | Good (0.6-0.8) | Excellent (0.8-1.0) |
|-----------|--------------|----------------|----------------|---------------------|
| Accuracy | Incorrect info | Partially correct | Mostly correct | Fully accurate |
| Completeness | Missing key points | Some gaps | Covers main points | Comprehensive |
| Relevance | Off-topic | Somewhat related | Directly relevant | Perfectly aligned |
| Clarity | Confusing | Understandable | Clear | Crystal clear |

#### Role Appropriateness

```python
# Example evaluation for role-based generation
role_evaluation = {
    "beginner": {
        "simplicity": 0.9,  # Uses simple language
        "length": 0.8,      # Concise
        "jargon": 0.1       # Minimal technical terms
    },
    "student": {
        "detail": 0.7,      # Moderate detail
        "examples": 0.8,    # Includes examples
        "structure": 0.9    # Well-organized
    },
    "teacher": {
        "depth": 0.9,       # Technical depth
        "comprehensiveness": 0.85,
        "accuracy": 0.95    # High accuracy
    }
}
```

---

## 📉 Visualization & Reporting

### Recommended Visualizations

The project includes a comprehensive visualization module. See `visualization/README.md` for details.

**Available Visualizers:**
- `MetricsVisualizer` - 6 plot types for evaluation metrics
- `RAGVisualizer` - 7 plot types for retrieval analysis
- `XAIVisualizer` - 7 plot types for explainability insights

**Quick Example:**
```python
from visualization import MetricsVisualizer, RAGVisualizer, XAIVisualizer

# Visualize metrics
metrics_viz = MetricsVisualizer()
metrics_viz.plot_retrieval_metrics(retrieval_metrics)
metrics_viz.plot_generation_metrics(generation_metrics)

# Visualize retrieval
rag_viz = RAGVisualizer()
rag_viz.plot_retrieval_scores(results)
rag_viz.plot_cross_lingual_comparison(language_scores)

# Visualize XAI
xai_viz = XAIVisualizer()
xai_viz.plot_shap_query_importance(query_importance)
xai_viz.plot_counterfactual_impact(counterfactuals)
```

Run example visualizations:
```bash
python visualization/example_usage.py
```

---

## 🎯 Experimental Scenarios

### Scenario 1: Baseline Performance
**Objective:** Establish baseline metrics across all languages

```python
test_queries = {
    "en": "What is Machine Learning?",
    "hi": "मशीन लर्निंग क्या है?",
    "bn": "মেশিন লার্নিং কি?",
    "ar": "ما هو تعلم الآلة?"
}

baseline_results = {}
for lang, query in test_queries.items():
    results = run_full_pipeline(query, lang)
    baseline_results[lang] = results['overall_score']
```

### Scenario 2: Retrieval Method Comparison
**Objective:** Compare semantic vs. BM25 vs. hybrid retrieval

```python
retrieval_methods = ['semantic', 'bm25', 'hybrid']
comparison_results = {}

for method in retrieval_methods:
    retriever.set_method(method)
    results = retriever.search(query, language, top_k=5)
    comparison_results[method] = results[0]['score']
```

### Scenario 3: XAI Method Comparison
**Objective:** Compare SHAP vs. Counterfactual explanations

```python
# Time and quality comparison
import time

# SHAP
start = time.time()
shap_results = shap_explainer.explain_query_importance(query, language)
shap_time = time.time() - start

# Counterfactual
start = time.time()
cf_results = cf_explainer.explain_query_words(query, language)
cf_time = time.time() - start

print(f"SHAP Time: {shap_time:.2f}s")
print(f"Counterfactual Time: {cf_time:.2f}s")
```

### Scenario 4: Role-Based Generation Quality
**Objective:** Evaluate appropriateness of role-specific answers

```python
roles = ["beginner", "student", "teacher"]
role_metrics = {}

for role in roles:
    answer = generator.generate_answer(query, context, role, language)
    metrics = evaluate_role_appropriateness(answer, role)
    role_metrics[role] = metrics
```

### Scenario 5: Stress Testing
**Objective:** Test system with edge cases

```python
edge_cases = [
    "a",  # Single character
    "What is the meaning of life, universe, and everything?",  # Complex
    "asdfghjkl",  # Nonsense
    "Machine Learning " * 50,  # Repetitive
]

for query in edge_cases:
    try:
        results = run_full_pipeline(query, "en")
        print(f"Query: {query[:50]}... | Score: {results['overall_score']:.3f}")
    except Exception as e:
        print(f"Query: {query[:50]}... | Error: {str(e)}")
```

---

## 📝 Reporting Template

### Executive Summary
- System overview
- Key findings (3-5 bullet points)
- Overall performance score

### Methodology
- Dataset description
- Experimental setup
- Evaluation metrics used

### Results
- Quantitative metrics (tables)
- Qualitative analysis
- Visualizations

### XAI Analysis
- SHAP insights
- Counterfactual findings
- Interpretability assessment

### Discussion
- Strengths and limitations
- Cross-lingual fairness analysis
- Comparison with baselines

### Conclusion
- Summary of findings
- Future improvements
- Recommendations

---

## 🔍 Key Performance Indicators (KPIs)

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| Retrieval Precision@5 | > 0.8 | 0.6-0.8 | < 0.6 |
| Answer Completeness | > 0.7 | 0.5-0.7 | < 0.5 |
| Context Utilization | > 0.6 | 0.4-0.6 | < 0.4 |
| Overall Score | > 0.75 | 0.6-0.75 | < 0.6 |
| XAI Execution Time | < 15s | 15-30s | > 30s |
| Cross-Lingual Variance | < 0.1 | 0.1-0.2 | > 0.2 |

---

## 🚀 Future Enhancements

1. **Advanced Metrics**
   - BLEU/ROUGE scores for answer quality
   - BERTScore for semantic similarity
   - Perplexity for fluency

2. **User Studies**
   - Human evaluation of answers
   - XAI interpretability assessment
   - Cross-lingual usability testing

3. **Automated Testing**
   - Continuous integration pipeline
   - Regression testing suite
   - Performance benchmarking

4. **Fairness Analysis**
   - Demographic parity across languages
   - Equal opportunity in retrieval
   - Bias detection in generation

---

## 📚 References

1. Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
2. Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions" (SHAP)
3. Wachter et al. (2017). "Counterfactual Explanations without Opening the Black Box"
4. Reimers & Gurevych (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Author:** Multilingual RAG QA System Team
