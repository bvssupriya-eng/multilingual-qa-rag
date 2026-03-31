# Multilingual QA RAG System - Solution Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Key Components](#key-components)
4. [Technical Implementation](#technical-implementation)
5. [Features & Capabilities](#features--capabilities)
6. [Setup & Installation](#setup--installation)
7. [Usage Guide](#usage-guide)
8. [Performance & Optimization](#performance--optimization)
9. [MLflow Tracking & Tracing](#mlflow-tracking--tracing)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Future Enhancements](#future-enhancements)

---

## 🎯 Project Overview

A production-ready **Multilingual Question-Answering (QA) system** using Retrieval-Augmented Generation (RAG) with explainability (XAI) and comprehensive experiment tracking.

### Supported Languages
- **English (en)**
- **Hindi (hi)**
- **Bengali (bn)**
- **Arabic (ar)**

### Key Capabilities
- ✅ Hybrid retrieval (Dense FAISS + Sparse BM25)
- ✅ Wikipedia fallback with semantic article matching
- ✅ Role-based answer generation (Beginner/Student/Teacher)
- ✅ Explainability (SHAP & Counterfactual)
- ✅ MLflow experiment tracking with distributed tracing
- ✅ Comprehensive evaluation metrics
- ✅ Multilingual translation support

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY (Any Language)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  LANGUAGE DETECTION                          │
│              (langdetect + normalization)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID RETRIEVAL                          │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Dense Search    │         │  Sparse Search   │          │
│  │  (FAISS HNSW)    │         │  (BM25)          │          │
│  │  Top-10          │         │  Top-10          │          │
│  └────────┬─────────┘         └────────┬─────────┘          │
│           │                            │                     │
│           └────────────┬───────────────┘                     │
│                        ▼                                     │
│              Merge & Score (Top-20)                          │
│              α=0.6 Dense + 0.4 Sparse                        │
│                        │                                     │
│                        ▼                                     │
│              Relevance Check                                 │
│              (Semantic Similarity)                           │
│                        │                                     │
│           ┌────────────┴────────────┐                        │
│           ▼                         ▼                        │
│    Local Results              Wikipedia Fallback            │
│    (Top-5)                    (Semantic Article Match)       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  ANSWER GENERATION                           │
│              (Mistral-7B GGUF Q4_K_M)                        │
│                                                              │
│  Step 1: Factual Answer (temp=0.3)                          │
│  Step 2: Role-based Styling (temp=0.25)                     │
│           - Beginner: Simple language                        │
│           - Student: Balanced detail                         │
│           - Teacher: Comprehensive                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRANSLATION                               │
│         (NLLB-200-distilled-600M if needed)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION                                │
│  • Retrieval Metrics (Precision@5, MRR, Quality)             │
│  • Generation Metrics (Completeness, Relevance, Fluency)     │
│  • Faithfulness Score (Token Overlap)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  XAI EXPLANATIONS                            │
│  • SHAP: Feature importance (query & context words)          │
│  • Counterfactual: Word removal impact analysis              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  MLFLOW TRACKING                             │
│  • Experiment Tracking (params, metrics, artifacts)          │
│  • Distributed Tracing (spans for each component)            │
│  • Artifact Storage (answer.txt, retrieval.json)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Key Components

### 1. **Retrieval System** (`retrieval/`)

#### Hybrid Search Architecture
- **Dense Search**: FAISS HNSW index with `paraphrase-multilingual-MiniLM-L12-v2` embeddings
- **Sparse Search**: BM25 for keyword matching
- **Scoring Formula**: `hybrid_score = α × dense_norm + (1-α) × sparse_norm`
  - Short queries (≤3 words): α = 0.5 (balanced)
  - Long queries (>3 words): α = 0.7 (favor semantic)

#### Wikipedia Fallback
Triggers when:
1. No local results found
2. Top hybrid_score < 0.60 threshold
3. Semantic relevance check fails (title similarity < 0.3)

**Semantic Article Matching**:
```python
final_score = 0.6 × embedding_similarity + 
              0.3 × word_overlap + 
              0.1 × exact_match_bonus - 
              disambiguation_penalty
```

Features:
- Query preprocessing (removes "Who is", "What is", etc.)
- Stopword removal for better matching
- Disambiguation page penalty (-0.5)
- Top-5 candidate evaluation

#### Files
- `search.py`: Main retriever with hybrid search & Wikipedia fallback
- `bm25_index.py`: BM25 sparse retrieval implementation
- `query_normalizer.py`: Code-mixed query handling
- `build_faiss.py`: FAISS index construction

---

### 2. **Generation System** (`generation/`)

#### Model: Mistral-7B-Instruct-v0.2 (GGUF Q4_K_M)
- **Quantization**: 4-bit for efficiency
- **Context Window**: 8192 tokens
- **Framework**: llama-cpp-python

#### Two-Step Generation Process

**Step 1: Factual Answer**
```python
temperature = 0.3
max_tokens = 512
prompt = f"Question: {question}\nContext: {context}\nAnswer:"
```

**Step 2: Role-Based Styling**
```python
temperature = 0.25
roles = {
    "beginner": "simple, easy language with examples",
    "student": "balanced, moderate detail",
    "teacher": "advanced, comprehensive with nuances"
}
```

#### Faithfulness Regeneration
- If faithfulness_score < 0.45, regenerate with stricter instructions
- Forces citation of source labels [S1], [S2], etc.

---

### 3. **Evaluation System** (`evaluation/`)

#### Retrieval Metrics
- **Precision@5**: Proportion of relevant docs in top-5
- **MRR (Mean Reciprocal Rank)**: Position of first relevant doc
- **Retrieval Quality**: Weighted score distribution
- **Retrieval Score**: Combined metric (0.4×P@5 + 0.3×MRR + 0.3×Quality)

#### Generation Metrics
- **Completeness**: Answer length adequacy (0.8-1.0 optimal)
- **Relevance**: Semantic similarity (query ↔ answer)
- **Context Use**: Token overlap (answer ∩ context)
- **Fluency**: Sentence structure quality
- **Generation Score**: Weighted average

#### Faithfulness Score
```python
faithfulness = token_overlap(answer, sources) × 
               sqrt(retrieval_score × generation_score)
```

Confidence levels:
- `high`: ≥ 0.60
- `medium`: 0.40 - 0.59
- `low`: < 0.40

---

### 4. **Explainability (XAI)** (`explainability/`)

#### SHAP Explainer
- **Query Importance**: Shapley values for each query word
- **Context Importance**: Key terms in retrieved documents
- **Samples**: 20 (query), 30 (context)
- **Time**: ~10-20 seconds

#### Counterfactual Explainer
- **Word Removal Impact**: Score change when removing each word
- **Context Usage**: Which context parts influenced the answer
- **Time**: ~3-5 seconds

---

### 5. **MLflow Tracking** (`run_phase5.py`)

#### Experiment Tracking
**Parameters Logged**:
- query, detected_language, answer_language
- role, xai_method
- retrieval_threshold, top_k

**Metrics Logged**:
- Retrieval: precision_at_5, mrr, retrieval_quality, retrieval_score
- Generation: completeness, relevance, context_use, fluency, generation_score
- Overall: overall_score, confidence
- Performance: retrieval_time, generation_time

**Artifacts Logged**:
- `answer.txt`: Generated answer
- `retrieval.json`: Retrieved documents with metadata

#### Distributed Tracing (NEW!)
**Trace Spans**:
1. **retrieval** span
   - Inputs: query, language
   - Outputs: num_results, source
   - Sub-span: **dense_search**
   
2. **xai_explanation** span
   - Inputs: method, query
   - Outputs: completed
   - Attributes: xai_time_ms

3. **generation** span
   - Inputs: query, role, language
   - Outputs: answer_length
   - Attributes: generation_time_ms

4. **translation** span (conditional)
   - Inputs: text_length, src, tgt
   - Outputs: translated_length

5. **evaluation** span
   - Outputs: overall_score, confidence
   - Attributes: evaluation_time_ms

**View Traces**:
```bash
mlflow ui
# Navigate to http://localhost:5000 → Select run → "Traces" tab
```

---

## 🛠️ Technical Implementation

### Configuration (`config.py`)

```python
# Language Codes (Single Source of Truth)
LANG_CODES = {
    "en": {"iso": "en", "nllb": "eng_Latn", "wiki": "en"},
    "hi": {"iso": "hi", "nllb": "hin_Deva", "wiki": "hi"},
    "bn": {"iso": "bn", "nllb": "ben_Beng", "wiki": "bn"},
    "ar": {"iso": "ar", "nllb": "arb_Arab", "wiki": "ar"}
}

# Retrieval Parameters
RETRIEVAL_THRESHOLD = 0.60
HYBRID_DENSE_TOP_K = 10
HYBRID_SPARSE_TOP_K = 10
HYBRID_MERGED_TOP_K = 20
FINAL_TOP_K = 5

# FAISS Index
FAISS_INDEX_TYPE = "hnsw"
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 80
HNSW_EF_SEARCH = 64

# Generation
MISTRAL_MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
REGENERATE_ON_LOW_FAITHFULNESS = True
FAITHFULNESS_RETRY_THRESHOLD = 0.45
```

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | Dense retrieval |
| Translation | facebook/nllb-200-distilled-600M | Cross-lingual support |
| Generation | Mistral-7B-Instruct-v0.2 (Q4_K_M) | Answer generation |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8+
- 8GB+ RAM
- 10GB+ disk space

### Installation Steps

```bash
# 1. Clone/Navigate to project
cd c:\Users\Win11\Desktop\multilingual_qa

# 2. Create virtual environment
python -m venv qa_env

# 3. Activate environment
qa_env\Scripts\activate  # Windows
# source qa_env/bin/activate  # Linux/Mac

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download Mistral model (if not present)
# Place mistral-7b-instruct-v0.2.Q4_K_M.gguf in models/

# 6. Build FAISS index (first time only)
python retrieval/build_faiss.py

# 7. Run the system
python main.py
```

---

## 📖 Usage Guide

### CLI Entry Point (`main.py`)

```bash
python main.py
```

**Menu Options**:
1. Phase 1: Build corpus and embeddings
2. Phase 2: Build FAISS index
3. Phase 3: Test retrieval only
4. Phase 4: Basic RAG (no XAI)
5. Phase 5: Full RAG with XAI & MLflow

### Phase 5 (Full System)

```bash
python run_phase5.py
```

**Interactive Prompts**:
1. Enter query (any language)
2. Select answer language (en/hi/bn/ar)
3. Select role (beginner/student/teacher)
4. Select XAI method (SHAP/Counterfactual/None)

**Example Session**:
```
Enter your query: Who is Elon Musk?
✓ Detected query language: English (en)

=== Answer Language Selection ===
Choose answer language (1/2/3/4): 1
✓ Answer will be in: English

=== Role Selection ===
Choose role (1/2/3): 2
✓ Selected role: STUDENT

=== XAI Method Selection ===
Choose XAI method (1/2/3): 1

[Retrieval → XAI → Generation → Evaluation]
```

---

## ⚡ Performance & Optimization

### Retrieval Optimizations
1. **HNSW Index**: Fast approximate nearest neighbor search
2. **Candidate Filtering**: 2x candidates → language filter → top-k
3. **Score Normalization**: Min-max scaling for fair hybrid scoring
4. **Early Stopping**: Skip low-quality results (score < 0.5)

### Generation Optimizations
1. **4-bit Quantization**: Reduces model size from 14GB → 4GB
2. **Context Truncation**: Max 1200 chars to fit context window
3. **Two-step Generation**: Separate factual + styling for better control

### Typical Latencies
- Retrieval: 0.5-1.5s
- Generation: 3-8s (depends on answer length)
- SHAP XAI: 10-20s
- Counterfactual XAI: 3-5s
- Total (with SHAP): ~15-30s

---

## 📊 MLflow Tracking & Tracing

### Experiment Structure
```
multilingual_qa_system/
├── Run 1 (query: "Who is Elon Musk?")
│   ├── Parameters: query, language, role, xai_method
│   ├── Metrics: retrieval_score, generation_score, overall_score
│   ├── Artifacts: answer.txt, retrieval.json
│   └── Traces:
│       ├── retrieval (1.2s)
│       │   └── dense_search (0.8s)
│       ├── xai_explanation (15.3s)
│       ├── generation (5.4s)
│       ├── translation (0.3s)
│       └── evaluation (0.2s)
├── Run 2 (query: "What is quantum computing?")
│   └── ...
```

### Viewing Results

```bash
# Start MLflow UI
mlflow ui

# Open browser
http://localhost:5000
```

**UI Features**:
- Compare runs side-by-side
- Filter by parameters/metrics
- Download artifacts
- View trace execution tree
- Export data for analysis

### Database Location
- **SQLite DB**: `mlflow.db` (run metadata)
- **Artifacts**: `mlruns/` directory

---

## 📈 Evaluation Metrics

### Metric Ranges & Interpretation

| Metric | Range | Good | Excellent |
|--------|-------|------|-----------|
| Precision@5 | 0-1 | >0.6 | >0.8 |
| MRR | 0-1 | >0.7 | >0.9 |
| Retrieval Score | 0-1 | >0.7 | >0.85 |
| Completeness | 0-1 | >0.8 | >0.9 |
| Relevance | 0-1 | >0.6 | >0.8 |
| Context Use | 0-1 | >0.4 | >0.6 |
| Fluency | 0-1 | >0.9 | >0.95 |
| Generation Score | 0-1 | >0.7 | >0.85 |
| Overall Score | 0-1 | >0.7 | >0.85 |
| Faithfulness | 0-1 | >0.6 | >0.8 |

### Metric Formulas

**Retrieval Score**:
```
retrieval_score = 0.4 × precision@5 + 0.3 × mrr + 0.3 × quality
```

**Generation Score**:
```
generation_score = 0.3 × completeness + 0.3 × relevance + 
                   0.2 × context_use + 0.2 × fluency
```

**Overall Score**:
```
overall_score = 0.5 × retrieval_score + 0.5 × generation_score
```

---

## 🔮 Future Enhancements

### Planned Features
1. ✅ ~~MLflow distributed tracing~~ (COMPLETED)
2. ✅ ~~Semantic Wikipedia article matching~~ (COMPLETED)
3. ⏳ Unit tests for all components
4. ⏳ Config validation on startup
5. ⏳ Query history logging system
6. ⏳ Batch query processing
7. ⏳ REST API endpoint
8. ⏳ Web UI dashboard
9. ⏳ Multi-turn conversation support
10. ⏳ Custom corpus upload

### Potential Improvements
- **Retrieval**: Add reranker model for better top-k selection
- **Generation**: Fine-tune Mistral on domain-specific data
- **XAI**: Add attention visualization for transformer models
- **Evaluation**: Add human evaluation interface
- **Scalability**: Migrate to vector database (Pinecone/Weaviate)

---

## 📝 Testing Queries

### Entity Disambiguation
```
1. Who is Elon Musk?
2. Who is Michael Jordan?
3. Tell me about Paris
4. What is Python?
5. Who is Taylor Swift?
```

### Multilingual
```
6. Qui est Albert Einstein? (French)
7. ما هي مصر (Arabic - What is Egypt?)
8. भारत क्या है (Hindi - What is India?)
```

### Semantic
```
9. Explain artificial intelligence
10. Describe quantum computing
11. What are black holes?
```

### Edge Cases
```
12. asdfghjkl random nonsense
13. Who is XYZ123NotReal?
14. [Empty query]
```

---

## 🤝 Contributing

### Code Structure
```
multilingual_qa/
├── config.py              # Central configuration
├── main.py                # CLI entry point
├── run_phase5.py          # Full RAG with MLflow
├── retrieval/             # Hybrid search + Wikipedia
├── generation/            # Mistral-7B generation
├── evaluation/            # Metrics computation
├── explainability/        # SHAP & Counterfactual
├── embeddings/            # Corpus embedding
├── datasets_loader/       # Data loading utilities
└── requirements.txt       # Dependencies
```

### Development Guidelines
1. Follow existing code style
2. Add docstrings to all functions
3. Update SOLUTION.md for major changes
4. Test with all 4 languages
5. Log experiments with MLflow

---

## 📄 License

This project is for research and educational purposes.

---

## 📧 Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.

---

**Last Updated**: 2024
**Version**: 1.0.0
**Status**: Production-Ready ✅
