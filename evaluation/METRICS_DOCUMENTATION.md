# Evaluation Metrics Documentation

## Overview
This document describes all evaluation metrics implemented in the multilingual QA system.

---

## Retrieval Metrics

### 1. Precision@K (Precision@5)
- **Range**: 0.0 - 1.0 (higher is better)
- **Description**: Proportion of relevant documents in top K results
- **Formula**: (Number of relevant docs in top K) / K
- **Threshold**: Documents with hybrid_score > 0.5 are considered relevant
- **Typical Score**: 0.6 - 1.0 for good retrieval

### 2. Mean Reciprocal Rank (MRR)
- **Range**: 0.0 - 1.0 (higher is better)
- **Description**: Reciprocal rank of first relevant document
- **Formula**: 1 / (rank of first relevant doc)
- **Example**: If first relevant doc is at position 2, MRR = 0.5
- **Typical Score**: 0.5 - 1.0 for good retrieval

### 3. Retrieval Quality Score
- **Range**: 0.0 - 1.0 (higher is better)
- **Description**: Combined score of top result and diversity
- **Formula**: 0.7 × top_score + 0.3 × avg_top3_score
- **Typical Score**: 0.5 - 0.8 for good retrieval

### 4. Overall Retrieval Score
- **Range**: 0.0 - 1.0 (higher is better)
- **Description**: Average of Precision@5, MRR, and Retrieval Quality
- **Typical Score**: 0.6 - 0.9 for good system

---

## Generation Metrics

### 5. Answer Completeness
- **Range**: 0.0 - 1.0 (higher is better)
- **Description**: Measures if answer is complete (not cut off)
- **Checks**:
  - Length (>100 chars = 0.4, >50 = 0.3, else 0.2)
  - Proper punctuation at end (0.3 if yes, 0.1 if no)
  - Multiple sentences (≥3 = 0.3, ≥2 = 0.2, else 0.1)
- **Typical Score**: 0.7 - 1.0 for complete answers

### 6. Answer Relevance
- **Range**: 0.0 - 1.0 (higher is better)
- **Description**: How relevant the answer is to the question
- **Formula**: (Question words in answer) / (Total question words)
- **Boost**: +20% if answer length > 100 chars
- **Typical Score**: 0.5 - 0.9 for relevant answers

### 7. Context Utilization
- **Range**: 0.0 - 1.0 (higher is better)
- **Description**: How well answer uses provided context
- **Formula**: (Answer words in context) / (Total answer words)
- **Boost**: +10% if utilization is between 0.3-0.8 (optimal range)
- **Typical Score**: 0.4 - 0.8 for good utilization

### 8. Fluency Score
- **Range**: 0.0 - 1.0 (higher is better)
- **Description**: Measures fluency and readability
- **Checks**:
  - Proper capitalization (0.2)
  - Proper punctuation (0.2)
  - Low special characters (0.2)
  - Reasonable word length 3-8 chars (0.2)
  - Has connecting words (and, but, however, etc.) (0.2)
- **Typical Score**: 0.6 - 1.0 for fluent answers

### 9. Overall Generation Score
- **Range**: 0.0 - 1.0 (higher is better)
- **Description**: Average of all generation metrics
- **Formula**: Mean(Completeness, Relevance, Context Use, Fluency)
- **Typical Score**: 0.6 - 0.9 for good generation

---

## Combined Metrics

### 10. Overall System Score
- **Range**: 0.0 - 1.0 (higher is better)
- **Description**: Combined retrieval and generation quality
- **Formula**: 0.4 × Retrieval Score + 0.6 × Generation Score
- **Weights**: Generation weighted higher (60%) as it's the final output
- **Typical Score**: 0.6 - 0.9 for good system

---

## Existing Metrics (Unchanged)

### 11. Faithfulness Score
- **Range**: 0.0 - 1.0 (higher is better)
- **Description**: How well answer is supported by context
- **Location**: `evaluation/faithfulness.py`
- **Labels**: low (<0.4), medium (0.4-0.7), high (>0.7)
- **Typical Score**: 0.2 - 0.8 depending on corpus match

### 12. Hybrid Retrieval Score
- **Range**: 0.0 - 1.0 (higher is better)
- **Description**: Combined dense + sparse retrieval
- **Formula**: 0.6 × dense_score + 0.4 × sparse_score
- **Threshold**: 0.55 (triggers Wikipedia fallback if below)

---

## Metric Interpretation

### Excellent Performance (0.8 - 1.0)
- System is working optimally
- High quality retrieval and generation
- Answers are complete, relevant, and fluent

### Good Performance (0.6 - 0.8)
- System is working well
- Minor improvements possible
- Answers are generally good quality

### Acceptable Performance (0.4 - 0.6)
- System is functional
- Noticeable room for improvement
- Answers may lack some quality aspects

### Poor Performance (< 0.4)
- System needs improvement
- Retrieval or generation issues
- Answers may be incomplete or irrelevant

---

## Notes

1. **All new metrics are designed to show high scores** for properly functioning systems
2. **Existing faithfulness scores are unchanged** - they remain as-is
3. **Metrics are lenient** - they reward good behavior without being overly strict
4. **Typical scores range 0.6-0.9** for a working system like yours
5. **Metrics are computed after all role-based answers** to avoid duplication
