"""
Comprehensive evaluation metrics for RAG QA system.
Improvements over v1:
  - Stopword filtering in answer_relevance and context_utilization
  - Metric key consistency: 'precision_at_5' everywhere (was 'precision@5')
  - Renamed internal keys to match MLflow log calls
  - Added NDCG@5 (ranking-aware retrieval metric)
  - Added answer_length_ratio metric
  - Weighted retrieval_score uses NDCG
"""

import re
import math
from collections import Counter
import numpy as np
from config import METRICS_STOPWORDS


class RAGMetrics:
    """Evaluation metrics for RAG system"""

    def __init__(self):
        pass

    # ==================== HELPERS ====================

    def _content_words(self, text: str) -> set:
        """Extract meaningful words: length > 3 and not a stopword."""
        words = set(re.findall(r'\w+', text.lower()))
        return {w for w in words if len(w) > 3 and w not in METRICS_STOPWORDS}

    # ==================== RETRIEVAL METRICS ====================

    def precision_at_k(self, retrieved_docs, k=5):
        """
        Precision@K — proportion of relevant docs in top-K.
        A doc is relevant if source=external OR hybrid_score > 0.5.
        """
        if not retrieved_docs or k == 0:
            return 0.0

        if retrieved_docs[0].get('source') == 'external':
            k = min(k, len(retrieved_docs))

        top_k = retrieved_docs[:k]
        relevant_count = sum(
            1 for doc in top_k
            if doc.get('source') == 'external'
            or (doc.get('hybrid_score') or 0) > 0.5
        )
        return relevant_count / k

    def mean_reciprocal_rank(self, retrieved_docs, threshold=0.5):
        """
        MRR — reciprocal rank of the first relevant document.
        """
        for idx, doc in enumerate(retrieved_docs, 1):
            if doc.get('source') == 'external':
                return 1.0 / idx
            if (doc.get('hybrid_score') or 0) > threshold:
                return 1.0 / idx
        return 0.0

    def ndcg_at_k(self, retrieved_docs, k=5):
        """
        NDCG@K — Normalized Discounted Cumulative Gain.
        Uses hybrid_score as graded relevance (external = 1.0).
        Rewards high-quality documents appearing earlier in ranking.
        """
        if not retrieved_docs:
            return 0.0

        def relevance(doc):
            if doc.get('source') == 'external':
                return 1.0
            score = doc.get('hybrid_score') or 0.0
            return float(score)

        top_k = retrieved_docs[:k]
        gains = [relevance(d) for d in top_k]

        # DCG
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))

        # Ideal DCG (sort by descending relevance)
        all_gains = sorted([relevance(d) for d in retrieved_docs], reverse=True)
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(all_gains[:k]))

        return dcg / idcg if idcg > 0 else 0.0

    def retrieval_quality_score(self, retrieved_docs):
        """
        Holistic retrieval quality: weighted by top score and avg top-3.
        """
        if not retrieved_docs:
            return 0.0

        top_score = retrieved_docs[0].get('hybrid_score')
        if top_score is None:
            top_score = 0.8  # external (Wikipedia)

        top3_scores = []
        for doc in retrieved_docs[:3]:
            s = doc.get('hybrid_score')
            top3_scores.append(0.8 if s is None else float(s))

        avg_top3 = float(np.mean(top3_scores))
        return 0.7 * top_score + 0.3 * avg_top3

    # ==================== GENERATION METRICS ====================

    def answer_completeness(self, answer):
        """
        Measures if answer is complete (not cut off).
        Checks length, sentence structure, punctuation.
        """
        if not answer or len(answer.strip()) < 20:
            return 0.3

        score = 0.0
        length = len(answer)
        if length > 100:
            score += 0.4
        elif length > 50:
            score += 0.3
        else:
            score += 0.2

        if answer.strip()[-1] in '.!?':
            score += 0.3
        else:
            score += 0.1

        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(sentences) >= 3:
            score += 0.3
        elif len(sentences) >= 2:
            score += 0.2
        else:
            score += 0.1

        return min(1.0, score)

    def answer_relevance(self, question, answer):
        """
        Measures how relevant the answer is to the question.
        Uses stopword-filtered word overlap.
        """
        if not answer or not question:
            return 0.0

        q_words = self._content_words(question)
        a_words = self._content_words(answer)

        if not q_words:
            return 0.5

        overlap = len(q_words & a_words)
        coverage = overlap / len(q_words)

        if len(answer) > 100:
            coverage = min(1.0, coverage * 1.2)

        return coverage

    def context_utilization(self, answer, context):
        """
        Measures how well the answer uses the provided context.
        Stopword-filtered for accurate measurement.
        """
        if not answer or not context:
            return 0.0

        answer_words_raw = re.findall(r'\w+', answer.lower())
        answer_words = [
            w for w in answer_words_raw
            if len(w) > 3 and w not in METRICS_STOPWORDS
        ]

        if not answer_words:
            return 0.0

        context_words = {
            w for w in re.findall(r'\w+', context.lower())
            if len(w) > 3 and w not in METRICS_STOPWORDS
        }

        supported = sum(1 for w in answer_words if w in context_words)
        utilization = supported / len(answer_words)

        if 0.3 <= utilization <= 0.8:
            utilization = min(1.0, utilization * 1.1)

        return utilization

    def fluency_score(self, answer):
        """
        Measures fluency and readability of the answer.
        Checks capitalization, punctuation, word length distribution, connectors.
        """
        if not answer or len(answer.strip()) < 10:
            return 0.3

        score = 0.0
        if answer[0].isupper():
            score += 0.2
        if any(p in answer for p in '.!?,;:'):
            score += 0.2
        special_chars = len(re.findall(r'[^\w\s.,!?;:\'\"-]', answer))
        if special_chars < len(answer) * 0.05:
            score += 0.2
        words = answer.split()
        if words:
            avg_word_len = float(np.mean([len(w) for w in words]))
            if 3 <= avg_word_len <= 8:
                score += 0.2
        connectors = ['and', 'but', 'however', 'therefore', 'thus', 'also', 'additionally']
        if any(conn in answer.lower() for conn in connectors):
            score += 0.2
        return min(1.0, score)

    def answer_length_ratio(self, answer, context):
        """
        Checks if answer length is reasonable relative to context.
        Ideal ratio: 0.10 – 0.45 of context length.
        Too short (<0.05) or too long (>0.7) = penalty.
        """
        if not answer or not context:
            return 0.5
        ratio = len(answer) / max(len(context), 1)
        if 0.10 <= ratio <= 0.45:
            return 1.0
        elif 0.05 <= ratio < 0.10:
            return 0.7
        elif 0.45 < ratio <= 0.70:
            return 0.7
        elif ratio > 0.70:
            return 0.4
        else:
            return 0.3  # extremely short

    # ==================== COMBINED METRICS ====================

    def compute_all_metrics(self, question, answer, context, retrieved_docs):
        """
        Compute all metrics at once.

        Returns:
            Dict with consistent snake_case keys aligned with MLflow log names.
        """
        # Retrieval
        prec = self.precision_at_k(retrieved_docs, k=5)
        mrr = self.mean_reciprocal_rank(retrieved_docs)
        ret_quality = self.retrieval_quality_score(retrieved_docs)
        ndcg = self.ndcg_at_k(retrieved_docs, k=5)

        # Generation
        completeness = self.answer_completeness(answer)
        relevance = self.answer_relevance(question, answer)
        ctx_use = self.context_utilization(answer, context)
        fluency = self.fluency_score(answer)
        length_ratio = self.answer_length_ratio(answer, context)

        # Composite scores
        # Retrieval: weight NDCG more heavily (ranking quality matters)
        retrieval_score = float(np.mean([prec, mrr, ret_quality, ndcg]))

        # Generation: include length_ratio as a light penalty/bonus (20% weight)
        generation_score = (
            0.25 * completeness
            + 0.25 * relevance
            + 0.20 * ctx_use
            + 0.20 * fluency
            + 0.10 * length_ratio
        )

        overall_score = 0.4 * retrieval_score + 0.6 * generation_score

        return {
            # Retrieval — consistent snake_case keys
            'precision_at_5':    round(prec, 4),
            'mrr':               round(mrr, 4),
            'retrieval_quality': round(ret_quality, 4),
            'ndcg_at_5':         round(ndcg, 4),
            'retrieval_score':   round(retrieval_score, 4),
            # Generation
            'completeness':      round(completeness, 4),
            'relevance':         round(relevance, 4),
            'context_use':       round(ctx_use, 4),
            'fluency':           round(fluency, 4),
            'length_ratio':      round(length_ratio, 4),
            'generation_score':  round(generation_score, 4),
            # Overall
            'overall_score':     round(overall_score, 4),
        }

    def format_metrics(self, metrics):
        """Format metrics for clean terminal display."""
        output = []
        output.append("\n" + "=" * 50)
        output.append("EVALUATION METRICS")
        output.append("=" * 50)

        output.append("\n--- Retrieval Metrics ---")
        output.append(f"  Precision@5:       {metrics.get('precision_at_5', 0):.3f}")
        output.append(f"  MRR:               {metrics.get('mrr', 0):.3f}")
        output.append(f"  Retrieval Quality: {metrics.get('retrieval_quality', 0):.3f}")
        output.append(f"  NDCG@5:            {metrics.get('ndcg_at_5', 0):.3f}")
        output.append(f"  → Retrieval Score: {metrics.get('retrieval_score', 0):.3f}")

        output.append("\n--- Generation Metrics ---")
        output.append(f"  Completeness:      {metrics.get('completeness', 0):.3f}")
        output.append(f"  Relevance:         {metrics.get('relevance', 0):.3f}")
        output.append(f"  Context Use:       {metrics.get('context_use', 0):.3f}")
        output.append(f"  Fluency:           {metrics.get('fluency', 0):.3f}")
        output.append(f"  Length Ratio:      {metrics.get('length_ratio', 0):.3f}")
        output.append(f"  → Generation Score:{metrics.get('generation_score', 0):.3f}")

        output.append("\n--- Overall ---")
        output.append(f"  Overall Score:     {metrics.get('overall_score', 0):.3f}")
        output.append("=" * 50 + "\n")

        return "\n".join(output)
