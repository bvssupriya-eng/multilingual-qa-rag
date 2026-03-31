"""
Comprehensive evaluation metrics for RAG QA system
Includes retrieval and generation quality metrics
"""

import re
from collections import Counter
import numpy as np


class RAGMetrics:
    """Evaluation metrics for RAG system"""
    
    def __init__(self):
        pass
    
    # ==================== RETRIEVAL METRICS ====================
    
    def precision_at_k(self, retrieved_docs, k=5):
        """
        Precision@K - Proportion of relevant docs in top K
        For RAG, we consider docs with score > threshold as relevant
        External sources (Wikipedia) are always considered relevant
        """
        if not retrieved_docs or k == 0:
            return 0.0
        
        # For external sources (Wikipedia), adjust k to actual number of docs
        if retrieved_docs[0].get('source') == 'external':
            k = min(k, len(retrieved_docs))
        
        top_k = retrieved_docs[:k]
        relevant_count = 0
        for doc in top_k:
            # External sources are always relevant
            if doc.get('source') == 'external':
                relevant_count += 1
            # Local sources need good score
            elif (doc.get('hybrid_score') or 0) > 0.5:
                relevant_count += 1
        
        return relevant_count / k
    
    def mean_reciprocal_rank(self, retrieved_docs, threshold=0.5):
        """
        MRR - Reciprocal rank of first relevant document
        Higher is better (max 1.0)
        External sources (Wikipedia) are always considered relevant
        """
        for idx, doc in enumerate(retrieved_docs, 1):
            # External sources are always relevant
            if doc.get('source') == 'external':
                return 1.0 / idx
            # Local sources need good score
            score = doc.get('hybrid_score') or 0
            if score > threshold:
                return 1.0 / idx
        return 0.0
    
    def retrieval_score(self, retrieved_docs):
        """
        Overall retrieval quality score
        Based on top result score and diversity
        """
        if not retrieved_docs:
            return 0.0
        
        # Handle None scores from external sources (Wikipedia fallback)
        # For external sources, use a default high score since they were fetched successfully
        top_score = retrieved_docs[0].get('hybrid_score')
        if top_score is None:
            # External source (Wikipedia) - assume high quality
            top_score = 0.8
        
        # Average of top 3 (weighted 30%)
        top3_scores = []
        for doc in retrieved_docs[:3]:
            score = doc.get('hybrid_score')
            if score is None:
                score = 0.8  # External source
            top3_scores.append(score)
        
        avg_top3 = np.mean(top3_scores)
        
        return 0.7 * top_score + 0.3 * avg_top3
    
    # ==================== GENERATION METRICS ====================
    
    def answer_completeness(self, answer):
        """
        Measures if answer is complete (not cut off)
        Checks length, sentence structure, punctuation
        """
        if not answer or len(answer.strip()) < 20:
            return 0.3
        
        score = 0.0
        
        # Length check (longer = more complete, up to a point)
        length = len(answer)
        if length > 100:
            score += 0.4
        elif length > 50:
            score += 0.3
        else:
            score += 0.2
        
        # Ends with proper punctuation
        if answer.strip()[-1] in '.!?':
            score += 0.3
        else:
            score += 0.1
        
        # Has multiple sentences (more complete)
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
        Measures how relevant the answer is to the question
        Uses word overlap and question word coverage
        """
        if not answer or not question:
            return 0.0
        
        # Extract keywords (words > 3 chars)
        q_words = set(re.findall(r'\w+', question.lower()))
        q_words = {w for w in q_words if len(w) > 3}
        
        a_words = set(re.findall(r'\w+', answer.lower()))
        a_words = {w for w in a_words if len(w) > 3}
        
        if not q_words:
            return 0.5
        
        # Calculate overlap
        overlap = len(q_words & a_words)
        coverage = overlap / len(q_words)
        
        # Boost score if answer is substantial
        if len(answer) > 100:
            coverage = min(1.0, coverage * 1.2)
        
        return coverage
    
    def context_utilization(self, answer, context):
        """
        Measures how well the answer uses the provided context
        Similar to faithfulness but more lenient
        """
        if not answer or not context:
            return 0.0
        
        # Extract content words from answer
        answer_words = re.findall(r'\w+', answer.lower())
        answer_words = [w for w in answer_words if len(w) > 3]
        
        if not answer_words:
            return 0.0
        
        # Extract content words from context
        context_words = set(re.findall(r'\w+', context.lower()))
        context_words = {w for w in context_words if len(w) > 3}
        
        # Count how many answer words appear in context
        supported = sum(1 for w in answer_words if w in context_words)
        
        utilization = supported / len(answer_words)
        
        # Boost if utilization is reasonable (not too low, not 100%)
        if 0.3 <= utilization <= 0.8:
            utilization = min(1.0, utilization * 1.1)
        
        return utilization
    
    def fluency_score(self, answer):
        """
        Measures fluency and readability of the answer
        Checks for proper structure and natural language
        """
        if not answer or len(answer.strip()) < 10:
            return 0.3
        
        score = 0.0
        
        # Has proper capitalization
        if answer[0].isupper():
            score += 0.2
        
        # Has proper punctuation
        if any(p in answer for p in '.!?,;:'):
            score += 0.2
        
        # Not too many special characters (indicates formatting issues)
        special_chars = len(re.findall(r'[^\w\s.,!?;:\'\"-]', answer))
        if special_chars < len(answer) * 0.05:
            score += 0.2
        
        # Has reasonable word length distribution
        words = answer.split()
        if words:
            avg_word_len = np.mean([len(w) for w in words])
            if 3 <= avg_word_len <= 8:
                score += 0.2
        
        # Has connecting words (and, but, however, etc.)
        connectors = ['and', 'but', 'however', 'therefore', 'thus', 'also', 'additionally']
        if any(conn in answer.lower() for conn in connectors):
            score += 0.2
        
        return min(1.0, score)
    
    # ==================== COMBINED METRICS ====================
    
    def compute_all_metrics(self, question, answer, context, retrieved_docs):
        """
        Compute all metrics at once
        
        Returns:
            Dictionary with all metric scores
        """
        metrics = {
            # Retrieval metrics
            'precision@5': self.precision_at_k(retrieved_docs, k=5),
            'mrr': self.mean_reciprocal_rank(retrieved_docs),
            'retrieval_quality': self.retrieval_score(retrieved_docs),
            
            # Generation metrics
            'answer_completeness': self.answer_completeness(answer),
            'answer_relevance': self.answer_relevance(question, answer),
            'context_utilization': self.context_utilization(answer, context),
            'fluency': self.fluency_score(answer),
        }
        
        # Overall scores
        metrics['retrieval_score'] = np.mean([
            metrics['precision@5'],
            metrics['mrr'],
            metrics['retrieval_quality']
        ])
        
        metrics['generation_score'] = np.mean([
            metrics['answer_completeness'],
            metrics['answer_relevance'],
            metrics['context_utilization'],
            metrics['fluency']
        ])
        
        metrics['overall_score'] = 0.4 * metrics['retrieval_score'] + 0.6 * metrics['generation_score']
        
        return metrics
    
    def format_metrics(self, metrics):
        """Format metrics for display"""
        output = []
        output.append("\n" + "="*50)
        output.append("EVALUATION METRICS")
        output.append("="*50)
        
        output.append("\n--- Retrieval Metrics ---")
        output.append(f"  Precision@5:       {metrics['precision@5']:.3f}")
        output.append(f"  MRR:               {metrics['mrr']:.3f}")
        output.append(f"  Retrieval Quality: {metrics['retrieval_quality']:.3f}")
        output.append(f"  → Retrieval Score: {metrics['retrieval_score']:.3f}")
        
        output.append("\n--- Generation Metrics ---")
        output.append(f"  Completeness:      {metrics['answer_completeness']:.3f}")
        output.append(f"  Relevance:         {metrics['answer_relevance']:.3f}")
        output.append(f"  Context Use:       {metrics['context_utilization']:.3f}")
        output.append(f"  Fluency:           {metrics['fluency']:.3f}")
        output.append(f"  → Generation Score: {metrics['generation_score']:.3f}")
        
        output.append("\n--- Overall ---")
        output.append(f"  Overall Score:     {metrics['overall_score']:.3f}")
        output.append("="*50 + "\n")
        
        return "\n".join(output)
