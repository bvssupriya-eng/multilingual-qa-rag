"""
SHAP Explainer for RAG System
Uses actual SHAP library for model interpretability
"""

import numpy as np
import re
import shap
import logging
from typing import Dict, List, Tuple


class RAGShapExplainer:
    """
    SHAP-based explainer using the official SHAP library
    Provides feature importance for query words and context
    """
    
    def __init__(self, generator, retriever):
        """
        Args:
            generator: QAGenerator instance
            retriever: Retriever instance
        """
        self.generator = generator
        self.retriever = retriever

    @staticmethod
    def _result_score(results) -> float:
        """Return a stable retrieval score for local or fallback results."""
        if not results:
            return 0.0

        top = results[0]
        if top.get("source") == "external":
            return 1.0

        score = top.get("hybrid_score")
        return float(score) if score is not None else 0.0
    
    def explain_query_importance(self, query: str, language: str = "en", num_samples: int = 15) -> Dict[str, float]:
        """
        Calculate SHAP values for query words using fast approximation
        
        Args:
            query: User query
            language: Query language
            num_samples: Number of samples for SHAP estimation (unused, kept for compatibility)
            
        Returns:
            Dictionary mapping word -> SHAP value
        """
        import warnings
        warnings.filterwarnings('ignore')
        
        # Temporarily suppress logging during SHAP analysis
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        
        words = query.split()
        # Keep all words except very short stopwords
        stopwords = ['is', 'are', 'was', 'were', 'the', 'a', 'an', 'of', 'to', 'in', 'on', 'at', 'by', 'for']
        content_words = [w for w in words if w.lower() not in stopwords]
        
        # If no content words after filtering, use all words with length > 2
        if len(content_words) < 1:
            content_words = [w for w in words if len(w) > 2]
        
        # Still no words? Return empty
        if len(content_words) < 1:
            logging.getLogger().setLevel(original_level)
            print(f"  ✗ No content words found in query: {query}")
            return {}
        
        # Limit words to avoid computational explosion (max 5 words for speed)
        if len(content_words) > 5:
            content_words = content_words[:5]
        
        print(f"  → Analyzing {len(content_words)} words: {content_words}...", flush=True)
        
        # Use fast manual calculation
        result = self._fast_shapley_approximation(query, language, content_words)
        logging.getLogger().setLevel(original_level)
        
        if not result:
            print(f"  ✗ SHAP approximation returned empty results", flush=True)
        
        return result
    
    def _fast_shapley_approximation(self, query: str, language: str, content_words: List[str]) -> Dict[str, float]:
        """
        Fast Shapley approximation using only key combinations
        Much faster than full SHAP - only tests each word individually
        
        Args:
            query: Original query
            language: Query language
            content_words: List of content words
            
        Returns:
            Dictionary mapping word -> approximate Shapley value
        """
        # Cache for retrieval results
        retrieval_cache = {}
        
        def cached_search(q):
            if q not in retrieval_cache:
                try:
                    logging.disable(logging.CRITICAL)
                    results = self.retriever.search(q, language=language, top_k=1)
                    logging.disable(logging.NOTSET)

                    score = self._result_score(results)
                except:
                    logging.disable(logging.NOTSET)
                    score = 0.0
                retrieval_cache[q] = score
            return retrieval_cache[q]
        
        # Get baseline score (all words)
        baseline_query = " ".join(content_words)
        baseline_score = cached_search(baseline_query)
        
        shapley_values = {}
        
        # For each word, calculate marginal contribution
        for target_word in content_words:
            # Score without this word
            words_without = [w for w in content_words if w != target_word]
            query_without = " ".join(words_without)
            score_without = cached_search(query_without) if query_without else 0.0
            
            # Marginal contribution = baseline - without
            marginal = baseline_score - score_without
            shapley_values[target_word] = marginal
        
        print(f"  ✓ SHAP analysis complete")
        return shapley_values
    
    def explain_context_importance(self, question: str, context: str, num_samples: int = 30) -> Dict[str, float]:
        """
        Calculate SHAP values for context words
        Uses word masking to determine importance
        
        Args:
            question: User question
            context: Retrieved context
            num_samples: Number of samples for SHAP
            
        Returns:
            Dictionary mapping word -> SHAP value
        """
        # Extract context words (limit to first 50 for efficiency)
        context_words = context.split()[:50]
        content_words = [w for w in context_words if len(re.sub(r'\W+', '', w)) > 3]
        
        if len(content_words) < 2:
            return {}
        
        # Limit to top 10 words for computational efficiency
        if len(content_words) > 10:
            content_words = content_words[:10]
        
        try:
            # Define prediction function
            def predict_answer_quality(word_mask):
                """
                Predict answer quality based on which context words are included
                """
                scores = []
                for mask in word_mask:
                    # Build context from masked words
                    selected_words = [w for w, m in zip(content_words, mask) if m > 0.5]
                    
                    if not selected_words:
                        scores.append(0.0)
                        continue
                    
                    masked_context = " ".join(selected_words)
                    
                    # Generate answer and score it
                    try:
                        answer = self.generator.generate_answer(
                            question=question,
                            context=masked_context,
                            role="student",
                            language="en"
                        )
                        # Score based on answer length and relevance
                        score = min(1.0, len(answer.split()) / 50.0)
                    except Exception:
                        score = 0.0
                    
                    scores.append(score)
                
                return np.array(scores)
            
            # Create background samples
            num_features = len(content_words)
            np.random.seed(42)
            background_samples = min(num_samples, 2 ** num_features)
            background = np.random.randint(0, 2, size=(background_samples, num_features)).astype(float)
            background[0] = np.zeros(num_features)
            background[1] = np.ones(num_features)
            
            # Current instance (all words present)
            instance = np.ones((1, num_features))
            
            # Create SHAP explainer
            explainer = shap.KernelExplainer(
                model=predict_answer_quality,
                data=background,
                link="identity"
            )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(
                instance,
                nsamples=num_samples,
                silent=True
            )
            
            # Map words to SHAP values
            word_importance = {}
            for word, shap_val in zip(content_words, shap_values[0]):
                word_importance[word] = float(shap_val)
            
            return word_importance
            
        except Exception as e:
            print(f"  ✗ Context SHAP failed: {str(e)[:80]}")
            return self._fallback_context_analysis(question, context)
    
    def _fallback_context_analysis(self, question: str, context: str) -> Dict[str, float]:
        """
        Fallback: Simple word overlap heuristic
        """
        question_words = set(re.findall(r'\w+', question.lower()))
        question_words = {w for w in question_words if len(w) > 3}
        
        context_words = context.split()[:50]
        word_importance = {}
        
        for word in context_words:
            clean_word = re.sub(r'\W+', '', word.lower())
            if len(clean_word) > 3:
                if clean_word in question_words:
                    word_importance[word] = 1.0
                else:
                    word_importance[word] = min(0.5, len(clean_word) / 20.0)
        
        return word_importance
    
    def get_summary(self, query_importance: Dict[str, float], context_importance: Dict[str, float]) -> str:
        """
        Get human-readable summary of SHAP analysis
        
        Args:
            query_importance: Dict from explain_query_importance
            context_importance: Dict from explain_context_importance
            
        Returns:
            Formatted summary string
        """
        summary = []
        
        summary.append("\n--- SHAP Query Word Importance ---")
        if query_importance:
            sorted_query = sorted(query_importance.items(), 
                                 key=lambda x: abs(x[1]), reverse=True)[:5]
            for word, score in sorted_query:
                summary.append(f"  {word}: {score:+.3f}")
        else:
            summary.append("  No results")
        
        summary.append("\n--- SHAP Context Word Importance ---")
        if context_importance:
            sorted_context = sorted(context_importance.items(),
                                   key=lambda x: abs(x[1]), reverse=True)[:5]
            for word, score in sorted_context:
                summary.append(f"  {word}: {score:.3f}")
        else:
            summary.append("  No results")
        
        return "\n".join(summary)
