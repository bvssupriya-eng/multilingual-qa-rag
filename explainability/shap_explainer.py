"""
SHAP Explainer for RAG System (FIXED VERSION)
Uses simple permutation-based Shapley values without numba
"""

import numpy as np
import re
from itertools import combinations


class RAGShapExplainer:
    """
    SHAP-based explainer using manual Shapley value calculation
    Avoids numba compilation issues
    """
    
    def __init__(self, generator, retriever):
        """
        Args:
            generator: QAGenerator instance
            retriever: Retriever instance
        """
        self.generator = generator
        self.retriever = retriever
    
    def explain_query_importance(self, query, language="en", num_samples=30):
        """
        Calculate Shapley values for query words manually
        
        Args:
            query: User query
            language: Query language
            num_samples: Number of permutations to sample
            
        Returns:
            Dictionary mapping word -> SHAP value
        """
        import warnings
        warnings.filterwarnings('ignore')
        
        # Temporarily suppress logging during SHAP analysis
        import logging
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        
        words = query.split()
        content_words = [w for w in words if len(w) > 2]
        
        if len(content_words) < 2:
            logging.getLogger().setLevel(original_level)
            return {}
        
        # Limit to avoid too many permutations
        if len(content_words) > 8:
            content_words = content_words[:8]
        
        print(f"  → Calculating Shapley values for {len(content_words)} words...")
        
        try:
            # Cache to avoid repeated retrieval
            retrieval_cache = {}
            
            def cached_search(q):
                if q not in retrieval_cache:
                    results = self.retriever.search(q, language=language, top_k=1)
                    if results:
                        score = results[0].get('hybrid_score')
                        if score is None:
                            score = 0.8  # External source
                    else:
                        score = 0.0
                    retrieval_cache[q] = score
                return retrieval_cache[q]
            
            # Calculate baseline (empty query)
            baseline_score = 0.0
            
            # Calculate full query score
            full_score = cached_search(query)
            
            # Calculate Shapley values using sampling
            shapley_values = {}
            
            for target_word in content_words:
                marginal_contributions = []
                
                # Sample different subsets
                other_words = [w for w in content_words if w != target_word]
                
                # Sample subsets of different sizes
                for subset_size in range(len(other_words) + 1):
                    if subset_size == 0:
                        # Subset without target word: empty
                        subset_without = []
                        subset_with = [target_word]
                    elif subset_size == len(other_words):
                        # Subset without target word: all others
                        subset_without = other_words
                        subset_with = content_words
                    else:
                        # Random subset
                        import random
                        random.seed(42)
                        subset_without = random.sample(other_words, subset_size)
                        subset_with = subset_without + [target_word]
                    
                    # Score without target word
                    query_without = " ".join(subset_without)
                    if query_without.strip():
                        score_without = cached_search(query_without)
                    else:
                        score_without = 0.0
                    
                    # Score with target word
                    query_with = " ".join(subset_with)
                    score_with = cached_search(query_with)
                    
                    # Marginal contribution
                    marginal = score_with - score_without
                    marginal_contributions.append(marginal)
                
                # Average marginal contribution is the Shapley value
                shapley_values[target_word] = np.mean(marginal_contributions)
            
            # Restore logging level
            logging.getLogger().setLevel(original_level)
            print(f"  ✓ SHAP analysis complete")
            return shapley_values
            
        except Exception as e:
            # Restore logging level on error
            logging.getLogger().setLevel(original_level)
            print(f"  ✗ SHAP failed: {str(e)[:80]}")
            print(f"  → Using fallback analysis...")
            return self._fallback_query_analysis(query)
    
    def explain_context_importance(self, question, context, num_samples=30):
        """
        Calculate importance of context words
        Uses word overlap heuristic for speed
        
        Args:
            question: User question
            context: Retrieved context
            num_samples: Not used (for compatibility)
            
        Returns:
            Dictionary mapping word -> importance score
        """
        # Extract question keywords
        question_words = set(re.findall(r'\w+', question.lower()))
        question_words = {w for w in question_words if len(w) > 3}
        
        # Extract context words
        context_words = context.split()[:50]  # Limit to 50 words
        
        word_importance = {}
        
        for word in context_words:
            clean_word = re.sub(r'\W+', '', word.lower())
            if len(clean_word) > 3:
                # Score based on overlap with question
                if clean_word in question_words:
                    word_importance[word] = 1.0
                else:
                    # Score by word length (longer = more important)
                    word_importance[word] = min(0.5, len(clean_word) / 20.0)
        
        return word_importance
    
    def get_summary(self, query_importance, context_importance):
        """
        Get human-readable summary
        
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
    
    def _fallback_query_analysis(self, query):
        """Simple fallback when SHAP fails"""
        words = query.split()
        word_importance = {}
        max_len = max([len(w) for w in words if len(w) > 2], default=1)
        
        for w in words:
            if len(w) > 2:
                word_importance[w] = len(w) / max_len
        
        return word_importance
