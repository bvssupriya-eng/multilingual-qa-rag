"""
Unified XAI Analysis
Runs both SHAP and Counterfactual explanations together
"""

from explainability.shap_explainer import RAGShapExplainer
from explainability.counterfactual_explainer import CounterfactualExplainer


class UnifiedXAI:
    """Run both SHAP and Counterfactual XAI methods together"""
    
    def __init__(self, generator, retriever):
        """
        Initialize both explainers
        
        Args:
            generator: QAGenerator instance
            retriever: Retriever instance
        """
        self.shap_explainer = RAGShapExplainer(generator, retriever)
        self.cf_explainer = CounterfactualExplainer(generator, retriever)
    
    def explain_all(self, query, context, language="en", answer=None):
        """
        Run both SHAP and Counterfactual explanations
        
        Args:
            query: User query
            context: Retrieved context
            language: Language code
            answer: Generated answer (optional, for counterfactual context analysis)
            
        Returns:
            Dict with all XAI results
        """
        print("\n" + "="*70)
        print("UNIFIED XAI ANALYSIS (SHAP + COUNTERFACTUAL)")
        print("="*70)
        
        results = {}
        
        # 1. SHAP Analysis
        print("\n[1/2] Running SHAP Analysis...")
        print("  → Analyzing query token importance...")
        
        try:
            query_importance = self.shap_explainer.explain_query_importance(
                query, language, num_samples=100
            )
            
            # Check if fallback was used
            if query_importance and len(query_importance) > 0:
                results['shap_query'] = query_importance
                
                # Show top 5 tokens
                sorted_tokens = sorted(query_importance.items(), 
                                      key=lambda x: abs(x[1]), reverse=True)[:5]
                print("\n  Top 5 Query Tokens (SHAP):")
                for word, value in sorted_tokens:
                    print(f"    • '{word}': {value:+.3f}")
            else:
                print("  ⚠ SHAP returned no results, skipping...")
                results['shap_query'] = {}
        
        except Exception as e:
            print(f"  ✗ SHAP failed: {str(e)[:100]}")
            print("  → Using fallback analysis instead...")
            # Use simple fallback
            results['shap_query'] = self._simple_token_importance(query)
        
        # 2. Counterfactual Analysis
        print("\n[2/2] Running Counterfactual Analysis...")
        print("  → Testing word removal impact...")
        
        try:
            query_cf = self.cf_explainer.explain_query_words(
                query, language=language, top_k=5
            )
            results['counterfactual'] = query_cf
            
            # Show top 3 impactful words
            counterfactuals = query_cf.get('counterfactuals', [])
            if counterfactuals:
                print("\n  Top 3 Impactful Words (Counterfactual):")
                for cf in counterfactuals[:3]:
                    word = cf['removed_word']
                    impact = cf['impact']
                    interp = cf['interpretation'].split(' - ')[0]
                    print(f"    • Remove '{word}': {impact:+.3f} ({interp})")
        
        except Exception as e:
            print(f"  ✗ Counterfactual failed: {e}")
            results['counterfactual'] = {}
        
        print("\n" + "="*70)
        print("XAI ANALYSIS COMPLETE")
        print("="*70)
        
        return results
    

    
    def format_report(self, results):
        """
        Generate formatted text report
        
        Args:
            results: Results from explain_all()
            
        Returns:
            Formatted string report
        """
        lines = []
        lines.append("\n" + "="*70)
        lines.append("XAI EXPLANATION REPORT")
        lines.append("="*70)
        
        # SHAP Section
        lines.append("\n[SHAP ANALYSIS: Query Token Importance]")
        lines.append("-" * 70)
        
        shap_query = results.get('shap_query', {})
        if shap_query:
            sorted_tokens = sorted(shap_query.items(), 
                                  key=lambda x: abs(x[1]), reverse=True)[:10]
            
            lines.append("\nTop 10 Most Important Tokens:")
            for i, (word, value) in enumerate(sorted_tokens, 1):
                impact = "POSITIVE" if value > 0 else "NEGATIVE"
                lines.append(f"  {i:2d}. '{word:15s}' → {value:+.3f} ({impact})")
            
            lines.append("\nInterpretation:")
            lines.append("  • Positive values: Token increases retrieval relevance")
            lines.append("  • Negative values: Token decreases retrieval relevance")
        else:
            lines.append("  No SHAP results available")
        
        # Counterfactual Section
        lines.append("\n[COUNTERFACTUAL ANALYSIS: Word Removal Impact]")
        lines.append("-" * 70)
        
        cf_data = results.get('counterfactual', {})
        counterfactuals = cf_data.get('counterfactuals', [])
        baseline_score = cf_data.get('baseline_score', 0)
        
        if counterfactuals:
            lines.append(f"\nBaseline Score: {baseline_score:.3f}")
            lines.append("\nWhat happens if we remove each word?")
            
            for i, cf in enumerate(counterfactuals[:10], 1):
                word = cf['removed_word']
                cf_score = cf['counterfactual_score']
                impact = cf['impact']
                
                # Categorize impact
                if impact > 0.1:
                    category = "CRITICAL"
                elif impact > 0.05:
                    category = "IMPORTANT"
                elif impact > 0.01:
                    category = "HELPFUL"
                elif impact > -0.01:
                    category = "NEUTRAL"
                else:
                    category = "NOISE"
                
                lines.append(f"\n  {i:2d}. Remove '{word}':")
                lines.append(f"      Score: {baseline_score:.3f} → {cf_score:.3f} (Δ {impact:+.3f})")
                lines.append(f"      Impact: {category}")
            
            lines.append("\nInterpretation:")
            lines.append("  • CRITICAL (>0.1): Essential for retrieval")
            lines.append("  • IMPORTANT (>0.05): Significantly affects results")
            lines.append("  • HELPFUL (>0.01): Minor positive contribution")
            lines.append("  • NEUTRAL: Minimal effect")
            lines.append("  • NOISE (<-0.01): Removal improves results")
        else:
            lines.append("  No counterfactual results available")
        
        # Summary
        lines.append("\n" + "="*70)
        lines.append("SUMMARY")
        lines.append("="*70)
        
        if shap_query and counterfactuals:
            # Find agreement between SHAP and Counterfactual
            shap_top = set([w for w, _ in sorted(shap_query.items(), 
                           key=lambda x: abs(x[1]), reverse=True)[:5]])
            cf_top = set([cf['removed_word'] for cf in counterfactuals[:5]])
            agreement = shap_top & cf_top
            
            lines.append(f"\nBoth methods agree on these important words:")
            if agreement:
                for word in agreement:
                    shap_val = shap_query.get(word, 0)
                    cf_item = next((cf for cf in counterfactuals if cf['removed_word'] == word), None)
                    cf_impact = cf_item['impact'] if cf_item else 0
                    lines.append(f"  • '{word}': SHAP={shap_val:+.3f}, CF Impact={cf_impact:+.3f}")
            else:
                lines.append("  (No strong agreement - methods capture different aspects)")
        
        lines.append("\n" + "="*70)
        
        return "\n".join(lines)
    
    def _simple_token_importance(self, query):
        """
        Simple fallback token importance when SHAP fails
        Based on word length and position
        """
        import re
        words = re.findall(r'\w+', query.lower())
        
        # Filter content words (length > 3)
        content_words = [w for w in words if len(w) > 3]
        
        if not content_words:
            return {}
        
        # Score by length (longer words tend to be more important)
        importance = {}
        max_len = max(len(w) for w in content_words)
        
        for word in content_words:
            # Normalize by max length
            importance[word] = len(word) / max_len
        
        return importance
