"""
Counterfactual Explainer for RAG System
Shows "what if" scenarios - how changes to query affect results
"""

import re
import numpy as np
from collections import defaultdict


class CounterfactualExplainer:
    """
    Counterfactual explanations for RAG system
    Shows how query modifications affect retrieval and answers
    """
    
    def __init__(self, generator, retriever):
        """
        Args:
            generator: QAGenerator instance
            retriever: Retriever instance
        """
        self.generator = generator
        self.retriever = retriever
    
    def _get_content_words(self, text):
        """Extract content words (length > 3)"""
        words = re.findall(r'\w+', text.lower())
        return [w for w in words if len(w) > 3]
    
    def explain_query_words(self, query, language="en", top_k=3):
        """
        Generate counterfactual explanations by removing each word
        Shows impact of each word on retrieval
        
        Args:
            query: User query
            language: Query language
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with counterfactual explanations
        """
        print(f"\n[COUNTERFACTUAL ANALYSIS: Testing query word importance...]")
        
        # Get baseline results
        try:
            baseline_results = self.retriever.search(query, language=language, top_k=top_k)
            baseline_score = baseline_results[0].get('hybrid_score', 0.0) if baseline_results else 0.0
            baseline_score = baseline_score if baseline_score is not None else 0.0
        except:
            baseline_score = 0.0
            baseline_results = []
        
        words = query.split()
        content_words = [w for w in words if len(w) > 3]
        
        if len(content_words) < 2:
            return {
                "baseline_score": baseline_score,
                "counterfactuals": [],
                "summary": "Query too short for counterfactual analysis"
            }
        
        counterfactuals = []
        
        # Test removing each content word
        for word in content_words:
            # Create counterfactual query (remove this word)
            cf_query = " ".join([w for w in words if w.lower() != word.lower()])
            
            if len(cf_query.strip()) < 3:
                continue
            
            try:
                cf_results = self.retriever.search(cf_query, language=language, top_k=top_k)
                cf_score = cf_results[0].get('hybrid_score', 0.0) if cf_results else 0.0
                cf_score = cf_score if cf_score is not None else 0.0
            except:
                cf_score = 0.0
            
            # Calculate impact
            impact = baseline_score - cf_score
            
            counterfactuals.append({
                "removed_word": word,
                "counterfactual_query": cf_query,
                "baseline_score": baseline_score,
                "counterfactual_score": cf_score,
                "impact": impact,
                "interpretation": self._interpret_impact(impact, word)
            })
        
        # Sort by absolute impact
        counterfactuals.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        print(f"  → Analysis complete")
        
        return {
            "baseline_score": baseline_score,
            "counterfactuals": counterfactuals,
            "summary": self._generate_summary(counterfactuals)
        }
    
    def explain_context_usage(self, question, context, answer):
        """
        Show how different parts of context contribute to answer
        
        Args:
            question: User question
            context: Retrieved context
            answer: Generated answer
            
        Returns:
            Dictionary with context counterfactuals
        """
        print(f"\n[COUNTERFACTUAL ANALYSIS: Testing context importance...]")
        
        # Split context into sentences
        sentences = re.split(r'(?<=[.!?])\s+', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) < 2:
            return {
                "counterfactuals": [],
                "summary": "Context too short for analysis"
            }
        
        # Limit to first 5 sentences for speed
        sentences = sentences[:5]
        
        # Get baseline answer length and content
        baseline_length = len(answer)
        baseline_words = set(self._get_content_words(answer))
        
        counterfactuals = []
        
        # Test removing each sentence
        for i, sentence in enumerate(sentences):
            # Create counterfactual context (remove this sentence)
            cf_context = " ".join([s for j, s in enumerate(sentences) if j != i])
            
            if len(cf_context.strip()) < 50:
                continue
            
            try:
                # Generate answer with modified context
                cf_answer = self.generator.generate_answer(
                    question=question,
                    context=cf_context,
                    role="student",
                    language="en"
                )
                
                cf_length = len(cf_answer)
                cf_words = set(self._get_content_words(cf_answer))
                
                # Calculate impact
                length_change = baseline_length - cf_length
                word_overlap = len(baseline_words & cf_words) / len(baseline_words) if baseline_words else 0
                
                counterfactuals.append({
                    "removed_sentence": sentence[:80] + "..." if len(sentence) > 80 else sentence,
                    "length_change": length_change,
                    "word_overlap": word_overlap,
                    "impact_score": abs(length_change) / baseline_length if baseline_length > 0 else 0,
                    "interpretation": self._interpret_context_impact(length_change, word_overlap)
                })
                
            except Exception as e:
                continue
        
        # Sort by impact
        counterfactuals.sort(key=lambda x: x['impact_score'], reverse=True)
        
        print(f"  → Analysis complete")
        
        return {
            "counterfactuals": counterfactuals[:3],  # Top 3 most impactful
            "summary": self._generate_context_summary(counterfactuals[:3])
        }
    
    def _interpret_impact(self, impact, word):
        """Generate human-readable interpretation of impact"""
        if impact > 0.1:
            return f"'{word}' is CRITICAL - removing it drops retrieval score significantly"
        elif impact > 0.05:
            return f"'{word}' is IMPORTANT - removing it reduces retrieval quality"
        elif impact > 0.01:
            return f"'{word}' is HELPFUL - removing it slightly affects retrieval"
        elif impact > -0.01:
            return f"'{word}' is NEUTRAL - removing it has minimal effect"
        else:
            return f"'{word}' is NOISE - removing it actually improves retrieval"
    
    def _interpret_context_impact(self, length_change, word_overlap):
        """Interpret context removal impact"""
        if length_change > 50 or word_overlap < 0.5:
            return "CRITICAL - This sentence is essential for the answer"
        elif length_change > 20 or word_overlap < 0.7:
            return "IMPORTANT - This sentence contributes significantly"
        elif length_change > 0:
            return "HELPFUL - This sentence adds some information"
        else:
            return "MINIMAL - This sentence has little impact"
    
    def _generate_summary(self, counterfactuals):
        """Generate summary of query counterfactuals"""
        if not counterfactuals:
            return "No counterfactuals generated"
        
        top_3 = counterfactuals[:3]
        summary_lines = ["Top impactful words:"]
        
        for cf in top_3:
            summary_lines.append(
                f"  • {cf['removed_word']}: {cf['interpretation']}"
            )
        
        return "\n".join(summary_lines)
    
    def _generate_context_summary(self, counterfactuals):
        """Generate summary of context counterfactuals"""
        if not counterfactuals:
            return "No counterfactuals generated"
        
        summary_lines = ["Most impactful context sentences:"]
        
        for i, cf in enumerate(counterfactuals, 1):
            summary_lines.append(
                f"  {i}. {cf['interpretation']}"
            )
        
        return "\n".join(summary_lines)
    
    def format_output(self, query_cf, context_cf=None):
        """
        Format counterfactual explanations for display
        
        Args:
            query_cf: Query counterfactuals
            context_cf: Context counterfactuals (optional)
            
        Returns:
            Formatted string
        """
        output = []
        
        output.append("\n--- Counterfactual: Query Word Impact ---")
        output.append(f"Baseline retrieval score: {query_cf['baseline_score']:.3f}")
        output.append("\nWhat if we remove each word?")
        
        for cf in query_cf['counterfactuals'][:5]:
            output.append(f"\n  Remove '{cf['removed_word']}':")
            output.append(f"    New score: {cf['counterfactual_score']:.3f} (impact: {cf['impact']:+.3f})")
            output.append(f"    → {cf['interpretation']}")
        
        if context_cf and context_cf['counterfactuals']:
            output.append("\n--- Counterfactual: Context Sentence Impact ---")
            output.append("\nWhat if we remove each sentence?")
            
            for i, cf in enumerate(context_cf['counterfactuals'], 1):
                output.append(f"\n  Sentence {i}: {cf['removed_sentence']}")
                output.append(f"    → {cf['interpretation']}")
        
        return "\n".join(output)
