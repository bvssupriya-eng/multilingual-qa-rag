"""
Counterfactual Explainer for RAG System
Shows "what if" scenarios - how changes to query affect results
"""

import re
import numpy as np
import logging
from collections import defaultdict
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional


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
        self._orig_cache = {}  # cache for original answers

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
    
    def _get_content_words(self, text):
        """Extract content words (length > 3)"""
        words = re.findall(r'\w+', text.lower())
        return [w for w in words if len(w) > 3]
    
    def _format_context(self, docs):
        """Format retrieved documents into a single context string"""
        if not docs:
            return ""
        context_parts = []
        for i, doc in enumerate(docs[:5], 1):
            text = doc.get('text', '')
            if text:
                context_parts.append(f"[S{i}] {text}")
        return "\n".join(context_parts)
    
    def _get_original_answer(self, query, language, target_lang="en"):
        """Helper to get original answer with caching"""
        key = (query, language, target_lang)
        if key not in self._orig_cache:
            docs = self.retriever.search(query, language=language, top_k=5)
            context = self._format_context(docs)
            answer = self.generator.generate_answer(
                question=query,
                context=context,
                role="student",
                language="en"
            )
            if target_lang != "en":
                answer = self.retriever.translate(answer, "en", target_lang)
            self._orig_cache[key] = answer
        return self._orig_cache[key]
    
    # ==================== EXISTING METHODS (unchanged) ====================
    
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
        
        # Get baseline results (suppress logging)
        try:
            logging.disable(logging.CRITICAL)
            baseline_results = self.retriever.search(query, language=language, top_k=top_k)
            logging.disable(logging.NOTSET)

            baseline_score = self._result_score(baseline_results)
        except:
            logging.disable(logging.NOTSET)
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
                logging.disable(logging.CRITICAL)
                cf_results = self.retriever.search(cf_query, language=language, top_k=top_k)
                logging.disable(logging.NOTSET)

                cf_score = self._result_score(cf_results)
            except:
                logging.disable(logging.NOTSET)
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
    
    # ==================== NEW METHODS FOR UI INTERACTION ====================
    
    def explain_manual_whatif(self, original_query: str, modified_query: str,
                              language: str, target_lang: str = "en") -> Dict[str, Any]:
        """
        Compare original answer with answer for a manually modified query.
        Used for interactive "what-if" in the UI.
        
        Args:
            original_query: The original user query
            modified_query: User-edited query (what-if scenario)
            language: Query language (e.g., 'en', 'hi')
            target_lang: Language for answer generation
            
        Returns:
            Dictionary with original and counterfactual answers and comparison metrics
        """
        print(f"\n[MANUAL COUNTERFACTUAL] Original: '{original_query}' → Modified: '{modified_query}'")
        
        # --- Original answer ---
        orig_docs = self.retriever.search(original_query, language=language, top_k=5)
        orig_context = self._format_context(orig_docs)
        orig_answer = self.generator.generate_answer(
            question=original_query,
            context=orig_context,
            role="student",
            language="en"
        )
        
        # --- Modified answer ---
        mod_docs = self.retriever.search(modified_query, language=language, top_k=5)
        mod_context = self._format_context(mod_docs)
        mod_answer = self.generator.generate_answer(
            question=modified_query,
            context=mod_context,
            role="student",
            language="en"
        )

        if target_lang != "en":
            orig_answer = self.retriever.translate(orig_answer, "en", target_lang)
            mod_answer = self.retriever.translate(mod_answer, "en", target_lang)
        
        # Compute similarity and deltas
        answer_similarity = SequenceMatcher(None, orig_answer, mod_answer).ratio()
        retrieval_changed = False
        if orig_docs and mod_docs:
            # Check if top document changed significantly
            orig_first = orig_docs[0].get('text', '')[:100]
            mod_first = mod_docs[0].get('text', '')[:100]
            retrieval_changed = orig_first != mod_first
        
        result = {
            "original_query": original_query,
            "modified_query": modified_query,
            "original_answer": orig_answer,
            "modified_answer": mod_answer,
            "answer_similarity": answer_similarity,
            "retrieval_changed": retrieval_changed,
            "original_num_docs": len(orig_docs),
            "modified_num_docs": len(mod_docs)
        }
        
        print(f"  → Answer similarity: {answer_similarity:.2f}, Retrieval changed: {retrieval_changed}")
        return result
    
    def suggest_counterfactuals(self, query: str, language: str, target_lang: str = "en",
                                top_k_suggestions: int = 3) -> List[Dict[str, Any]]:
        """
        Automatically generate minimal query edits that would change the answer.
        Uses heuristic perturbations (entity replacement, negation, year flipping).
        
        Args:
            query: Original user query
            language: Query language (currently supports English primarily)
            target_lang: Language for answer generation
            top_k_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of counterfactual suggestions, each with modified query and new answer
        """
        print(f"\n[AUTO COUNTERFACTUAL] Generating suggestions for: '{query}'")
        suggestions = []
        tokens = query.split()
        
        # Helper to generate answer for a modified query
        def get_answer_for_query(q):
            docs = self.retriever.search(q, language=language, top_k=5)
            if not docs:
                return None
            context = self._format_context(docs)
            return self.generator.generate_answer(
                question=q,
                context=context,
                role="student",
                language=target_lang
            )
        
        # Get original answer for comparison
        orig_answer = self._get_original_answer(query, language, target_lang)
        
        # --- 1. Entity replacement: replace key words with generic alternatives ---
        for i, word in enumerate(tokens):
            # Skip very short words or common stopwords
            if len(word) < 4 or word.lower() in {"what", "who", "is", "are", "the", "and", "of", "to"}:
                continue
            
            # Try different replacements
            replacements = [f"someone else", f"another", f"not {word}"]
            for repl in replacements:
                new_tokens = tokens.copy()
                new_tokens[i] = repl
                new_query = " ".join(new_tokens)
                # Avoid trivial changes (high similarity)
                if SequenceMatcher(None, query, new_query).ratio() > 0.85:
                    continue
                
                new_answer = get_answer_for_query(new_query)
                if new_answer and SequenceMatcher(None, orig_answer, new_answer).ratio() < 0.7:
                    suggestions.append({
                        "type": "entity_replacement",
                        "original_query": query,
                        "modified_query": new_query,
                        "modified_answer": new_answer,
                        "change_description": f"Replaced '{word}' with '{repl}'"
                    })
                if len(suggestions) >= top_k_suggestions:
                    return suggestions
        
        # --- 2. Negation insertion ---
        negation_words = ["is", "are", "was", "were", "does", "do", "has", "have"]
        for i, word in enumerate(tokens):
            if word.lower() in negation_words:
                new_tokens = tokens[:i+1] + ["not"] + tokens[i+1:]
                new_query = " ".join(new_tokens)
                # Avoid double negation
                if "not not" in new_query:
                    continue
                new_answer = get_answer_for_query(new_query)
                if new_answer and SequenceMatcher(None, orig_answer, new_answer).ratio() < 0.7:
                    suggestions.append({
                        "type": "negation",
                        "original_query": query,
                        "modified_query": new_query,
                        "modified_answer": new_answer,
                        "change_description": f"Added 'not' after '{word}'"
                    })
                if len(suggestions) >= top_k_suggestions:
                    return suggestions
        
        # --- 3. Year flipping (if a year is present) ---
        year_pattern = r'\b(?:19|20)\d{2}\b'
        years = re.findall(year_pattern, query)
        if years:
            for year_match in years:
                year = year_match  # e.g., '1999'
                try:
                    year_int = int(year)
                    # Flip to a different decade
                    if year_int < 2000:
                        flipped = str(year_int + 100)
                    else:
                        flipped = str(year_int - 100)
                    new_query = re.sub(r'\b' + year + r'\b', flipped, query)
                    new_answer = get_answer_for_query(new_query)
                    if new_answer and SequenceMatcher(None, orig_answer, new_answer).ratio() < 0.7:
                        suggestions.append({
                            "type": "year_flip",
                            "original_query": query,
                            "modified_query": new_query,
                            "modified_answer": new_answer,
                            "change_description": f"Changed year {year} → {flipped}"
                        })
                except:
                    pass
                if len(suggestions) >= top_k_suggestions:
                    return suggestions
        
        # If no strong counterfactuals found, return empty list
        print(f"  → Generated {len(suggestions)} suggestions")
        return suggestions
