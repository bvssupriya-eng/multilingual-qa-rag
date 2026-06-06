"""
Faithfulness computation for RAG QA answers.
Improvements over v1:
  - Stopword-filtered token overlap (no more inflation from "the", "is", etc.)
  - Bigram overlap added for phrase-level grounding (combined: 60% unigram + 40% bigram)
  - Consistent confidence: derived from token overlap, not retrieval+generation scores
  - Explicit 'overlap_score' field for transparency
"""

import re
from config import FAITHFULNESS_STOPWORDS


def _tokenize(text: str) -> list:
    """Tokenize to lowercase words, filtering stopwords and short tokens."""
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return [t for t in tokens if len(t) > 2 and t not in FAITHFULNESS_STOPWORDS]


def _bigrams(tokens: list) -> set:
    """Generate bigrams from a token list."""
    return {f"{a}_{b}" for a, b in zip(tokens, tokens[1:])}


def compute_faithfulness(answer: str, contexts: list, retrieval_score=None, generation_score=None) -> dict:
    """
    Compute how faithfully the answer is grounded in the retrieved contexts.

    Args:
        answer: Generated answer text.
        contexts: List of source text strings (retrieved chunks).
        retrieval_score: Unused (kept for backward-compat signature). Deprecated.
        generation_score: Unused (kept for backward-compat signature). Deprecated.

    Returns:
        dict with keys:
            faithfulness_score  – combined unigram+bigram overlap (0–1)
            unigram_overlap     – raw token-level overlap (0–1)
            bigram_overlap      – phrase-level overlap (0–1)
            confidence          – "high" / "medium" / "low"
            supported           – bool, True if faithfulness_score >= 0.40
    """
    answer_tokens = _tokenize(answer)

    if not answer_tokens:
        return {
            "faithfulness_score": 0.0,
            "unigram_overlap": 0.0,
            "bigram_overlap": 0.0,
            "confidence": "low",
            "supported": False,
        }

    # Aggregate all context tokens
    context_tokens: list = []
    for ctx in contexts:
        context_tokens.extend(_tokenize(ctx))
    context_token_set = set(context_tokens)

    # ── Unigram overlap ──────────────────────────────────────────────────────
    answer_token_set = set(answer_tokens)
    if answer_token_set:
        unigram_overlap = len(answer_token_set & context_token_set) / len(answer_token_set)
    else:
        unigram_overlap = 0.0

    # ── Bigram overlap ───────────────────────────────────────────────────────
    answer_bigrams = _bigrams(answer_tokens)
    context_bigrams = _bigrams(context_tokens)
    if answer_bigrams:
        bigram_overlap = len(answer_bigrams & context_bigrams) / len(answer_bigrams)
    else:
        bigram_overlap = 0.0

    # ── Combined faithfulness score ──────────────────────────────────────────
    faithfulness_score = 0.6 * unigram_overlap + 0.4 * bigram_overlap

    # ── Confidence: derived consistently from faithfulness_score ─────────────
    if faithfulness_score >= 0.60:
        confidence = "high"
    elif faithfulness_score >= 0.40:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "faithfulness_score": round(faithfulness_score, 4),
        "unigram_overlap": round(unigram_overlap, 4),
        "bigram_overlap": round(bigram_overlap, 4),
        "confidence": confidence,
        "supported": faithfulness_score >= 0.40,
    }
