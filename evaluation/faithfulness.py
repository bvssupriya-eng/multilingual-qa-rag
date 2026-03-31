import re


def _tokenize(text):
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def compute_faithfulness(answer, contexts, retrieval_score=None, generation_score=None):
    answer_tokens = _tokenize(answer)
    context_tokens = set()

    for context in contexts:
        context_tokens.update(_tokenize(context))

    if not answer_tokens:
        return {
            "faithfulness_score": 0.0,
            "confidence": "low",
            "supported": False
        }

    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    
    # If retrieval and generation scores provided, use weighted confidence
    if retrieval_score is not None and generation_score is not None:
        combined_score = 0.5 * retrieval_score + 0.5 * generation_score
        
        if combined_score >= 0.75:
            confidence = "high"
        elif combined_score >= 0.50:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        # Fallback to overlap-based confidence
        if overlap >= 0.75:
            confidence = "high"
        elif overlap >= 0.45:
            confidence = "medium"
        else:
            confidence = "low"

    return {
        "faithfulness_score": overlap,
        "confidence": confidence,
        "supported": overlap >= 0.45
    }
