import re


COMMON_CODE_MIX_MAP = {
    "kya": "what",
    "ka": "",
    "ki": "",
    "ke": "",
    "hai": "is",
    "kaun": "who",
    "kahan": "where",
    "kab": "when",
    "kyun": "why",
    "paisa": "money",
    "bazaar": "market",
    "market": "market",
    "trend": "trend",
    "stock": "stock",
    "sarkar": "government",
    "bharat": "india",
    "bharatiya": "indian",
    "cricket": "cricket",
    "kitna": "how much",
    "kaise": "how",
    "aur": "and"
}


def detect_code_mix(text, language="en"):
    has_latin = bool(re.search(r"[a-zA-Z]", text))
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    has_bengali = bool(re.search(r"[\u0980-\u09FF]", text))
    has_arabic = bool(re.search(r"[\u0600-\u06FF]", text))
    if has_latin and (has_devanagari or has_bengali or has_arabic):
        return True
    if language in {"hi", "bn", "ar"} and has_latin:
        return True
    return False


def normalize_code_mixed_query(query, language):
    cleaned = re.sub(r"\s+", " ", query.strip())
    lowered = cleaned.lower()

    transliterated_hint = lowered
    if language in {"hi", "bn"}:
        words = []
        for token in re.findall(r"\w+|\S", lowered, flags=re.UNICODE):
            replacement = COMMON_CODE_MIX_MAP.get(token, token)
            if replacement:
                words.append(replacement)
        transliterated_hint = " ".join(words)

    query_variants = [cleaned]

    if transliterated_hint and transliterated_hint != lowered:
        query_variants.append(transliterated_hint)

    if detect_code_mix(cleaned, language):
        query_variants.append(lowered)

    deduped = []
    seen = set()
    for variant in query_variants:
        normalized = variant.strip()
        if normalized and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)

    return {
        "original_query": cleaned,
        "normalized_query": deduped[0] if deduped else cleaned,
        "query_variants": deduped or [cleaned],
        "code_mixed": detect_code_mix(cleaned, language)
    }
