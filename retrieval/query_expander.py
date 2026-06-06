"""
Multilingual Query Expander
Expands queries with synonyms, acronyms, and related terms before retrieval.
Zero external dependencies — uses static dictionaries for all 4 supported languages.
"""

# ──────────────────────────────────────────────────────────
# ENGLISH: Acronym expansion + common synonyms
# ──────────────────────────────────────────────────────────
EN_EXPANSIONS = {
    # Technology acronyms
    "ai":    "artificial intelligence",
    "ml":    "machine learning",
    "dl":    "deep learning",
    "nlp":   "natural language processing",
    "cv":    "computer vision",
    "rl":    "reinforcement learning",
    "llm":   "large language model",
    "gpt":   "generative pre-trained transformer",
    "nn":    "neural network",
    "cnn":   "convolutional neural network",
    "rnn":   "recurrent neural network",
    "gan":   "generative adversarial network",
    "rag":   "retrieval augmented generation",
    "gpu":   "graphics processing unit",
    "cpu":   "central processing unit",
    "api":   "application programming interface",
    "db":    "database",
    "sql":   "structured query language",
    "http":  "hypertext transfer protocol",
    "html":  "hypertext markup language",
    "os":    "operating system",
    "ui":    "user interface",
    "ux":    "user experience",
    "iot":   "internet of things",
    "ar":    "augmented reality",
    "vr":    "virtual reality",
    "xr":    "extended reality",
    # Science
    "dna":   "deoxyribonucleic acid",
    "rna":   "ribonucleic acid",
    "ai":    "artificial intelligence",
    "qm":    "quantum mechanics",
    "qc":    "quantum computing",
    # Geography/Countries
    "usa":   "united states of america",
    "uk":    "united kingdom",
    "uae":   "united arab emirates",
    "eu":    "european union",
    "un":    "united nations",
    # Common synonyms (expands query coverage)
    "explain":   "describe what is",
    "define":    "what is definition of",
    "difference":"compare contrast between",
    "vs":        "versus compared to difference between",
    "advantages":"benefits pros",
    "history":   "origin evolution background",
    "founder":   "creator inventor who founded",
    "capital":   "capital city seat of government",
}

# ──────────────────────────────────────────────────────────
# HINDI: Romanized common terms → English equivalents
# ──────────────────────────────────────────────────────────
HI_EXPANSIONS = {
    # Native-script question/entity anchors
    "क्या":       "क्या what",
    "कौन":       "कौन who",
    "कहाँ":      "कहाँ where",
    "कब":        "कब when",
    "क्यों":      "क्यों why",
    "कैसे":       "कैसे how",
    "भारत":      "भारत इंडिया",
    "राजधानी":   "राजधानी नई दिल्ली दिल्ली",
    "राष्ट्रीय":  "राष्ट्रीय",
    "फूल":       "फूल",
    # Question words
    "kya":       "what",
    "kaun":      "who",
    "kahan":     "where",
    "kab":       "when",
    "kyun":      "why",
    "kaise":     "how",
    "kitna":     "how much how many",
    # Common nouns
    "bharat":    "india",
    "bharatiya": "indian",
    "desh":      "country nation",
    "sarkar":    "government",
    "rajdhani":  "capital city",
    "itihaas":   "history",
    "vigyan":    "science",
    "ganit":     "mathematics",
    "bhasha":    "language",
    "shiksha":   "education",
    "swasthya":  "health",
    "arthvyavastha": "economy",
    "rajniti":   "politics",
    "samaj":     "society",
    "sanskriti": "culture",
    "kala":      "art",
    "sahitya":   "literature",
    "khel":      "sport game",
    "cricket":   "cricket sport",
    "vishwa":    "world global",
    "manav":     "human",
    "parivaar":  "family",
    "paryavaran":"environment",
    "takneek":   "technology",
    "janvar":    "animal",
    "prani":     "living being organism",
    "paani":     "water",
    "vayu":      "air atmosphere",
    "agni":      "fire",
    "prithvi":   "earth",
    # Technology
    "ganaka":    "computer",
    "antarjal":  "internet",
    "soochna":   "information",
    "vigyan":    "science",
    "yantra":    "machine device",
}

# ──────────────────────────────────────────────────────────
# BENGALI: Romanized common terms → English equivalents
# ──────────────────────────────────────────────────────────
BN_EXPANSIONS = {
    # Native-script question/entity anchors
    "কী":         "কী কি what",
    "কি":         "কি what",
    "কে":         "কে who",
    "কোথায়":     "কোথায় where",
    "কবে":        "কবে when",
    "কেন":        "কেন why",
    "বাংলাদেশ":   "বাংলাদেশ",
    "বাংলাদেশের": "বাংলাদেশ জাতীয় ফুল শাপলা",
    "জাতীয়":     "জাতীয়",
    "ফুল":        "ফুল শাপলা",
    # Question words
    "ki":        "what",
    "ke":        "who",
    "kothay":    "where",
    "kobe":      "when",
    "keno":      "why",
    "kivabe":    "how",
    "koto":      "how much how many",
    # Common nouns
    "bangladesh": "bangladesh country",
    "bharat":    "india",
    "desh":      "country nation",
    "sarkar":    "government",
    "rajdhani":  "capital city",
    "itihas":    "history",
    "bigyan":    "science",
    "gonit":     "mathematics",
    "bhasha":    "language",
    "shikkha":   "education",
    "swasthya":  "health",
    "orthonoiti":"economy",
    "rajnithi":  "politics",
    "shomaj":    "society",
    "shongskriti":"culture",
    "shilpo":    "art industry",
    "sahitto":   "literature",
    "khela":     "sport game",
    "bishwo":    "world global",
    "manush":    "human person",
    "poribesh":  "environment",
    "projukti":  "technology",
    "prani":     "animal living being",
    "pani":      "water",
    "batas":     "air",
    "prithibi":  "earth",
    # Technology
    "computer":  "computer computing",
    "internet":  "internet network",
    "tottho":    "information data",
    "jantra":    "machine device",
}

# ──────────────────────────────────────────────────────────
# ARABIC: Romanized common terms → English equivalents
# ──────────────────────────────────────────────────────────
AR_EXPANSIONS = {
    # Native-script question/entity anchors
    "ما":         "ما ماذا",
    "من":         "من",
    "أين":        "أين",
    "متى":        "متى",
    "لماذا":      "لماذا",
    "كيف":        "كيف",
    "الطاقة":     "الطاقة",
    "الشمسية":    "الشمسية الشمس",
    # Question words
    "ma":        "what",
    "man":       "who",
    "ayna":      "where",
    "mata":      "when",
    "limatha":   "why",
    "kayfa":     "how",
    "kam":       "how much how many",
    # Common nouns
    "misr":      "egypt",
    "saudi":     "saudi arabia",
    "arabia":    "arab region",
    "dawla":     "state country",
    "hukuma":    "government",
    "asima":     "capital city",
    "tarikh":    "history",
    "ilm":       "science knowledge",
    "riyadiaat": "mathematics",
    "lugha":     "language",
    "talim":     "education",
    "sihha":     "health",
    "iqtisad":   "economy",
    "siyasa":    "politics",
    "mujtama":   "society",
    "thaqafa":   "culture",
    "fann":      "art",
    "adab":      "literature",
    "riyada":    "sport",
    "alam":      "world global",
    "insan":     "human person",
    "biya":      "environment",
    "taqniya":   "technology",
    "hayawan":   "animal",
    "maa":       "water",
    "hawa":      "air atmosphere",
    "ard":       "earth land",
    # Technology
    "hasib":     "computer",
    "internet":  "internet network",
    "maluma":    "information data",
    "ala":       "machine device",
    "dhakaa":    "artificial intelligence",
    "taallum":   "machine learning",
}

# Map language codes to expansion dictionaries
_EXPANSION_MAP = {
    "en": EN_EXPANSIONS,
    "hi": HI_EXPANSIONS,
    "bn": BN_EXPANSIONS,
    "ar": AR_EXPANSIONS,
}


def expand_query(query: str, language: str = "en") -> list:
    """
    Expand a query using language-specific synonym/acronym dictionaries.

    Args:
        query: The original user query string
        language: ISO language code ('en', 'hi', 'bn', 'ar')

    Returns:
        List of query variants: [original, expanded_version] (deduplicated)
        If no expansion found, returns [original]
    """
    expansions = _EXPANSION_MAP.get(language, EN_EXPANSIONS)
    tokens = query.strip().split()
    expanded_tokens = []
    changed = False

    for token in tokens:
        clean = token.lower().rstrip("?.,!;:؟।॥")
        if clean in expansions:
            replacement = expansions[clean]
            expanded_tokens.append(replacement)
            changed = True
        else:
            expanded_tokens.append(token)

    if not changed:
        return [query]

    expanded_query = " ".join(expanded_tokens)

    # Return both original and expanded as variants
    variants = [query]
    if expanded_query.strip() and expanded_query != query:
        variants.append(expanded_query)

    return variants


def get_expansion_info(query: str, language: str = "en") -> dict:
    """
    Returns a dict showing which tokens were expanded and to what.
    Useful for debugging and XAI display.
    """
    expansions = _EXPANSION_MAP.get(language, EN_EXPANSIONS)
    tokens = query.strip().split()
    mapping = {}

    for token in tokens:
        clean = token.lower().rstrip("?.,!;:؟।॥")
        if clean in expansions:
            mapping[token] = expansions[clean]

    return mapping
