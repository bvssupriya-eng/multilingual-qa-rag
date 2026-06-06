
# ===== Languages =====
LANGUAGES = ["en", "hi", "bn", "ar"]

# Language code mappings (single source of truth)
LANG_CODES = {
    "en": {"iso": "en", "nllb": "eng_Latn", "name": "English", "wiki": "en"},
    "hi": {"iso": "hi", "nllb": "hin_Deva", "name": "Hindi", "wiki": "hi"},
    "bn": {"iso": "bn", "nllb": "ben_Beng", "name": "Bengali", "wiki": "bn"},
    "ar": {"iso": "ar", "nllb": "arb_Arab", "name": "Arabic", "wiki": "ar"}
}

# ===== Chunking =====
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
MAX_DOCS_PER_LANGUAGE = 5000

# ===== Paths =====
PROCESSED_DATA_PATH = "data/processed"
INDEX_PATH = "data/indexes"

FAISS_INDEX_PATH = "data/indexes/faiss.index"
METADATA_PATH = "data/indexes/metadata.pkl"

# ===== Models =====
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# ===== Retrieval =====
RETRIEVAL_THRESHOLD = 0.60
HYBRID_DENSE_TOP_K = 10
HYBRID_SPARSE_TOP_K = 10
HYBRID_MERGED_TOP_K = 20
FINAL_TOP_K = 5
FAISS_INDEX_TYPE = "hnsw"
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 80
HNSW_EF_SEARCH = 64

# ===== Relevance Checking =====
# Title-level cosine similarity below this → consider irrelevant
# Raised from 0.30 to 0.38: avoid tangentially related docs
# (e.g. 'Analysis of variance' passing for 'AI vs ML' queries)
RELEVANCE_TITLE_THRESHOLD = 0.38
# Content-level cosine similarity below this (avg top-3 snippets) → consider irrelevant
# Both title AND content must fail to trigger fallback
# Raised from 0.25 to 0.32 for the same reason
RELEVANCE_CONTENT_THRESHOLD = 0.32

# ===== MMR Diversity =====
# 0.5 = balanced relevance/diversity; lower = more diverse results
MMR_LAMBDA = 0.5

# ===== Query Expansion =====
QUERY_EXPANSION_ENABLED = True

# ===== Faithfulness & Regeneration =====
REGENERATE_ON_LOW_FAITHFULNESS = True
FAITHFULNESS_RETRY_THRESHOLD = 0.45
FAITHFULNESS_MAX_RETRIES = 2  # Maximum regeneration attempts

# Stop-words filtered out before faithfulness token overlap computation
FAITHFULNESS_STOPWORDS = {
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "from", "into", "through", "during", "before", "after", "above",
    "below", "between", "each", "and", "but", "or", "nor", "so", "yet",
    "both", "either", "not", "only", "own", "same", "than", "too", "very",
    "just", "as", "if", "then", "that", "this", "these", "those", "it",
    "its", "also", "such", "even", "more", "most", "other", "some", "all",
    # Hindi common function words (romanized)
    "hai", "hain", "ka", "ki", "ke", "mein", "par", "aur", "se", "ko",
    "ne", "jo", "kya", "yah", "wah", "ek", "iske",
    # Bengali function words (romanized)
    "ei", "tai", "aar", "theke", "ebong", "je", "hoy", "ache", "hobe",
    # Arabic function words (romanized)
    "al", "wa", "fi", "min", "ila", "an", "ala", "ma", "li", "hatha"
}

# ===== Metrics Stopwords =====
# Filtered in answer_relevance and context_utilization
METRICS_STOPWORDS = {
    "this", "that", "with", "from", "have", "will", "what", "when",
    "where", "which", "there", "their", "about", "would", "could",
    "should", "these", "those", "then", "than", "they", "them",
    "also", "been", "were", "have", "just", "over", "such", "very",
    "into", "your", "more", "some", "like", "only", "both", "each",
    "most", "other", "even", "through", "during", "before", "after"
}

# ===== Model =====
MISTRAL_MODEL_PATH = "models\\qwen2.5-1.5b-instruct-q4_k_m.gguf"

# ===== Generation Performance =====
# Optimized for CPU inference speed while maintaining quality
GENERATION_N_CTX = 1536          # Smaller runtime context for faster CPU inference
GENERATION_N_THREADS = 8          # Optimal for most CPUs (was 16 - over-threading causes slowdown)
GENERATION_N_BATCH = 256          # Conservative batch size for CPU compatibility
GENERATION_MAX_TOKENS = 140       # Balanced answer length/speed target
GENERATION_TEMPERATURE = 0.2      # Lower = faster sampling, more deterministic
GENERATION_TOP_P = 0.85           # Reduced from 0.9 for faster token selection
GENERATION_REPEAT_PENALTY = 1.1   # Reduced from 1.2 for speed
GENERATION_CONTEXT_LIMIT = 1200   # Smaller prompt for faster generation while keeping evidence
