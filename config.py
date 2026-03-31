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
CHUNK_SIZE = 600  # Increased from 400 for better context
CHUNK_OVERLAP = 100  # Increased from 50 for better continuity
MAX_DOCS_PER_LANGUAGE = 5000    #increase from 500

# ===== Paths =====
PROCESSED_DATA_PATH = "data/processed"
INDEX_PATH = "data/indexes"

FAISS_INDEX_PATH = "data/indexes/faiss.index"
METADATA_PATH = "data/indexes/metadata.pkl"

# ===== Models =====
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# ===== Retrieval =====
RETRIEVAL_THRESHOLD = 0.60  # Lower threshold to trigger fallback for irrelevant queries
HYBRID_DENSE_TOP_K = 10
HYBRID_SPARSE_TOP_K = 10
HYBRID_MERGED_TOP_K = 20
FINAL_TOP_K = 5
FAISS_INDEX_TYPE = "hnsw"
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 80
HNSW_EF_SEARCH = 64
REGENERATE_ON_LOW_FAITHFULNESS = True
FAITHFULNESS_RETRY_THRESHOLD = 0.45
MISTRAL_MODEL_PATH = "models\\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
