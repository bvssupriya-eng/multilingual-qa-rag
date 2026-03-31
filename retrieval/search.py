import os
import faiss
import pickle
import numpy as np
import requests
import logging

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_THRESHOLD,
    HYBRID_DENSE_TOP_K,
    HYBRID_SPARSE_TOP_K,
    HYBRID_MERGED_TOP_K,
    FINAL_TOP_K,
    TRANSLATION_MODEL_NAME,
    HNSW_EF_SEARCH,
    LANG_CODES
)
from retrieval.bm25_index import BM25Index
from retrieval.query_normalizer import normalize_code_mixed_query

# Set environment variables to suppress warnings and enable offline mode
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Suppress sentence-transformers progress bars
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Retriever:
    def __init__(self):
        logging.info("Initializing Retriever...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = HNSW_EF_SEARCH

        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)

        logging.info("Loading embedding model...")
        try:
            self.model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                local_files_only=True
            )
            # Disable progress bars for encoding
            self.model.encode = self._wrap_encode(self.model.encode)
        except Exception as e:
            logging.debug(f"Trying without local_files_only flag...")
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            self.model.encode = self._wrap_encode(self.model.encode)

        logging.info("Building BM25 index...")
        self.bm25 = BM25Index([item.get("text", "") for item in self.metadata])

        self.translator_tokenizer = None
        self.translator_model = None
        
        # Use centralized language codes from config
        self.lang_code_map = {lang: LANG_CODES[lang]["nllb"] for lang in LANG_CODES}
    
    def _wrap_encode(self, original_encode):
        """Wrapper to disable progress bars in sentence-transformers encode"""
        def encode_wrapper(*args, **kwargs):
            kwargs['show_progress_bar'] = False
            return original_encode(*args, **kwargs)
        return encode_wrapper

    # -------------------------------------------------
    # Translation fallback
    # -------------------------------------------------
    def _ensure_translation_model(self):
        if self.translator_tokenizer and self.translator_model:
            return True

        try:
            print("Loading translation model for fallback...")
            # Suppress warnings during model loading
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning)
            
            try:
                self.translator_tokenizer = AutoTokenizer.from_pretrained(
                    TRANSLATION_MODEL_NAME,
                    local_files_only=True
                )
                self.translator_model = AutoModelForSeq2SeqLM.from_pretrained(
                    TRANSLATION_MODEL_NAME,
                    local_files_only=True
                )
            except Exception:
                # Fallback without local_files_only
                self.translator_tokenizer = AutoTokenizer.from_pretrained(
                    TRANSLATION_MODEL_NAME
                )
                self.translator_model = AutoModelForSeq2SeqLM.from_pretrained(
                    TRANSLATION_MODEL_NAME
                )
            return True
        except Exception as exc:
            print("Translation model unavailable, continuing without it:", exc)
            return False

    def translate(self, text, src_lang, tgt_lang):
        if src_lang == tgt_lang:
            return text

        if src_lang not in self.lang_code_map or tgt_lang not in self.lang_code_map:
            return text

        if not self._ensure_translation_model():
            return text

        tokenizer = self.translator_tokenizer
        model = self.translator_model
        tokenizer.src_lang = self.lang_code_map[src_lang]

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        forced_bos_token_id = tokenizer.convert_tokens_to_ids(self.lang_code_map[tgt_lang])
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512
        )

        return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    # -------------------------------------------------
    # Wikipedia fallback (multilingual)
    # -------------------------------------------------
    def fetch_wikipedia_summary(self, query, language="en"):
        """Fetch Wikipedia summary in the requested language (requires internet)"""
        # Use centralized language codes
        wiki_lang = LANG_CODES.get(language, {}).get("wiki", "en")
        
        headers = {
            "User-Agent": "MultilingualQAAssistant/1.0 (research project)"
        }

        try:
            # Preprocess query: remove question words for better Wikipedia search
            clean_query = query
            question_words = ['who is', 'what is', 'where is', 'when is', 'why is', 'how is',
                            'who are', 'what are', 'where are', 'when are', 'why are', 'how are',
                            'tell me about', 'explain', 'describe']
            for qw in question_words:
                if clean_query.lower().startswith(qw):
                    clean_query = clean_query[len(qw):].strip()
                    break
            # Remove trailing question marks
            clean_query = clean_query.rstrip('?').strip()
            
            # Step 1: Search for the best matching article
            search_url = f"https://{wiki_lang}.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": clean_query,
                "format": "json",
                "srlimit": 5  # Get top 5 results for better matching
            }
            
            search_response = requests.get(search_url, params=search_params, headers=headers, timeout=5)
            
            if search_response.status_code != 200:
                raise Exception(f"Search failed with status {search_response.status_code}")
            
            search_data = search_response.json()
            
            if not search_data.get("query") or not search_data["query"]["search"]:
                raise Exception("No search results found")
            
            # Get the best matching article title using semantic similarity
            search_results = search_data["query"]["search"]
            best_title = None
            best_match_score = -1
            
            # Compute query embedding once
            query_embedding = self.model.encode([clean_query], normalize_embeddings=True)[0]
            clean_query_lower = clean_query.lower()
            
            # Define stopwords for better text normalization
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were'}
            
            for result in search_results:
                title = result["title"]
                title_lower = title.lower()
                
                # 1. Semantic similarity using embeddings
                title_embedding = self.model.encode([title], normalize_embeddings=True)[0]
                embedding_score = float(np.dot(query_embedding, title_embedding))
                
                # 2. Word overlap score (normalized, stopwords removed)
                query_words = set(w for w in clean_query_lower.split() if w not in stopwords)
                title_words = set(w for w in title_lower.split() if w not in stopwords)
                overlap_score = 0.0
                if query_words and title_words:
                    overlap_score = len(query_words & title_words) / len(query_words)
                
                # 3. Exact match bonus
                exact_match_bonus = 1.0 if clean_query_lower == title_lower else 0.0
                
                # 4. Penalize disambiguation pages
                disambiguation_penalty = 0.5 if "disambiguation" in title_lower else 0.0
                
                # 5. Combined score (weighted)
                final_score = (
                    0.6 * embedding_score +
                    0.3 * overlap_score +
                    0.1 * exact_match_bonus -
                    disambiguation_penalty
                )
                
                logging.debug(f"Wikipedia candidate: '{title}' | emb={embedding_score:.3f} overlap={overlap_score:.3f} final={final_score:.3f}")
                
                if final_score > best_match_score:
                    best_match_score = final_score
                    best_title = title
            
            # Fallback to first result if no good match found
            if best_title is None:
                best_title = search_results[0]["title"]
            
            logging.info(f"Selected Wikipedia article: '{best_title}' (score={best_match_score:.3f})")
            
            # Step 2: Fetch the summary for that article
            import urllib.parse
            encoded_title = urllib.parse.quote(best_title)
            summary_url = f"https://{wiki_lang}.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
            
            response = requests.get(summary_url, headers=headers, timeout=5)

            if response.status_code == 200:
                data = response.json()

                if "extract" in data:
                    logging.info(f"Wikipedia fallback successful ({wiki_lang})")
                    return [{
                        "source": "external",
                        "title": data.get("title", ""),
                        "text": data["extract"],
                        "language": language,
                        "score": None,
                        "dense_score": None,
                        "sparse_score": None,
                        "hybrid_score": None,
                        "rerank_score": None,
                        "retrieval_stage": "fallback"
                    }]

        except requests.exceptions.RequestException as exc:
            logging.warning(f"Wikipedia fallback failed: No internet connection")
        except Exception as exc:
            logging.error(f"Wikipedia fallback error: {str(exc)[:50]}")

        # Return offline fallback message
        return [{
            "source": "local",
            "title": "Not Found",
            "text": f"No relevant information found in local corpus for this query. Wikipedia fallback requires internet connection.",
            "language": language,
            "score": 0.0,
            "dense_score": 0.0,
            "sparse_score": 0.0,
            "hybrid_score": 0.0,
            "rerank_score": None,
            "retrieval_stage": "fallback_offline"
        }]

    def _normalize_scores(self, pairs):
        if not pairs:
            return {}

        values = [score for _, score in pairs]
        min_score = min(values)
        max_score = max(values)

        if max_score - min_score < 1e-9:
            return {idx: 1.0 for idx, _ in pairs}

        return {
            idx: (score - min_score) / (max_score - min_score)
            for idx, score in pairs
        }

    def _dense_search(self, query_variants, top_k, language=None):
        query_embedding = self.model.encode(
            query_variants,
            normalize_embeddings=True
        )
        query_embedding = np.array(query_embedding).astype("float32")
        # Retrieve more candidates to allow for language filtering (optimized)
        search_k = top_k * 2 if language else top_k
        scores, indices = self.index.search(query_embedding, search_k)

        best_scores = {}
        for query_scores, query_indices in zip(scores, indices):
            for score, idx in zip(query_scores, query_indices):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                if len(self.metadata[idx].get("text", "")) < 200:
                    continue
                # Filter by language if specified
                if language and self.metadata[idx].get("language") != language:
                    continue
                best_scores[idx] = max(best_scores.get(idx, float("-inf")), float(score))

        return sorted(best_scores.items(), key=lambda item: item[1], reverse=True)[:top_k], query_embedding[0]

    def _sparse_search(self, query_variants, top_k, language=None):
        best_scores = {}
        # Retrieve more candidates to allow for language filtering (optimized)
        search_k = top_k * 2 if language else top_k
        for query in query_variants:
            ranked = self.bm25.search(query, top_k=search_k)
            for idx, score in ranked:
                if idx >= len(self.metadata) or len(self.metadata[idx].get("text", "")) < 200:
                    continue
                # Filter by language if specified
                if language and self.metadata[idx].get("language") != language:
                    continue
                best_scores[idx] = max(best_scores.get(idx, float("-inf")), float(score))
        return sorted(best_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def _merge_results(self, dense_hits, sparse_hits, query, language):
        dense_scores = self._normalize_scores(dense_hits)
        sparse_scores = self._normalize_scores(sparse_hits)

        merged = {}
        for idx, score in dense_hits:
            merged[idx] = {
                "dense_score": score,
                "sparse_score": 0.0,
                "dense_norm": dense_scores.get(idx, 0.0),
                "sparse_norm": 0.0
            }

        for idx, score in sparse_hits:
            merged.setdefault(idx, {
                "dense_score": 0.0,
                "sparse_score": 0.0,
                "dense_norm": 0.0,
                "sparse_norm": 0.0
            })
            merged[idx]["sparse_score"] = score
            merged[idx]["sparse_norm"] = sparse_scores.get(idx, 0.0)

        results = []
        for idx, scores in merged.items():
            item = self.metadata[idx]
            
            # Adaptive weighting: favor dense for semantic queries
            query_length = len(query.split())
            if query_length <= 3:
                # Short queries: balance dense and sparse
                alpha = 0.5
            else:
                # Long queries: favor dense (semantic)
                alpha = 0.7
            
            hybrid_score = alpha * scores["dense_norm"] + (1 - alpha) * scores["sparse_norm"]
            
            # Filter out results where both scores are very low
            if scores["dense_norm"] < 0.3 and scores["sparse_norm"] < 0.3:
                continue
            
            results.append({
                "source": "local",
                "title": item.get("title"),
                "text": item.get("text"),
                "language": item.get("language"),
                "score": hybrid_score,
                "dense_score": scores["dense_score"],
                "sparse_score": scores["sparse_score"],
                "hybrid_score": hybrid_score,
                "rerank_score": None,
                "docid": item.get("docid"),
                "original_lang": language,
                "query_used": query,
                "retrieval_stage": "hybrid"
            })

        results.sort(key=lambda item: item["hybrid_score"], reverse=True)
        return results[:HYBRID_MERGED_TOP_K]

    def _check_relevance(self, query, results, query_embedding=None):
        """Check if retrieved documents are semantically relevant to query"""
        if not results:
            return False
        
        # Reuse query embedding if provided, otherwise compute it
        if query_embedding is None:
            query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        
        # Check top result's title relevance
        top_title = results[0].get('title', '')
        if not top_title or top_title == 'Not Found':
            return False
        
        title_embedding = self.model.encode([top_title], normalize_embeddings=True)[0]
        title_similarity = float(np.dot(query_embedding, title_embedding))
        
        # If title similarity is very low (<0.3), likely irrelevant
        # This catches cases like "World War 1" query matching "Animal Farm" title
        if title_similarity < 0.3:
            logging.debug(f"Relevance check failed: title_similarity={title_similarity:.3f} for '{top_title}'")
            return False
        
        return True

    # -------------------------------------------------
    # Search
    # -------------------------------------------------
    def search(self, query, language="en", top_k=FINAL_TOP_K):
        normalized = normalize_code_mixed_query(query, language)
        query_variants = normalized["query_variants"]
        dense_hits, query_embedding = self._dense_search(query_variants, max(top_k * 2, HYBRID_DENSE_TOP_K), language=language)
        sparse_hits = self._sparse_search(query_variants, max(top_k * 2, HYBRID_SPARSE_TOP_K), language=language)
        results = self._merge_results(dense_hits, sparse_hits, normalized["normalized_query"], language)

        for item in results:
            item["normalized_query"] = normalized["normalized_query"]
            item["query_variants"] = query_variants
            item["code_mixed"] = normalized["code_mixed"]

        if not results:
            logging.info(f"No results found. Using {language.upper()} Wikipedia fallback...")
            return self.fetch_wikipedia_summary(normalized["normalized_query"], language=language)

        top_score = results[0]["hybrid_score"]
        logging.debug(f"Top retrieval score: {top_score}")
        
        # Check relevance before proceeding (reuse query_embedding)
        is_relevant = self._check_relevance(normalized["normalized_query"], results, query_embedding=query_embedding)
        if not is_relevant:
            logging.info(f"Retrieved documents not relevant to query. Using {language.upper()} Wikipedia fallback...")
            return self.fetch_wikipedia_summary(normalized["normalized_query"], language=language)

        # Check if fallback is needed (low score + low variance)
        if top_score < RETRIEVAL_THRESHOLD:
            # Calculate score variance to avoid fallback when results are decent
            scores = [r.get("hybrid_score", 0) for r in results[:5]]
            score_std = np.std(scores) if len(scores) > 1 else 0
            
            # If score is very low (< 0.5), always fallback regardless of variance
            # If score is moderate (0.5-0.65), check variance
            if top_score < 0.5 or score_std < 0.15:
                logging.info(f"Low similarity detected (score={top_score:.2f}, std={score_std:.2f}). Using {language.upper()} Wikipedia fallback...")
                return self.fetch_wikipedia_summary(normalized["normalized_query"], language=language)

        # Filter out low-quality results (score < 0.5) to reduce noise
        filtered_results = [r for r in results if r.get("hybrid_score", 0) >= 0.5]
        
        # If filtering removes too many, keep top 3 at minimum
        if len(filtered_results) < 3 and len(results) >= 3:
            filtered_results = results[:3]
        elif len(filtered_results) == 0:
            filtered_results = results[:top_k]
        
        return filtered_results[:top_k]
