import os
import faiss
import pickle
from config import *
from embeddings.embed_corpus import generate_embeddings


def build_index():
    embeddings, metadata = generate_embeddings()

    dimension = embeddings.shape[1]

    print("Building FAISS index...")
    index_type = FAISS_INDEX_TYPE.lower()

    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = HNSW_EF_SEARCH
    elif index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
    else:
        raise ValueError(f"Unsupported FAISS_INDEX_TYPE: {FAISS_INDEX_TYPE}")

    index.add(embeddings)

    # Ensure directory exists
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    print("Saving index...")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("Saving metadata...")
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("Index and metadata saved successfully.")
    print("Index type:", FAISS_INDEX_TYPE)
    print("Total vectors indexed:", index.ntotal)
