import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from config import *

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def load_chunks():
    all_chunks = []
    metadata = []

    for lang in LANGUAGES:
        file_path = os.path.join(PROCESSED_DATA_PATH, f"{lang}_chunks.jsonl")

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                all_chunks.append(record["text"])
                metadata.append(record)

    return all_chunks, metadata


def generate_embeddings():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)

    texts, metadata = load_chunks()

    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True
    )

    return np.array(embeddings), metadata
