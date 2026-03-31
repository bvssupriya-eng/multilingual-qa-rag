import os
import json
import time
from datasets import load_dataset
from transformers import AutoTokenizer
from config import *

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

try:
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", local_files_only=True)
except Exception as e:
    print(f"Warning: Could not load mt5-small tokenizer locally: {e}")
    print("Falling back to basic tokenization...")
    tokenizer = None


def chunk_text(text):
    if tokenizer is None:
        # Fallback: simple character-based chunking
        # Approximate: 1 token ≈ 4 characters for multilingual text
        char_chunk_size = CHUNK_SIZE * 4
        char_overlap = CHUNK_OVERLAP * 4
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + char_chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += char_chunk_size - char_overlap
        
        return chunks
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def stream_and_chunk():
    for lang in LANGUAGES:
        print(f"\nStreaming MIRACL corpus for language: {lang}")

        output_file = os.path.join(PROCESSED_DATA_PATH, f"{lang}_chunks.jsonl")

        doc_count = 0
        chunk_count = 0

        while True:
            try:
                dataset = load_dataset(
                    "miracl/miracl-corpus",
                    lang,
                    split="train",
                    streaming=True,
                    trust_remote_code=True
                )

                with open(output_file, "w", encoding="utf-8") as f:
                    for doc in dataset:
                        if doc_count >= MAX_DOCS_PER_LANGUAGE:
                            break

                        try:
                            chunks = chunk_text(doc["text"])
                        except Exception:
                            continue  # skip bad doc

                        for chunk in chunks:
                            record = {
                                "docid": doc["docid"],
                                "title": doc["title"],
                                "language": lang,
                                "text": chunk
                            }
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            chunk_count += 1

                        doc_count += 1

                break  # successful completion → exit retry loop

            except Exception as e:
                print(f"Network error for {lang}. Retrying in 5 seconds...")
                time.sleep(5)
                continue

        print(f"{lang}: {doc_count} docs → {chunk_count} chunks saved")

    print("\nPhase 1 complete.")
