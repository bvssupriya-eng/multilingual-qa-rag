from retrieval.search import Retriever

if __name__ == "__main__":
    retriever = Retriever()

    query = input("Enter your query: ")
    lang = input("Language code (en/hi/bn/ar): ").strip().lower()
    
    # Validate language code
    if lang not in ["en", "hi", "bn", "ar"]:
        print(f"Error: Invalid language code '{lang}'. Must be one of: en, hi, bn, ar")
        import sys
        sys.exit(1)
    
    # Validate query length
    if len(query.strip()) < 3:
        print("Error: Query too short. Please enter at least 3 characters.")
        import sys
        sys.exit(1)
    
    if len(query) > 500:
        print("Warning: Query is very long. Truncating to 500 characters.")
        query = query[:500]

    results = retriever.search(query, top_k=5, language=lang)

    print("\nResults:\n")

    for i, r in enumerate(results, 1):
        print(f"Result {i}")
        print("Source:", r.get("source"))
        print("Title:", r.get("title"))
        print("Text:", r.get("text")[:400])
        print("-" * 60)
