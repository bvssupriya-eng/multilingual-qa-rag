import math
import re
import heapq
from collections import Counter, defaultdict


class BM25Index:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = [self._tokenize(doc) for doc in documents]
        self.doc_count = len(self.documents)
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )
        self.term_frequencies = []
        self.document_frequencies = defaultdict(int)
        self.term_postings = defaultdict(list)

        for idx, tokens in enumerate(self.documents):
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            for token, frequency in frequencies.items():
                self.document_frequencies[token] += 1
                self.term_postings[token].append((idx, frequency))

    def _tokenize(self, text):
        punctuation = " \t\r\n?.,!;:؟।॥()[]{}\"'“”‘’`"
        return [
            token
            for token in (part.strip(punctuation) for part in re.split(r"\s+", text.lower()))
            if token
        ]

    def get_scores(self, query):
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.doc_count

        if not query_tokens or not self.doc_count:
            return scores

        for token in query_tokens:
            doc_frequency = self.document_frequencies.get(token, 0)
            if doc_frequency == 0:
                continue

            idf = math.log(1 + (self.doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5))

            for idx, term_frequency in self.term_postings.get(token, []):
                doc_length = self.doc_lengths[idx] or 1
                numerator = term_frequency * (self.k1 + 1)
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * (doc_length / max(self.avg_doc_length, 1e-9))
                )
                scores[idx] += idf * (numerator / denominator)

        return scores

    def search(self, query, top_k=10):
        scores = self.get_scores(query)
        return heapq.nlargest(top_k, enumerate(scores), key=lambda item: item[1])
