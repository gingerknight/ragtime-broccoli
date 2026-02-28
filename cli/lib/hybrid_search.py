import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


def normalize(values: list[float]) -> list[float]:
    # Min-max feature scaling: normalize scores to [0, 1]
    # (score - min_score) / (max_score - min_score)
    max_val = max(values)
    min_val = min(values)
    denominator = max_val - min_val
    if denominator == 0:
        return [1.0] * len(values)
    normalized_scores = [(x - min_val) / (denominator) for x in values]
    return normalized_scores


def pretty_print(values: list[float]) -> None:
    # Pretty print values
    print("Normalized values:")
    for _, score in enumerate(values):
        print(f"* {score:.4f}")


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
