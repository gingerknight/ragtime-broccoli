from .gemini_client import GeminiClient
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


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def pretty_print(values: list[float]) -> None:
    # Pretty print values
    print("Normalized values:")
    for _, score in enumerate(values):
        print(f"* {score:.4f}")


def print_hybrid_results(results: list, n: int = 5) -> None:
    for num, val in enumerate(results[:n]):
        print(f"{num + 1}. {val[1]['title']}")
        print(f"\tHybrid Score: {val[1]['hybrid_score']:.4f}")
        print(f"\tBM25: {val[1]['bm25_normalized_score']:.4f}, Semantic: {val[1]['semantic_normalized_score']:.4f}")
        print(f"\t{val[1]['document']}")


def print_rrf_results(results: list, n: int = 5) -> None:
    for num, val in enumerate(results[:n]):
        print(f"{num + 1}. {val[1]['title']}")
        print(f"\tRRF Score: {val[1]['rrf_score']:.4f}")
        print(f"\tBM25 Rank: {val[1]['bm25_rank']}, Semantic Rank: {val[1]['semantic_rank']}")
        print(f"\t{val[1]['document'][:100]}")


def alpha_bar(alpha: float, width: int = 20) -> str:
    # clamp to [0.0, 1.0]
    alpha = max(0.0, min(1.0, alpha))

    filled = round(alpha * width)
    empty = width - filled
    bar = "█" * filled + "-" * empty

    pct_keyword = int(round(alpha * 100))
    pct_semantic = 100 - pct_keyword

    if alpha == 0.5:
        label = "50/50 Split"
    elif alpha == 1.0:
        label = "100% Keyword"
    elif alpha == 0.0:
        label = "100% Semantic"
    else:
        label = f"{pct_keyword}% Keyword, {pct_semantic}% Semantic"

    return f"α = {alpha:.1f}: [{bar}] {label}"


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex.from_cache()
        if not self.idx.index:
            self.idx.build(self.documents)
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        # call _bm25 method to get keyword search
        hybrid_dict = {}
        print(alpha_bar(alpha))
        keyword_results = self._bm25_search(query, 500)
        self.semantic_search.load_or_create_chunk_embeddings(self.documents)
        semantic_results = self.semantic_search.search_chunks(query, 500)

        """
        Keyword Results: [(1771, 'Paddington', 10.489448461845111)]
        Semantic Results: [
            {
                'id': 2784,
                'title': 'Legends of the Fall',
                'document': 'Sick of betrayals the United States government perpetrated on the Native Americans, Colonel William ',
                'score': np.float32(0.5236),
                'metadata': {}
            },
        """
        # normalize scores
        semantic_normalized = normalize([r["score"] for r in semantic_results])
        bm25_normalized = normalize([r[2] for r in keyword_results])

        # iterate over bm25, add to combined dict
        for (id, title, _score), norm in zip(keyword_results, bm25_normalized, strict=True):
            hybrid_dict[id] = {
                "title": title,
                "bm25_normalized_score": norm,
                "semantic_normalized_score": 0.0,
                "document": self.idx.docmap[id]["description"][:100],
            }

        # iterate over semantic, add to combined dict
        for r, norm in zip(semantic_results, semantic_normalized, strict=True):
            if r["id"] not in hybrid_dict:
                hybrid_dict[r["id"]] = {
                    "title": r["title"],
                    "document": r["document"],
                    "semantic_normalized_score": norm,
                    "bm25_normalized_score": 0.0,
                }
            else:
                hybrid_dict[r["id"]]["semantic_normalized_score"] = norm

        for _doc_id, row in hybrid_dict.items():
            bm = row.get("bm25_normalized_score", 0.0)
            sem = row.get("semantic_normalized_score", 0.0)
            row["hybrid_score"] = hybrid_score(bm, sem, alpha)
        print("=" * 80)
        sorted_results = sorted(hybrid_dict.items(), key=lambda item: float(item[1]["hybrid_score"]), reverse=True)
        print_hybrid_results(sorted_results, limit)

    def rrf_search(self, query, k, gather_limit=5):  # 500
        """
        Reciprocal rank fusion (RRF) is a method for combining multiple result sets with different relevance indicators into a single result set.
        RRF requires no tuning, and the different relevance indicators do not have to be related to each other to achieve high-quality results.

        RRF uses the following formula to determine the score for ranking each document:
        score = 0.0
        for q in queries:
            if d in result(q):
                score += 1.0 / ( k + rank( result(q), d ) )
        return score

        # where
        # k is a ranking constant
        # q is a query in the set of queries
        # d is a document in the result set of q
        # result(q) is the result set of q
        # rank( result(q), d ) is d's rank within the result(q) starting from 1
        """
        # call _bm25 method to get keyword search
        rrf_dict = {}
        keyword_results = self._bm25_search(query, gather_limit)
        self.semantic_search.load_or_create_chunk_embeddings(self.documents)
        semantic_results = self.semantic_search.search_chunks(query, gather_limit)

        # iterate over bm25, add to rrf dict
        # keyword results is sorted descendin
        i = 1
        for id, title, _score in keyword_results:
            rrf_dict[id] = {
                "title": title,
                "bm25_rank": i,
                "rrf_score": self.rrf_score(i, k),
                "semantic_rank": None,
                "document": self.idx.docmap[id]["description"],
            }
            i += 1

        # iterate over semantic, add to rrf dict
        i = 1
        for r in semantic_results:
            if r["id"] not in rrf_dict:
                rrf_dict[r["id"]] = {
                    "title": r["title"],
                    "document": r["document"],
                    "semantic_rank": i,
                    "rrf_score": self.rrf_score(i, k),
                    "bm25_rank": None,
                }
            else:
                rrf_dict[r["id"]]["semantic_rank"] = i
                rrf_dict[r["id"]]["rrf_score"] += self.rrf_score(i, k)

            i += 1

        sorted_results = sorted(rrf_dict.items(), key=lambda item: float(item[1]["rrf_score"]), reverse=True)
        return sorted_results
        # print_rrf_results(sorted_results, limit)

    def rrf_score(self, rank, k=60) -> float:
        return 1 / (k + rank)


    def enhanced_query(self, query: str, choice="spell"):
        # do an enhanced query with the gemini api
        gem_client = GeminiClient()
        match choice:
            case "spell":
                response = gem_client.spell_check(query)
                print(f"Enhanced query ({choice}): '{query}' -> '{response.text}'\n")
                return response
            case "rewrite":
                response = gem_client.rewrite_query(query)
                print(f"Enhanced query ({choice}): '{query}' -> '{response.text}'\n")
                return response
            case "expand":
                response = gem_client.expand_query(query)
                print(f"Enhanced query ({choice}): '{query}' -> '{response.text}'\n")
                return response
            case _:
                raise NotImplementedError("Not an option that is implemented yet...")
