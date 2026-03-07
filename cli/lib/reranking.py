import json

from .gemini_client import GeminiClient
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


def print_rrf_reranked(results: list, query: str, k, n: int = 5) -> None:
    print(f"Re-ranking top {n} results using individual method...")
    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k})")
    for num, val in enumerate(results[:n]):
        print(f"{num + 1}. {val[1]['title']}")
        print(f"Re-rank Score: {val[1]['rerank_score']}/10")
        print(f"\tRRF Score: {val[1]['rrf_score']:.4f}")
        print(f"\tBM25 Rank: {val[1]['bm25_rank']}, Semantic Rank: {val[1]['semantic_rank']}")
        print(f"\t{val[1]['document'][:100]}")


def print_rrf_reranked_batched(results: list, query: str, k, n: int = 5) -> None:
    print(f"Re-ranking top {n} results using batch method...")
    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k})")
    for num, val in enumerate(results[:n]):
        print(f"{num + 1}. {val[1]['title']}")
        print(f"\tRe-rank Rank: {num + 1}")
        print(f"\tRRF Score: {val[1]['rrf_score']:.4f}")
        print(f"\tBM25 Rank: {val[1]['bm25_rank']}, Semantic Rank: {val[1]['semantic_rank']}")
        print(f"\t{val[1]['document'][:100]}")

def rerank_gemini(query, results: list, q_method: str, k, limit):
        llm_client = GeminiClient()
        """
        results is a list of tuples. [0] is doc id, [1] is the dictionary
        (doc_id#,
            {
                'title': 'string',
                'bm25_rank': #,
                'bm25_rrf_score': float,
                'semantic_rank': #,
                'document': 'string',
                'rrf_score_total': float,
                'semantic_rrf_score': float,
                }
            )
        """
        match q_method:
            case "individual":
                for _, val in enumerate(results):
                    raw_score = llm_client.rerank_doc_individual(query, val[1])
                    try:
                        score = float(str(raw_score).strip().split()[0])
                    except (ValueError, IndexError):
                        score = 0.0

                    val[1]["rerank_score"] = score
                    # print(f"Score from Gemini: {score}")
                    # print(f"Sleeping...")
                    # sleep(5)
                # print(f"Results: {results[:4]}")
                sorted_results = sorted(
                    results,
                    key=lambda item: float(item[1].get("rerank_score", float("-inf"))),
                    reverse=True,
                )
                print_rrf_reranked(sorted_results, query, k, limit)
            case "batch":
                # build hashmap for results list with docID as key and tuple as value
                ref_result = {}
                for val in results:
                    ref_result[val[0]] = val
                raw_score = llm_client.rerank_docs_batch(query, results)
                reranked_results = []
                scores = json.loads(raw_score)
                print(f"Score list: {scores}")
                for score in scores:
                    reranked_results.append(ref_result[score])
                print_rrf_reranked_batched(reranked_results, query, k, limit)
            case _:
                raise NotImplementedError("Not implemented rerank method for LLM processing...")