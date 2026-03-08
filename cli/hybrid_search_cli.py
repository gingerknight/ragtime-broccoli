import argparse
from dataclasses import dataclass

from lib.cross_encoding import cross_encoding
from lib.helpers import load_movies
from lib.hybrid_search import HybridSearch, normalize, pretty_print, print_rrf_results
from lib.reranking import (
    print_rrf_reranked,
    print_rrf_reranked_batched,
    print_rrf_reranked_cross_encoder,
    rerank_gemini,
)


@dataclass(frozen=True)
class RRF_Search_Request:
    query: str
    enhance: str | None
    rerank_method: str | None
    limit: int = 5
    k: int = 60


def handle_rrf_search(args: argparse.Namespace, hybrid: HybridSearch) -> None:
    # RRF search has multiple options; route with small helper steps.
    req = RRF_Search_Request(
        query=args.query, limit=args.limit, k=args.k, enhance=args.enhance, rerank_method=args.rerank_method
    )
    query = _resolve_query(req, hybrid)
    gather_limit = 500 if req.enhance else (5 * req.limit) if req.rerank_method else req.limit
    results = hybrid.rrf_search(query, req.k, gather_limit)
    results = _maybe_rerank(req, query, results)
    _output_rrf(results, req)


def _resolve_query(req: RRF_Search_Request, hs: HybridSearch):
    # either do enhanced or standard rrf-search
    if req.enhance:
        new_query = hs.enhanced_query(choice=req.enhance, query=req.query)
        return new_query.text
    return req.query


def _maybe_rerank(req, query, results):
    # do reranking if that method is included
    if req.rerank_method:
        if req.rerank_method == "cross_encoder":
            results = cross_encoding(results, query)
        else:
            results = rerank_gemini(query, results, req.rerank_method, req.k, req.limit)
    return results


def _output_rrf(results: list, req: RRF_Search_Request):
    if req.rerank_method:
        if req.rerank_method == "individual":
            print_rrf_reranked(results, req.query, req.k, req.limit)
        elif req.rerank_method == "batch":
            print_rrf_reranked_batched(results, req.query, req.k, req.limit)
        elif req.rerank_method == "cross_encoder":
            print_rrf_reranked_cross_encoder(results, req.query, req.k, req.limit)
    else:
        print_rrf_results(results, req.limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_scores = subparsers.add_parser("normalize", help="Max-Min normalize float values")
    normalize_scores.add_argument("scores", type=float, nargs="+", help="Space seperated float values to normalize")

    weighted_search = subparsers.add_parser("weighted-search", help="Perform weighted bm25 and semantic search")
    weighted_search.add_argument("query", type=str, help="Search string")
    weighted_search.add_argument(
        "--alpha", type=float, nargs="?", default=0.5, help="Weighting value between keyword and semantic search"
    )
    weighted_search.add_argument("--limit", type=int, nargs="?", default=5, help="Top 'N' matches to return")

    rrf_parser = subparsers.add_parser("rrf-search", help="Perform Reciprocal Rank Fusion search")
    rrf_parser.add_argument("query", type=str, help="Query text parameter")
    rrf_parser.add_argument(
        "-k",
        type=int,
        nargs="?",
        default=60,
        help="Weight parameter: lower value provide more weight to top ranks, higher value a more gradual drop off",
    )
    rrf_parser.add_argument("--limit", type=int, nargs="?", default=5, help="How many results to return")
    rrf_parser.add_argument(
        "--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method"
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Rerank method for the documents after the RRF search queries.",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            values = normalize(args.scores)
            pretty_print(values)
        case "weighted-search":
            movies = load_movies()
            hybrid_instance = HybridSearch(movies)
            hybrid_instance.weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            movies = load_movies()
            hybrid_instance = HybridSearch(movies)
            handle_rrf_search(args, hybrid_instance)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
