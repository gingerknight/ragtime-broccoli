import argparse

from lib.helpers import load_movies
from lib.hybrid_search import HybridSearch, normalize, pretty_print, print_rrf_results


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
        choices=["individual", "batch"],
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
            if args.enhance:
                new_query = hybrid_instance.enhanced_query(choice=args.enhance, query=args.query)
                sorted_results = hybrid_instance.rrf_search(new_query.text, args.k, 500)
                print_rrf_results(sorted_results, args.limit)
            elif args.rerank_method:
                sorted_results = hybrid_instance.rrf_search(args.query, args.k, 5 * (args.limit))
                hybrid_instance.rerank_gemini(args.query, sorted_results, args.rerank_method, args.k, args.limit)
                # raise NotImplementedError("Not implemented yet.")
            else:
                sorted_results = hybrid_instance.rrf_search(args.query, args.k, args.limit)
                print_rrf_results(sorted_results, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
