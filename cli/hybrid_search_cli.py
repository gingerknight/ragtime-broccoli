import argparse

from lib.helpers import load_movies
from lib.hybrid_search import HybridSearch, normalize, pretty_print


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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            values = normalize(args.scores)
            pretty_print(values)
        case "weighted-search":
            movies = load_movies()
            hybrid_instance = HybridSearch(movies)
            hybrid_instance.weighted_search(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
