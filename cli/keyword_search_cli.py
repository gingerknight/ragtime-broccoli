#!/usr/bin/env python3

import argparse

# Script vs module import setup
try:
    from search_cls import MovieSearch
except ImportError:
    from .search_cls import MovieSearch


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search Query")
    search_parser.add_argument("--path", default="data/movies.json")

    args = parser.parse_args()
    try:
        ms = MovieSearch.from_file(args.path)
    except Exception:
        print("Unable to load data file...check your movies.json file: {e}")

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            # ms.sample_data()
            titles = ms.find_titles(args.query)
            ms.print_results(titles)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
