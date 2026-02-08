#!/usr/bin/env python3

import argparse
from search_cls import MovieSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search Query")
    search_parser.add_argument("--path", default="data/movies.json")

    args = parser.parse_args()

    ms = MovieSearch.from_file()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            # ms.sample_data()
            ms.find_titles(args.query)       
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()