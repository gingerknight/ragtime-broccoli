#!/usr/bin/env python3

import argparse

# Script vs module import setup
try:
    from search_cls import MovieSearch, InvertedIndex
except ImportError:
    from .search_cls import MovieSearch, InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search Query")
    search_parser.add_argument("--path", default="data/movies.json")

    subparsers.add_parser("build", help="Build Inverse index artifacts")

    args = parser.parse_args()
    try:
        ms = MovieSearch.from_file("data/movies.json")
    except Exception as e:
        print(f"Unable to load data file...check your movies.json file: {e}")

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            # ms.sample_data()
            titles = ms.find_titles(args.query)
            ms.print_results(titles)
        case "build":
            try:
                inv = InvertedIndex()
                inv.build(ms._movies)
                # Debug statement
                merida_list = inv.get_documents("merida")
                print(f"First document for token 'merida' = {merida_list[0]}")
            except Exception as e:
                print(f"Unable to build index and/or docmap: {e}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
