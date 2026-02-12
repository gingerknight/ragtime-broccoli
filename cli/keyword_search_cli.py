#!/usr/bin/env python3

import argparse

from search_cls import MovieSearch, InvertedIndex
from errors.exception_handling import SearchEngineError


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search Query")

    subparsers.add_parser("build", help="Build Inverse index artifacts")
    subparsers.add_parser("load", help="Load pickle cache files for processed data")

    args = parser.parse_args()
    try:
        ms = MovieSearch.from_file()
    except Exception as e:
        print(f"Unable to load data file...check your movies.json file: {e}")
        return 2

    match args.command:
        case "search":
            print("Loading cache files...")
            try:
                idx_cache, docmap_cache = InvertedIndex.load()
            except SearchEngineError as e:
                print(f"Error: {e}")
                return 2
            print(f"Searching for: {args.query}")
            # ms.sample_data()
            titles = ms.find_titles(
                args.query, idx_cache=idx_cache, docmap_cache=docmap_cache
            )
            ms.print_results(titles)
        case "build":
            try:
                inv = InvertedIndex()
                inv.build(ms._movies)
                # Debug statement
                # merida_list = inv.get_documents("merida")
                # print(f"First document for token 'merida' = {merida_list[0]}")
            except Exception as e:
                print(f"Unable to build index and/or docmap: {e}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
