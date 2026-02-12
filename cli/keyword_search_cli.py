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
    
    term_frequency_parser = subparsers.add_parser("tf", help="Fetch term frequency in the related doc")
    term_frequency_parser.add_argument("id", type=int, help="Docuemnt ID in the Inverse Index cache")
    term_frequency_parser.add_argument("term", type=str, help="Search term you are lookin for")

    inverse_doc_freq_parser = subparsers.add_parser("idf", help="Return the inverse document frequency value for a search term")
    inverse_doc_freq_parser.add_argument("term", type=str, help="Search term to look for")

    tf_idf_parser = subparsers.add_parser("tfidf", help="Caclulate the TF-IDF for a document and term")
    tf_idf_parser.add_argument("id", type=int, help="Document ID to search")
    tf_idf_parser.add_argument("term", type=str, help="Search terms to calculate the TF-IDFs with")

    args = parser.parse_args()
    # initialize inv object
    inv = InvertedIndex()
    try:
        ms = MovieSearch.from_file()
    except Exception as e:
        print(f"Unable to load data file...check your movies.json file: {e}")
        return 2

    match args.command:
        case "search":
            print("Loading cache files...")
            try:
                idx_cache, docmap_cache, tf_cache = InvertedIndex.load()
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
                inv.build(ms._movies)
                # Debug statement
                # merida_list = inv.get_documents("merida")
                # print(f"First document for token 'merida' = {merida_list[0]}")
            except Exception as e:
                print(f"Unable to build index and/or docmap: {e}")
        case "tf":
            _, _, tf_cache = InvertedIndex.load()
            inv.term_frequencies = tf_cache
            print(f"Fetching term frequency with params {args.id} -- {args.term} ")
            num = inv.get_tf(args.id, args.term)
            print(f"Term frequency for {args.term} --> {num}")
        case "idf":
            idx_cache, docmap_cache, _ = InvertedIndex.load()
            idf = inv.calculate_idf(idx_cache, docmap_cache, args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            idx_cache, docmap_cache, tf_cache = InvertedIndex.load()
            inv.term_frequencies = tf_cache
            tf = inv.get_tf(args.id, args.term)
            idf = inv.calculate_idf(idx_cache, docmap_cache, args.term)
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.id}': {tf_idf:.2f}")
        case _:
            parser.print_help(args.id, args.term)


if __name__ == "__main__":
    main()
