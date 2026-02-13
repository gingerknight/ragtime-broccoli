#!/usr/bin/env python3

import argparse

from errors.exception_handling import SearchEngineError
from helpers import BM25_B, BM25_K1
from search_cls import InvertedIndex, MovieSearch


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

    inverse_doc_freq_parser = subparsers.add_parser(
        "idf", help="Return the inverse document frequency value for a search term"
    )
    inverse_doc_freq_parser.add_argument("term", type=str, help="Search term to look for")

    tf_idf_parser = subparsers.add_parser("tfidf", help="Caclulate the TF-IDF for a document and term")
    tf_idf_parser.add_argument("id", type=int, help="Document ID to search")
    tf_idf_parser.add_argument("term", type=str, help="Search terms to calculate the TF-IDFs with")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    inv = InvertedIndex()
    try:
        ms = MovieSearch.from_file()
    except Exception as e:
        print(f"Unable to load data file...check your movies.json file: {e}")
        return 2

    cache_commands = {"search", "load", "tf", "idf", "tfidf", "bm25idf", "bm25tf", "bm25search"}
    if args.command in cache_commands:
        print("Loading cache files...")
        try:
            inv = InvertedIndex.from_cache()
        except SearchEngineError as e:
            print(f"Error: {e}")
            return 2

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            titles = ms.find_titles(args.query, idx_cache=inv.index, docmap_cache=inv.docmap)
            ms.print_results(titles)
        case "build":
            try:
                inv.build(ms._movies)
                # Debug statement
                # merida_list = inv.get_documents("merida")
                # print(f"First document for token 'merida' = {merida_list[0]}")
            except Exception as e:
                print(f"Unable to build index and/or docmap: {e}")
        case "load":
            print("Cache loaded successfully.")
        case "tf":
            print(f"Fetching term frequency with params {args.id} -- {args.term} ")
            num = inv.get_tf(args.id, args.term)
            print(f"Term frequency for {args.term} --> {num}")
        case "idf":
            idf = inv.calculate_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf = inv.get_tf(args.id, args.term)
            idf = inv.calculate_idf(args.term)
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25idf = inv.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = inv.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            bm_list = inv.bm25_search(args.query)
            for bm_item in bm_list:
                print(f"({bm_item[0]}) {bm_item[1]} - Score: {bm_item[2]:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
