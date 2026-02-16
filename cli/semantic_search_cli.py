#!/usr/bin/env python3

import argparse

from helpers import load_movies
from lib.semantic_search import SemanticSearch, embed_query_text, embed_text, verify_embeddings, verify_model


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the embedding model loaded")

    embedding = subparsers.add_parser("embed_text", help="Command to calculate and return embedding value of string")
    embedding.add_argument("input", type=str, help="Text top caluclate embedding from")

    subparsers.add_parser(
        "verify_embeddings", help="Verify we have built and/or loaded embedding values from documents"
    )

    embed_query = subparsers.add_parser("embedquery", help="Calculate embeddings NDArray for user query")
    embed_query.add_argument("query", type=str, help="User string query to embed and create embedding vector")

    search = subparsers.add_parser("search", help="Seamtic search the documents (movies.json)")
    search.add_argument(
        "user_query", type=str, help="String query to search via embedding and semantic vector calculations"
    )
    search.add_argument("--limit", type=int, nargs="?", default=5, help="Limit return to N values")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.input)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            sem_search = SemanticSearch()
            movies = load_movies()
            sem_search.load_or_create_embeddings(movies)
            result_list = sem_search.search(args.user_query, args.limit)
            for i in range(len(result_list)):
                print(f"{i + 1}. {result_list[i][1]} (score: {result_list[i][0]:.2f})\n\t{result_list[i][2][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
