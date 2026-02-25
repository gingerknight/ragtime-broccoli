#!/usr/bin/env python3

import argparse

from helpers import load_movies
from lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    embed_query_text,
    embed_text,
    pretty_display_chunks,
    size_defined_chunking,
    verify_embeddings,
    verify_model,
    searching_chunks
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the embedding model loaded")
    subparsers.add_parser("embed_chunks", help="Load or create semantic chunk embeddins")

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

    chunking = subparsers.add_parser("chunk", help="Chunk text into N sized chunks")
    chunking.add_argument("text", type=str, help="Text to chunk")
    chunking.add_argument(
        "--chunk-size", type=int, nargs="?", default=200, help="Number represented as an int of the chunk size"
    )
    chunking.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=0,
        help="Overlap of words between the chunks to preserve word meaning between chunks",
    )

    semantic_chunking = subparsers.add_parser("semantic_chunk", help="Semantic chunking: split text on sentecnes")
    semantic_chunking.add_argument("text", type=str, help="Text to chunk")
    semantic_chunking.add_argument(
        "--max-chunk-size", type=int, nargs="?", default=4, help="number of sentences per chunk"
    )
    semantic_chunking.add_argument(
        "--overlap", type=int, nargs="?", default=0, help="number of sentences to overlap for each chunk"
    )

    search_chunked = subparsers.add_parser("search_chunked", help="String to query with")
    search_chunked.add_argument("query", type=str, help="Search query for semantic chunking search")
    search_chunked.add_argument("--limit", type=int, nargs="?", default=10, help="Number of results to limit")

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
        case "chunk":
            chunks = size_defined_chunking(args.text, args.chunk_size, args.overlap)
            pretty_display_chunks(chunks, len(args.text))
        case "semantic_chunk":
            chunks = size_defined_chunking(args.text, args.max_chunk_size, args.overlap, semantic=True)
            pretty_display_chunks(chunks, len(args.text), semantic=True)
        case "embed_chunks":
            sem_chunk_search = ChunkedSemanticSearch()
            movies = load_movies()
            embeddings = sem_chunk_search.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")
            results = sem_chunk_search.search_chunks("superhero action movie")
            print(results)
        case "search_chunked":
            searching_chunks(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
