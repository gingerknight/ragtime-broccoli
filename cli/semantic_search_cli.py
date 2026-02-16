#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_text, verify_embeddings, verify_model


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the embedding model loaded")

    embedding = subparsers.add_parser("embed_text", help="Command to calculate and return embedding value of string")
    embedding.add_argument("input", type=str, help="Text top caluclate embedding from")

    subparsers.add_parser(
        "verify_embeddings", help="Verify we have built and/or loaded embedding values from documents"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.input)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
