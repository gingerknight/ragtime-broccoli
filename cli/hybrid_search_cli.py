import argparse

from lib.hybrid_search import normalize, pretty_print


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_scores = subparsers.add_parser("normalize", help="Max-Min normalize float values")
    normalize_scores.add_argument("scores", type=float, nargs="+", help="Space seperated float values to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            values = normalize(args.scores)
            pretty_print(values)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
