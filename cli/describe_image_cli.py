import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    describe_image = subparsers.add_parser("describe", help="Query prompt for image description")
    describe_image.add_argument("--image", type=str, help="Path to image")
    describe_image.add_argument("--query", type=str, help="Text query to rewrite based on the image")

    args = parser.parse_args()

    match args.command:
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
