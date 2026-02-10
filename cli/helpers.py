import json
from typing import Any

DEFAULT_MAX_TITLES = 5


# load stop words from file
def load_stopwords(filePath: str = "data/stopwords.txt") -> set[str]:
    with open(filePath, "r") as stopFile:
        return set(stopFile.read().splitlines())


# load json movie data into dict
def load_movies(movie_path: str) -> list[dict[str, Any]]:
    try:
        with open(movie_path, "r") as movie_json:
            payload = json.load(movie_json)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing file: {movie_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Bad json in {movie_path}") from e
    return payload["movies"]
