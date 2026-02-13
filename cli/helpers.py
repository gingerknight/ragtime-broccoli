import json
import string
from typing import Any

from nltk.stem import PorterStemmer

DEFAULT_MAX_TITLES = 5

INDEX_PATH = "./cache/index.pkl"
DOCMAP_PATH = "./cache/docmap.pkl"
TF_PATH = "./cache/term_frequencies.pkl"
DOC_LENGTHS_PATH = "./cache/doc_lengths.pkl"

MOVIES_PATH: str = "data/movies.json"
STOP_PATH: str = "data/stopwords.txt"

BM25_K1 = 1.5
BM25_B = 0.75


# load stop words from file
def load_stopwords() -> set[str]:
    with open(STOP_PATH) as stopFile:
        return set(stopFile.read().splitlines())


# load json movie data into dict
def load_movies(movie_path: str = MOVIES_PATH) -> list[dict[str, Any]]:
    try:
        with open(movie_path) as movie_json:
            payload = json.load(movie_json)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing file: {movie_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Bad json in {movie_path}") from e
    return payload["movies"]


def normalize(title: str) -> list[str]:
    # remove stardard punctuation from movie or query title string
    # drop case to lower
    valid_tokens = []
    drop_punc_title = title.translate(str.maketrans("", "", string.punctuation))
    drop_lower = drop_punc_title.lower()
    # tokenize string
    text = drop_lower.split()
    for word in text:
        if word:
            valid_tokens.append(word)
    # drop stop words from tokens
    stop_words = load_stopwords()
    drop_stops = [x for x in valid_tokens if x not in stop_words]
    # stem the words in the tokens list
    stemmer = PorterStemmer()
    stemmed = []
    for word in drop_stops:
        stemmed.append(stemmer.stem(word))
    return stemmed
