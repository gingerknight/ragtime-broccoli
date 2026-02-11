# keyword search
"""
* Load movies.json into dictionary
* Iterate over movies section and match title field
* Order return by IDs ascending
"""

from typing import Any, Dict
from operator import itemgetter
import pickle
import os

import string
from nltk.stem import PorterStemmer


try:
    # Package import when run as module
    from .helpers import DEFAULT_MAX_TITLES, load_stopwords, load_movies
except ImportError:
    # Script execution fallback
    from helpers import DEFAULT_MAX_TITLES, load_stopwords, load_movies


class MovieSearch:
    def __init__(self, movies: list[dict[str, Any]], stopwords: set[str]):
        self._movies = movies
        self._stopwords = stopwords or set()

    @classmethod
    def from_file(
        cls,
        movies_path: str = "data/movies.json",
        stop_path: str = "data/stopwords.txt",
    ) -> "MovieSearch":
        movies = load_movies(movies_path)
        stopwords = load_stopwords(stop_path)
        return cls(movies=movies, stopwords=stopwords)

    def sample_data(self, n: int = DEFAULT_MAX_TITLES) -> None:
        print("Printing Sample Titles and Movie Ids:...")
        for movie in self._movies[:n]:
            movie_id = movie.get("id")
            title = movie.get("title")
            print(f"{movie_id}\t{title}")

    def find_titles(self, query: str) -> list[tuple[int | str]]:
        matched_titles = []
        q_token = self._tokenize_query(self._normalize(query))
        q_token_stops = self._drop_stopwords(q_token)
        q_stemmed = self._stem_tokenizer(q_token_stops)

        for movie in self._movies:
            title = movie.get("title", "")
            m_id = movie.get("id")
            if any(q in self._normalize(title) for q in q_stemmed):
                matched_titles.append((title, m_id))

        # in place sort list of tuples by second element (id)
        matched_titles.sort(key=itemgetter(1))
        # grab titles
        found_titles = [x[0] for x in matched_titles]

        return found_titles

    def print_results(self, titles: list, n: int = DEFAULT_MAX_TITLES) -> None:
        for num, title in enumerate(titles[:n]):
            print(f"{num + 1}. {title}")

    def _normalize(self, title: str) -> str:
        # remove stardard punctuation from movie or query title string
        # drop case to lower
        drop_punc_title = title.translate(str.maketrans("", "", string.punctuation))
        return drop_punc_title.lower()

    def _drop_stopwords(self, tokens: list[str]) -> list[str]:
        # drop stop words from user/title list
        return list(set(tokens) - self._stopwords)

    def _stem_tokenizer(self, tokens: list[str]) -> list[str]:
        # stem the words in the tokens list
        stemmer = PorterStemmer()
        stemmed = []
        for word in tokens:
            stemmed.append(stemmer.stem(word))
        return stemmed

    def _tokenize_query(self, query: str) -> list[str]:
        # tokenize query string on whitespace
        # no stemming
        return query.split()


class InvertedIndex:
    """
    Inverted index (also referred to as a postings list, postings file, or inverted file) is a database index storing a mapping from content, such as words or numbers, to its locations in a table, or in a document or a set of documents (named in contrast to a forward index, which maps from documents to content).[1] The purpose of an inverted index is to allow fast full-text searches, at a cost of increased processing when a document is added to the database.[2] The inverted file may be the database file itself, rather than its index. It is the most popular data structure used in document retrieval systems,[3] used on a large scale for example in search engines. Additionally, several significant general-purpose mainframe-based database management systems have used inverted list architectures, including ADABAS, DATACOM/DB, and Model 204.
    """

    def __init__(self):
        # dictionary mapping tokens to sets of doc Ids
        self.index: Dict[str, set[int]] = {}
        # dictionary mapping doc Ids to their full doc objects
        self.docmap: Dict[int, Dict[str, Any]] = {}

    def _normalize(self, text: str) -> str:
        # Keep indexing token rules consistent with query normalization.
        drop_punc_text = text.translate(str.maketrans("", "", string.punctuation))
        return drop_punc_text.lower()

    def _add_document(self, doc_id: int, text: str):
        # tokenize and lower the text (title and desc)
        # iterate over and add the id to the set values of the "word" key
        for word in self._normalize(text).split():
            self.index.setdefault(word, set()).add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        # Get set of doc_ids for given token
        # return as a list sorted ascending
        normalized_term = self._normalize(term)
        return sorted(self.index.get(normalized_term, set()))

    def build(self, movies: list[dict]) -> None:
        #  iterate over all the movies and add them to both the index and the docmap.
        print("Building inverse index...")
        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"

            # build docmap
            self.docmap[doc_id] = movie
            # build inverse index
            self._add_document(doc_id=doc_id, text=text)
        print("Done!")
        print("Saving index and docmap to pickle files")
        self.save()

    def save(self) -> None:
        # use pickle.dump method to save index and docmap to files
        index_path = "./cache/index.pkl"
        docmap_path = "./cache/docmap.pkl"
        if not os.path.isdir("./cache"):
            os.mkdir("./cache")
        with open(index_path, "wb") as index_fp:
            pickle.dump(self.index, index_fp)
        with open(docmap_path, "wb") as docmap_fp:
            pickle.dump(self.docmap, docmap_fp)
        """
        # JSON cannot encode sets directly, so serialize posting lists as sorted arrays.
        index_for_json = {token: sorted(doc_ids) for token, doc_ids in self.index.items()}
        with open("./cache/index.json", "w") as ifp:
            json.dump(index_for_json, ifp, ensure_ascii=False, indent=2)
        with open("./cache/docmap.json", "w") as dfp:
            json.dump(self.docmap, dfp, ensure_ascii=False, indent=2)
        """
