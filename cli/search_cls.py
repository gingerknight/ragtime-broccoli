# keyword search
"""
* Load movies.json into dictionary
* Iterate over movies section and match title field
* Order return by IDs ascending
"""

import json
import math
import os
import pickle
from collections import Counter
from typing import Any

from errors.exception_handling import (
    CacheIOError,
    DataLoadError,
    IndexBuildError,
    InvalidTerm,
)
from helpers import (
    BM25_B,
    BM25_K1,
    DEFAULT_MAX_TITLES,
    DOC_LENGTHS_PATH,
    DOCMAP_PATH,
    INDEX_PATH,
    TF_PATH,
    load_movies,
    normalize,
)


class MovieSearch:
    def __init__(self, movies: list[dict[str, Any]]):
        self._movies = movies

    @classmethod
    def from_file(cls) -> "MovieSearch":
        movies = load_movies()
        return cls(movies=movies)

    def sample_data(self, n: int = DEFAULT_MAX_TITLES) -> None:
        print("Printing Sample Titles and Movie Ids:...")
        for movie in self._movies[:n]:
            movie_id = movie.get("id")
            title = movie.get("title")
            print(f"{movie_id}\t{title}")

    def find_titles(
        self,
        query: str,
        idx_cache: dict[str, list[int] | set[int]],
        docmap_cache: dict[int | str, dict[str, Any]],
    ) -> list[str]:
        q_tokens = normalize(query)

        found_titles: list[str] = []
        inverse_idx_matches: list[int] = []
        for token in q_tokens:
            inverse_idx_matches.extend(idx_cache.get(token, ""))
        unique_ids = sorted(set(inverse_idx_matches))
        for doc_id in unique_ids:
            found_titles.append(docmap_cache[doc_id]["title"])
        return found_titles
        """
        matched_titles: list[tuple[str, int]] = []
        for movie in self._movies:
            title = movie.get("title", "")
            m_id = movie.get("id")
            if any(q in self._normalize(title) for q in q_stemmed):
                matched_titles.append((title, m_id))

        matched_titles.sort(key=itemgetter(1))
        return [x[0] for x in matched_titles]
        """

    def print_results(self, titles: list, n: int = DEFAULT_MAX_TITLES) -> None:
        for num, title in enumerate(titles[:n]):
            print(f"{num + 1}. {title}")


class InvertedIndex:
    """
    Inverted index (also referred to as a postings list, postings file, or inverted file) is a database index storing a mapping from content, such as words or numbers, to its locations in a table, or in a document or a set of documents (named in contrast to a forward index, which maps from documents to content).[1] The purpose of an inverted index is to allow fast full-text searches, at a cost of increased processing when a document is added to the database.[2] The inverted file may be the database file itself, rather than its index. It is the most popular data structure used in document retrieval systems,[3] used on a large scale for example in search engines. Additionally, several significant general-purpose mainframe-based database management systems have used inverted list architectures, including ADABAS, DATACOM/DB, and Model 204.
    """

    def __init__(self):
        # dictionary mapping tokens to sets of doc Ids
        self.index: dict[str, set[int]] = {}
        # dictionary mapping doc Ids to their full doc objects
        self.docmap: dict[int, dict[str, Any]] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

    @classmethod
    def from_cache(cls) -> "InvertedIndex":
        idx_cache, docmap_cache, tf_cache, doclength_cache = cls.load()
        inv = cls()
        inv.index = idx_cache
        inv.docmap = docmap_cache
        inv.term_frequencies = tf_cache
        inv.doc_lengths = doclength_cache
        return inv

    def get_documents(self, term: str) -> list[int]:
        # Get set of doc_ids for given token
        # return as a list sorted ascending
        normalized_term = term.lower()
        return sorted(self.index.get(normalized_term, set()))

    def build(self, movies: list[dict]) -> None:
        #  iterate over all the movies and add them to both the index and the docmap.
        print("Building inverse index...")
        try:
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
        except KeyError as e:
            raise IndexBuildError(f"Missing required movie field: {e}") from e
        except OSError as e:
            raise CacheIOError(f"Failed writing cache files: {e}") from e

    def save(self) -> None:
        # use pickle.dump method to save index and docmap to files
        if not os.path.isdir("./cache"):
            os.mkdir("./cache")
        with open(INDEX_PATH, "wb") as index_fp:
            pickle.dump(self.index, index_fp)
        with open(DOCMAP_PATH, "wb") as docmap_fp:
            pickle.dump(self.docmap, docmap_fp)
        with open(TF_PATH, "wb") as tf_fp:
            pickle.dump(self.term_frequencies, tf_fp)
        with open(DOC_LENGTHS_PATH, "wb") as doclength_fp:
            pickle.dump(self.doc_lengths, doclength_fp)

    def _add_document(self, doc_id: int, text: str) -> None:
        cnt = Counter()
        tokens = normalize(text)
        self.doc_lengths[doc_id] = len(tokens)
        for word in tokens:
            self.index.setdefault(word, set()).add(doc_id)
            cnt[word] += 1

        # add to counter dictionary
        self.term_frequencies[doc_id] = cnt

    def __get_avg_doc_length(self) -> float:
        # calculate and return average document length across all documents
        lengths = self.doc_lengths.values()
        return (sum(lengths) / len(lengths)) if self.doc_lengths else 0.0

    def get_tf(self, doc_id, term) -> int:
        # return the times the token term appears in the document with given ID
        token = normalize(term)
        if len(token) > 1:
            raise InvalidTerm("Expected sinle word term, not multiple tokens")
        # sets num to 0 if term doesn't appear in Counter iterable
        num = self.term_frequencies[doc_id][token[0]]
        return num

    def calculate_idf(self, term) -> float:
        # calculate the idf using in-memory index and docmap
        token = normalize(term)[0]
        num_docs = len(self.docmap)
        occurance = self.index.get(token, set())
        idf = math.log((int(num_docs) + 1) / (len(occurance) + 1))
        return idf

    def get_bm25_idf(self, term: str) -> float:
        # calculate the bm25 of a normalized string and return value
        # log((N - df + 0.5) / (df + 0.5) + 1)
        token = normalize(term)
        if len(token) > 1:
            raise InvalidTerm("Expected sinle word term, not multiple tokens")
        num_docs = len(self.docmap)
        occurance = self.index.get(token[0], set())
        bm25 = math.log((int(num_docs) - len(occurance) + 0.5) / (len(occurance) + 0.5) + 1)
        return bm25

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        # length normalization factor
        # length_norm = 1 - b + b * (doc_length / avg_doc_length)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length == 0:
            return 0.0
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        num = self.get_tf(doc_id, term)
        bm25_tf = (num * (k1 + 1)) / (num + k1 * length_norm)
        return bm25_tf

    def _debug_cache(self) -> None:
        # For Dev: debug cache contents and structure
        # JSON cannot encode sets directly, so serialize posting lists as sorted arrays.
        index_for_json = {token: sorted(doc_ids) for token, doc_ids in self.index.items()}
        with open("./cache/index.json", "w") as ifp:
            json.dump(index_for_json, ifp, ensure_ascii=False, indent=2)
        with open("./cache/docmap.json", "w") as dfp:
            json.dump(self.docmap, dfp, ensure_ascii=False, indent=2)

    @staticmethod
    def load():
        # Load pickle cache files from disk
        try:
            with open(INDEX_PATH, "rb") as ifp:
                index_cache = pickle.load(ifp)
            with open(DOCMAP_PATH, "rb") as rfp:
                docmap_cache = pickle.load(rfp)
            with open(TF_PATH, "rb") as tfp:
                tf_cache = pickle.load(tfp)
            with open(DOC_LENGTHS_PATH, "rb") as dfp:
                doclength_cache = pickle.load(dfp)
        except FileNotFoundError as e:
            raise DataLoadError(f"Unable to load cache files: {e}") from e
        return index_cache, docmap_cache, tf_cache, doclength_cache
