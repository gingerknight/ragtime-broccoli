# keyword search
"""
* Load movies.json into dictionary
* Iterate over movies section and match title field
* Order return by IDs ascending
"""

from typing import Any, Dict, List
import pickle
import os
import json


from helpers import (
    normalize,
    DEFAULT_MAX_TITLES,
    load_movies,
    INDEX_PATH,
    DOCMAP_PATH,
)
from errors.exception_handling import DataLoadError, IndexBuildError, CacheIOError


class MovieSearch:
    def __init__(self, movies: List[Dict[str, Any]]):
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

    """
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
    """


class InvertedIndex:
    """
    Inverted index (also referred to as a postings list, postings file, or inverted file) is a database index storing a mapping from content, such as words or numbers, to its locations in a table, or in a document or a set of documents (named in contrast to a forward index, which maps from documents to content).[1] The purpose of an inverted index is to allow fast full-text searches, at a cost of increased processing when a document is added to the database.[2] The inverted file may be the database file itself, rather than its index. It is the most popular data structure used in document retrieval systems,[3] used on a large scale for example in search engines. Additionally, several significant general-purpose mainframe-based database management systems have used inverted list architectures, including ADABAS, DATACOM/DB, and Model 204.
    """

    def __init__(self):
        # dictionary mapping tokens to sets of doc Ids
        self.index: Dict[str, set[int]] = {}
        # dictionary mapping doc Ids to their full doc objects
        self.docmap: Dict[int, Dict[str, Any]] = {}

    def _add_document(self, doc_id: int, text: str):
        for word in normalize(text):
            self.index.setdefault(word, set()).add(doc_id)

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

    def _debug_cache(self) -> None:
        # For Dev: debug cache contents and structure
        # JSON cannot encode sets directly, so serialize posting lists as sorted arrays.
        index_for_json = {
            token: sorted(doc_ids) for token, doc_ids in self.index.items()
        }
        with open("./cache/index.json", "w") as ifp:
            json.dump(index_for_json, ifp, ensure_ascii=False, indent=2)
        with open("./cache/docmap.json", "w") as dfp:
            json.dump(self.docmap, dfp, ensure_ascii=False, indent=2)

    @staticmethod
    def load() -> tuple[dict[str, set[int]], dict[int, dict[str, Any]]]:
        # Load pickle cache files from disk
        try:
            with open(INDEX_PATH, "rb") as ifp:
                index_cache = pickle.load(ifp)
            with open(DOCMAP_PATH, "rb") as rfp:
                docmap_cache = pickle.load(rfp)
        except FileNotFoundError as e:
            raise DataLoadError(f"Unable to load cache files: {e}") from e
        return index_cache, docmap_cache
