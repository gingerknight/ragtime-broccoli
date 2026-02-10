# keyword search 
"""
* Load movies.json into dictionary
* Iterate over movies section and match title field
* Order return by IDs ascending
"""

from typing import Any
from operator import itemgetter
import string

from .utils.helpers import DEFAULT_MAX_TITLES, load_stopwords, load_movies

class MovieSearch:
    def __init__(self, movies: list[dict[str, Any]], stopwords: set[str]):
        self._movies = movies
        self._stopwords = stopwords or set()

    @classmethod
    def from_file(cls, movies_path: str = "data/movies.json", stop_path: str = "data/stopwords.txt") -> "MovieSearch":
        movies = load_movies(movies_path)
        stopwords = load_stopwords(stop_path)
        return cls(movies=movies, stopwords=stopwords)

    def sample_data(self, n: int = DEFAULT_MAX_TITLES) -> None:
        print("Printing Sample Titles and Movie Ids:...")
        for movie in self._movies[:n]:
            movie_id = movie.get("id")
            title = movie.get("title")
            print(f"{movie_id}\t{title}")

    def find_titles(self, query: str) -> list[tuple[int|str]]:
        matched_titles = []
        q_token = self._tokenize_query(self._normalize(query))
        q_token_stops = self._drop_stopwords(q_token)

        for movie in self._movies:
            title = movie.get("title", "")
            m_id = movie.get("id")
            if any(q in self._normalize(title) for q in q_token_stops):
                matched_titles.append((title, m_id))
        
        # in place sort list of tuples by second element (id)
        matched_titles.sort(key=itemgetter(1))
        # grab titles
        found_titles = [x[0] for x in matched_titles]
            
        return found_titles
    
    def print_results(self, titles: list, n: int = DEFAULT_MAX_TITLES) -> None:
        for num, title in enumerate(titles[:n]):
            print(f"{num+1}. {title}")

    def _normalize(self, title: str) -> str:
        # remove stardard punctuation from movie or query title string
        # drop case to lower
        drop_punc_title = title.translate(str.maketrans('','',string.punctuation))
        return drop_punc_title.lower()
    
    def _drop_stopwords(self, tokens: list[str]) -> list[str]:
        # drop stop words from user/title list
        return list( set(tokens) - self._stopwords)
    
    
    def _tokenize_query(self, query: str) -> list[str]:
        # tokenize query string on whitespace
        # no stemming
        return query.split()
