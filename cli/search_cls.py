# keyword search 
"""
* Load movies.json into dictionary
* Iterate over movies section and match title field
* Order return by IDs ascending
"""
import json
from typing import Any
from operator import itemgetter
import string

class MovieSearch:
    def __init__(self, movies: list[dict[str, Any]]):
        self._movies = movies

    @classmethod
    def from_file(cls, path: str = "data/movies.json") -> "MovieSearch":
        try:
            with open(path, 'r') as movie_json:
                payload = json.load(movie_json)

        except FileNotFoundError:
            raise FileNotFoundError("I couldn't find the file captain")
        except json.JSONDecodeError as e:
            raise ValueError("Bad json captain")
        movies = payload["movies"]
        return cls(movies)

    def sample_data(self, n: int = 5) -> None:
        print("Printing Sample Titles and Movie Ids:...")
        for movie in self._movies[:n]:
            movie_id = movie.get("id")
            title = movie.get("title")
            print(f"{movie_id}\t{title}")

    def find_titles(self, query: str) -> list[str]:
        found_titles: list[str] = []
        title_tuple = []
        query_lower_no_punc = self._remove_punctuation(query.lower())
        q_token = self._tokenize_query(query_lower_no_punc)

        for movie in self._movies:
            title = movie.get("title")
            m_id = movie.get("id")
            if any(q in self._remove_punctuation(title.casefold()) for q in q_token):
                title_tuple.append((title, m_id))
        
        # in place sort list of tuples by second element (id)
        title_tuple.sort(key=itemgetter(1))
        # grab titles
        found_titles = [x[0] for x in title_tuple]
            
        return self._truncate_data(found_titles)
    
    def _truncate_data(self, titles: list, n: int = 5) -> None:
        for num, title in enumerate(titles[:n]):
            print(f"{num+1}. {title}")

    def _remove_punctuation(self, title: str) -> str:
        # remove stardard punctuation from movie or query title string
        return title.translate(str.maketrans('','',string.punctuation))
    
    def _tokenize_query(self, query: str) -> list[str]:
        # tokenize query string on whitespace
        # no stemming
        return query.split()