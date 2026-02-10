from __future__ import annotations

# Run with uv run pytest as module
from cli.search_cls import MovieSearch


def make_movies():
    return [
        {"id": 2, "title": "Star Wars"},
        {"id": 1, "title": "Star Trek"},
        {"id": 3, "title": "The Matrix"},
    ]


def test_find_titles_orders_by_id_and_matches_tokens():
    ms = MovieSearch(make_movies(), stopwords=set())
    titles = ms.find_titles("star")
    assert titles == ["Star Trek", "Star Wars"]


def test_find_titles_removes_stopwords():
    ms = MovieSearch(make_movies(), stopwords={"the"})
    titles = ms.find_titles("the matrix")
    assert titles == ["The Matrix"]


def test_print_results_outputs_numbered_list(capsys):
    ms = MovieSearch(make_movies(), stopwords=set())
    ms.print_results(["A", "B"], n=2)
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["1. A", "2. B"]
