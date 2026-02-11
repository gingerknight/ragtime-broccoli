from __future__ import annotations

# Run with uv run pytest as module
from cli.search_cls import InvertedIndex, MovieSearch


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


def test_inverted_index_normalizes_keys_and_get_documents():
    inv = InvertedIndex()
    movies = [
        {"id": 2, "title": "Kaakha..Kaakha: The Police", "description": "A cop story."},
        {"id": 1, "title": "The Police Story", "description": "Another case."},
    ]

    inv.build(movies)

    assert "kaakhakaakha" in inv.index
    assert "kaakha..kaakha:" not in inv.index
    assert inv.get_documents("Police") == [1, 2]
    assert inv.get_documents("police.") == [1, 2]
