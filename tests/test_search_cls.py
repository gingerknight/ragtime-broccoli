from __future__ import annotations

import pytest

from cli.search_cls import InvertedIndex, MovieSearch
from errors.exception_handling import DataLoadError, IndexBuildError, InvalidTerm


def make_movies():
    return [
        {"id": 2, "title": "Star Wars", "description": "Space opera"},
        {"id": 1, "title": "Star Trek", "description": "Space crew"},
        {"id": 3, "title": "The Matrix", "description": "Simulation"},
    ]


def test_find_titles_orders_by_id_and_matches_tokens(monkeypatch):
    ms = MovieSearch(make_movies())
    idx_cache = {"star": [2, 1]}
    docmap_cache = {1: {"title": "Star Trek"}, 2: {"title": "Star Wars"}}

    monkeypatch.setattr("cli.search_cls.normalize", lambda q: ["star"])

    titles = ms.find_titles("star", idx_cache=idx_cache, docmap_cache=docmap_cache)

    assert titles == ["Star Trek", "Star Wars"]


def test_find_titles_merges_multiple_tokens_without_duplicates(monkeypatch):
    ms = MovieSearch(make_movies())
    idx_cache = {
        "space": [2, 1],
        "crew": [1],
    }
    docmap_cache = {
        1: {"title": "Star Trek"},
        2: {"title": "Star Wars"},
    }

    monkeypatch.setattr("cli.search_cls.normalize", lambda q: ["space", "crew"])

    titles = ms.find_titles("space crew", idx_cache=idx_cache, docmap_cache=docmap_cache)

    assert titles == ["Star Trek", "Star Wars"]


def test_print_results_outputs_numbered_list(capsys):
    ms = MovieSearch(make_movies())
    ms.print_results(["A", "B"], n=2)
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["1. A", "2. B"]


def test_inverted_index_build_populates_and_save_called(monkeypatch):
    inv = InvertedIndex()
    movies = [
        {"id": 2, "title": "Kaakha..Kaakha: The Police", "description": "A cop story."},
        {"id": 1, "title": "The Police Story", "description": "Another case."},
    ]

    save_calls = {"count": 0}
    monkeypatch.setattr("cli.search_cls.normalize", lambda text: text.lower().split())
    monkeypatch.setattr(inv, "save", lambda: save_calls.__setitem__("count", save_calls["count"] + 1))

    inv.build(movies)

    assert inv.docmap[1]["title"] == "The Police Story"
    assert inv.get_documents("police") == [1, 2]
    assert save_calls["count"] == 1


def test_inverted_index_build_raises_for_missing_fields(monkeypatch):
    inv = InvertedIndex()
    monkeypatch.setattr("cli.search_cls.normalize", lambda text: text.lower().split())

    with pytest.raises(IndexBuildError):
        inv.build([{"id": 1, "title": "Missing Description"}])


def test_load_raises_data_load_error_when_cache_missing(monkeypatch):
    def raise_missing(*_args, **_kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr("builtins.open", raise_missing)

    with pytest.raises(DataLoadError):
        InvertedIndex.load()


def test_get_tf_returns_normalized_token_count(monkeypatch):
    inv = InvertedIndex()
    inv.term_frequencies = {424: {"trapper": 4, "bear": 1}}

    monkeypatch.setattr("cli.search_cls.normalize", lambda term: ["trapper"])

    assert inv.get_tf(424, "trappers") == 4


def test_get_tf_raises_for_multiword_term(monkeypatch):
    inv = InvertedIndex()
    inv.term_frequencies = {1: {"brave": 1}}

    monkeypatch.setattr("cli.search_cls.normalize", lambda term: ["one", "two"])

    with pytest.raises(InvalidTerm):
        inv.get_tf(1, "one two")
