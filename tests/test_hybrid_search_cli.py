from __future__ import annotations

from argparse import Namespace

import cli.hybrid_search_cli as cli_mod


class _FakeHybridSearch:
    def __init__(self, movies):
        self.movies = movies
        self.weighted_args = None
        self.rrf_args = None
        self.enhanced_args = None

    def weighted_search(self, query, alpha, limit):
        self.weighted_args = (query, alpha, limit)

    def enhanced_query(self, *, choice, query):
        self.enhanced_args = (choice, query)
        return type("Response", (), {"text": "enhanced query"})()

    def rrf_search(self, query, k, limit):
        self.rrf_args = (query, k, limit)
        return [
            (
                10,
                {
                    "title": "string",
                    "bm25_rank": 4,
                    "bm25_rrf_score": 0.01234,
                    "semantic_rank": 5,
                    "document": "Movies are cool yo",
                    "rrf_score": 0.03456,
                    "semantic_rrf_score": 0.9876,
                },
            )
        ]


def test_main_normalize_path(monkeypatch):
    captured = {"scores": None, "pretty": None}

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command="normalize", scores=[4.0, 2.0, 1.0]),
    )
    monkeypatch.setattr(
        cli_mod,
        "normalize",
        lambda scores: captured.__setitem__("scores", scores) or [1.0, 0.33, 0.0],
    )
    monkeypatch.setattr(cli_mod, "pretty_print", lambda values: captured.__setitem__("pretty", values))

    rc = cli_mod.main()

    assert rc is None
    assert captured["scores"] == [4.0, 2.0, 1.0]
    assert captured["pretty"] == [1.0, 0.33, 0.0]


def test_main_weighted_search_path(monkeypatch):
    fake_movies = [{"id": 1, "title": "A", "description": "Desc"}]
    fake_hybrid = _FakeHybridSearch(fake_movies)

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command="weighted-search", query="bear", alpha=0.7, limit=3),
    )
    monkeypatch.setattr(cli_mod, "load_movies", lambda: fake_movies)
    monkeypatch.setattr(cli_mod, "HybridSearch", lambda movies: fake_hybrid)

    rc = cli_mod.main()

    assert rc is None
    assert fake_hybrid.movies == fake_movies
    assert fake_hybrid.weighted_args == ("bear", 0.7, 3)


def test_main_rrf_enhance_path(monkeypatch):
    fake_movies = [{"id": 1, "title": "A", "description": "Desc"}]
    fake_hybrid = _FakeHybridSearch(fake_movies)

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(
            command="rrf-search",
            query="ber",
            enhance="spell",
            rerank_method=None,
            k=60,
            limit=2,
        ),
    )
    monkeypatch.setattr(cli_mod, "load_movies", lambda: fake_movies)
    monkeypatch.setattr(cli_mod, "HybridSearch", lambda movies: fake_hybrid)

    rc = cli_mod.main()

    assert rc is None
    assert fake_hybrid.enhanced_args == ("spell", "ber")
    assert fake_hybrid.rrf_args == ("enhanced query", 60, 500)


def test_main_rrf_rerank_path(monkeypatch):
    fake_movies = [{"id": 1, "title": "A", "description": "Desc"}]
    fake_hybrid = _FakeHybridSearch(fake_movies)

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(
            command="rrf-search",
            query="bear",
            enhance=None,
            rerank_method="individual",
            k=60,
            limit=2,
        ),
    )
    monkeypatch.setattr(cli_mod, "load_movies", lambda: fake_movies)
    monkeypatch.setattr(cli_mod, "HybridSearch", lambda movies: fake_hybrid)

    rc = cli_mod.main()

    assert rc is None
    assert fake_hybrid


def test_main_rrf_default_path(monkeypatch):
    fake_movies = [{"id": 1, "title": "A", "description": "Desc"}]
    fake_hybrid = _FakeHybridSearch(fake_movies)

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(
            command="rrf-search",
            rrf_mode="enhance",
            query="bear",
            enhance=None,
            rerank_method=None,
            k=42,
            limit=6,
        ),
    )
    monkeypatch.setattr(cli_mod, "load_movies", lambda: fake_movies)
    monkeypatch.setattr(cli_mod, "HybridSearch", lambda movies: fake_hybrid)

    rc = cli_mod.main()

    assert rc is None
    assert fake_hybrid.rrf_args == ("bear", 42, 6)


def test_main_unknown_command_prints_help(monkeypatch):
    printed = {"count": 0}

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command=None),
    )
    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "print_help",
        lambda _self: printed.__setitem__("count", printed["count"] + 1),
    )

    rc = cli_mod.main()

    assert rc is None
    assert printed["count"] == 1
