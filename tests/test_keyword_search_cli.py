from __future__ import annotations

from argparse import Namespace

from errors.exception_handling import SearchEngineError

import cli.keyword_search_cli as cli_mod


class _FakeMovieSearch:
    def __init__(self):
        self._movies = [{"id": 1, "title": "Any", "description": "Any"}]
        self.printed = None

    def find_titles(self, query, idx_cache, docmap_cache):
        self.last_query = query
        self.last_idx = idx_cache
        self.last_docmap = docmap_cache
        return ["Brave"]

    def print_results(self, titles):
        self.printed = titles


def test_main_search_path_runs_query(monkeypatch, capsys):
    fake_ms = _FakeMovieSearch()

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command="search", query="brave"),
    )
    monkeypatch.setattr(cli_mod.MovieSearch, "from_file", classmethod(lambda _cls: fake_ms))
    monkeypatch.setattr(
        cli_mod.InvertedIndex,
        "from_cache",
        classmethod(
            lambda _cls: type(
                "LoadedInv",
                (),
                {"index": {"brave": [1]}, "docmap": {1: {"title": "Brave"}}},
            )()
        ),
    )

    rc = cli_mod.main()

    out = capsys.readouterr().out
    assert rc is None
    assert "Loading cache files..." in out
    assert "Searching for: brave" in out
    assert fake_ms.printed == ["Brave"]


def test_main_build_path_calls_build(monkeypatch):
    fake_ms = _FakeMovieSearch()
    build_calls = {"count": 0, "movies": None}

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command="build", query=None),
    )
    monkeypatch.setattr(cli_mod.MovieSearch, "from_file", classmethod(lambda _cls: fake_ms))

    class _FakeInv:
        def build(self, movies):
            build_calls["count"] += 1
            build_calls["movies"] = movies

    monkeypatch.setattr(cli_mod, "InvertedIndex", _FakeInv)

    rc = cli_mod.main()

    assert rc is None
    assert build_calls["count"] == 1
    assert build_calls["movies"] == fake_ms._movies


def test_main_returns_2_when_loading_movies_fails(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command="search", query="x"),
    )
    monkeypatch.setattr(
        cli_mod.MovieSearch,
        "from_file",
        classmethod(lambda _cls: (_ for _ in ()).throw(RuntimeError("boom"))),
    )

    rc = cli_mod.main()

    out = capsys.readouterr().out
    assert rc == 2
    assert "Unable to load data file" in out


def test_main_returns_2_when_cache_load_fails(monkeypatch, capsys):
    fake_ms = _FakeMovieSearch()

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command="search", query="brave"),
    )
    monkeypatch.setattr(cli_mod.MovieSearch, "from_file", classmethod(lambda _cls: fake_ms))
    monkeypatch.setattr(
        cli_mod.InvertedIndex,
        "from_cache",
        classmethod(lambda _cls: (_ for _ in ()).throw(SearchEngineError("cache fail"))),
    )

    rc = cli_mod.main()

    out = capsys.readouterr().out
    assert rc == 2
    assert "Error: cache fail" in out


def test_main_tf_path_prints_frequency(monkeypatch, capsys):
    fake_ms = _FakeMovieSearch()

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command="tf", id=424, term="trapper"),
    )
    monkeypatch.setattr(cli_mod.MovieSearch, "from_file", classmethod(lambda _cls: fake_ms))

    class _FakeInv:
        def get_tf(self, doc_id, term):
            return 4

    monkeypatch.setattr(cli_mod.InvertedIndex, "from_cache", classmethod(lambda _cls: _FakeInv()))

    rc = cli_mod.main()

    out = capsys.readouterr().out
    assert rc is None
    assert "Fetching term frequency with params 424 -- trapper" in out
    assert "Term frequency for trapper --> 4" in out
