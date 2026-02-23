from __future__ import annotations

from argparse import Namespace

import cli.semantic_search_cli as cli_mod


class _FakeSemanticSearch:
    def __init__(self):
        self.loaded_docs = None
        self.search_args = None

    def load_or_create_embeddings(self, documents):
        self.loaded_docs = documents

    def search(self, query, limit):
        self.search_args = (query, limit)
        return [
            (0.91, "Brave", "A princess fights fate."),
            (0.82, "Moana", "A girl voyages across the ocean."),
        ]


def test_main_verify_invokes_verify_model(monkeypatch):
    calls = {"count": 0}

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command="verify"),
    )
    monkeypatch.setattr(cli_mod, "verify_model", lambda: calls.__setitem__("count", calls["count"] + 1))

    rc = cli_mod.main()

    assert rc is None
    assert calls["count"] == 1


def test_main_embed_text_invokes_embed_text(monkeypatch):
    captured = {"value": None}

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command="embed_text", input="hello world"),
    )
    monkeypatch.setattr(cli_mod, "embed_text", lambda text: captured.__setitem__("value", text))

    rc = cli_mod.main()

    assert rc is None
    assert captured["value"] == "hello world"


def test_main_embedquery_invokes_embed_query_text(monkeypatch):
    captured = {"value": None}

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command="embedquery", query="adventure"),
    )
    monkeypatch.setattr(cli_mod, "embed_query_text", lambda query: captured.__setitem__("value", query))

    rc = cli_mod.main()

    assert rc is None
    assert captured["value"] == "adventure"


def test_main_verify_embeddings_invokes_verify_embeddings(monkeypatch):
    calls = {"count": 0}

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command="verify_embeddings"),
    )
    monkeypatch.setattr(
        cli_mod,
        "verify_embeddings",
        lambda: calls.__setitem__("count", calls["count"] + 1),
    )

    rc = cli_mod.main()

    assert rc is None
    assert calls["count"] == 1


def test_main_search_loads_movies_and_prints_results(monkeypatch, capsys):
    fake_search = _FakeSemanticSearch()
    fake_movies = [{"id": 1, "title": "Brave", "description": "A princess fights fate."}]

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command="search", user_query="princess", limit=2),
    )
    monkeypatch.setattr(cli_mod, "SemanticSearch", lambda: fake_search)
    monkeypatch.setattr(cli_mod, "load_movies", lambda: fake_movies)

    rc = cli_mod.main()

    out = capsys.readouterr().out
    assert rc is None
    assert fake_search.loaded_docs == fake_movies
    assert fake_search.search_args == ("princess", 2)
    assert "1. Brave (score: 0.91)" in out
    assert "2. Moana (score: 0.82)" in out


def test_main_chunk_uses_chunking_and_pretty_display(monkeypatch):
    seen = {"chunk": None, "display": None}

    monkeypatch.setattr(
        cli_mod.argparse.ArgumentParser,
        "parse_args",
        lambda _self: Namespace(command="chunk", text="a b c d", chunk_size=2, overlap=1),
    )

    def _fake_chunk(text, chunk_size, overlap):
        seen["chunk"] = (text, chunk_size, overlap)
        return ["a b", "b c", "c d"]

    def _fake_display(chunks, text_length):
        seen["display"] = (chunks, text_length)

    monkeypatch.setattr(cli_mod, "size_defined_chunking", _fake_chunk)
    monkeypatch.setattr(cli_mod, "pretty_display_chunks", _fake_display)

    rc = cli_mod.main()

    assert rc is None
    assert seen["chunk"] == ("a b c d", 2, 1)
    assert seen["display"] == (["a b", "b c", "c d"], len("a b c d"))


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
