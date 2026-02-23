from __future__ import annotations

import json

import pytest

import cli.helpers as helpers


def test_load_stopwords_reads_configured_file(tmp_path, monkeypatch):
    stop_path = tmp_path / "stopwords.txt"
    stop_path.write_text("the\na\nand\n", encoding="utf-8")
    monkeypatch.setattr(helpers, "STOP_PATH", str(stop_path))

    words = helpers.load_stopwords()

    assert words == {"the", "a", "and"}


def test_load_movies_returns_movies_from_payload(tmp_path):
    movie_path = tmp_path / "movies.json"
    payload = {"movies": [{"id": 1, "title": "Brave", "description": "Pixar"}]}
    movie_path.write_text(json.dumps(payload), encoding="utf-8")

    movies = helpers.load_movies(str(movie_path))

    assert movies == payload["movies"]


def test_load_movies_raises_clear_error_on_missing_file(tmp_path):
    missing = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match=f"Missing file: {missing}"):
        helpers.load_movies(str(missing))


def test_load_movies_raises_clear_error_on_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match=f"Bad json in {bad}"):
        helpers.load_movies(str(bad))


def test_normalize_removes_punctuation_lowercases_removes_stops_and_stems(monkeypatch):
    monkeypatch.setattr(helpers, "load_stopwords", lambda: {"the", "and"})

    tokens = helpers.normalize("The CATS, and DOGS!!!")

    assert tokens == ["cat", "dog"]


def test_normalize_returns_empty_list_for_only_stopwords(monkeypatch):
    monkeypatch.setattr(helpers, "load_stopwords", lambda: {"the", "and"})

    tokens = helpers.normalize("The and the")

    assert tokens == []
