from __future__ import annotations

import numpy as np
import pytest

import cli.lib.semantic_search as sem_mod


class _FakeSentenceTransformer:
    def __init__(self, model_name, cache_folder=None, local_files_only=None):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.local_files_only = local_files_only
        self.max_seq_length = 256

    def encode(self, payload, show_progress_bar=False):
        if isinstance(payload, list):
            if payload and isinstance(payload[0], str):
                if len(payload) == 1:
                    return np.array([[1.0, 2.0, 3.0]])
                return np.array([[float(i), float(i + 1)] for i in range(len(payload))])
        raise TypeError("unexpected payload")


def _make_docs():
    return [
        {"id": 1, "title": "Brave", "description": "Princess adventure"},
        {"id": 2, "title": "Moana", "description": "Ocean journey"},
    ]


def test_cosine_similarity_handles_normal_and_zero_vectors():
    assert sem_mod.cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert sem_mod.cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_size_defined_chunking_with_overlap_and_guardrails():
    text = "one two three four five"
    chunks = sem_mod.size_defined_chunking(text, chunk_size=3, overlap=1)

    assert chunks == ["one two three", "three four five"]

    with pytest.raises(ValueError):
        sem_mod.size_defined_chunking(text, chunk_size=3, overlap=3)

    with pytest.raises(ValueError):
        sem_mod.size_defined_chunking(text, chunk_size=3, overlap=-1)


def test_semantic_chunking_groups_sentences_and_applies_overlap():
    text = (
        "Ted is a 2012 comedy film directed by Seth MacFarlane. "
        "The story follows John Bennett and his magical teddy bear. "
        "The film explores themes of friendship and growing up. "
        "John must choose between his relationship and Ted."
    )

    chunks = sem_mod.size_defined_chunking(text, chunk_size=2, overlap=1, semantic=True)

    assert chunks == [
        "Ted is a 2012 comedy film directed by Seth MacFarlane. The story follows John Bennett and his magical teddy bear.",
        "The story follows John Bennett and his magical teddy bear. The film explores themes of friendship and growing up.",
        "The film explores themes of friendship and growing up. John must choose between his relationship and Ted.",
    ]


def test_semantic_chunking_rejects_invalid_overlap():
    text = "One. Two. Three."

    with pytest.raises(ValueError):
        sem_mod.size_defined_chunking(text, chunk_size=2, overlap=2, semantic=True)

    with pytest.raises(ValueError):
        sem_mod.size_defined_chunking(text, chunk_size=2, overlap=-1, semantic=True)


def test_pretty_display_chunks_prints_expected_format(capsys):
    sem_mod.pretty_display_chunks(["a b", "c d"], text_length=7)
    out = capsys.readouterr().out.splitlines()

    assert out == ["Chunking 7 characters", "1. a b", "2. c d"]


def test_generate_embedding_rejects_blank_and_returns_vector(monkeypatch, tmp_path):
    monkeypatch.setattr(sem_mod, "SentenceTransformer", _FakeSentenceTransformer)
    search = sem_mod.SemanticSearch(cache_dir=tmp_path)

    vector = search.generate_embedding("hello")

    assert np.array_equal(vector, np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError):
        search.generate_embedding("   ")


def test_build_embeddings_populates_docs_map_and_saves(monkeypatch, tmp_path):
    monkeypatch.setattr(sem_mod, "SentenceTransformer", _FakeSentenceTransformer)
    search = sem_mod.SemanticSearch(cache_dir=tmp_path)
    docs = _make_docs()

    calls = {"count": 0}
    monkeypatch.setattr(search, "save", lambda: calls.__setitem__("count", calls["count"] + 1))

    embeddings = search.build_embeddings(docs)

    assert embeddings.shape == (2, 2)
    assert search.documents == docs
    assert search.document_map[1]["title"] == "Brave"
    assert calls["count"] == 1


def test_load_or_create_embeddings_loads_from_cache_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(sem_mod, "SentenceTransformer", _FakeSentenceTransformer)
    search = sem_mod.SemanticSearch(cache_dir=tmp_path)
    docs = _make_docs()

    cache_file = tmp_path / "movie_embeddings.npy"
    monkeypatch.setattr(sem_mod, "EMBEDDING_CACHE", str(cache_file))

    expected = np.array([[9.0, 8.0]])
    with open(cache_file, "wb") as fp:
        np.save(fp, expected)

    search.load_or_create_embeddings(docs)

    assert np.array_equal(search.embeddings, expected)
    assert search.document_map[2]["title"] == "Moana"


def test_load_or_create_embeddings_builds_when_cache_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(sem_mod, "SentenceTransformer", _FakeSentenceTransformer)
    search = sem_mod.SemanticSearch(cache_dir=tmp_path)
    docs = _make_docs()

    cache_file = tmp_path / "missing.npy"
    monkeypatch.setattr(sem_mod, "EMBEDDING_CACHE", str(cache_file))

    calls = {"count": 0}

    def _fake_build(documents):
        calls["count"] += 1
        assert documents == docs
        return np.array([[1.0, 2.0]])

    monkeypatch.setattr(search, "build_embeddings", _fake_build)

    search.load_or_create_embeddings(docs)

    assert calls["count"] == 1
    assert np.array_equal(search.embeddings, np.array([[1.0, 2.0]]))


def test_save_and_load_round_trip_embeddings(monkeypatch, tmp_path):
    monkeypatch.setattr(sem_mod, "SentenceTransformer", _FakeSentenceTransformer)
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(sem_mod, "EMBEDDING_CACHE", str(cache_dir / "movie_embeddings.npy"))

    search = sem_mod.SemanticSearch(cache_dir=tmp_path)
    search.embeddings = np.array([[3.0, 4.0], [5.0, 6.0]])

    # save() writes a relative ./cache folder; scope cwd to tmp path for isolation.
    monkeypatch.chdir(tmp_path)
    search.save()
    loaded = search.load()

    assert np.array_equal(loaded, np.array([[3.0, 4.0], [5.0, 6.0]]))
