from __future__ import annotations

import pytest

import cli.lib.hybrid_search as hybrid_mod


class _FakeSemanticSearch:
    def __init__(self):
        self.loaded_docs = None
        self.search_args = None

    def load_or_create_chunk_embeddings(self, documents):
        self.loaded_docs = documents
        return [{"id": 1}]

    def search_chunks(self, query, limit):
        self.search_args = (query, limit)
        return [
            {"id": 2, "title": "B", "document": "semantic b", "score": 0.1},
            {"id": 3, "title": "C", "document": "semantic c", "score": 0.9},
        ]


class _FakeIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {
            1: {"description": "doc one"},
            2: {"description": "doc two"},
            3: {"description": "doc three"},
        }
        self.loaded = False
        self.built_docs = None
        self.saved = False

    def build(self, documents):
        self.built_docs = documents
        self.index = {"ready": [1]}

    def save(self):
        self.saved = True

    def load(self):
        self.loaded = True

    def bm25_search(self, query, limit):
        return [(1, "A", 2.0), (2, "B", 1.0)]


def test_normalize_handles_mixed_values():
    assert hybrid_mod.normalize([10.0, 7.0, 2.0, 0.0]) == [1.0, 0.7, 0.2, 0.0]


def test_normalize_all_same_value():
    assert hybrid_mod.normalize([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]


def test_hybrid_score():
    assert hybrid_mod.hybrid_score(1.0, 0.0, alpha=0.25) == 0.25


def test_alpha_bar_labels():
    assert "50/50 Split" in hybrid_mod.alpha_bar(0.5)
    assert "100% Keyword" in hybrid_mod.alpha_bar(1.0)
    assert "100% Semantic" in hybrid_mod.alpha_bar(0.0)


def test_print_helpers(capsys):
    hybrid_mod.pretty_print([1.0, 0.5])
    hybrid_mod.print_hybrid_results(
        [
            (
                1,
                {
                    "title": "A",
                    "hybrid_score": 0.9,
                    "bm25_normalized_score": 1.0,
                    "semantic_normalized_score": 0.8,
                    "document": "x",
                },
            ),
        ],
        n=1,
    )
    hybrid_mod.print_rrf_results(
        [
            (2, {"title": "B", "rrf_score": 0.1, "bm25_rank": 1, "semantic_rank": 2, "document": "y"}),
        ],
        n=1,
    )
    out = capsys.readouterr().out
    assert "Normalized values:" in out
    assert "Hybrid Score:" in out
    assert "RRF Score:" in out


def test_hybrid_search_init_builds_cache_when_index_empty(monkeypatch):
    fake_sem = _FakeSemanticSearch()
    fake_idx = _FakeIndex()
    docs = [{"id": 1, "title": "A", "description": "doc one"}]

    class _FakeInvClass:
        @classmethod
        def from_cache(cls):
            return fake_idx

    monkeypatch.setattr(hybrid_mod, "ChunkedSemanticSearch", lambda: fake_sem)
    monkeypatch.setattr(hybrid_mod, "InvertedIndex", _FakeInvClass)

    hs = hybrid_mod.HybridSearch(docs)

    assert hs.documents == docs
    assert fake_sem.loaded_docs == docs
    assert fake_idx.built_docs == docs
    assert fake_idx.saved is True


def test_bm25_search_calls_index_load(monkeypatch):
    fake_sem = _FakeSemanticSearch()
    fake_idx = _FakeIndex()

    class _FakeInvClass:
        @classmethod
        def from_cache(cls):
            return fake_idx

    monkeypatch.setattr(hybrid_mod, "ChunkedSemanticSearch", lambda: fake_sem)
    monkeypatch.setattr(hybrid_mod, "InvertedIndex", _FakeInvClass)

    hs = hybrid_mod.HybridSearch([{"id": 1, "title": "A", "description": "doc one"}])
    result = hs._bm25_search("bear", 5)

    assert fake_idx.loaded is True
    assert result == [(1, "A", 2.0), (2, "B", 1.0)]


def test_weighted_search_builds_scores_and_prints(monkeypatch):
    hs = object.__new__(hybrid_mod.HybridSearch)
    hs.documents = [{"id": 1, "title": "A", "description": "doc one"}]
    hs.semantic_search = _FakeSemanticSearch()
    hs.idx = _FakeIndex()
    hs._bm25_search = lambda _query, _limit: [(1, "A", 2.0), (2, "B", 1.0)]

    captured = {"results": None, "limit": None}
    monkeypatch.setattr(
        hybrid_mod, "print_hybrid_results", lambda results, n: captured.update({"results": results, "limit": n})
    )

    hs.weighted_search("bear", alpha=0.75, limit=2)

    assert captured["limit"] == 2
    result_ids = [doc_id for doc_id, _row in captured["results"]]
    assert result_ids == [1, 3, 2]
    assert hs.semantic_search.search_args == ("bear", 500)


def test_rrf_search_combines_and_sorts(monkeypatch):
    hs = object.__new__(hybrid_mod.HybridSearch)
    hs.documents = [{"id": 1, "title": "A", "description": "doc one"}]
    hs.semantic_search = _FakeSemanticSearch()
    hs.idx = _FakeIndex()
    hs._bm25_search = lambda _query, _limit: [(1, "A", 2.0), (2, "B", 1.0)]

    results = hs.rrf_search("bear", k=60, gather_limit=100)

    assert results[0][0] == 2
    assert results[0][1]["title"] == "B"
    assert results[0][1]["document"] == "doc two"
    assert hs.semantic_search.search_args == ("bear", 100)


def test_rrf_score():
    hs = object.__new__(hybrid_mod.HybridSearch)
    assert hs.rrf_score(1, 60) == pytest.approx(1 / 61)


def test_enhanced_query_routes_calls(monkeypatch):
    hs = object.__new__(hybrid_mod.HybridSearch)

    class _FakeGeminiClient:
        def spell_check(self, query):
            return type("Resp", (), {"text": f"spell:{query}"})()

        def rewrite_query(self, query):
            return type("Resp", (), {"text": f"rewrite:{query}"})()

        def expand_query(self, query):
            return type("Resp", (), {"text": f"expand:{query}"})()

    monkeypatch.setattr(hybrid_mod, "GeminiClient", _FakeGeminiClient)

    assert hs.enhanced_query("bear", "spell").text == "spell:bear"
    assert hs.enhanced_query("bear", "rewrite").text == "rewrite:bear"
    assert hs.enhanced_query("bear", "expand").text == "expand:bear"

    with pytest.raises(NotImplementedError):
        hs.enhanced_query("bear", "unsupported")
