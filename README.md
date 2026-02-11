# Learnin RAG basics

A learning‑driven project that starts with classic keyword search and evolves into a full Retrieval‑Augmented Generation (RAG) pipeline. The focus is not just building features, but building them **correctly**: clear interfaces, testable components, and measurable tradeoffs.

**Status (Today)**
- Keyword search over movie titles
- Normalization, tokenization, stopword filtering
- CLI‑first interface
- Pytest coverage for core search behavior

**Why this project matters**
This codebase is a deliberate, incremental build. Each step is implemented simply first, then upgraded once a real limitation appears. This mirrors how production search systems are built: the constraints drive the architecture, not the other way around.

**What it will become**
- TF‑IDF and BM25 scoring
- Semantic search with embeddings
- Chunking + document segmentation
- Hybrid retrieval (lexical + semantic)
- Reranking pipelines
- LLM‑powered RAG responses

---

## Project Structure
- `cli/` command‑line interface and search logic
- `cli/utils/` helpers (data loading, stopwords, constants)
- `tests/` pytest tests
- `data/` datasets (movies, stopwords)

---

## Quick Start
1. Install dependencies
```bash
uv lock
uv sync --dev
```

2. Run search
```bash
uv run python -m cli.keyword_search_cli search "star wars"
```

3. Run tests
```bash
uv run pytest
```

---

## Engineering Principles (Portfolio‑Ready)
- **Separation of concerns**: data loading, preprocessing, search, and presentation are isolated.
- **Testability first**: search returns data; the CLI owns formatting.
- **Preprocessing pipeline**: normalization + tokenization + stopword filtering is explicit and reusable.
- **Incremental complexity**: each new feature justifies its added complexity and cost.

---

## Roadmap (High‑Level)
1. **TF‑IDF**
   - Build term frequency + inverse document frequency
   - Add scoring, ranking, and evaluation metrics
2. **BM25**
   - Replace TF‑IDF scoring with BM25
   - Compare precision/recall tradeoffs
3. **Chunking**
   - Support longer documents beyond titles
   - Introduce chunk boundaries and overlap strategies
4. **Semantic Search**
   - Embed documents and queries
   - Add vector similarity search
5. **Hybrid Search**
   - Combine lexical + semantic scores
   - Tune weights and evaluate
6. **Reranking**
   - Add a cross‑encoder for final ordering
   - Compare quality vs latency
7. **RAG Integration**
   - Retrieve context
   - Generate grounded responses with an LLM

---


