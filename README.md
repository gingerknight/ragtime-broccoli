# RAG Search Engine (Lexical Baseline)

This project is a learning-first retrieval engine over a movie dataset. It currently implements a lexical pipeline with an inverted index, term frequencies, inverse document frequency, and simple TF-IDF scoring exposed through a CLI.

## Current Features
- Build an inverted index from `data/movies.json`
- Normalize text (lowercase, punctuation removal, stopword filtering, stemming)
- Search movie titles/descriptions by normalized tokens
- Query term frequency (TF) for a document-term pair
- Query inverse document frequency (IDF) for a term
- Compute TF-IDF for a document-term pair
- Pytest coverage for search/index/CLI flows

## Project Layout
- `cli/keyword_search_cli.py`: CLI entrypoint
- `cli/search_cls.py`: `MovieSearch` and `InvertedIndex`
- `cli/helpers.py`: normalization + file/cache constants
- `cli/errors/exception_handling.py`: custom exceptions
- `data/`: source dataset and stopwords
- `cache/`: generated index artifacts
- `tests/`: pytest test suite

## Requirements
- Python `>=3.13`
- `uv`

Install dependencies:
```bash
uv sync --dev
```

## CLI Usage
Run commands from project root (`rag-search-engine/`).

Build cache artifacts first:
```bash
uv run cli/keyword_search_cli.py build
```

Search:
```bash
uv run cli/keyword_search_cli.py search "brave bear"
```

Term frequency for one document/term:
```bash
uv run cli/keyword_search_cli.py tf 424 trapper
```

Inverse document frequency:
```bash
uv run cli/keyword_search_cli.py idf trapper
```

TF-IDF score:
```bash
uv run cli/keyword_search_cli.py tfidf 424 trapper
```

## Cache Artifacts
`build` writes pickle files under `cache/`:
- `index.pkl`: token -> set of doc IDs
- `docmap.pkl`: doc ID -> movie object
- `term_frequencies.pkl`: doc ID -> token frequency counter

## Running Tests
```bash
uv run pytest
```

Pytest is configured with:
- `--import-mode=importlib`
- `pythonpath = [".", "cli"]`

(from `pyproject.toml`)

## Notes
- The CLI currently imports modules as script-local imports (`from search_cls import ...`), so invoking commands as shown above (`uv run cli/keyword_search_cli.py ...`) is the expected runtime path.
- The retrieval/scoring implementation is intentionally simple and serves as a baseline for future BM25/semantic retrieval work.
