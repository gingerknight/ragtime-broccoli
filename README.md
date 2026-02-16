# RAG Search Engine

Learning-focused retrieval over a movie dataset with both lexical and semantic search CLIs.

## Current Functionality
- Build/load a lexical inverted index from `data/movies.json`
- Normalize text with lowercasing, punctuation removal, stopword filtering, and stemming
- Run token-based title lookup via the inverted index
- Inspect TF, IDF, TF-IDF, and BM25 components
- Run full BM25 ranking search
- Build/load semantic embedding vectors for movies
- Run cosine-similarity semantic search over embeddings

## Project Layout
- `cli/keyword_search_cli.py`: lexical/BM25 CLI entrypoint
- `cli/semantic_search_cli.py`: semantic embedding/search CLI entrypoint
- `cli/search_cls.py`: `MovieSearch` and `InvertedIndex`
- `cli/lib/semantic_search.py`: `SemanticSearch` model, embedding, cosine similarity helpers
- `cli/helpers.py`: dataset paths, cache paths, normalization helpers, constants
- `cli/errors/exception_handling.py`: custom exception classes
- `data/`: source dataset and stopword list
- `cache/`: generated lexical + embedding artifacts
- `tests/`: pytest suite (lexical and keyword CLI focused)

## Requirements
- Python `>=3.13`
- `uv`

Install dependencies:
```bash
uv sync --dev
```

Run commands from project root (`rag-search-engine/`).

## Keyword / BM25 CLI
Build lexical cache artifacts:
```bash
uv run cli/keyword_search_cli.py build
```

Token-based search:
```bash
uv run cli/keyword_search_cli.py search "brave bear"
```

Load existing lexical cache files:
```bash
uv run cli/keyword_search_cli.py load
```

Term frequency:
```bash
uv run cli/keyword_search_cli.py tf 424 trapper
```

Inverse document frequency:
```bash
uv run cli/keyword_search_cli.py idf trapper
```

TF-IDF:
```bash
uv run cli/keyword_search_cli.py tfidf 424 trapper
```

BM25 IDF:
```bash
uv run cli/keyword_search_cli.py bm25idf trapper
```

BM25 TF:
```bash
uv run cli/keyword_search_cli.py bm25tf 424 trapper
```

Full BM25 ranking search:
```bash
uv run cli/keyword_search_cli.py bm25search "brave bear"
```

## Semantic CLI
Verify model can be loaded:
```bash
uv run cli/semantic_search_cli.py verify
```

Embed one text string:
```bash
uv run cli/semantic_search_cli.py embed_text "A princess with a bow"
```

Embed a user query:
```bash
uv run cli/semantic_search_cli.py embedquery "rebellious princess adventure"
```

Build/load movie embeddings and verify shape:
```bash
uv run cli/semantic_search_cli.py verify_embeddings
```

Semantic search:
```bash
uv run cli/semantic_search_cli.py search "family animation adventure" --limit 5
```

## Cache Artifacts
Keyword `build` writes:
- `cache/index.pkl`: token -> set of document IDs
- `cache/docmap.pkl`: document ID -> movie object
- `cache/term_frequencies.pkl`: document ID -> token frequency counter
- `cache/doc_lengths.pkl`: document ID -> normalized token count

Semantic embedding flow writes:
- `cache/movie_embeddings.npy`: embedding matrix aligned with movie ordering from `data/movies.json`

## Model Notes
- Semantic search uses SentenceTransformers model `all-MiniLM-L6-v2`.
- Model loading is currently configured with `local_files_only=True`.
- Default model cache folder is `data/.cache/huggingface`.
- Ensure model files are already available locally before running semantic commands.

## Running Tests
```bash
uv run pytest
```

Current tests cover keyword/index flows and keyword CLI paths.
