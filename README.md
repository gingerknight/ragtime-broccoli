# RAG Search Engine

Learning-focused retrieval over a movie dataset with three CLIs:
- keyword/BM25 search
- semantic embedding search
- weighted hybrid search

## Features
- Inverted index build/load from `data/movies.json`
- Text normalization (punctuation removal, lowercase, stopword filtering, stemming)
- TF, IDF, TF-IDF, BM25 scoring inspection
- BM25 ranking search over all documents
- Embedding generation with SentenceTransformers
- Cosine-similarity semantic retrieval
- Chunking utilities (fixed-size and sentence-based semantic chunking)
- Weighted hybrid scoring with configurable `alpha`

## Project Layout
- `cli/keyword_search_cli.py`: keyword/BM25 CLI
- `cli/semantic_search_cli.py`: semantic/chunking CLI
- `cli/hybrid_search_cli.py`: hybrid CLI
- `cli/lib/keyword_search.py`: `MovieSearch`, `InvertedIndex`, BM25 logic
- `cli/lib/semantic_search.py`: embedding model, chunking, semantic retrieval
- `cli/lib/hybrid_search.py`: min-max normalization + weighted fusion
- `cli/lib/helpers.py`: shared constants, data/cache paths, normalization utilities
- `cli/errors/exception_handling.py`: custom exceptions
- `data/`: source files (`movies.json`, `stopwords.txt`)
- `cache/`: generated index + embedding artifacts
- `tests/`: pytest suite

## Requirements
- Python `>=3.13`
- `uv`

Install dependencies:
```bash
uv sync --dev
```

Run all commands from project root (`rag-search-engine/`).

## Quick Start
```bash
# 1) Build keyword cache
uv run cli/keyword_search_cli.py build

# 2) Verify semantic model / embeddings
uv run cli/semantic_search_cli.py verify
uv run cli/semantic_search_cli.py verify_embeddings

# 3) Run hybrid weighted search
uv run cli/hybrid_search_cli.py weighted-search "bear in the wilderness" --alpha 0.7 --limit 5
```

## Keyword CLI (`cli/keyword_search_cli.py`)
```bash
uv run cli/keyword_search_cli.py build
uv run cli/keyword_search_cli.py load
uv run cli/keyword_search_cli.py search "brave bear"
uv run cli/keyword_search_cli.py tf 424 trapper
uv run cli/keyword_search_cli.py idf trapper
uv run cli/keyword_search_cli.py tfidf 424 trapper
uv run cli/keyword_search_cli.py bm25idf trapper
uv run cli/keyword_search_cli.py bm25tf 424 trapper
uv run cli/keyword_search_cli.py bm25tf 424 trapper 1.2 0.75
uv run cli/keyword_search_cli.py bm25search "adventure family"
```

## Semantic CLI (`cli/semantic_search_cli.py`)
```bash
uv run cli/semantic_search_cli.py verify
uv run cli/semantic_search_cli.py embed_text "A princess with a bow"
uv run cli/semantic_search_cli.py embedquery "rebellious princess adventure"
uv run cli/semantic_search_cli.py verify_embeddings
uv run cli/semantic_search_cli.py search "family animation adventure" --limit 5
uv run cli/semantic_search_cli.py chunk "one two three four five" --chunk-size 3 --overlap 1
uv run cli/semantic_search_cli.py semantic_chunk "Sentence one. Sentence two. Sentence three." --max-chunk-size 2 --overlap 1
uv run cli/semantic_search_cli.py embed_chunks
uv run cli/semantic_search_cli.py search_chunked "wilderness revenge story" --limit 10
```

## Hybrid CLI (`cli/hybrid_search_cli.py`)
```bash
# Min-max normalize arbitrary values
uv run cli/hybrid_search_cli.py normalize 10 7 2 0

# Weighted hybrid search
uv run cli/hybrid_search_cli.py weighted-search "bear adventure" --alpha 0.5 --limit 5
uv run cli/hybrid_search_cli.py weighted-search "bear adventure" --alpha 0.8 --limit 5
```

`alpha` weighting:
- `alpha = 1.0`: keyword-only weight
- `alpha = 0.0`: semantic-only weight
- values in between blend both scores

## Cache Artifacts
Keyword artifacts:
- `cache/index.pkl`
- `cache/docmap.pkl`
- `cache/term_frequencies.pkl`
- `cache/doc_lengths.pkl`

Semantic artifacts:
- `cache/movie_embeddings.npy`
- `cache/chunk_embeddings.npy`
- `cache/chunk_metadata.json`

## Model Notes
- Embedding model: `all-MiniLM-L6-v2`
- Uses `local_files_only=True` in `SentenceTransformer`
- Default model cache folder: `data/.cache/huggingface`
- Ensure model files exist locally before semantic commands

## Tests
```bash
uv run pytest
```
