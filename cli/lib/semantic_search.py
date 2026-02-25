import json
import os
import re
from pathlib import Path

import numpy as np
from helpers import CHUNK_EMBEDDINGS_CACHE, CHUNK_METADATA_JSON, EMBEDDING_CACHE, load_movies
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / ".cache" / "huggingface"


hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


def verify_model():
    # Load the model (downloads automatically the first time)
    embedding_model = SemanticSearch()
    MODEL = embedding_model.model
    MAX_LENGTH = embedding_model.model.max_seq_length

    print(f"Model loaded: {MODEL}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"Cache directory: {embedding_model.cache_dir}")


def embed_text(text: str):
    model = SemanticSearch()
    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    model = SemanticSearch()
    documents = load_movies()
    model.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {model.embeddings.shape[0]} vectors in {model.embeddings.shape[1]} dimensions")


def embed_query_text(query: str):
    # create instance of SemSearch
    # call generate embedding
    # print info about query
    model = SemanticSearch()
    embedding = model.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def size_defined_chunking(text: str, chunk_size: int, overlap: int, semantic: bool = False) -> list[str]:
    # chunk text into size N chunks and return a list of the chunks
    chunks = []
    words = re.split(r"(?<=[.!?])\s+", text) if semantic else text.split()
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    i = 0
    step = chunk_size - overlap
    while i < len(words):
        chunks_words = words[i : i + chunk_size]
        # chunk_size = 5, overlap = 2, step = 3
        # remaining only 2 words, we alreadyt captured previously break
        # also do not break if we haven't captured first chunks array
        if chunks and len(chunks_words) <= overlap:
            break
        chunks.append(" ".join(chunks_words))
        i += step
    # for i in range(0, len(words), chunk_size):
    # chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks


def pretty_display_chunks(chunks: list[str], text_length: int, semantic: bool = False) -> None:
    # Pretty print the list of chunks in specific format
    preamble = "Semantically chunking" if semantic else "Chunking"
    print(f"{preamble} {text_length} characters")
    for i, word in enumerate(chunks):
        print(f"{i + 1}. {word}")

def searching_chunks(query: str, limit: int = 10):
    chunk_sem = ChunkedSemanticSearch()
    movies = load_movies()
    embeddings = chunk_sem.load_or_create_chunk_embeddings(movies)
    result = chunk_sem.search_chunks(query, limit)
    for i, result_dict in enumerate(result):
            print(f"\n{i}. {result_dict["title"]} (score: {result_dict["score"]:.4f})")
            print(f"   {result_dict["document"]}...")


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str | Path = DEFAULT_CACHE_DIR):
        cache_path = Path(cache_dir).expanduser().resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_path
        self.model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_path),
            local_files_only=True,
        )
        self.embeddings = None
        self.documents = None
        self.document_map = {}  # emtpy dict

    def search(self, query: str, limit=5):
        embed_similarity = []  # list of tuples storing similiary_score, document
        if self.embeddings is not None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        # making a pretty big assumption the data is created in order like movies.json for embedding...
        # need to look at fixing this later for other docs and not groomed json...
        for doc_id, vector in zip(self.documents, self.embeddings, strict=True):
            # print(f"doc_id troulbeshooting: {doc_id}")
            embed_similarity.append(
                (cosine_similarity(query_embedding, vector), doc_id["title"], doc_id["description"])
            )
        embed_similarity_sorted = sorted(embed_similarity, key=lambda tup: tup[0], reverse=True)
        return embed_similarity_sorted[:limit]
        # print(embed_similarity_sorted[:limit])

    def build_embeddings(self, documents: list):
        # documents is a list of dict (movies.json?)
        # each document, add a key id of the doc and value is the doc values itself
        if not (self.documents or self.document_map):
            doc_list = []
            self.documents = documents
            for doc in self.documents:
                self.document_map[doc["id"]] = doc
                doc_list.append(f"{doc['title']} {doc['description']}")
            self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
            self.save(EMBEDDING_CACHE)
            return self.embeddings

    def save(self, path) -> None:
        # use np.save methods to write files
        if not os.path.isdir("./cache"):
            os.mkdir("./cache")
        with open(path, "wb") as efp:
            np.save(efp, self.embeddings)

    @staticmethod
    def load(path):
        # Load npy cache files from disk
        with open(path, "rb") as efp:
            embed_cache = np.load(efp)
        return embed_cache

    def load_or_create_embeddings(self, documents):
        # build embeddings only once, load from disk if exists
        doc_list = []
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']} {doc['description']}")
        file_path = Path(EMBEDDING_CACHE)
        if file_path.exists():
            self.embeddings = self.load(EMBEDDING_CACHE)
        else:
            self.embeddings = self.build_embeddings(self.documents)

    def generate_embedding(self, text: str) -> np.typing.NDArray:
        if not text or text.isspace():
            raise ValueError("Missing text")
        embedding = self.model.encode([text])
        return embedding[0]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list):
        # documents is a list of dict (movies.json?)
        # each document, add a key id of the doc and value is the doc values itself
        # print(f"Building Chunk embeddings")
        chunk_metadata_list = []
        all_chunks = []
        self.documents = documents
        for idx, doc in enumerate(self.documents):
            # skip if missing descritption (important for semantic embedding)
            if not doc["description"]:
                # print(f"Skipping, no description")
                continue
            self.document_map[doc["id"]] = doc
            # chunk description
            doc_chunks = size_defined_chunking(doc["description"], 4, 1, True)
            num_chunks = len(doc_chunks)
            # print(f"Number of chunks in doc description: {num_chunks}")
            for ch_idx, _ in enumerate(doc_chunks):
                chunk_metadata_list.append({"movie_idx": idx, "chunk_idx": ch_idx, "total_chunks": num_chunks})
            all_chunks.extend(doc_chunks)
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_save(CHUNK_EMBEDDINGS_CACHE)
        self.chunk_metadata = chunk_metadata_list
        self.save_metadata(all_chunks)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        # build embeddings only once, load from disk if exists
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        embed_path = Path(CHUNK_EMBEDDINGS_CACHE)
        metadata_path = Path(CHUNK_METADATA_JSON)
        if embed_path.exists() and metadata_path.exists():
            self.chunk_embeddings = self.chunk_load(CHUNK_EMBEDDINGS_CACHE)
            self.chunk_metadata = self.load_metadata(CHUNK_METADATA_JSON)
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(self.documents)
        
    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunk_metadata = self.chunk_metadata["chunks"]
        # Build chunk-level scores.
        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            chunk_meta = chunk_metadata[i]
            score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": chunk_meta["chunk_idx"],
                    "movie_idx": chunk_meta["movie_idx"],
                    "score": score,
                }
            )
        # Keep only the best chunk score per movie.
        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score
        # Sort movie scores from highest to lowest and apply limit.
        sorted_movie_scores = sorted(movie_scores.items(), key=lambda item: item[1], reverse=True)
        # format return to be like 
        """
        {
        "id": doc_id,
        "title": title,
        "document": document[:100],
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata or {}
        }
        """
        results = []
        for i in range(0,limit):
            val = {
                "id": self.documents[sorted_movie_scores[i][0]]["id"],
                "title": self.documents[sorted_movie_scores[i][0]]["title"],
                "document": self.documents[sorted_movie_scores[i][0]]["description"][:100],
                "score": round(sorted_movie_scores[i][1], 4),
                "metadata": {}
            }
            # print(val)
            results.append(val)
        return results
        


    def save_metadata(self, chunk_list: list):
        # use np.save methods to write files
        if not os.path.isdir("./cache"):
            os.mkdir("./cache")
        with open(CHUNK_METADATA_JSON, "w") as fp:
            # save chunk metadata
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunk_list)}, fp, indent=2)

    def load_metadata(self, json_path: str):
        with open(json_path) as json_fp:
            return json.load(json_fp)

    def chunk_save(self, path) -> None:
        # use np.save methods to write files
        if not os.path.isdir("./cache"):
            os.mkdir("./cache")
        with open(path, "wb") as efp:
            np.save(efp, self.chunk_embeddings)

    @staticmethod
    def chunk_load(path):
        # Load npy cache files from disk
        with open(path, "rb") as efp:
            embed_cache = np.load(efp)
        return embed_cache
