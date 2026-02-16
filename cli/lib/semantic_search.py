import os
from pathlib import Path

import numpy as np
from helpers import EMBEDDING_CACHE, load_movies
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

    def build_embeddings(self, documents):
        # documents is a list of dict (movies.json?)
        # each document, add a key id of the doc and value is the doc values itself
        if not (self.documents or self.document_map):
            doc_list = []
            self.documents = documents
            for doc in self.documents:
                self.document_map[doc["id"]] = doc
                doc_list.append(f"{doc['title']} {doc['description']}")
            self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
            self.save()
            return self.embeddings

    def save(self) -> None:
        # use np.save methods to write files
        if not os.path.isdir("./cache"):
            os.mkdir("./cache")
        with open(EMBEDDING_CACHE, "wb") as efp:
            np.save(efp, self.embeddings)

    @staticmethod
    def load():
        # Load npy cache files from disk
        with open(EMBEDDING_CACHE, "rb") as efp:
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
            self.embeddings = self.load()
        else:
            self.embeddings = self.build_embeddings(self.documents)

    def generate_embedding(self, text: str):
        if not text or text.isspace():
            raise ValueError("Missing text")
        embedding = self.model.encode([text])
        return embedding[0]
