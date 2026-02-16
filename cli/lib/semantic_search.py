from pathlib import Path
import os
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / ".cache" / "huggingface"


os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from huggingface_hub.utils import disable_progress_bars
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

disable_progress_bars()
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

def verify_model():
    # Load the model (downloads automatically the first time)
    embedding_model = SemanticSearch()
    MODEL = embedding_model.model
    MAX_LENGTH = embedding_model.model.max_seq_length

    print(f"Model loaded: {MODEL}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"Cache directory: {embedding_model.cache_dir}")

def embed_text(text:str):
    model = SemanticSearch()
    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str | Path = DEFAULT_CACHE_DIR):
        cache_path = Path(cache_dir).expanduser().resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_path
        self.model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_path),
        )

    def generate_embedding(self, text: str):
        if not text or text.isspace():
            raise ValueError("Missing text")
        embedding = self.model.encode([text])
        return embedding[0]
