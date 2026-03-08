import os
from pathlib import Path

from sentence_transformers import CrossEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / ".cache" / "huggingface"


def cross_encoding(results, query):
    # init cross encoder
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_files_only = os.getenv("HF_HUB_OFFLINE", "0") == "1"
    cross_encoder = CrossEncoder(
        "cross-encoder/ms-marco-TinyBERT-L2-v2",
        cache_folder=str(DEFAULT_CACHE_DIR),
        local_files_only=local_files_only,
    )

    pairs = []
    # Create a new list of "pairs" lists, where the first element is the query,
    # and the second element is the document, in this stringified format
    for val in results:
        pairs.append([query, f"{val[1].get('title', '')} - {val[1].get('document', '')}"])

    # pass and score the pairs to the encoder
    # `predict` returns a list of numbers, one for each pair
    scores = cross_encoder.predict(pairs)
    for doc, score in zip(results, scores, strict=True):
        doc[1]["cross_encoder_score"] = float(score)

    sorted_results = sorted(
        results,
        key=lambda item: float(item[1].get("cross_encoder_score", float("-inf"))),
        reverse=True,
    )
    return sorted_results
