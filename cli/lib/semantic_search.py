from sentence_transformers import SentenceTransformer


def verify_model():
    # Load the model (downloads automatically the first time)
    embedding_model = SemanticSearch()
    MODEL = embedding_model.model
    MAX_LENGTH = embedding_model.model.max_seq_length


    print(f"Model loaded: {MODEL}")
    print(f"Max sequence length: {MAX_LENGTH}")


class SemanticSearch:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
