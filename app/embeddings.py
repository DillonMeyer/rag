from sentence_transformers import SentenceTransformer

# BGE-small (384 dimensional)
MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model: SentenceTransformer | None = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_text(text: str) -> list[float]:
    model = get_model()
    vec = model.encode([text], normalize_embeddings=True)[0].tolist()
    # return a list of floats instead of a numpy array to ensure compatibility with pgvector
    return vec

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    # return list of lists instead of numpy array for better compatibility with SQLModel + pgvector
    return [v.tolist() for v in vecs]

def embed_query(text: str) -> list[float]:
    return embed_text(text)