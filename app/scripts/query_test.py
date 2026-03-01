from sqlmodel import Session, text
from sentence_transformers import SentenceTransformer
from app.db import engine

MODEL_NAME = "BAAI/bge-small-en-v1.5"

def main():
    query = "Where should I go for anime and electronics?"
    model = SentenceTransformer(MODEL_NAME)
    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()

    qvec_str = "[" + ",".join(map(str, qvec)) + "]"

    sql = text("""
        SELECT chunk_id, chunk_index, text,
            (embedding <-> CAST(:qvec AS vector)) AS distance
        FROM chunks
        ORDER BY embedding <-> CAST(:qvec AS vector)
        LIMIT 5;
    """)

    with Session(engine) as session:
        rows = session.execute(sql, {"qvec": qvec_str}).all()

    print("Query:", query)
    for (chunk_id, chunk_index, text_val, distance) in rows:
        print(f"- id={chunk_id} idx={chunk_index} dist={distance:.4f} :: {text_val}")

if __name__ == "__main__":
    main()