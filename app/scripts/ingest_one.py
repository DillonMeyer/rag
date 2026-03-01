from __future__ import annotations

from sqlmodel import Session
from sentence_transformers import SentenceTransformer

from app.db import engine
from app.models import Document, Chunk

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main():
    text = (
        "Tokyo has many districts. Akihabara is known for electronics and anime. "
        "Shinjuku is a major transit hub. This is a test document for ingestion."
    )
    
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        sentences,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    ).tolist()
    
    with Session(engine) as session:
        doc = Document(source="local:test", title="Test Doc (BGE)")
        session.add(doc)
        session.commit()
        session.refresh(doc)

        for i, (sentence, vec) in enumerate(zip(sentences, vectors)):
            chunk = Chunk(
                document_id=doc.document_id,
                chunk_index=i,
                text=sentence,
                token_count=None,
                embedding=vec,
                embedding_model_version_id=1,
            )
            session.add(chunk)

        session.commit()

    print(f"Inserted doc with BGE embeddings: chunks={len(sentences)}")


if __name__ == "__main__":
    main()