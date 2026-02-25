from __future__ import annotations
import random
from sqlmodel import Session
from app.db import engine
from app.models import Document, Chunk

def random_vec(dim: int = 384) -> list[float]:
    # deterministic-ish makes debugging easier
    random.seed(42)
    return [random.random() for _ in range(dim)]

def main():
    text = (
        "Tokyo has many districts. Akihabara is known for electronics and anime. "
        "Shinjuku is a major transit hub. This is a test document for ingestion."
    )

    with Session(engine) as session:
        doc = Document(source="local:test", title="Test Doc")
        session.add(doc)
        session.commit()
        session.refresh(doc)

        # dumb chunking: split by sentences
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        for i, s in enumerate(sentences):
            chunk = Chunk(
                document_id=doc.document_id,
                chunk_index=i,
                text=s,
                token_count=None,
                embedding=random_vec(384),
                embedding_model_version_id=1,
            )
            session.add(chunk)

        session.commit()
        print(f"Inserted doc_id={doc.document_id}, chunks={len(sentences)}")

if __name__ == "__main__":
    main()
