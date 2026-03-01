from fastapi import FastAPI
from sqlmodel import Session, SQLModel
from sqlalchemy import text
from .db import engine
from .embeddings import embed_query
from .schemas import AskRequest, AskResponse, ChunkHit
from .db import engine

app = FastAPI(title="RAG (Observable)")

@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/dbcheck")
def dbcheck():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1;"))
        result = conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname='vector';")
        ).fetchone()
    return {"db_ok": True, "pgvector_enabled": bool(result)}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    qvec = embed_query(req.question)
    qvec_str = "[" + ",".join(map(str, qvec)) + "]"

    sql = text("""
        SELECT
            c.chunk_id,
            c.chunk_index,
            c.text,
            (c.embedding <-> CAST(:qvec AS vector)) AS distance,
            d.document_id,
            d.title,
            d.source
        FROM chunks c
        JOIN documents d ON d.document_id = c.document_id
        ORDER BY c.embedding <-> CAST(:qvec AS vector)
        LIMIT :top_k;
    """)

    with Session(engine) as session:
        rows = session.execute(sql, {"qvec": qvec_str, "top_k": req.top_k}).all()

    hits = [
        ChunkHit(
            chunk_id=r[0],
            chunk_index=r[1],
            text=r[2],
            distance=float(r[3]),
            document_id=r[4],
            title=r[5],
            source=r[6],
        )
        for r in rows
    ]

    return AskResponse(question=req.question, hits=hits)