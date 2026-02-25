from fastapi import FastAPI
from sqlmodel import text
from .db import engine

app = FastAPI(title="RAG (Observable)")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/dbcheck")
def dbcheck():
    # Verifies DB connection + pgvector extension availability
    with engine.connect() as conn:
        conn.execute(text("SELECT 1;"))
        # check that the extension exists
        result = conn.execute(text("SELECT extname FROM pg_extension WHERE extname='vector';")).fetchone()
        return {"db_ok": True, "pgvector_enabled": bool(result)}