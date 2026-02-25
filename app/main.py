from fastapi import FastAPI
from sqlmodel import SQLModel, text

from .db import engine
from . import models  # important: ensures models are registered

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