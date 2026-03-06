import time
import requests
import re

from fastapi import FastAPI
from sqlmodel import SQLModel, Session

from app.llm_local import generate_answer_with_citations_local
from app.schemas import Citation as CitationSchema

from .db import engine
from .embeddings import embed_query
from .retrieval import retrieve_hits
from .schemas import AskRequest, AskResponse, ChunkHit, HitPreview
from . import models
from sqlalchemy import text


app = FastAPI(title="RAG")


@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)


@app.get("/health")
def health():
    with Session(engine) as session:
        session.exec(text("SELECT 1"))
    return {"ok": True}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    request_start = time.perf_counter()

    qvec = embed_query(req.question)

    with Session(engine) as session:
        # retrieval
        retrieval_start = time.perf_counter()

        hit_rows = retrieve_hits(
            session,
            qvec=qvec,
            top_k=req.top_k,
            max_chunks_per_doc=2
        )

        hits = [
            ChunkHit(
                chunk_id=h["chunk_id"],
                chunk_index=h["chunk_index"],
                text=h["text"],
                distance=h["distance"],
                document_id=h["document_id"],
                title=h["title"],
                source=h["source"],
            )
            for h in hit_rows
        ]

        retrieval_latency_ms = int((time.perf_counter() - retrieval_start) * 1000)

        answer: str | None = None
        citations: list[CitationSchema] = []

        gen_latency_ms = None
        answer_length = None
        citations_present = None

        # generation
        if req.generate:
            gen_start = time.perf_counter()

            answer_text, cite_meta = generate_answer_with_citations_local(
                question=req.question,
                hits=hit_rows,
                model=req.model or "llama3.2:latest"
            )

            gen_latency_ms = int((time.perf_counter() - gen_start) * 1000)

            answer = answer_text
            answer_length = len(answer_text or "")
            citations_present = len(cite_meta) > 0

            citations = [
                CitationSchema(
                    n=c.n,
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    title=c.title,
                    source=c.source,
                    chunk_index=c.chunk_index,
                )
                for c in cite_meta
            ]

        # logging
        q = models.Query(
            question_text=req.question,
            embedding_model_version_id=1
        )
        session.add(q)
        session.flush()

        rlog = models.Retrieval(
            query_id=q.query_id,
            top_k=req.top_k,
            retrieval_latency_ms=retrieval_latency_ms,
            embedding_model_version_id=1,
        )
        session.add(rlog)
        session.flush()

        for rank, h in enumerate(hits, start=1):
            session.add(
                models.RetrievalResult(
                    retrieval_id=rlog.retrieval_id,
                    chunk_id=h.chunk_id,
                    rank=rank,
                    distance=h.distance,
                )
            )

        if req.generate and answer is not None:
            session.add(
                models.Generation(
                    query_id=q.query_id,
                    model_name=req.model or "llama3.2:latest",
                    generation_latency_ms=gen_latency_ms,
                    answer_length_chars=answer_length,
                    citations_present=citations_present,
                )
            )

        session.commit()

    response_hits = None
    if req.include_hits:
        response_hits = [
            HitPreview(
                chunk_id=h.chunk_id,
                chunk_index=h.chunk_index,
                distance=h.distance,
                document_id=h.document_id,
                title=h.title,
                source=h.source,
                preview=h.text[:220].replace("\n", " ").strip(),
            )
            for h in hits
        ]

    return AskResponse(
        question=req.question,
        answer=answer,
        citations=citations,
        hits=response_hits,
    )