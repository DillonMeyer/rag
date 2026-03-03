import time
from fastapi import FastAPI
from sqlmodel import SQLModel, Session
from app.llm_local import generate_answer_with_citations_local
from app.schemas import Citation as CitationSchema

from .db import engine
from .embeddings import embed_query
from .retrieval import retrieve_hits
from .schemas import AskRequest, AskResponse, ChunkHit
from . import models

app = FastAPI(title="RAG")

@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t0 = time.perf_counter()

    qvec = embed_query(req.question)

    with Session(engine) as session:
        hit_rows = retrieve_hits(session, qvec=qvec, top_k=req.top_k, max_chunks_per_doc=2)

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

        answer: str | None = None
        citations: list[CitationSchema] = []

        if req.generate:
            answer_text, cite_meta = generate_answer_with_citations_local(
                question=req.question,
                hits=hit_rows,
                model=req.model or "llama3.2:latest"
            )
            answer = answer_text
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

        latency_ms = int((time.perf_counter() - t0) * 1000)

        q = models.Query(question_text=req.question, embedding_model_version_id=1)
        session.add(q)
        session.flush()

        rlog = models.Retrieval(
            query_id=q.query_id,
            top_k=req.top_k,
            retrieval_latency_ms=latency_ms,
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

        session.commit()

    return AskResponse(question=req.question, answer=answer, citations=citations, hits=hits)