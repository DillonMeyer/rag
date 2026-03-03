from __future__ import annotations

from collections import defaultdict
from typing import TypedDict

from sqlalchemy import text
from sqlmodel import Session


class HitRow(TypedDict):
    chunk_id: int
    document_id: int
    chunk_index: int
    distance: float
    text: str
    title: str | None
    source: str


def retrieve_hits(
    session: Session,
    qvec: list[float],
    top_k: int,
    *,
    max_chunks_per_doc: int = 2,
    fetch_k: int | None = None,
) -> list[HitRow]:
    # Shared retriever used by both /ask and eval.
    # Uses cosine distance operator (<=>) so it matches ivfflat index (vector_cosine_ops).
    # Overfetches then diversifies (max N chunks per doc).
    qvec_str = "[" + ",".join(map(str, qvec)) + "]"
    fetch_k = fetch_k or max(top_k * 10, 50)

    sql = text("""
        SELECT
            c.chunk_id,
            c.document_id,
            c.chunk_index,
            (c.embedding <=> CAST(:qvec AS vector)) AS distance,
            c.text,
            d.title,
            d.source
        FROM chunks c
        JOIN documents d ON d.document_id = c.document_id
        ORDER BY c.embedding <=> CAST(:qvec AS vector)
        LIMIT :fetch_k;
    """)

    rows = session.execute(sql, {"qvec": qvec_str, "fetch_k": fetch_k}).all()

    per_doc = defaultdict(int)
    hits: list[HitRow] = []

    for r in rows:
        doc_id = int(r[1])
        if per_doc[doc_id] >= max_chunks_per_doc:
            continue
        per_doc[doc_id] += 1

        hits.append(
            {
                "chunk_id": int(r[0]),
                "document_id": doc_id,
                "chunk_index": int(r[2]),
                "distance": float(r[3]),
                "text": r[4],
                "title": r[5],
                "source": r[6],
            }
        )

        if len(hits) >= top_k:
            break

    return hits


def retrieve_chunk_ids(
    session: Session,
    qvec: list[float],
    top_k: int,
    *,
    max_chunks_per_doc: int = 2,
    fetch_k: int | None = None,
) -> list[int]:
    # Convenience wrapper around retrieve_hits that just returns chunk IDs, used for eval.
    if fetch_k is None:
        fetch_k = max(top_k * 10, 50)
    hits = retrieve_hits(
        session,
        qvec=qvec,
        top_k=top_k,
        max_chunks_per_doc=max_chunks_per_doc,
        fetch_k=fetch_k,
    )
    return [h["chunk_id"] for h in hits]