from __future__ import annotations

from datetime import datetime, UTC
from typing import Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import Column
from pgvector.sqlalchemy import Vector

class Document(SQLModel, table=True):
    __tablename__ = "documents"

    document_id: Optional[int] = Field(default=None, primary_key=True)
    source: str
    title: Optional[str] = None
    metadata_json: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Chunk(SQLModel, table=True):
    __tablename__ = "chunks"

    chunk_id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.document_id")
    chunk_index: int
    text: str
    token_count: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    
    embedding: list[float] = Field(sa_column=Column(Vector(384)))
    
    embedding_model_version_id: Optional[int] = Field(default=None, index=True)


class Query(SQLModel, table=True):
    __tablename__ = "queries"

    query_id: Optional[int] = Field(default=None, primary_key=True)
    question_text: str
    embedding_model_version_id: Optional[int] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Retrieval(SQLModel, table=True):
    __tablename__ = "retrievals"

    retrieval_id: Optional[int] = Field(default=None, primary_key=True)
    query_id: int = Field(foreign_key="queries.query_id", index=True)

    top_k: int
    retrieval_latency_ms: int
    embedding_model_version_id: Optional[int] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RetrievalResult(SQLModel, table=True):
    __tablename__ = "retrieval_results"

    retrieval_result_id: Optional[int] = Field(default=None, primary_key=True)
    retrieval_id: int = Field(foreign_key="retrievals.retrieval_id", index=True)

    chunk_id: int = Field(foreign_key="chunks.chunk_id", index=True)
    rank: int
    distance: float

class EvalSet(SQLModel, table=True):
    __tablename__ = "eval_sets"
    eval_set_id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

class EvalQuestion(SQLModel, table=True):
    __tablename__ = "eval_questions"

    eval_question_id: Optional[int] = Field(default=None, primary_key=True)
    eval_set_id: int = Field(foreign_key="eval_sets.eval_set_id", index=True)
    question_text: str
    
    gold_chunk_id: Optional[int] = Field(default=None, foreign_key="chunks.chunk_id", index=True)
    
    gold_chunk_ids: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))