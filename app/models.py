from __future__ import annotations

from datetime import datetime
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
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(SQLModel, table=True):
    __tablename__ = "chunks"

    chunk_id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.document_id")
    chunk_index: int
    text: str
    token_count: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # pgvector column (BGE-small-en -> 384 dims)
    embedding: list[float] = Field(sa_column=Column(Vector(384)))

    # track which embedding model produced this
    embedding_model_version_id: Optional[int] = Field(default=None, index=True)