from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, List


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    generate: bool = False
    model: str = "llama3.2:latest"
    max_tokens: int = 256
    temperature: float = 0.2
    include_hits: bool = False


class ChunkHit(BaseModel):
    chunk_id: int
    chunk_index: int
    text: str
    distance: float
    document_id: int
    title: Optional[str] = None
    source: str


class Citation(BaseModel):
    n: int
    chunk_id: int
    document_id: int
    chunk_index: int
    title: Optional[str] = None
    source: str

class HitPreview(BaseModel):
    chunk_id: int
    chunk_index: int
    distance: float
    document_id: int
    title: Optional[str] = None
    source: str
    preview: str

class AskResponse(BaseModel):
    question: str
    hits: List[ChunkHit]
    answer: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    hits: Optional[List[HitPreview]] = None