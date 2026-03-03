from pydantic import BaseModel, Field
from typing import List, Optional

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    generate: bool = True
    model: str | None = None  # optional override


class ChunkHit(BaseModel):
    chunk_id: int
    chunk_index: int
    text: str
    distance: float
    document_id: int
    title: str | None = None
    source: str


class Citation(BaseModel):
    n: int
    chunk_id: int
    document_id: int
    title: str | None = None
    source: str
    chunk_index: int


class AskResponse(BaseModel):
    question: str
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    hits: list[ChunkHit]

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    generate: bool = False
    model: str = "llama3.2:latest"
    max_tokens: int = 256
    temperature: float = 0.2

class ChunkHit(BaseModel):
    chunk_id: int
    chunk_index: int
    text: str
    distance: float
    document_id: int
    title: Optional[str] = None
    source: str

class AskResponse(BaseModel):
    question: str
    hits: List[ChunkHit]