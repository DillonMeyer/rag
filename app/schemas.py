from pydantic import BaseModel
from typing import List, Optional


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


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