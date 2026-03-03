from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import requests


@dataclass(frozen=True)
class Citation:
    n: int
    chunk_id: int
    document_id: int
    title: str | None
    source: str
    chunk_index: int


def _build_context(hits: Sequence[dict]) -> tuple[str, list[Citation]]:
    citations: list[Citation] = []
    parts: list[str] = []

    for i, h in enumerate(hits, start=1):
        citations.append(
            Citation(
                n=i,
                chunk_id=int(h["chunk_id"]),
                document_id=int(h["document_id"]),
                title=h.get("title"),
                source=str(h["source"]),
                chunk_index=int(h["chunk_index"]),
            )
        )

        title = h.get("title") or "Untitled"
        parts.append(
            f"[{i}] title: {title}\n"
            f"    source: {h['source']}\n"
            f"    chunk_id: {h['chunk_id']}  chunk_index: {h['chunk_index']}\n"
            f"    text:\n{h['text']}\n"
        )

    return "\n---\n".join(parts), citations


def generate_answer_with_citations_local(
    question: str,
    hits: Sequence[dict],
    *,
    model: str = "llama3.2",
    ollama_url: str = "http://127.0.0.1:11434",
    timeout_s: int = 120,
) -> tuple[str, list[Citation]]:
    """
    Ollama local generation. Returns (answer, citations metadata).
    Answer should include inline citations like [1] [2].
    """
    context, citations = _build_context(hits)

    system = (
        "You are a precise assistant doing retrieval-augmented QA.\n"
        "You MUST only use the provided Sources.\n"
        "If the Sources do not contain enough information, say so.\n"
        "When you state a claim, add inline citations like [1] or [2][4].\n"
        "Do not invent citations.\n"
    )

    prompt = (
        f"{system}\n\n"
        f"Question:\n{question}\n\n"
        f"Sources:\n{context}\n\n"
        "Write a concise answer grounded in the sources."
    )

    resp = requests.post(
        f"{ollama_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()

    answer = (data.get("response") or "").strip()
    return answer, citations