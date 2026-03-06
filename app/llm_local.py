from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import requests
import re
from app.settings import OLLAMA_BASE_URL, DEFAULT_MODEL

def extract_used_citation_numbers(answer: str) -> list[int]:
    nums = re.findall(r"\[(\d+)\]", answer or "")
    seen: list[int] = []
    for n in nums:
        i = int(n)
        if i not in seen:
            seen.append(i)
    return seen

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
    model: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_BASE_URL,
    timeout_s: int = 120,
) -> tuple[str, list[Citation]]:
    """
    Ollama local generation. Returns (answer, citations metadata).
    Answer should include inline citations like [1] [2].
    """

    # if not hits:
    #     return "I could not find relevant grounded sources for this question.", []

    context, citations = _build_context(hits)

    system = (
        "You are a precise assistant doing retrieval-augmented QA.\n"
        "You MUST only use the provided Sources.\n"
        "If the Sources do not contain enough information, say so briefly.\n"
        "Answer in 2-4 sentences when possible.\n"
        "Start with a direct definition or answer.\n"
        "When you state a claim, add inline citations like [1] or [2][4].\n"
        "Do not invent citations.\n"
        "Do NOT output a References section.\n"
        "Only use bracket citations like [1].\n"
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

    used_nums = extract_used_citation_numbers(answer)
    filtered_citations = [c for c in citations if c.n in used_nums]

    return answer, filtered_citations