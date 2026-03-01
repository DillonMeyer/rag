from __future__ import annotations

import os
import re
import json
import time
import pathlib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

import fitz
from sqlmodel import Session, select

from app.db import engine
from app.models import Document, Chunk
from app.embeddings import embed_text


ARXIV_API = "http://export.arxiv.org/api/query"


def slugify(s: str) -> str:
    # Converts string to lowercase, replaces non-alphanumeric with hyphens, and truncates to 120 chars.
    # This is used to create filesystem-friendly names for PDFs based on their titles.
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:120]


def arxiv_search(query: str, max_results: int = 25, start: int = 0) -> list[dict]:
    # Calls arXiv Atom API. Returns list of dict with id, title, pdf_url, summary, authors, published, categories.
    params = {
        "search_query": query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = ARXIV_API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as resp:
        xml_text = resp.read()

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_text)

    out: list[dict] = []
    for entry in root.findall("atom:entry", ns):
        arxiv_id_url = entry.findtext("atom:id", default="", namespaces=ns)
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = entry.findtext("atom:published", default="", namespaces=ns) or ""
        authors = [a.findtext("atom:name", default="", namespaces=ns) for a in entry.findall("atom:author", ns)]
        categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", ns)]
        
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if not pdf_url and arxiv_id_url:
            
            pdf_url = arxiv_id_url.replace("/abs/", "/pdf/")

        out.append(
            {
                "arxiv_id_url": arxiv_id_url,
                "title": title,
                "summary": summary,
                "published": published,
                "authors": authors,
                "categories": categories,
                "pdf_url": pdf_url,
            }
        )
    return out


def download_pdf(pdf_url: str, dest_path: pathlib.Path) -> None:
    # Downloads PDF from pdf_url to dest_path. Skips if file already exists and is non-empty.
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "rag-project/0.1"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())


def pdf_to_text(pdf_path: pathlib.Path, max_pages: int | None = None) -> str:
    # Extract text from PDF using PyMuPDF (fitz). Returns concatenated text of all pages up to max_pages.
    doc = fitz.open(pdf_path)
    texts: list[str] = []
    n_pages = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
    for i in range(n_pages):
        page = doc.load_page(i)
        texts.append(page.get_text("text"))
    doc.close()
    
    text = "\n".join(texts)
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, target_chars: int = 1200, overlap_chars: int = 200) -> list[str]:
    # Splits text into paragraphs, then groups them into chunks of approximately target_chars, with optional overlap.
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= target_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
        
    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = []
    prev = ""
    for c in chunks:
        if prev:
            overlapped.append((prev[-overlap_chars:] + "\n\n" + c).strip())
        else:
            overlapped.append(c)
        prev = c
    return overlapped


def main():
    
    query = 'all:"retrieval augmented generation" OR all:"dense retrieval" OR all:"vector search" OR all:"RAG"'
    max_results = int(os.getenv("ARXIV_MAX_RESULTS", "25"))
    data_dir = pathlib.Path("data/papers")

    papers = arxiv_search(query=query, max_results=max_results)
    print(f"Found {len(papers)} arXiv entries")

    with Session(engine) as session:
        for i, p in enumerate(papers, start=1):
            title = p["title"]
            pdf_url = p["pdf_url"]
            arxiv_id_url = p["arxiv_id_url"]
            
            existing = session.exec(select(Document).where(Document.source == pdf_url)).first()
            if existing:
                chunk_ct = session.exec(
                    select(Chunk).where(Chunk.document_id == existing.document_id)
                ).first()
                if chunk_ct:
                    print(f"[{i}/{len(papers)}] SKIP already ingested: {title[:80]}")
                    continue
                else:
                    print(f"[{i}/{len(papers)}] REPAIR missing chunks: {title[:80]}")
                    session.delete(existing)
                    session.commit()

            filename = slugify(title) or f"paper_{i}"
            pdf_path = data_dir / f"{filename}.pdf"

            print(f"[{i}/{len(papers)}] Downloading: {title[:80]}")
            download_pdf(pdf_url, pdf_path)
            time.sleep(0.5)

            print(f"    Extracting text: {pdf_path.name}")
            text = pdf_to_text(pdf_path, max_pages=None)
            if len(text) < 2000:
                print("    WARN: extracted text is short; skipping (maybe scanned/empty).")
                continue

            doc = Document(
                source=pdf_url,
                title=title,
                metadata_json=json.dumps(
                    {
                        "arxiv_id_url": arxiv_id_url,
                        "pdf_path": str(pdf_path),
                        "published": p["published"],
                        "authors": p["authors"],
                        "categories": p["categories"],
                        "summary": p["summary"][:2000],
                        "ingested_at": datetime.utcnow().isoformat(),
                    }
                ),
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)

            chunks = chunk_text(text, target_chars=1200, overlap_chars=200)

            print(f"    Chunking -> {len(chunks)} chunks; embedding…")
            for idx, ch in enumerate(chunks):
                vec = embed_text(ch)
                session.add(
                    Chunk(
                        document_id=doc.document_id,
                        chunk_index=idx,
                        text=ch,
                        token_count=None,
                        embedding=vec,
                        embedding_model_version_id=1,
                    )
                )
            session.commit()
            print(f"    Inserted document_id={doc.document_id}, chunks={len(chunks)}")

    print("Done.")


if __name__ == "__main__":
    main()