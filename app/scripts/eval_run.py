from __future__ import annotations

import argparse
from dataclasses import dataclass

from sqlmodel import Session, select

from app.db import engine
from app.embeddings import embed_query
from app import models
from app.retrieval import retrieve_chunk_ids

def retrieve_ids(session: Session, question: str, top_k: int) -> list[int]:
    qvec = embed_query(question)
    # align with /ask:
    return retrieve_chunk_ids(session, qvec=qvec, top_k=top_k, max_chunks_per_doc=2)

def parse_gold_ids(q: models.EvalQuestion) -> set[int]:
    gold: set[int] = set()
    if getattr(q, "gold_chunk_id", None) is not None:
        gold.add(int(q.gold_chunk_id))  # legacy
    if getattr(q, "gold_chunk_ids", None):
        parts = [p.strip() for p in (q.gold_chunk_ids or "").split(",") if p.strip()]
        for p in parts:
            gold.add(int(p))
    return gold


@dataclass
class Metrics:
    questions: int
    recall_at_k: float
    mrr: float


def compute_metrics(session: Session, eval_set_name: str, top_k: int, verbose: bool) -> Metrics:
    es = session.exec(select(models.EvalSet).where(models.EvalSet.name == eval_set_name)).first()
    if not es:
        raise RuntimeError(f"Eval set not found: {eval_set_name}")

    questions = session.exec(
        select(models.EvalQuestion).where(models.EvalQuestion.eval_set_id == es.eval_set_id)
    ).all()

    if not questions:
        raise RuntimeError(f"No eval questions found for eval set: {eval_set_name}")

    hits = 0
    rr_sum = 0.0

    for i, q in enumerate(questions, start=1):
        gold_ids = parse_gold_ids(q)
        if not gold_ids:
            raise RuntimeError(f"Eval question {q.eval_question_id} has no gold_chunk_id(s).")

        retrieved = retrieve_ids(session, q.question_text, top_k=top_k)

        rank = None
        for idx, cid in enumerate(retrieved, start=1):
            if cid in gold_ids:
                rank = idx
                break

        if rank is not None:
            hits += 1
            rr_sum += 1.0 / rank

        if verbose:
            status = "status good" if rank is not None else "status ERROR"
            gold_str = ",".join(map(str, sorted(gold_ids)))
            print(f"[{i}] {status} rank={rank} gold={gold_str} :: {q.question_text}")
            print(f"     retrieved={retrieved}")

    n = len(questions)
    return Metrics(
        questions=n,
        recall_at_k=hits / n,
        mrr=rr_sum / n,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", required=True, help="Eval set name, e.g. toy_v1")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with Session(engine) as session:
        m = compute_metrics(session, eval_set_name=args.eval_set, top_k=args.top_k, verbose=args.verbose)

    print()
    print(f"Eval set: {args.eval_set}")
    print(f"Questions scored: {m.questions}")
    print(f"Recall@{args.top_k}: {m.recall_at_k:.3f}")
    print(f"MRR: {m.mrr:.3f}")


if __name__ == "__main__":
    main()