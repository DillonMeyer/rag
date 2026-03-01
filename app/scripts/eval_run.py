from __future__ import annotations

import argparse
from dataclasses import dataclass

from sqlmodel import Session, text, select

from app.db import engine
from app.embeddings import embed_query
from app.models import EvalSet, EvalQuestion


@dataclass
class Metrics:
    n: int
    recall_at_k: float
    mrr: float


def parse_gold_ids(q: EvalQuestion) -> set[int]:
    """
    Supports:
      - gold_chunk_ids: "2,6"
      - fallback to gold_chunk_id: 2
    """
    ids: set[int] = set()
    if getattr(q, "gold_chunk_ids", None):
        raw = q.gold_chunk_ids or ""
        for part in raw.split(","):
            part = part.strip()
            if part:
                ids.add(int(part))
    if q.gold_chunk_id is not None:
        ids.add(int(q.gold_chunk_id))
    return ids


def retrieve_top_k(session: Session, question: str, top_k: int) -> list[int]:
    qvec = embed_query(question)
    qvec_str = "[" + ",".join(map(str, qvec)) + "]"

    sql = text("""
        SELECT chunk_id
        FROM chunks
        ORDER BY embedding <-> CAST(:qvec AS vector)
        LIMIT :top_k;
    """)
    rows = session.execute(sql, {"qvec": qvec_str, "top_k": top_k}).all()
    return [r[0] for r in rows]


def compute_metrics(session: Session, eval_set_name: str, top_k: int, verbose: bool) -> Metrics:
    eval_set = session.exec(select(EvalSet).where(EvalSet.name == eval_set_name)).first()
    if not eval_set:
        raise RuntimeError(f"Eval set not found: {eval_set_name}")

    questions = session.exec(
        select(EvalQuestion).where(EvalQuestion.eval_set_id == eval_set.eval_set_id)
    ).all()

    if not questions:
        raise RuntimeError(f"No eval questions found for eval set: {eval_set_name}")

    n = 0
    hits = 0
    rr_sum = 0.0

    for i, q in enumerate(questions, start=1):
        gold_ids = parse_gold_ids(q)
        if not gold_ids:
            continue

        retrieved = retrieve_top_k(session, q.question_text, top_k=top_k)
        
        best_rank = None
        for gid in gold_ids:
            if gid in retrieved:
                r = retrieved.index(gid) + 1
                best_rank = r if best_rank is None else min(best_rank, r)

        ok = best_rank is not None
        n += 1
        if ok:
            hits += 1
            rr_sum += 1.0 / best_rank

        if verbose:
            gold_str = ",".join(map(str, sorted(gold_ids)))
            if ok:
                print(f"[{i}] rank={best_rank} gold={gold_str} :: {q.question_text}")
            else:
                print(f"[{i}] rank=NA gold={gold_str} :: {q.question_text}")
            print(f"     retrieved={retrieved}")

    if n == 0:
        raise RuntimeError("No scorable eval questions (missing gold labels).")

    return Metrics(n=n, recall_at_k=hits / n, mrr=rr_sum / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="toy_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with Session(engine) as session:
        m = compute_metrics(session, eval_set_name=args.eval_set, top_k=args.top_k, verbose=args.verbose)

    print()
    print(f"Eval set: {args.eval_set}")
    print(f"Questions scored: {m.n}")
    print(f"Recall@{args.top_k}: {m.recall_at_k:.3f}")
    print(f"MRR: {m.mrr:.3f}")


if __name__ == "__main__":
    main()