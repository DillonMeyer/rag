from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sqlmodel import Session, select

from app.db import engine
from app import models


def ensure_eval_set(session: Session, name: str) -> models.EvalSet:
    es = session.exec(
        select(models.EvalSet).where(models.EvalSet.name == name)
    ).first()

    if es:
        return es

    es = models.EvalSet(name=name)
    session.add(es)
    session.commit()
    session.refresh(es)
    return es


def chunk_exists(session: Session, chunk_id: int) -> bool:
    result = session.exec(
        select(models.Chunk).where(models.Chunk.chunk_id == chunk_id)
    ).first()
    return result is not None


def question_exists(session: Session, eval_set_id: int, question_text: str) -> bool:
    result = session.exec(
        select(models.EvalQuestion)
        .where(models.EvalQuestion.eval_set_id == eval_set_id)
        .where(models.EvalQuestion.question_text == question_text)
    ).first()
    return result is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--eval-set", required=True, help="Eval set name")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise RuntimeError(f"CSV file not found: {csv_path}")

    with Session(engine) as session:
        eval_set = ensure_eval_set(session, args.eval_set)

        inserted = 0
        skipped = 0

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                question = row["question_text"].strip()
                gold_chunk_id = int(row["gold_chunk_id"])

                if not chunk_exists(session, gold_chunk_id):
                    raise RuntimeError(
                        f"Chunk ID {gold_chunk_id} does not exist."
                    )

                if question_exists(session, eval_set.eval_set_id, question):
                    skipped += 1
                    continue

                q = models.EvalQuestion(
                    eval_set_id=eval_set.eval_set_id,
                    question_text=question,
                    gold_chunk_id=gold_chunk_id,
                )

                session.add(q)
                inserted += 1

        session.commit()

    print()
    print(f"Eval set: {args.eval_set}")
    print(f"Inserted: {inserted}")
    print(f"Skipped (duplicates): {skipped}")


if __name__ == "__main__":
    main()