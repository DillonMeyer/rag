# RAG Project — Dev Notes

## pgvector — IVFFLAT indexing

### Check extension
```bash
psql "postgresql://rag:rag@localhost:5432/rag" -c "SELECT extname FROM pg_extension WHERE extname='vector';"
```

### Create IVFFLAT index (cosine distance)
```bash
psql "postgresql://rag:rag@localhost:5432/rag" -c "CREATE INDEX IF NOT EXISTS idx_chunks_embedding
ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);"
```

### After ingestion (recommended)
```bash
psql "postgresql://rag:rag@localhost:5432/rag" -c "ANALYZE chunks;"
```

### Optional tuning (per session)
Higher = better recall, slower query.
```bash
psql "postgresql://rag:rag@localhost:5432/rag" -c "SET ivfflat.probes = 10;"
```

### Rebuild index (only if needed)
```bash
psql "postgresql://rag:rag@localhost:5432/rag" -c "REINDEX INDEX idx_chunks_embedding;"
```

### Run evaluation
```bash
python -m app.scripts.eval_run \
  --eval-set rag_core_v1 \
  --top-k 5 \
  --verbose
```

## API

### Start API
```bash
uvicorn app.main:app --reload
```

### Test /ask
```bash
curl -X POST "http://127.0.0.1:8000/ask"   -H "Content-Type: application/json"   -d '{"question":"What is agentic RAG?","top_k":5}'
```

Swagger UI:
- http://127.0.0.1:8000/docs

---

## Docker

List containers:
```bash
docker ps
```

Start:
```bash
docker compose up -d
```

Stop:
```bash
docker compose down
```

Wipe DB volume:
```bash
docker compose down -v
```

---

## PostgreSQL

Connect:
```bash
psql "postgresql://rag:rag@localhost:5432/rag"
```

List tables:
```bash
psql "postgresql://rag:rag@localhost:5432/rag" -c "\dt"
```

Describe schema:
```bash
psql "postgresql://rag:rag@localhost:5432/rag" -c "\d documents"
```

Peek rows:
```bash
psql "postgresql://rag:rag@localhost:5432/rag" -c "SELECT document_id, title FROM documents LIMIT 5;"
```

Clean DB:
```bash
psql "postgresql://rag:rag@localhost:5432/rag" -c "TRUNCATE chunks, documents RESTART IDENTITY CASCADE;"
```

---

## Scripts

Run ingestion:
```bash
python -m app.scripts.ingest_arxiv
```

Run eval:
```bash
python -m app.scripts.eval_run --eval-set toy_v1 --top-k 5 --verbose
```

Load eval questions from csv:
```bash
python -m app.scripts.load_eval_csv \
  --csv rag_core_v1.csv \
  --eval-set rag_core_v1
```

to do
update deps and bundle with docker for deployment - local llm, db build?, gold quetsions?, start ollama

What to implement (practical)

1) Update schemas

Add to AskResponse:
	•	answer: str | None
	•	citations: list[Citation] | None

Where Citation includes:
	•	ref: int (1..N)
	•	chunk_id
	•	document_id
	•	title
	•	source

2) Implement ollama_generate(prompt, model, max_tokens, temperature)

Use requests (simplest) or httpx.

3) Wire it into /ask

If req.generate:
	•	build prompt from hits
	•	call ollama
	•	return answer + citations