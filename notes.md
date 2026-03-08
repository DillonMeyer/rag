# RAG Project — Dev Notes

### Start full stack
```bash
docker compose up --build
```

### **Stop full stack**

```
docker compose down
```

### **Stop full stack and wipe DB**

```
docker compose down -v
```

---

## **Health + API checks**

### **Health check**

```
curl http://localhost:8000/health
```

### **Ask a question (default: answer only)**

```
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is agentic RAG?"}'
```

### **Ask with citations**

```
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is agentic RAG?","include_citations":true}'
```

### **Ask with hit previews**

```
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is agentic RAG?","include_hits":true}'
```

### **Ask with citations + hit previews**

```
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is agentic RAG?","include_citations":true,"include_hits":true}'
```

### **Swagger UI**

- http://localhost:8000/docs

---

## **Ollama**

### **Pull model into Ollama container**

```
docker exec -it rag-ollama ollama pull llama3.2
```

### **List models in Ollama container**

```
docker exec -it rag-ollama ollama list
```

---

## **Ingestion**

### **Run ingestion in API container**

```
docker exec -it rag-api python -m app.scripts.ingest_one
```

### **Current PDF source folder**

```
data/papers
```

---

## **Evaluation**

### **Run eval in API container**

```
docker exec -it rag-api python -m app.scripts.eval_run --eval-set rag_core_v1 --top-k 5
```

### **Verbose eval**

```
docker exec -it rag-api python -m app.scripts.eval_run --eval-set rag_core_v1 --top-k 5 --verbose
```

### **Current benchmark snapshot**

- Eval set: rag_core_v1
- Questions scored: 27
- Recall@5: ~0.407
- MRR: ~0.355

---

## **PostgreSQL / pgvector**

### **Connect to DB**

```
psql "postgresql://rag:rag@localhost:5432/rag"
```

### **List tables**

```
psql "postgresql://rag:rag@localhost:5432/rag" -c "\dt"
```

### **Show documents**

```
psql "postgresql://rag:rag@localhost:5432/rag" -c "SELECT document_id, title FROM documents LIMIT 20;"
```

### **Clean DB**

```
psql "postgresql://rag:rag@localhost:5432/rag" -c "TRUNCATE chunks, documents RESTART IDENTITY CASCADE;"
```

### **Check pgvector extension**

```
psql "postgresql://rag:rag@localhost:5432/rag" -c "SELECT extname FROM pg_extension WHERE extname='vector';"
```

### **Create IVFFLAT index**

```
psql "postgresql://rag:rag@localhost:5432/rag" -c "CREATE INDEX IF NOT EXISTS idx_chunks_embedding
ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);"
```

### **Analyze chunks after ingestion**

```
psql "postgresql://rag:rag@localhost:5432/rag" -c "ANALYZE chunks;"
```

### **Optional ivfflat probes tuning**

```
psql "postgresql://rag:rag@localhost:5432/rag" -c "SET ivfflat.probes = 10;"
```

### **Rebuild index if needed**

```
psql "postgresql://rag:rag@localhost:5432/rag" -c "REINDEX INDEX idx_chunks_embedding;"
```

---

## **Notes**

- Default /ask returns only answer
- include_citations=true adds citation metadata
- include_hits=true adds retrieved hit previews
- Current eval metrics are tied to the current corpus snapshot
- Broad definitional questions can retrieve bibliography-like chunks from research-paper corpora