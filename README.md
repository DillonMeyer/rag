# Retrieval-Augmented Generation System

A containerized Retrieval-Augmented Generation (RAG) system for answering questions over research papers.

The system retrieves relevant document chunks using pgvector and generates answers using a local LLM served through Ollama.

The project includes ingestion, vector retrieval, answer generation, logging, evaluation, and multi-container deployment.

```bash
Client → API → embed query (BGE-small) → retrieve chunks (pgvector) → generate answer (Ollama)
```

Each stage leaves a record.

Responses include citations pointing to the retrieved sources.

**Example Query**

```bash
curl -X POST http://localhost:8000/ask \
-H "Content-Type: application/json" \
-d '{"question":"What is agentic RAG?"}'
```

**Example Response**

```JSON
{
  "answer": "Agentic RAG refers to a paradigm of retrieval-augmented generation systems that incorporate autonomous agents capable of dynamic decision-making and workflow optimization."
}
```

*Optional debugging flags allow returning citations or retrieved chunks.*

**Key technologies:**

- FastAPI
- BGE-small
- PostgreSQL + pgvector
- Ollama (local LLM inference)
- Docker

## Architecture

The system embeds incoming questions, retrieves relevant document chunks using pgvector, and generates answers with a local LLM.

Query and retrieval metadata are logged to support evaluation and debugging.

**System Flow**
![System Architecture](E2E_RAG.png)

**Database Schema**
![System Architecture](RAG_ERD.png)

### Tables
- `documents`
  - Metadata about each source document.
- `chunks`
  - Each document divided into chunks.
  - Each chunk is embedded with BGE-small and stored with its vector representation.
- `queries`
  - Records each call to the /ask endpoint.
  - Serves as the root event for retrieval and generation logging.
- `retrievals`
  - Records the retrieval step for a query.
  - Stored separately because retrieval settings can vary.
- `retrieval_results`
  - Records which chunks were returned.
  - Assists with inspecting and debugging retrieval.
-  `generations`
  - Records metadata about the LLM generation step.

## Retrieval Pipeline

Documents are chunked into overlapping segments and embedded using the **BGE-small** embedding model.

Embeddings are stored in **PostgreSQL** with **pgvector** and queried using cosine similarity.

Retrieval is configured with:

- IVFFLAT vector indexing
  - Groups embeddings into clusters to reduce the number of vectors scanned during similarity search.
- configurable top-k retrieval
- chunk limits per document
  - Ensures all the top_k results don’t come from a single document.

This allows efficient semantic search over the corpus.

## Retrieval Evaluation

To measure retrieval quality, a small benchmark dataset was created.

**Questions**: 27

**Recall@5**: 0.407 *(whether the relevant chunk appears in the retrieved results)*

**MRR**: 0.355 *(how early the relevant result appears)*

## Design Decisions
pgvector was selected over a separate vector database to keep retrieval, metadata, and logging in one system.

Ollama for local model serving to avoid external API dependencies.

Dockerized services to make deployment reproducible.

## Challenges Encountered

- retrieval frequently surfaced bibliography or reference sections
- embedding similarity does not always correlate with answer relevance
- citation filtering required parsing the model output
- containerizing LLM services introduced resource management challenges

## Limitations

- semantic retrieval can surface bibliography-style chunks
- evaluation dataset is small and tied to the current corpus snapshot
- no reranking stage is implemented

## Future Improvements

- reranking model for improved retrieval quality
- query rewriting
- larger evaluation dataset
- streaming responses
- automated corpus ingestion

## Local Setup

Start the stack:

```bash
docker compose up --build
```

Pull the model:

```bash
docker exec -it rag-ollama ollama pull llama3.2
```

Ingest documents:

```bash
docker exec -it rag-api python -m app.scripts.ingest_one
```