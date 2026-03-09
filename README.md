# Retrieval-Augmented Generation System

A containerized Retrieval-Augmented Generation (RAG) system for answering questions over research papers.

The system retrieves relevant document chunks using pgvector and generates grounded answers using a local LLM served through Ollama. Answers include citation references to the retrieved sources.  

<br>

## Features

```
• pgvector semantic retrieval
• citation-grounded answers
• evaluation harness (Recall@k / MRR)
• containerized deployment
• local LLM inference with Ollama
```

<br>

## Architecture

### End-to-End Data Flow
![System Architecture](E2E_RAG.png)

### ERD
![System Architecture](RAG_ERD.png)

<br>

## Example Query

```
curl -X POST http://localhost:8000/ask \
-H "Content-Type: application/json" \
-d '{"question":"What is agentic RAG?"}'
```

<br>

## Example Response

```
{
  "answer": "Agentic RAG refers to a paradigm of retrieval-augmented generation systems that incorporate autonomous agents capable of dynamic decision-making and workflow optimization."
}
```

Optional debugging flags allow returning citations or retrieved chunks.

<br>

## Retrieval Pipeline

Documents are chunked into overlapping segments and embedded using the **BGE-small** embedding model.

Embeddings are stored in **PostgreSQL with pgvector** and queried using cosine similarity.

To improve retrieval performance, the system uses:

- IVFFLAT vector indexing
- configurable top-k retrieval
- chunk limits per document

This allows efficient semantic search over the document corpus while keeping the system simple and reproducible.

<br>

## Evaluation

To measure retrieval quality, a small benchmark dataset was created.

```
Eval set: rag_core_v1
Questions: 27

Recall@5: 0.407
MRR: 0.355
```

Metrics include:

- **Recall@k** – whether the relevant chunk appears in the retrieved results
- **MRR (Mean Reciprocal Rank)** – how early the relevant result appears

This provides a basic signal of retrieval quality and allows experimentation with indexing parameters.

<br>

## Tech Stack

| Component | Technology |
| --- | --- |
| API | FastAPI |
| Vector database | PostgreSQL + pgvector |
| Embeddings | BGE-small |
| LLM | Llama3.2 via Ollama |
| Infrastructure | Docker |
| Hosting | Northflank |

The system is fully containerized to ensure reproducible environments and avoid dependency drift.

<br>

## Engineering Decisions

**PostgreSQL + pgvector**

Chosen for simplicity and strong ecosystem support. Using pgvector allows vector search without introducing a separate vector database.

**Local LLM via Ollama**

Using Ollama allows running local models without external API dependencies while keeping the deployment relatively simple.

**Containerized stack**

Docker ensures the system runs consistently across local development and cloud deployment.

<br>

## Challenges Encountered

Several practical issues emerged during development:

- retrieval frequently surfaced bibliography or reference sections
- embedding similarity does not always correlate with answer relevance
- citation filtering required parsing the model output
- containerizing LLM services introduced resource management challenges

These issues highlight common limitations in simple RAG pipelines.

<br>

## Limitations

Current limitations include:

- semantic retrieval can surface bibliography-style chunks
- evaluation dataset is small and tied to the current corpus snapshot
- no reranking stage is implemented

<br>

## Future Improvements

Possible improvements include:

- reranking model for improved retrieval quality
- query rewriting
- larger evaluation dataset
- streaming responses
- automated corpus ingestion

<br>

## Deployment

The system runs as a multi-container stack:

```
API container
PostgreSQL + pgvector container
Ollama model server
```

The stack can be deployed using Docker Compose and hosted on platforms such as Northflank.