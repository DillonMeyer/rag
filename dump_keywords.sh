#!/bin/bash

KEYWORDS=(
"agentic"
"reflection"
"planning"
"multi-agent"
"retrieval"
"rerank"
"multi-hop"
"evaluation"
"recall"
"mrr"
"latency"
"embedding"
"cosine"
"vector"
"hallucination"
"domain"
"graph"
"benchmark"
"adaptation"
"collaboration"
)

for kw in "${KEYWORDS[@]}"; do
  echo "Dumping keyword: $kw"

  psql "postgresql://rag:rag@localhost:5432/rag" \
    --tuples-only \
    --no-align \
    -c "
SELECT
  c.chunk_id,
  c.document_id,
  d.title,
  c.chunk_index,
  left(encode(c.text::bytea, 'escape'), 400)
FROM chunks c
JOIN documents d ON d.document_id = c.document_id
WHERE c.text ILIKE '%$kw%'
ORDER BY c.document_id, c.chunk_index
LIMIT 50;
" > "dump/dump_${kw}.txt"

done

echo "Done."