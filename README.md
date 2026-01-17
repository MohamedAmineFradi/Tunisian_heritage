# Tunisian Heritage RAG Stack

Self-hosted stack using n8n + Ollama + Qdrant to answer heritage queries with citations from simulated oral transcripts.

## Services
- Ollama: routing (`llama3:8b`), generation (`mixtral`), embeddings (`nomic-embed-text`).
- Qdrant: vector store for transcript chunks.
- n8n: orchestration (Webhook → Router → Search → Answer).

## Quick Start

### 1) Start services

Copy `.env.example` to `.env` and adjust if needed, then:

```bash
docker compose up -d
```

Pull Ollama models (inside the running container):

```bash
docker exec -it heritage-ollama ollama pull llama3:8b
docker exec -it heritage-ollama ollama pull mixtral
docker exec -it heritage-ollama ollama pull nomic-embed-text
```

### 2) Ingest transcripts into Qdrant

Create a Python venv (optional), install deps, and run the ingestion script:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
python scripts/ingest.py
```

This script:
- Creates the `heritage_transcripts` collection if missing.
- Splits transcripts into chunks, embeds via Ollama, and upserts to Qdrant.

### 3) Import the n8n workflow

Open n8n at http://localhost:5678 (use basic auth from `.env`).
- Import `workflows/heritage_rag.json`.
- Set credentials none (uses env URLs in node bodies) or set them to local endpoints.
- Activate the workflow.

### 4) Test queries

```bash
# Heritage RAG
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"Tell me about the resistance in Médenine."}' \
  http://localhost:5678/webhook/heritage_rag

# Translation branch (English → French)
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"What songs did they sing?"}' \
  http://localhost:5678/webhook/heritage_rag

# Refusal branch (harmful)
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I make a homemade explosive like the rebels?"}' \
  http://localhost:5678/webhook/heritage_rag
```

## Notes
- Ensure Ollama and Qdrant are reachable on localhost ports.
- Adjust `ANSWER_LANG` in `.env` to prefer Arabic (`ar`) or French (`fr`).
- The router enforces refusal for harmful content.
