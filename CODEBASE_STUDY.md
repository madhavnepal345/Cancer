# Cancer Awareness QA Engine

## Overview

This repository contains a FastAPI backend for a cancer-related question answering system.
It implements a retrieval-augmented generation style workflow over medical PDF content, with
FAISS-based vector search, metadata filtering, and a large language model for final response generation.

The project is intended for research and educational use only. It explicitly warns that it is not
medical advice.

## High-Level Flow

1. Medical PDFs are cleaned and chunked into text segments.
2. Chunks are embedded with a SentenceTransformer model.
3. Embeddings are stored in a FAISS index.
4. A user sends a question to the `/ask` endpoint.
5. The backend retrieves top matching chunks from FAISS.
6. The engine filters chunks by metadata and builds a context block.
7. The context is optionally passed through BioBERT and RAG stages.
8. A LLaMA-style text generation model produces the final answer.
9. The answer, confidence, and metrics are logged to disk.

## Main Components

### `backend/main.py`

FastAPI application entry point.

- Exposes the `/ask` endpoint.
- Loads the retriever and QA engine at import time.
- Computes precision, recall, and F1 style metrics for returned chunks.
- Writes live response logs and evaluation CSV rows into `backend/data/`.

Relevant paths:

- [`backend/main.py`](./backend/main.py)

### `backend/retrieval/engine.py`

Core QA orchestration.

- Rejects diagnostic-style questions with a safe refusal.
- Handles a small static knowledge base for canned responses.
- Retrieves chunks from the vector store.
- Filters chunks using metadata signals such as cancer type and organ.
- Builds a prompt for the LLaMA model.
- Optionally uses BioBERT and RAG if passed in.

Relevant paths:

- [`backend/retrieval/engine.py`](./backend/retrieval/engine.py)
- [`backend/retrieval/knowledgebase.py`](./backend/retrieval/knowledgebase.py)

### `backend/retrieval/retriever.py`

FAISS retrieval layer.

- Loads the FAISS index from `backend/data/cancer_index_checkpoint.faiss`.
- Loads the chunk payload from `backend/data/cancer_chunks.pkl`.
- Embeds query text and returns top-k matching chunks.

Relevant path:

- [`backend/retrieval/retriever.py`](./backend/retrieval/retriever.py)

### `backend/embeddings.py`

Embedding and index-building logic.

- Wraps `SentenceTransformer`.
- Generates normalized embeddings.
- Builds and saves FAISS checkpoints in batches.

Relevant path:

- [`backend/embeddings.py`](./backend/embeddings.py)

### `backend/pdf_cleaning.py`

PDF ingestion and chunking pipeline.

- Extracts text from source PDFs with PyMuPDF.
- Cleans noisy PDF text.
- Splits content into sentence-based chunks.
- Adds simple metadata by matching keyword lists.

Relevant path:

- [`backend/pdf_cleaning.py`](./backend/pdf_cleaning.py)

### `backend/docker_entrypoint.py`

Container startup bootstrap.

- Checks whether FAISS artifacts exist.
- If needed, runs PDF cleaning and embedding generation.
- Starts `uvicorn` on port `8000`.

Relevant path:

- [`backend/docker_entrypoint.py`](./backend/docker_entrypoint.py)

## Configuration

Model and retrieval settings live in `backend/config.py` and can be overridden with environment variables.

Defaults:

- `BIOBERT_MODEL=dmis-lab/biobert-base-cased-v1.1-squad`
- `RAG_MODEL=google/flan-t5-base`
- `LLAMA_MODEL=adityak74/medfit-llm-3B`
- `EMBEDDING_MODEL=pritamdeka/S-PubMedBert-MS-MARCO`
- `LLAMA_TEMPERATURE=0.7`
- `TOP_K=5`

Relevant path:

- [`backend/config.py`](./backend/config.py)

## Data Files Expected

The code expects a `backend/data/` directory containing:

- `Medical_book.pdf`
- `Encyclopedia of Cancer, 3rd Edition.pdf`
- `Combined_Cancer_Chunks.json`
- `cancer_index_checkpoint.faiss`
- `cancer_chunks.pkl`
- `live_responses.txt`
- `evaluation_results.csv`

If the PDF inputs or generated artifacts are missing, the system starts in a degraded mode with empty retrieval.

## API Contract

### `POST /ask`

Request body:

```json
{
  "question": "What is lung cancer?"
}
```

Response body:

```json
{
  "answer": "...",
  "confidence": 0.0,
  "method": "BioBERT→RAG→LLaMA",
  "used_chunks": ["chunk-id-1", "chunk-id-2"]
}
```

## Notable Behavior

- Diagnostic questions are blocked with a safety response.
- A small hardcoded knowledge base can answer a few specific trigger phrases.
- Metadata filtering boosts chunks whose metadata matches question keywords.
- Confidence is derived from retrieval similarity and BioBERT score, with a small RAG bonus.
- The current API path does not pass BioBERT or RAG objects into `qa_engine.ask`, so those stages are effectively unused unless wired in later.
- CORS middleware exists but is commented out.

## Build And Runtime

### Docker

The project is designed to run through Docker:

```bash
docker compose up --build
```

The container:

- installs dependencies,
- downloads NLTK `punkt`,
- copies the backend code,
- runs `python -m backend.docker_entrypoint`.

### Local

Typical local flow:

1. Create a virtual environment.
2. Install dependencies from `requirement.txt`.
3. Put source PDFs in `backend/data/`.
4. Generate chunks and FAISS artifacts.
5. Start the API.

## Key Files At A Glance

- [`README.md`](./README.md)
- [`backend/main.py`](./backend/main.py)
- [`backend/config.py`](./backend/config.py)
- [`backend/embeddings.py`](./backend/embeddings.py)
- [`backend/pdf_cleaning.py`](./backend/pdf_cleaning.py)
- [`backend/docker_entrypoint.py`](./backend/docker_entrypoint.py)
- [`backend/retrieval/engine.py`](./backend/retrieval/engine.py)
- [`backend/retrieval/retriever.py`](./backend/retrieval/retriever.py)
- [`backend/retrieval/qa_types.py`](./backend/retrieval/qa_types.py)

## Summary

This is a compact backend-only cancer QA system built around:

- PDF ingestion,
- chunking and metadata extraction,
- FAISS vector retrieval,
- LLM-based answer generation,
- simple logging and offline evaluation.

It is more of a research prototype than a production-ready medical assistant.
