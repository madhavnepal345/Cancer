# Major Changes

## 1. Backend packaging and import cleanup
- Updated backend modules to use package-relative imports instead of mixed root-level imports.
- This makes the backend more reliable when running as a package, from Docker, or from different entry points.

Files touched:
- `backend/main.py`
- `backend/embeddings.py`
- `backend/pdf_cleaning.py`
- `backend/vector_db.py`
- `backend/retrieval/biobert_qa.py`
- `backend/retrieval/engine.py`
- `backend/retrieval/knowledgebase.py`
- `backend/retrieval/retriever.py`
- `backend/retrieval/Rag.py`

## 2. Centralized backend data paths
- Added `Path`/`os.path` based resolution so backend data files are loaded from `backend/data/` consistently.
- This replaces hardcoded relative paths like `./data/...`, which are fragile when the working directory changes.

Examples of affected files:
- `backend/main.py`
- `backend/embeddings.py`
- `backend/pdf_cleaning.py`
- `backend/retrieval/knowledgebase.py`
- `backend/retrieval/retriever.py`

## 3. Retrieval pipeline improvements
- The QA engine now has cleaner keyword handling and metadata-based chunk filtering.
- The retrieval flow still supports:
  - BioBERT extractive QA
  - RAG generation
  - final LLaMA-based reasoning
- Diagnostic-style questions continue to be blocked with a safe medical disclaimer response.

Files touched:
- `backend/retrieval/engine.py`
- `backend/retrieval/qa_types.py`
- `backend/retrieval/biobert_qa.py`
- `backend/retrieval/Rag.py`
- `backend/retrieval/retriever.py`
- `backend/retrieval/knowledgebase.py`

## 4. Documentation updates
- Expanded the README with Docker usage instructions.
- Added notes about where the FAISS index, chunk pickle, and response logs live.
- Added optional model configuration examples for `.env`.

Files touched:
- `README.md`

## 5. Dependency updates
- Added the core runtime dependencies needed by the backend and retrieval stack.

Added packages:
- `faiss-cpu`
- `sentence-transformers`
- `torch`
- `fastapi`
- `uvicorn`
- `accelerate>=0.26.0`
- `python-dotenv`

Files touched:
- `requirement.txt`

## 6. Ignore rules
- Expanded `.gitignore` to cover common local artifacts and generated files.
- This includes:
  - Python caches and test/tool caches
  - local environment files
  - frontend build output and `node_modules`
  - backend model/checkpoint artifacts

Files touched:
- `.gitignore`

## 7. Frontend worktree state
- The current worktree shows a large set of frontend files as removed in Git status, including `frontend/node_modules/`, Vite cache files, and many tracked frontend source files.
- If that cleanup was intentional, it should be committed separately.
- If it was accidental, it should be reviewed before finalizing the repo state.

## Summary
- The main theme of the change set is backend hardening: cleaner imports, safer path handling, more robust retrieval behavior, and better packaging for Docker/local runs.
- The repo also got documentation, dependency, and ignore-rule updates to support that workflow.
