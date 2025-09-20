# app/main.py
import os
import traceback
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle

from embeddings import EmbeddingGenerator, build_and_save_faiss
from retrieval.engine import CancerQAEngine
from retrieval.qa_types import QAResult, RetrievedChunk
from retrieval.retriever import Retriever  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cancer Awareness QA API",
    description="Provides answers to cancer-related questions using FAISS embeddings and LLMs",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    confidence: float
    method: str
    used_chunks: List[str] = []


# --- Initialize FAISS embedding & retriever ---
DATA_PKL = "./data/cancer_chunks.pkl"
TOP_K = 5

try:
    # Load prebuilt FAISS index + chunks
    with open(DATA_PKL, "rb") as f:
        faiss_data = pickle.load(f)

    index = faiss_data["index"]
    chunks = faiss_data["chunks"]

    # Embedding generator
    embedder = EmbeddingGenerator()

    # Retriever wrapper
    retriever = Retriever(faiss_pkl_path=DATA_PKL, top_k=TOP_K)

    # QA Engine
    qa_engine = CancerQAEngine(vectordb=retriever, retriever_k=TOP_K)

    logger.info("QA pipeline initialized successfully.")

except Exception as e:
    logger.error("Failed to initialize QA pipeline: %s", e)
    traceback.print_exc()
    raise e


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        qa_result: QAResult = qa_engine.ask(question)
        return AskResponse(
            answer=qa_result.answer,
            confidence=qa_result.confidence,
            method=qa_result.method,
            used_chunks=[c.id for c in qa_result.used_chunks] if qa_result.used_chunks else []
        )
    except Exception as e:
        logger.error("Error in /ask endpoint: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/health")
def health_check():
    try:
        count = len(chunks)
        return {"status": "ok", "collection_count": count}
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {"status": "error", "collection_count": 0}


@app.on_event("startup")
def startup_event():
    logger.info("FastAPI Cancer QA API is starting up.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
