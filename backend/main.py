# app/main.py
import os
import traceback
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from vector_db import VectorDB
from retrieval.engine import CancerQAEngine
from retrieval.retriever import Retriever
from retrieval.Rag import RAG
from retrieval.qa_types import QAResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cancer Awareness QA API",
    description="Provides answers to cancer-related questions using PDF-based knowledge and LLMs",
    version="1.0.0",
)



DB_PATH = "./chroma_db"
JSON_PATH = "./data/Preprocessed_Medical_Book.json"

try:
    vectordb = VectorDB(persist_directory=DB_PATH, collection_name="cancer_awareness", auto_load=True)
    retriever = Retriever(vectordb=vectordb, top_k=5)
    qa_engine = CancerQAEngine(vectordb=vectordb, retriever_k=5)
    rag_model = RAG(model_name="google/flan-t5-base")
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
        used_chunks = [c.id for c in qa_result.used_chunks] if qa_result.used_chunks else []

        return AskResponse(
            answer=qa_result.answer,
            confidence=qa_result.confidence,
            method=qa_result.method,
            used_chunks=used_chunks
        )
    except Exception as e:
        logger.error("Error in /ask endpoint: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/reindex")
def reindex_documents():
    try:
        vectordb.auto_load_documents(search_paths=["./data/"])
        return {"status": "success", "message": "Documents reindexed."}
    except Exception as e:
        logger.error("Error in /reindex endpoint: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to reindex documents.")

@app.get("/health")
def health_check():
    try:
        count = vectordb.collection.count()
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
