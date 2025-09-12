# app/main.py
import os
import traceback
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List

from vector_db import VectorDB
from retrieval.engine import CancerQAEngine
from retrieval.retriever import Retriever
from retrieval.Rag import RAG
from retrieval.qa_types import QAResult, RetrievedChunk


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cancer_qa_api")


app = FastAPI(
    title="Cancer Awareness QA API",
    description="Provides answers to cancer-related questions using PDF-based knowledge and LLMs",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    confidence: float
    method: str
    used_chunks: List[str] = []


DB_PATH = "./chroma_db"

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
async def ask_question(req: AskRequest, request: Request):
    question = req.question.strip()
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Received question from {client_ip}: {question}")

    if not question:
        logger.warning("Empty question received")
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        qa_result: QAResult = qa_engine.ask(question)
        logger.info(f"QA Result -> method: {qa_result.method}, confidence: {qa_result.confidence}")

        used_chunks_ids = [c.id for c in qa_result.used_chunks] if qa_result.used_chunks else []
        logger.debug(f"Used chunk IDs: {used_chunks_ids}")

        return AskResponse(
            answer=qa_result.answer,
            confidence=qa_result.confidence,
            method=qa_result.method,
            used_chunks=used_chunks_ids
        )
    except Exception as e:
        logger.error("Error processing question: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/reindex")
async def reindex_documents():
    try:
        vectordb.auto_load_documents(search_paths=["./data/"])
        logger.info("Documents reindexed successfully.")
        return {"status": "success", "message": "Documents reindexed."}
    except Exception as e:
        logger.error("Error reindexing documents: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to reindex documents.")

@app.get("/health")
async def health_check():
    try:
        count = vectordb.collection.count()
        logger.info(f"Health check: collection_count={count}")
        return {"status": "ok", "collection_count": count}
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {"status": "error", "collection_count": 0}


@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI Cancer QA API is starting up.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="debug")
