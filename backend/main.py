import os
import json
import logging
import traceback
import csv
import re
from typing import List, Set
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from threading import Lock
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .retrieval.engine import CancerQAEngine, ALL_KEYWORDS, map_to_canonical
from .retrieval.qa_types import QAResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cancer Awareness QA API",
    description="Provides answers to cancer-related questions using FAISS embeddings and LLMs (OncoBot)",
)

FAISS_INDEX_PATH = "./data/cancer_index_checkpoint.faiss"
CHUNKS_PKL_PATH = "./data/cancer_chunks.pkl"
TOP_K = config.TOP_K
EVAL_K = TOP_K
LIVE_RESPONSES_PATH = "./data/live_responses.txt"
EVAL_RESULTS_PATH = "./data/evaluation_results.csv"
RELEVANCE_FIELDS = ("cancer_types", "organs_affected", "tumor_characteristics", "treatments")

file_lock = Lock()
chunk_relevance_index = []

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    confidence: float
    method: str
    used_chunks: List[str] = []


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # React frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

def save_response_log(entry: dict):
    try:
        line = json.dumps(entry, ensure_ascii=False)
        with file_lock:
            with open(LIVE_RESPONSES_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        logger.exception("Failed to save API response: %s", e)

def _extract_query_keywords(question: str) -> Set[str]:
    words = re.findall(r"\b\w+\b", question.lower())
    keywords = set()

    for word in words:
        canonical = map_to_canonical(word)
        if canonical in ALL_KEYWORDS:
            keywords.add(canonical)

    q_lower = question.lower()
    for keyword in ALL_KEYWORDS:
        if keyword in q_lower:
            keywords.add(keyword)

    return keywords


def _build_chunk_search_blob(chunk: dict) -> str:
    metadata = chunk.get("metadata") or {}
    values = [chunk.get("text", "").lower()]

    for field in RELEVANCE_FIELDS:
        field_value = metadata.get(field) or []
        if isinstance(field_value, str):
            field_value = [field_value]
        values.extend(str(v).lower() for v in field_value)

    return " ".join(values)


def compute_recall_at_k(question: str, used_chunks: List[str], k: int = EVAL_K) -> float:
    if not used_chunks or not chunk_relevance_index:
        return 0.0

    query_keywords = _extract_query_keywords(question)
    if not query_keywords:
        return 0.0

    relevant_ids = {
        chunk_id
        for chunk_id, searchable_blob in chunk_relevance_index
        if any(keyword in searchable_blob for keyword in query_keywords)
    }
    if not relevant_ids:
        return 0.0

    retrieved_top_k = set(used_chunks[:k])
    hits = len(retrieved_top_k.intersection(relevant_ids))
    recall = hits / len(relevant_ids)
    return round(recall, 6)


def log_evaluation_metrics(question, answer, confidence, method, used_chunks, recall_at_k):
    fallback_triggered = "RAG" in method or "LLaMA" in method
    with file_lock:
        file_exists = os.path.exists(EVAL_RESULTS_PATH)
        with open(EVAL_RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question", "answer", "confidence", "method", "used_chunks",
                "fallback_triggered", "recall_at_k", "k"
            ])
            if not file_exists or os.stat(EVAL_RESULTS_PATH).st_size == 0:
                writer.writeheader()
            writer.writerow({
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "method": method,
                "used_chunks": "|".join(used_chunks),
                "fallback_triggered": fallback_triggered,
                "recall_at_k": recall_at_k,
                "k": EVAL_K
            })

try:
    from .embeddings import EmbeddingGenerator
    embedder = EmbeddingGenerator()

    from .retrieval.retriever import Retriever
    retriever = Retriever(
        faiss_index_path=FAISS_INDEX_PATH,
        chunks_pkl_path=CHUNKS_PKL_PATH,
        top_k=TOP_K
    )
    chunk_relevance_index = [
        (chunk.get("id", str(i)), _build_chunk_search_blob(chunk))
        for i, chunk in enumerate(retriever.chunks)
    ]

    qa_engine = CancerQAEngine(vectordb=retriever, retriever_k=TOP_K)
    logger.info("QA pipeline initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize QA pipeline.")
    raise

os.makedirs(os.path.dirname(LIVE_RESPONSES_PATH) or ".", exist_ok=True)
os.makedirs(os.path.dirname(EVAL_RESULTS_PATH) or ".", exist_ok=True)

if not os.path.exists(LIVE_RESPONSES_PATH):
    with open(LIVE_RESPONSES_PATH, "w") as f:
        f.write("")

@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        qa_result: QAResult = qa_engine.ask(question)
        answer = qa_result.answer
        confidence = getattr(qa_result, "confidence", 0.0)
        method = getattr(qa_result, "method", "")
        used_chunks = [c.id for c in getattr(qa_result, "used_chunks", [])]
        recall_at_k = compute_recall_at_k(question, used_chunks, EVAL_K)

        # Save live response log
        log_entry = {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "method": method,
            "used_chunks": used_chunks,
            "recall_at_k": recall_at_k
        }
        save_response_log(log_entry)

        # log evaluation metrics
        log_evaluation_metrics(
            question=question,
            answer=answer,
            confidence=confidence,
            method=method,
            used_chunks=used_chunks,
            recall_at_k=recall_at_k
        )

        return AskResponse(
            answer=answer,
            confidence=confidence,
            method=method,
            used_chunks=used_chunks
        )

    except Exception as e:
        logger.error("Error in /ask endpoint: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
