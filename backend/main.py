import os
import json
import logging
import traceback
import csv
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from threading import Lock
from fastapi.middleware.cors import CORSMiddleware


from retrieval.engine import CancerQAEngine
from retrieval.qa_types import QAResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cancer Awareness QA API",
    description="Provides answers to cancer-related questions using FAISS embeddings and LLMs (OncoBot)",
)

FAISS_INDEX_PATH = "./data/cancer_index_checkpoint.faiss"
CHUNKS_PKL_PATH = "./data/cancer_chunks.pkl"
TOP_K = 5
LIVE_RESPONSES_PATH = "./data/live_responses.txt"
EVAL_RESULTS_PATH = "./data/evaluation_results.csv"

file_lock = Lock()

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    confidence: float
    method: str
    used_chunks: List[str] = []


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def save_response_log(entry: dict):
    try:
        line = json.dumps(entry, ensure_ascii=False)
        with file_lock:
            with open(LIVE_RESPONSES_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        logger.exception("Failed to save API response: %s", e)

def log_evaluation_metrics(question, answer, confidence, method, used_chunks):
    fallback_triggered = "RAG" in method or "LLaMA" in method
    with file_lock:
        file_exists = os.path.exists(EVAL_RESULTS_PATH)
        with open(EVAL_RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question","answer","confidence","method","used_chunks","fallback_triggered"])
            if not file_exists or os.stat(EVAL_RESULTS_PATH).st_size == 0:
                writer.writeheader()
            writer.writerow({
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "method": method,
                "used_chunks": "|".join(used_chunks),
                "fallback_triggered": fallback_triggered
            })

try:
    from embeddings import EmbeddingGenerator
    embedder = EmbeddingGenerator()

    from retrieval.retriever import Retriever
    retriever = Retriever(
        faiss_index_path=FAISS_INDEX_PATH,
        chunks_pkl_path=CHUNKS_PKL_PATH,
        top_k=TOP_K
    )

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

        # Save live response log
        log_entry = {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "method": method,
            "used_chunks": used_chunks
        }
        save_response_log(log_entry)

        # log evaluation metrics
        log_evaluation_metrics(question, answer, confidence, method, used_chunks)

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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
