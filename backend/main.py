# app/main.py
import os
import json
import logging
import traceback
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from threading import Lock
import rouge_score

from retrieval.engine import CancerQAEngine
from retrieval.qa_types import QAResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cancer Awareness QA API",
    description="Provides answers to cancer-related questions using FAISS embeddings and LLMs (OncoBot)",
    version="1.0.0",
)

FAISS_INDEX_PATH = "./data/cancer_index_checkpoint.faiss"
CHUNKS_PKL_PATH = "./data/cancer_chunks.pkl"
TOP_K = 5

EVAL_DATA_PATH = "./data/eval_data.json"            
LIVE_LOG_PATH = "./data/live_eval_log.jl"          
ENABLE_BLEU_ROUGE = True                            

class AskRequest(BaseModel):
    question: str


class Metrics(BaseModel):
    accuracy: Optional[int] = None            
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    bleu: Optional[float] = None
    rouge_1_f: Optional[float] = None
    rouge_l_f: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    confidence: float
    method: str
    used_chunks: List[str] = []
    metrics: Optional[Metrics] = None


file_lock = Lock()
eval_data_map: Dict[str, str] = {}   

_sentence_bleu = None
_Rouge = None
if ENABLE_BLEU_ROUGE:
    try:
        from nltk.translate.bleu_score import sentence_bleu
        _sentence_bleu = sentence_bleu
    except Exception:
        logger.warning("nltk BLEU not available. BLEU will be skipped.")
        _sentence_bleu = None

    try:
        from rouge import Rouge
        _Rouge = Rouge
    except Exception:
        try:
            from rouge_score import rouge_scorer
            _Rouge = rouge_scorer  
        except Exception:
            logger.warning("ROUGE not available. ROUGE will be skipped.")
            _Rouge = None


def ensure_data_paths():
    os.makedirs(os.path.dirname(EVAL_DATA_PATH) or ".", exist_ok=True)
    if not os.path.exists(LIVE_LOG_PATH):
        with open(LIVE_LOG_PATH, "w") as f:
            f.write("") 


def load_eval_data():
    global eval_data_map
    if not os.path.exists(EVAL_DATA_PATH):
        logger.warning("Evaluation dataset not found at %s. Live evaluation will be disabled.", EVAL_DATA_PATH)
        eval_data_map = {}
        return

    try:
        with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            eval_data_map = {item["question"].strip().lower(): item["answer"].strip() for item in data}
            logger.info("Loaded %d eval Q/A pairs from %s", len(eval_data_map), EVAL_DATA_PATH)
    except Exception as e:
        logger.exception("Failed to load eval data: %s", e)
        eval_data_map = {}


def token_overlap_metrics(gt: str, pred: str):
    """Compute token-overlap precision, recall, f1 (simple and robust)."""
    gt_tokens = {t for t in gt.lower().split() if t}
    pred_tokens = {t for t in pred.lower().split() if t}

    if not gt_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0

    tp = len(gt_tokens & pred_tokens)
    precision = tp / len(pred_tokens) if pred_tokens else 0.0
    recall = tp / len(gt_tokens) if gt_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_metrics_if_available(question: str, predicted: str) -> Dict:
    """Return metrics dict or empty dict if no ground-truth for the question."""
    q_lower = question.strip().lower()
    gt = eval_data_map.get(q_lower)
    if not gt:
        return {}

    metrics = {}
   
    if gt.strip().lower() in predicted.strip().lower() or predicted.strip().lower() in gt.strip().lower():
        metrics["accuracy"] = 1
    else:
        metrics["accuracy"] = 0

    precision, recall, f1 = token_overlap_metrics(gt, predicted)
    metrics["precision"] = round(precision, 4)
    metrics["recall"] = round(recall, 4)
    metrics["f1"] = round(f1, 4)

    if _sentence_bleu is not None:
        try:
            bleu = _sentence_bleu([gt.split()], predicted.split())
            metrics["bleu"] = round(float(bleu), 4)
        except Exception:
            metrics["bleu"] = None

    if _Rouge is not None:
        try:
            if hasattr(_Rouge, "__call__") or _Rouge.__name__ == "Rouge":
                if isinstance(_Rouge, type):
                    r = _Rouge()
                else:
                    r = _Rouge()
                rouge_scores = r.get_scores(predicted, gt)[0]
                metrics["rouge_1_f"] = round(rouge_scores["rouge-1"]["f"], 4)
                metrics["rouge_l_f"] = round(rouge_scores["rouge-l"]["f"], 4)
            else:
                # rouge_score.scorer interface
                scorer = _Rouge.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
                scores = scorer.score(gt, predicted)
                metrics["rouge_1_f"] = round(scores["rouge1"].fmeasure, 4)
                metrics["rouge_l_f"] = round(scores["rougeL"].fmeasure, 4)
        except Exception:
            metrics.setdefault("rouge_1_f", None)
            metrics.setdefault("rouge_l_f", None)

    return metrics


def append_log(entry: dict):
    """Append a JSON object as a line in LIVE_LOG_PATH in a thread-safe manner."""
    try:
        line = json.dumps(entry, ensure_ascii=False)
        with file_lock:
            with open(LIVE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        logger.exception("Failed to append log entry: %s", e)


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

ensure_data_paths()
load_eval_data()


# ---------- Routes ----------
@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        qa_result: QAResult = qa_engine.ask(question)   # your existing pipeline call
        answer = qa_result.answer
        confidence = qa_result.confidence if hasattr(qa_result, "confidence") else 0.0
        method = qa_result.method if hasattr(qa_result, "method") else ""
        used_chunks = [c.id for c in qa_result.used_chunks] if getattr(qa_result, "used_chunks", None) else []

        # Compute metrics if we have ground-truth for this question
        metrics_raw = compute_metrics_if_available(question, answer)

        # Prepare log entry (including input, output, metrics, model info, confidence)
        log_entry = {
            "question": question,
            "ground_truth": eval_data_map.get(question.strip().lower()),
            "predicted": answer,
            "confidence": confidence,
            "method": method,
            "used_chunks": used_chunks,
            "metrics": metrics_raw,
            "error": None,
        }
        append_log(log_entry)

        # Return response including metrics (converted to Metrics model)
        metrics_model = Metrics(**metrics_raw) if metrics_raw else None

        return AskResponse(
            answer=answer,
            confidence=confidence,
            method=method,
            used_chunks=used_chunks,
            metrics=metrics_model
        )

    except Exception as e:
        logger.error("Error in /ask endpoint: %s", e)
        traceback.print_exc()

        # Log the error for this request
        error_log = {
            "question": question,
            "predicted": None,
            "confidence": None,
            "method": None,
            "used_chunks": None,
            "metrics": None,
            "error": str(e)
        }
        append_log(error_log)

        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/health")
def health_check():
    try:
        count = len(retriever.chunks)
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
