from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import os

try:
    from backend import config  # pragma: no cover
except ImportError:  # pragma: no cover
    import config

class BioBertQA:
    def __init__(self, model_name=None, device=-1):
        model_path = model_name or config.BIOBERT_MODEL

        self.qa = pipeline(
            "question-answering",
            model=AutoModelForQuestionAnswering.from_pretrained(model_path),
            tokenizer=AutoTokenizer.from_pretrained(model_path,use_fast=False),
            device=device
        )

    def answer(self, question: str, context: str):
        
        #Returns a dict with 'answer' and 'score'. Handles list outputs safely.
      
        if not context.strip():
            return {"answer": "", "score": 0.0}

        qa_out = self.qa(question=question, context=context)

        # Handle list outputs from some pipelines
        if isinstance(qa_out, list):
            if len(qa_out) == 0:
                return {"answer": "", "score": 0.0}
            qa_out = qa_out[0]

        if not isinstance(qa_out, dict):
            return {"answer": "", "score": 0.0}

        return {"answer": qa_out.get("answer", ""), "score": float(qa_out.get("score", 0.0))}
