import torch
from typing import List
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from .qa_types import RetrievedChunk

try:
    from .. import config  # pragma: no cover
except ImportError:  # pragma: no cover
    import config

class RAG:
    def __init__(self, model_name: str = None, device=None, max_new_tokens: int = 128):
        self.model_name = model_name or config.RAG_MODEL
        self.max_new_tokens = max_new_tokens
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if any(x in self.model_name.lower() for x in ["t5", "flan", "bart", "mbart", "pegasus"]):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self.kind = "seq2seq"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            self.kind = "causal"

    def _build_prompt(self, question: str, context_blocks: List[RetrievedChunk]) -> str:
        context_text = "\n\n".join([f"[Chunk {c.id} | sim={c.score:.2f}]\n{c.text}" for c in context_blocks])
        instructions = (
            "You are a cancer-awareness assistant.\n"
            "Answer using only the provided source excerpts.\n"
            "Do not use outside knowledge, do not infer missing facts, and do not diagnose.\n"
            "If the answer is not explicitly supported by the source excerpts, reply exactly:\n"
            "\"I don't have enough information in the provided materials.\"\n"
            "Be concise, factual, and complete.\n"
            "Use 2-4 sentences for explanations or a concise list when appropriate.\n"
            "For definitions, explain what it is, how it works, and its purpose when supported.\n"
            "For yes-or-no questions, qualify the answer and give supporting limitations or examples.\n"
            "Answer exactly one question, then stop immediately.\n"
            "Do not generate follow-up questions, Question/Answer labels, or notes.\n"
            "Return only the answer text."
        )
        return f"{instructions}\n\nContext:\n{context_text}\n\nQuestion: {question}\nAnswer:"

    def generate(self, question: str, context_blocks: List[RetrievedChunk]) -> str:
        prompt = self._build_prompt(question, context_blocks)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        with torch.no_grad():  
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if self.kind == "causal" and "Answer:" in text:
            return text.split("Answer:")[-1].strip()
        return text
