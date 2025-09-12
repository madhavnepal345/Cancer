import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from retrieval.qa_types import RetrievedChunk

class RAG:
    def __init__(self, model_name: str = "google/flan-t5-base", device=None, max_new_tokens: int = 256):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if any(x in model_name.lower() for x in ["t5", "flan", "bart", "mbart", "pegasus"]):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.kind = "seq2seq"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.kind = "causal"

    def _build_prompt(self, question: str, context_blocks: List[RetrievedChunk]) -> str:
        context_text = "\n\n".join([f"[Chunk {c.id} | sim={c.score:.2f}]\n{c.text}" for c in context_blocks])
        instructions = (
            "You are a careful cancer-awareness assistant. Answer the question using ONLY the provided context. "
            "If the answer is not present, say 'I don't have enough information in the provided materials.' "
            "Be concise, factual, and non-diagnostic."
        )
        return f"{instructions}\n\nContext:\n{context_text}\n\nQuestion: {question}\nAnswer:"

    def generate(self, question: str, context_blocks: List[RetrievedChunk]) -> str:
        prompt = self._build_prompt(question, context_blocks)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        with torch.no_grad():  # CPU-friendly
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if self.kind == "causal" and "Answer:" in text:
            return text.split("Answer:")[-1].strip()
        return text
